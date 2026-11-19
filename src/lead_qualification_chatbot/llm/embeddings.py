from langchain_core.embeddings import Embeddings

from lead_qualification_chatbot.config import Settings
from lead_qualification_chatbot.exceptions import ConfigError

_SUPPORTED = ("openai", "gemini", "ollama")


def _provider(cfg: Settings) -> str:
    provider = cfg.embedding_provider or cfg.llm_provider
    if provider not in _SUPPORTED:
        raise ConfigError(
            f"Set EMBEDDING_PROVIDER to one of {', '.join(_SUPPORTED)} "
            f"(LLM_PROVIDER '{cfg.llm_provider}' has no embeddings)"
        )
    return provider


def _model(cfg: Settings, provider: str) -> str:
    if cfg.embedding_model:
        return cfg.embedding_model
    defaults = {"openai": "text-embedding-3-small", "ollama": "nomic-embed-text"}
    if provider not in defaults:
        raise ConfigError(f"EMBEDDING_MODEL is required for provider '{provider}'")
    return defaults[provider]


def embedding_id(cfg: Settings) -> str:
    """Stable id of the embedding setup; part of the Chroma collection name."""
    provider = _provider(cfg)
    return f"{provider}:{_model(cfg, provider)}"


def build_embeddings(cfg: Settings) -> Embeddings:
    provider = _provider(cfg)
    model = _model(cfg, provider)
    key = cfg.embedding_api_key or cfg.llm_api_key
    if provider != "ollama" and not key.get_secret_value():
        raise ConfigError(f"EMBEDDING_API_KEY (or LLM_API_KEY) is required for '{provider}'")

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=model, base_url=cfg.embedding_base_url or "http://localhost:11434"
        )
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=key)
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(  # type: ignore[call-arg]
        model=model,
        api_key=key,
        base_url=cfg.embedding_base_url,
        timeout=cfg.llm_timeout_s,
        max_retries=0,
    )
