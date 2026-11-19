from langchain_core.language_models.chat_models import BaseChatModel

from lead_qualification_chatbot.config import Settings
from lead_qualification_chatbot.exceptions import ConfigError

# OpenAI-compatible endpoints reached through langchain-openai.
_COMPAT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
}


def build_chat_model(cfg: Settings) -> BaseChatModel:
    """Build the provider chat model. SDK-level retries are off: tenacity owns retries."""
    provider = cfg.llm_provider
    if provider != "ollama" and not cfg.llm_api_key.get_secret_value():
        raise ConfigError(f"LLM_API_KEY is required for provider '{provider}'")

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=cfg.llm_model,
            base_url=cfg.llm_base_url or "http://localhost:11434",
            temperature=cfg.llm_temperature,
            client_kwargs={"timeout": cfg.llm_timeout_s},
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=cfg.llm_model,
            google_api_key=cfg.llm_api_key,
            temperature=cfg.llm_temperature,
            timeout=cfg.llm_timeout_s,
            max_retries=0,
        )
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(  # type: ignore[call-arg]
            model=cfg.llm_model,
            api_key=cfg.llm_api_key,
            temperature=cfg.llm_temperature,
            timeout=cfg.llm_timeout_s,
            max_retries=0,
        )
    if provider == "openai" or provider in _COMPAT_BASE_URLS:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.llm_model,
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_base_url or _COMPAT_BASE_URLS.get(provider),
            temperature=cfg.llm_temperature,
            timeout=cfg.llm_timeout_s,
            max_retries=0,
        )
    raise ConfigError(f"Unsupported LLM_PROVIDER '{provider}'")
