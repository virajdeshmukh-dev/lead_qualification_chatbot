import pytest
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from lead_qualification_chatbot.config import Settings
from lead_qualification_chatbot.exceptions import ConfigError
from lead_qualification_chatbot.llm.embeddings import build_embeddings, embedding_id


def cfg(**kw):
    base = {"llm_provider": "ollama", "llm_model": "llama3.1", **kw}
    return Settings(_env_file=None, **base)


def test_ollama_default_model():
    assert isinstance(build_embeddings(cfg()), OllamaEmbeddings)
    assert embedding_id(cfg()) == "ollama:nomic-embed-text"


def test_openai_needs_key_and_uses_default_model():
    with pytest.raises(ConfigError):
        build_embeddings(cfg(llm_provider="openai"))
    c = cfg(llm_provider="openai", llm_api_key="sk-test")
    assert isinstance(build_embeddings(c), OpenAIEmbeddings)
    assert embedding_id(c) == "openai:text-embedding-3-small"


def test_provider_without_embeddings_must_be_set_explicitly():
    with pytest.raises(ConfigError):
        build_embeddings(cfg(llm_provider="groq", llm_api_key="k"))
    c = cfg(llm_provider="groq", llm_api_key="k", embedding_provider="ollama")
    assert isinstance(build_embeddings(c), OllamaEmbeddings)


def test_gemini_requires_explicit_model():
    with pytest.raises(ConfigError):
        build_embeddings(cfg(llm_provider="gemini", llm_api_key="k"))
