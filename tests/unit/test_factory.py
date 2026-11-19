import pytest
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from lead_qualification_chatbot.config import Settings
from lead_qualification_chatbot.exceptions import ConfigError
from lead_qualification_chatbot.llm.factory import build_chat_model


def cfg(**kw):
    base = {"llm_provider": "ollama", "llm_model": "llama3.1", **kw}
    return Settings(_env_file=None, **base)


def test_ollama_needs_no_key():
    assert isinstance(build_chat_model(cfg()), ChatOllama)


def test_hosted_provider_requires_key():
    with pytest.raises(ConfigError):
        build_chat_model(cfg(llm_provider="openai"))


def test_openai():
    assert isinstance(
        build_chat_model(cfg(llm_provider="openai", llm_api_key="sk-test")), ChatOpenAI
    )


def test_openrouter_uses_compat_base_url():
    model = build_chat_model(cfg(llm_provider="openrouter", llm_api_key="k"))
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
