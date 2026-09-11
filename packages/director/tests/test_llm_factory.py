"""Factory tests: provider selection yields the right client (or None)."""

from __future__ import annotations

from director.llm import GeminiLLMClient, LLMClient, OpenAICompatClient, get_llm_client
from director.settings import DirectorSettings


def test_provider_none_returns_none() -> None:
    settings = DirectorSettings(llm_provider="none")
    assert get_llm_client(settings) is None


def test_provider_gemini_returns_gemini_adapter() -> None:
    settings = DirectorSettings(gemini_api_key=None)
    client = get_llm_client(settings)
    assert isinstance(client, GeminiLLMClient)
    assert isinstance(client, LLMClient)  # runtime-checkable protocol


def test_provider_openai_compatible_returns_compat_client() -> None:
    settings = DirectorSettings(llm_provider="openai_compatible", llm_api_key=None)
    client = get_llm_client(settings)
    assert isinstance(client, OpenAICompatClient)
    assert isinstance(client, LLMClient)


def test_default_provider_is_gemini() -> None:
    assert DirectorSettings().llm_provider == "gemini"
