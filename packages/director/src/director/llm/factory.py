"""Factory: resolve the configured LLM provider into a client (or None)."""

from __future__ import annotations

from ..settings import DirectorSettings
from .base import LLMClient
from .gemini import GeminiLLMClient
from .openai_compat import OpenAICompatClient


def get_llm_client(settings: DirectorSettings) -> LLMClient | None:
    """Build the client for ``settings.llm_provider``; None means offline mode."""
    if settings.llm_provider == "none":
        return None
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatClient(settings=settings)
    return GeminiLLMClient(settings=settings)


__all__ = ["get_llm_client"]
