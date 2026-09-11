"""Provider-agnostic LLM layer (plan.md Phase 5).

Agents depend only on :class:`LLMClient`; providers plug in via
:func:`get_llm_client`. ``--fast`` / provider ``none`` yield None, keeping the
deterministic offline paths working with no API key.
"""

from __future__ import annotations

from .base import LLMClient, LLMError
from .factory import get_llm_client
from .gemini import GeminiLLMClient
from .openai_compat import OpenAICompatClient

__all__ = [
    "GeminiLLMClient",
    "LLMClient",
    "LLMError",
    "OpenAICompatClient",
    "get_llm_client",
]
