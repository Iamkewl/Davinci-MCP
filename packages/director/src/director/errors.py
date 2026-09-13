"""Shared exception base, so provider failures are catchable in one place.

The two provider families raise unrelated types — ``GeminiError`` from
:mod:`director.ingestion.gemini_client` and ``LLMError`` from
:mod:`director.llm.base`. The agents only ever caught the Gemini one, so a
transport failure on the openai-compatible path (a wrong base URL, a rejected
key, a rate limit) escaped as a raw SDK exception and printed a traceback where
every other provider problem is reported as one line of JSON.

This module imports nothing from the rest of the package on purpose:
``director.llm`` imports the Gemini client and ``director.llm.openai_compat``
imports ``director.agents.base``, so a shared base defined in either of those
places would close an import cycle.
"""

from __future__ import annotations

__all__ = ["ProviderError"]


class ProviderError(RuntimeError):
    """Base for "the model provider could not give us an answer"."""
