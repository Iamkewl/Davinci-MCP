"""Provider-agnostic LLM contract (plan.md Phase 5).

Agents depend on this *structural* interface only — never on a concrete
provider SDK. Two adapters implement it today:

* :class:`director.llm.gemini.GeminiLLMClient` — wraps the legacy
  :class:`director.ingestion.gemini_client.GeminiClient` untouched.
* :class:`director.llm.openai_compat.OpenAICompatClient` — any
  OpenAI-compatible chat-completions endpoint (OpenRouter, vLLM, …).

Error types are deliberately NOT part of the protocol: each adapter surfaces
its own transport errors (``GeminiError`` / :class:`LLMError`) and both are
caught next to :class:`director.agents.base.InvalidModelOutput` by callers.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from ..schemas import PerClipMap

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when an LLM provider call fails in some non-recoverable way."""


@runtime_checkable
class LLMClient(Protocol):
    """Structural interface every provider adapter must satisfy."""

    async def generate_json(self, *, system: str, user: str, response_schema: type[T]) -> T:
        """Generate a value validated against ``response_schema``.

        Implementations must NOT silently coerce: a schema mismatch raises
        (``InvalidModelOutput`` for the compat adapter, ``ValidationError``
        propagation for Gemini).
        """
        ...

    async def analyze_video(self, *, clip_path: str, clip_id: str, prompt: str) -> PerClipMap:
        """Analyze one video clip; return a populated :class:`PerClipMap`."""
        ...

    async def aclose(self) -> None:
        """Release underlying HTTP resources; safe to call repeatedly."""
        ...


__all__ = ["LLMClient", "LLMError"]
