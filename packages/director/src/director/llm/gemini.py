"""Gemini adapter for the :class:`director.llm.base.LLMClient` protocol.

Thin composition over the legacy :class:`GeminiClient` (left untouched in
``ingestion/``): both protocol methods delegate 1:1, dropping the legacy
optional ``model=`` override. ``aclose`` is a no-op — the google-genai client
holds no per-instance resources we manage. ``GeminiError`` propagates
unwrapped so existing caller handling keeps working unchanged.
"""

from __future__ import annotations

from ..ingestion.gemini_client import GeminiClient
from ..schemas import PerClipMap
from ..settings import DirectorSettings
from .base import T


class GeminiLLMClient:
    """Protocol surface over the wrapped Gemini SDK wrapper."""

    def __init__(self, *, settings: DirectorSettings) -> None:
        self._inner = GeminiClient(settings=settings)

    async def generate_json(self, *, system: str, user: str, response_schema: type[T]) -> T:
        return await self._inner.generate_json(
            system=system,
            user=user,
            response_schema=response_schema,
        )

    async def analyze_video(self, *, clip_path: str, clip_id: str, prompt: str) -> PerClipMap:
        return await self._inner.analyze_video(
            clip_path=clip_path,
            clip_id=clip_id,
            prompt=prompt,
        )

    async def aclose(self) -> None:
        return None


__all__ = ["GeminiLLMClient"]
