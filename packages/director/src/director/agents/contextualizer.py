"""Contextualizer: per-clip vision + audio analysis into :class:`PerClipMap`."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from pydantic import ValidationError

from ..ingestion.audio_analyzer import TrackAnalysis, analyze_track
from ..ingestion.gemini_client import GeminiClient, GeminiError
from ..schemas import PerClipMap
from ..settings import DirectorSettings
from .base import Agent, raise_or_rethrow_validation


@dataclass
class ContextResult:
    """Aggregated output across all clips in the run."""

    per_clip: list[PerClipMap]
    music_analysis: TrackAnalysis | None


class Contextualizer(Agent[ContextResult]):
    """Run vision analysis on every video clip and audio analysis on the music."""

    def __init__(
        self,
        *,
        gemini: GeminiClient | None,
        settings: DirectorSettings,
    ) -> None:
        super().__init__(gemini=gemini, settings=settings)

    async def run(
        self,
        clip_paths: list[str],
        music_path: str | None = None,
    ) -> ContextResult:
        per_clip: list[PerClipMap] = []
        for path in clip_paths:
            clip_id = f"clip_{uuid.uuid4().hex[:8]}"
            per_clip.append(await self._analyze_clip(clip_id, path))

        music_analysis: TrackAnalysis | None = None
        if music_path is not None:
            try:
                music_analysis = analyze_track(music_path)
            except Exception as exc:
                # Audio failure is non-fatal — the planner will fall back to a
                # uniform beat grid. We log and continue.
                await _log_warn(f"audio analysis failed for {music_path}: {exc}")
        return ContextResult(per_clip=per_clip, music_analysis=music_analysis)

    async def _analyze_clip(self, clip_id: str, path: str) -> PerClipMap:
        if self._gemini is None:
            # Without a Gemini client we cannot do vision analysis. Return a
            # placeholder map carrying only the path. Tests inject a fake gemini
            # for full coverage.
            return PerClipMap(clip_id=clip_id, source_path=path, duration_seconds=0.0)

        try:
            pcm = await self._gemini.analyze_video(
                clip_path=path,
                clip_id=clip_id,
                prompt=(
                    "Analyze this video clip. Identify dominant shot type, "
                    "two to three key moments with timestamps and a one-line "
                    "visual summary. Respond with valid JSON conforming to the schema."
                ),
            )
        except GeminiError as exc:
            await _log_warn(f"vision analyze failed for {path}: {exc}")
            return PerClipMap(clip_id=clip_id, source_path=path, duration_seconds=0.0)
        try:
            return pcm.model_copy(update={"clip_id": clip_id, "source_path": path})
        except ValidationError as err:
            raise_or_rethrow_validation(err, context="contextualizer")
            return PerClipMap(clip_id=clip_id, source_path=path, duration_seconds=0.0)  # unreachable


async def _log_warn(message: str) -> None:
    # Tiny helper for now; orchestrator routes these via structlog.
    from .logging_setup import get_logger

    get_logger("director.contextualizer").warning(message)


__all__ = ["ContextResult", "Contextualizer"]
