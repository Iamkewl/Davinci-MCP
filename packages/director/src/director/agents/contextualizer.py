"""Contextualizer: per-clip vision + probe, plus audio analysis of the music.

Every clip is probed with ffprobe regardless of whether a vision model is
configured, because the planner cannot cut to length without knowing how long
the footage is — and the file is more trustworthy than a model's guess, so a
probed duration/fps overrides whatever the vision pass reported.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..ingestion.audio_analyzer import TrackAnalysis, analyze_track
from ..ingestion.gemini_client import GeminiClient, GeminiError
from ..ingestion.media_probe import MediaInfo, probe_media
from ..schemas import PerClipMap
from ..settings import DirectorSettings
from .base import Agent, raise_or_rethrow_validation

if TYPE_CHECKING:
    from ..llm.base import LLMClient


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
        gemini: GeminiClient | None = None,
        llm: LLMClient | None = None,
        settings: DirectorSettings,
    ) -> None:
        super().__init__(gemini=gemini, llm=llm, settings=settings)

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
                # Audio failure is non-fatal — the planner falls back to an even
                # grid — but the run must say so rather than silently losing sync.
                await _log_warn(f"audio analysis failed for {music_path}: {exc}")
        return ContextResult(per_clip=per_clip, music_analysis=music_analysis)

    async def _analyze_clip(self, clip_id: str, path: str) -> PerClipMap:
        probe = probe_media(path)
        if self._llm is None:
            # No vision model: the probe alone still gives the planner what it
            # needs most (length, frame rate, whether there is audio).
            return _from_probe(clip_id, path, probe)

        try:
            pcm = await self._llm.analyze_video(
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
            return _from_probe(clip_id, path, probe)
        try:
            merged = pcm.model_copy(update=_probe_overrides(clip_id, path, probe, pcm))
        except ValidationError as err:
            raise_or_rethrow_validation(err, context="contextualizer")
            return _from_probe(clip_id, path, probe)  # unreachable
        return merged


def _from_probe(clip_id: str, path: str, probe: MediaInfo) -> PerClipMap:
    return PerClipMap(
        clip_id=clip_id,
        source_path=path,
        duration_seconds=probe.duration_seconds,
        fps=probe.fps,
        has_audio=probe.has_audio,
    )


def _probe_overrides(
    clip_id: str, path: str, probe: MediaInfo, pcm: PerClipMap
) -> dict[str, object]:
    """Identity and hard facts come from us and the file, not from the model."""
    overrides: dict[str, object] = {"clip_id": clip_id, "source_path": path}
    if probe.duration_seconds > 0:
        overrides["duration_seconds"] = probe.duration_seconds
    if probe.fps:
        overrides["fps"] = probe.fps
    if probe.has_audio:
        overrides["has_audio"] = True
    # Drop key moments the file cannot contain (a model may hallucinate timestamps).
    if probe.duration_seconds > 0 and pcm.key_moments:
        kept = [m for m in pcm.key_moments if m.position_seconds <= probe.duration_seconds]
        if len(kept) != len(pcm.key_moments):
            overrides["key_moments"] = kept
    return overrides


async def _log_warn(message: str) -> None:
    from .logging_setup import get_logger

    get_logger("director.contextualizer").warning(message)


__all__ = ["ContextResult", "Contextualizer"]
