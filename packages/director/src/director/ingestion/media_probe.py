"""Read real durations out of media files.

The planner cannot cut to the beat if it does not know how long a clip is, and
guessing is worse than admitting ignorance: an over-long source range is
rejected by Resolve at execution time. ``ffprobe`` is the only dependency-free
way to get this (FFmpeg is already a recommended install), so we shell out to it
when it is on PATH and report ``duration_seconds == 0.0`` — meaning *unknown*,
never *empty* — when it isn't.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["MEDIA_EXTENSIONS", "MediaInfo", "is_media_file", "probe_media"]

PROBE_TIMEOUT_SECONDS = 20.0

VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mxf", ".mts", ".m2ts", ".mpg", ".mpeg", ".wmv", ".flv"}
)
AUDIO_EXTENSIONS = frozenset(
    {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".aif", ".aiff", ".wma"}
)
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".dpx", ".exr"})
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | IMAGE_EXTENSIONS


@dataclass(frozen=True)
class MediaInfo:
    """What we could learn about one file. ``duration_seconds == 0.0`` = unknown."""

    path: str
    duration_seconds: float = 0.0
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    has_video: bool = False
    has_audio: bool = False

    @property
    def is_known(self) -> bool:
        return self.duration_seconds > 0.0


def is_media_file(path: str) -> bool:
    """True for files a video editor would treat as footage/audio/stills."""
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


def probe_media(path: str) -> MediaInfo:
    """Best-effort probe. Never raises: an unreadable file is simply 'unknown'."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None or not Path(path).is_file():
        return MediaInfo(path=path)
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-show_entries",
                "stream=codec_type,avg_frame_rate,width,height,duration",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
        if proc.returncode != 0:
            return MediaInfo(path=path)
        data = json.loads(proc.stdout or "{}")
    except (OSError, ValueError, subprocess.SubprocessError):
        return MediaInfo(path=path)

    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = _as_float((data.get("format") or {}).get("duration"))
    if duration <= 0 and video is not None:
        duration = _as_float(video.get("duration"))
    if duration <= 0 and audio is not None:
        duration = _as_float(audio.get("duration"))
    return MediaInfo(
        path=path,
        duration_seconds=max(0.0, duration),
        fps=_parse_fraction(video.get("avg_frame_rate")) if video else None,
        width=_as_int(video.get("width")) if video else None,
        height=_as_int(video.get("height")) if video else None,
        has_video=video is not None,
        has_audio=audio is not None,
    )


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_fraction(value: Any) -> float | None:
    """ffprobe reports rates as "24000/1001"; 0/0 means 'not applicable'."""
    if not isinstance(value, str) or "/" not in value:
        return None
    num, _, den = value.partition("/")
    try:
        numerator, denominator = float(num), float(den)
    except ValueError:
        return None
    if denominator <= 0 or numerator <= 0:
        return None
    return round(numerator / denominator, 6)
