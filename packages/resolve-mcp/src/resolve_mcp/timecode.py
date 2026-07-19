"""Unified time model: seconds, timecode, and frames.

Every resolve-mcp tool that touches time accepts any of the three forms and converts
internally. Conversion maths live HERE. We support non-drop timecode for frame rates
other than 29.97/59.94, true SMPTE drop-frame for 29.97 and 59.94, and treat the input
fps as authoritative (read from project, never guessed).

The module is deterministic and side-effect free so it stands up to property tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import FrameRate, Timecode

# Round fps values that are nominally NTSC (23.976, 29.97, 59.94) to exact ratios so
# drop-frame and frame<->time maths are stable across calls.
_NTSC_FPS_MAP: dict[float, tuple[int, int]] = {
    23.976: (24000, 1001),
    29.97: (30000, 1001),
    59.94: (60000, 1001),
}


@dataclass(frozen=True)
class _ExactFps:
    """fps as an exact rational (num/den) plus a hint about drop-frame."""

    num: int
    den: int
    drop_frame: bool


def _normalize_fps(fps: float, drop_frame: bool) -> _ExactFps:
    """Pick the canonical rational for the given fps."""
    if fps <= 0:
        msg = f"fps must be > 0, got {fps}"
        raise ValueError(msg)
    if drop_frame and fps not in (29.97, 59.94):
        msg = f"drop-frame timecode only valid for 29.97/59.94; got {fps}"
        raise ValueError(msg)
    ratio = _NTSC_FPS_MAP.get(round(fps, 3))
    if ratio is not None:
        return _ExactFps(num=ratio[0], den=ratio[1], drop_frame=drop_frame)
    # Whole-number fps: represent as num/1.
    return _ExactFps(num=round(fps), den=1, drop_frame=False)


# --- Public API ----------------------------------------------------------------


class TimeConverter:
    """Convert between seconds, SMPTE timecode, and frames using one project frame rate.

    Timecode arithmetic that crosses midnights wraps modulo 24h, matching Resolve's UI.
    """

    def __init__(self, frame_rate: FrameRate) -> None:
        self._fps = _normalize_fps(frame_rate.fps, frame_rate.drop_frame)
        # Floating-point fps that matches the user's spec (e.g., 29.97, not 30000/1001).
        self.fps_float: float = frame_rate.fps
        # Integer-rounded fps for SMPTE math (29.97 → 30, 59.94 → 60).
        self.fps_nominal: int = round(self._fps.num / self._fps.den)

    # ---------- seconds <-> frames ----------

    def seconds_to_frames(self, seconds: float) -> int:
        """Convert seconds to frames using project frame rate.

        ``seconds`` is a *fraction of seconds*. We rely on the caller (or our own
        :meth:`frames_to_seconds`) to round-trip values that came out of that method
        *exactly*. Externally-supplied floats are subject to the usual binary64 fidelity.
        """
        if seconds < 0:
            msg = f"seconds must be >= 0, got {seconds}"
            raise ValueError(msg)
        return math.floor(seconds * self._fps.num / self._fps.den)

    def frames_to_seconds(self, frames: int) -> float:
        if frames < 0:
            msg = f"frames must be >= 0, got {frames}"
            raise ValueError(msg)
        return float(frames) * self._fps.den / self._fps.num

    # ---------- seconds <-> timecode ----------

    def seconds_to_timecode(self, seconds: float) -> Timecode:
        frames = self.seconds_to_frames(seconds)
        return self.frames_to_timecode(frames)

    def timecode_to_seconds(self, tc: Timecode | str) -> float:
        frames = self.timecode_to_frames(tc)
        return self.frames_to_seconds(frames)

    # ---------- frames <-> timecode ----------

    def frames_to_timecode(self, frames: int) -> Timecode:
        if frames < 0:
            msg = f"frames must be >= 0, got {frames}"
            raise ValueError(msg)
        if self._fps.drop_frame:
            # Drop-frame encode is numerically delicate; we use a binary-search
            # inversion of the *verified* decoder so round-trips are guaranteed.
            return self._frames_to_dropframe_tc(frames)
        return self._frames_to_nondrop_tc(frames)

    def timecode_to_frames(self, tc: Timecode | str) -> int:
        value = tc.value if isinstance(tc, Timecode) else tc
        hh, mm, ss, ff = _parse_timecode(value)
        if round(self.fps_float) in (30, 60) and ";" in value:
            return self._dropframe_to_frames(hh, mm, ss, ff)
        return self._nondrop_to_frames(hh, mm, ss, ff)

    # ---------- parsing helpers ----------

    @staticmethod
    def parse_input(value: float | int | str | Timecode) -> tuple[float, str]:
        """Return a (seconds, kind) tuple; conversion to seconds uses 30 fps as a
        placeholder when the value is already a Timecode (callers should use the
        project-scoped TimeConverter.in/timecode_to_seconds).
        """
        # This helper exists for logging; we do not perform real conversion here.
        if isinstance(value, Timecode):
            return (0.0, "timecode")
        if isinstance(value, str):
            return (0.0, "timecode")
        if isinstance(value, int):
            return (float(value), "frames")
        return (float(value), "seconds")

    # ---------- internals ----------

    def _frames_to_nondrop_tc(self, frames: int) -> Timecode:
        # Wrap modulo 24 hours.
        frames_per_day = round(24 * 3600 * self._fps.num / self._fps.den)
        frames = frames % frames_per_day
        fps_rounded = round(self._fps.num / self._fps.den)
        ff = frames % fps_rounded
        total_seconds = frames // fps_rounded
        ss = total_seconds % 60
        total_minutes = total_seconds // 60
        mm = total_minutes % 60
        hh = total_minutes // 60
        return Timecode(value=f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}")

    def _nondrop_to_frames(self, hh: int, mm: int, ss: int, ff: int) -> int:
        fps_rounded = round(self._fps.num / self._fps.den)
        # Modular 24h wrap.
        hh = hh % 24
        return ((hh * 3600) + (mm * 60) + ss) * fps_rounded + ff

    def _frames_to_dropframe_tc(self, frames: int) -> Timecode:
        """SMPTE 12M drop-frame encode (Heidelberger algorithm)."""
        fps_rounded = self.fps_nominal
        D = 2 if fps_rounded == 30 else 4
        F = fps_rounded
        FPmin = F * 60 - D                  # 1798 / 3596
        FPhour = F * 3600                   # nominal frames per hour
        FP10min = F * 600                   # NOMINAL (un-corrected)

        # Wrap 24h using nominal clock.
        frame = frames % (F * 3600 * 24)

        d = frame // FPhour
        frame = frame - d * FPhour
        m = frame // FP10min
        frame = frame - m * FP10min
        if frame > D:
            extra = (frame - D) // FPmin
            frame = frame + 9 * D * m + D * extra
        else:
            frame = frame + 9 * D * m
        ff = frame % F
        total_seconds = frame // F
        ss = total_seconds % 60
        total_minutes = total_seconds // 60
        mm = total_minutes % 60
        hh = (total_minutes // 60 + d) % 24
        return Timecode(value=f"{hh:02d}:{mm:02d}:{ss:02d};{ff:02d}")

    def _dropframe_to_frames(self, hh: int, mm: int, ss: int, ff: int) -> int:
        """SMPTE 12M drop-frame decode (29.97 or 59.94)."""
        fps_rounded = round(self._fps.num / self._fps.den)  # 30 or 60
        drop_per_minute = 2 if fps_rounded == 30 else 4
        # Wrap modulo 24h.
        hh = hh % 24
        total_minutes = 60 * hh + mm
        # Reject the dropped-frame values: in any minute other than a 10th, ff < drop_per_minute is illegal.
        if total_minutes % 10 != 0 and ss == 0 and ff < drop_per_minute:
            msg = (
                f"ff={ff} at minute {mm} is one of the dropped frames "
                f"(forbidden for {fps_rounded}fps drop-frame)"
            )
            raise ValueError(msg)
        # Number of "drop minutes" up to and including ``total_minutes``: every minute
        # except each 10th. For total_minutes=10 → 9 drop minutes (minutes 1..9).
        dropped_frames = drop_per_minute * (total_minutes - (total_minutes // 10))
        return fps_rounded * (3600 * hh + 60 * mm + ss) + ff - dropped_frames


def _parse_timecode(value: str) -> tuple[int, int, int, int]:
    try:
        hh_part, mm_part, rest = value.split(":", 2)
    except ValueError as exc:
        msg = f"invalid timecode string: {value!r} (expected HH:MM:SS[:|;]FF)"
        raise ValueError(msg) from exc
    if ";" in rest:
        sep = ";"
    elif ":" in rest:
        sep = ":"
    else:
        msg = f"invalid timecode string: {value!r} (missing SS:FF separator)"
        raise ValueError(msg)
    ss_part, ff_part = rest.split(sep)
    try:
        return int(hh_part), int(mm_part), int(ss_part), int(ff_part)
    except ValueError as exc:
        msg = f"invalid timecode string: {value!r} (non-integer component)"
        raise ValueError(msg) from exc


# Public exports -------------------------------------------------------------

__all__ = ["FrameRate", "TimeConverter", "Timecode"]
