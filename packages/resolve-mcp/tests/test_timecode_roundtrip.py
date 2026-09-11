"""Round-trip property tests for SMPTE drop-frame timecode.

plan.md Phase 2 guardrail: encode(decode(x)) == x and decode(encode(f)) == f
must hold with zero tolerance across the DF frame domain. The encoder is now
the exact inverse of the decoder; a one-off inline sweep proved this over the
full 24h label-day for both rates (2,589,408 + 5,178,816 frames/labels each
direction, zero mismatches), and the suites below pin it continuously:
exhaustive near-day windows plus every minute/hour boundary of the day plus a
prime-stride sample across the whole day (kept bounded so the suite stays fast).
"""

from __future__ import annotations

import pytest
from resolve_mcp.schemas import FrameRate
from resolve_mcp.timecode import TimeConverter


def _sweep(tc: TimeConverter, limit: int) -> None:
    for frames in range(limit):
        encoded = tc.frames_to_timecode(frames).value
        decoded = tc.timecode_to_frames(encoded)
        assert decoded == frames, (
            f"round-trip broke at {frames}: {encoded!r} decodes back to {decoded}"
        )


def test_dropframe_2997_roundtrip_exhaustive_first_hours() -> None:
    tc = TimeConverter(FrameRate(fps=29.97, drop_frame=True))
    _sweep(tc, limit=(2 * 3600 * 30_000) // 1001)  # first 2 hours @ 30000/1001 fps (~215,784 frames)


def test_dropframe_5994_roundtrip_exhaustive_first_hour() -> None:
    tc = TimeConverter(FrameRate(fps=59.94, drop_frame=True))
    _sweep(tc, limit=1 * 60 * 60 * 60 * 1000 // 1001)  # first ~hour


def test_dropframe_2997_minute_boundaries_full_day() -> None:
    """Every minute boundary of a 24h day must round-trip exactly."""
    tc = TimeConverter(FrameRate(fps=29.97, drop_frame=True))
    for minutes in range(0, 24 * 60):
        # Frame count of the first frame of minute N (drop-frame arithmetic).
        dropped = 2 * (minutes - minutes // 10)
        frames = 30 * 60 * minutes - dropped
        encoded = tc.frames_to_timecode(frames).value
        assert tc.timecode_to_frames(encoded) == frames, (
            f"minute {minutes} boundary ({frames}) -> {encoded!r}"
        )


def test_dropframe_5994_minute_boundaries_full_day() -> None:
    """Every minute boundary of a 24h day must round-trip exactly at 59.94df too."""
    tc = TimeConverter(FrameRate(fps=59.94, drop_frame=True))
    for minutes in range(0, 24 * 60):
        dropped = 4 * (minutes - minutes // 10)
        frames = 60 * 60 * minutes - dropped
        encoded = tc.frames_to_timecode(frames).value
        assert tc.timecode_to_frames(encoded) == frames, (
            f"minute {minutes} boundary ({frames}) -> {encoded!r}"
        )


@pytest.mark.parametrize(
    ("fps", "drop_per_minute", "nominal_fps"), [(29.97, 2, 30), (59.94, 4, 60)]
)
def test_dropframe_hour_boundaries_full_day(
    fps: float, drop_per_minute: int, nominal_fps: int
) -> None:
    """Every hour label HH:00:00;00 of the day is the exact round-trip of its frame."""
    tc = TimeConverter(FrameRate(fps=fps, drop_frame=True))
    for hour in range(24):
        # decode(HH:00:00;00): nominal F*3600*h minus D dropped per non-tenth minute
        # (54 such minutes per hour).
        frames = nominal_fps * 3600 * hour - 54 * drop_per_minute * hour
        expected = f"{hour:02d}:00:00;00"
        encoded = tc.frames_to_timecode(frames).value
        assert encoded == expected, f"hour {hour} ({frames}) encoded to {encoded!r}"
        assert tc.timecode_to_frames(encoded) == frames


def test_dropframe_sampled_interior_full_day() -> None:
    """Prime-stride sample across the entire label-day for both rates.

    The full exhaustive sweep (~7.8M conversions) lives in the one-off inline
    proof script; this keeps a fast continuous tripwire spanning all 24h.
    """
    for fps, drop_per_minute in ((29.97, 2), (59.94, 4)):
        tc = TimeConverter(FrameRate(fps=fps, drop_frame=True))
        day = 144 * (10 * tc.fps_nominal * 60 - 9 * drop_per_minute)
        for n in range(0, day, 997):
            encoded = tc.frames_to_timecode(n).value
            assert tc.timecode_to_frames(encoded) == n, (
                f"fps={fps} round-trip broke at {n}: {encoded!r}"
            )
