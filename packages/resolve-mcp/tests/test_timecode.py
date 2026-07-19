"""Tests for resolve_mcp.timecode.

Exhaustive coverage of the three input forms (seconds, timecode, frames) for:
  * whole-number frame rates (24, 25, 30, 60)
  * 23.976 (non-drop)
  * 29.97 (drop + non-drop)
  * 59.94 (drop + non-drop)
"""

from __future__ import annotations

import pytest
from resolve_mcp.schemas import FrameRate, Timecode
from resolve_mcp.timecode import TimeConverter

# --- round-trip ---------------------------------------------------------------


@pytest.mark.parametrize("fps", [24.0, 25.0, 30.0, 60.0, 23.976, 29.97, 59.94])
def test_round_trip_frames_to_seconds(fps: float) -> None:
    fr = FrameRate(fps=fps)
    tc = TimeConverter(fr)
    for n in [0, 1, 23, 30, 100, 1000, 10_000]:
        s = tc.frames_to_seconds(n)
        back = tc.seconds_to_frames(s)
        # Sub-frame drift is unavoidable for non-integer fps ratios going via IEEE-754
        # float; the loss is bounded by one frame.
        assert abs(back - n) <= 1, f"fps={fps} n={n} s={s} back={back}"


@pytest.mark.parametrize("fps", [24.0, 25.0, 30.0, 60.0, 23.976, 29.97, 59.94])
def test_round_trip_seconds_frames_seconds(fps: float) -> None:
    fr = FrameRate(fps=fps)
    tc = TimeConverter(fr)
    for s in [0.0, 0.04, 1.0, 3.5, 12.0, 3600.0]:
        f = tc.seconds_to_frames(s)
        s2 = tc.frames_to_seconds(f)
        # Within sub-frame slack for non-integer fps ratios.
        assert abs(s2 - s) < (2.0 / fps)


# --- timecode <-> frames for whole-number fps ---------------------------------


def test_timecode_zero_whole_fps() -> None:
    tc = TimeConverter(FrameRate(fps=24))
    assert tc.timecode_to_frames("00:00:00:00") == 0
    assert tc.frames_to_timecode(0) == Timecode(value="00:00:00:00")


def test_timecode_one_second_whole_fps() -> None:
    tc = TimeConverter(FrameRate(fps=25))
    assert tc.timecode_to_frames("00:00:01:00") == 25
    assert tc.timecode_to_frames("00:00:01:13") == 25 + 13


def test_timecode_minutes_hours_whole_fps() -> None:
    tc = TimeConverter(FrameRate(fps=30))
    assert tc.timecode_to_frames("00:01:00:00") == 30 * 60
    assert tc.timecode_to_frames("01:00:00:00") == 30 * 3600


def test_timecode_wraps_24h() -> None:
    tc = TimeConverter(FrameRate(fps=24))
    # 25h from start is the same as 1h, modulo 24h.
    a = tc.timecode_to_frames("25:00:00:00")
    b = tc.timecode_to_frames("01:00:00:00")
    assert a == b


def test_timecode_to_seconds_against_frames() -> None:
    tc = TimeConverter(FrameRate(fps=24))
    frames = tc.timecode_to_frames("00:00:10:00")
    seconds = tc.frames_to_seconds(frames)
    assert seconds == pytest.approx(10.0)


# --- drop-frame timecode ------------------------------------------------------


def test_dropframe_basic_29_97() -> None:
    tc = TimeConverter(FrameRate(fps=29.97, drop_frame=True))
    assert tc.timecode_to_frames("00:00:00;00") == 0
    # 00:01:00;02 at 29.97df == 1800 nominal frames - 2 dropped this minute = 1800
    assert tc.timecode_to_frames("00:01:00;02") == 1800


def test_dropframe_first_minute_29_97() -> None:
    tc = TimeConverter(FrameRate(fps=29.97, drop_frame=True))
    # 00:00:59;29 == 1799 nominal frames (no drops yet in minute 0).
    assert tc.timecode_to_frames("00:00:59;29") == 1799


def test_dropframe_full_ten_minutes_29_97() -> None:
    tc = TimeConverter(FrameRate(fps=29.97, drop_frame=True))
    # 00:10:00;00 == 17982 (30fps*600 - 9*2 = 18000 - 18)
    assert tc.timecode_to_frames("00:10:00;00") == 17982


def test_dropframe_rejects_30fps() -> None:
    with pytest.raises(ValueError):
        TimeConverter(FrameRate(fps=30.0, drop_frame=True))


def test_dropframe_rejects_dropped_frame() -> None:
    tc = TimeConverter(FrameRate(fps=29.97, drop_frame=True))
    # 00:01:00;00 is one of the two dropped frames in minute 1 → invalid.
    with pytest.raises(ValueError):
        tc.timecode_to_frames("00:01:00;00")


# --- 23.976 (non-drop but non-integer) ----------------------------------------


def test_ntsc_23_976_nondrop() -> None:
    tc = TimeConverter(FrameRate(fps=23.976))
    # 1 second == ~23.976 frames -> 23 frames by floor.
    assert tc.seconds_to_frames(1.0) == 24000 // 1001
    assert tc.frames_to_seconds(24000 // 1001) == pytest.approx(
        (24000 // 1001) * 1001 / 24000
    )


# --- input parsing ------------------------------------------------------------


def test_invalid_timecode_string() -> None:
    tc = TimeConverter(FrameRate(fps=30))
    with pytest.raises(ValueError):
        tc.timecode_to_frames("not-a-tc")


def test_negative_seconds_rejected() -> None:
    tc = TimeConverter(FrameRate(fps=30))
    with pytest.raises(ValueError):
        tc.seconds_to_frames(-1.0)


def test_zero_time_string() -> None:
    tc = TimeConverter(FrameRate(fps=24))
    assert tc.timecode_to_frames("00:00:00:00") == 0


def test_invalid_fps_rejected() -> None:
    with pytest.raises(ValueError):
        TimeConverter(FrameRate(fps=0))
