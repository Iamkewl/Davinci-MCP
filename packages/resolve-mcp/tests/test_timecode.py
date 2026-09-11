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


# --- drop-frame encode reference vectors (plan.md Phase 2) ---------------------


def test_dropframe_encode_reference_vectors_2997() -> None:
    """Encoder output pinned against the independently verified decoder.

    Each expected label is derived from decode(label) == frames, e.g. 17982 is
    30*600 nominal minus 9*2 dropped over minutes 1-9 ('00:10:00;00' -> 17982).
    """
    tc = TimeConverter(FrameRate(fps=29.97, drop_frame=True))
    day = 144 * (10 * 30 * 60 - 9 * 2)  # 2,589,408 labels per 24h
    vectors = [
        (0, "00:00:00;00"),  # first label, no drops elapsed
        (900, "00:00:30;00"),  # mid-minute interior, unaffected by drops
        (1799, "00:00:59;29"),  # last label before minute 1's drop pair
        (1800, "00:01:00;02"),  # first label after the 2 dropped frames of minute 1
        (17982, "00:10:00;00"),  # tenth minute realigns: 18000 - 9*2 (regression vector)
        (35964, "00:20:00;00"),  # two blocks: 36000 - 18*2
        (53946, "00:30:00;00"),  # three blocks: 54000 - 27*2
        (107891, "00:59:59;29"),  # last label of hour 0: 108000 - 54*2 - 1
        (107892, "01:00:00;00"),  # hour boundary: 6 blocks * 17982
        (2589407, "23:59:59;29"),  # last label of the day: 144*17982 - 1
        (2589408, "00:00:00;00"),  # day wrap modulo 24h
    ]
    for frames, label in vectors:
        assert tc.frames_to_timecode(frames).value == label, f"frames={frames}"
        assert tc.timecode_to_frames(label) == frames % day, f"label={label}"


def test_dropframe_encode_reference_vectors_5994() -> None:
    """Same contract at 59.94df (F=60, D=4)."""
    tc = TimeConverter(FrameRate(fps=59.94, drop_frame=True))
    day = 144 * (10 * 60 * 60 - 9 * 4)  # 5,178,816 labels per 24h
    vectors = [
        (0, "00:00:00;00"),
        (1800, "00:00:30;00"),
        (3599, "00:00:59;59"),  # last label before minute 1's drop quartet
        (3600, "00:01:00;04"),  # first label after the 4 dropped frames of minute 1
        (35964, "00:10:00;00"),  # tenth minute realigns: 36000 - 9*4
        (71928, "00:20:00;00"),  # two blocks: 72000 - 18*4
        (215783, "00:59:59;59"),  # last label of hour 0: 216000 - 54*4 - 1
        (215784, "01:00:00;00"),  # hour boundary: 6 blocks * 35964
        (5178815, "23:59:59;59"),  # last label of the day: 144*35964 - 1
        (5178816, "00:00:00;00"),  # day wrap modulo 24h
    ]
    for frames, label in vectors:
        assert tc.frames_to_timecode(frames).value == label, f"frames={frames}"
        assert tc.timecode_to_frames(label) == frames % day, f"label={label}"


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
