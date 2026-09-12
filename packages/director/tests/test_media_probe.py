"""Media probing decides how long a clip is, so it must never lie or explode.

"Unknown" (0.0) is a legitimate answer — the planner treats it as "assume it
fits" and the editor clamps at execution time — but a wrong number would make
the planner cut past the end of the footage.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from director.ingestion.media_probe import MediaInfo, is_media_file, probe_media

HAS_FFPROBE = shutil.which("ffprobe") is not None
HAS_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("shot.mp4", True),
        ("shot.MOV", True),
        ("take.mkv", True),
        ("track.wav", True),
        ("track.mp3", True),
        ("frame.png", True),
        ("notes.txt", False),
        ("project.drp", False),
        ("noextension", False),
        (".DS_Store", False),
    ],
)
def test_is_media_file(name: str, expected: bool) -> None:
    assert is_media_file(name) is expected


def test_missing_file_is_unknown_not_an_error(tmp_path: Path) -> None:
    info = probe_media(str(tmp_path / "nope.mp4"))
    assert isinstance(info, MediaInfo)
    assert info.duration_seconds == 0.0
    assert info.is_known is False


def test_garbage_file_is_unknown_not_an_error(tmp_path: Path) -> None:
    junk = tmp_path / "broken.mp4"
    junk.write_bytes(b"not really a video")
    info = probe_media(str(junk))
    assert info.duration_seconds == 0.0
    assert info.is_known is False


@pytest.mark.skipif(not (HAS_FFPROBE and HAS_FFMPEG), reason="ffmpeg/ffprobe not installed")
def test_reads_real_duration_and_rate(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    made = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25:duration=3",
            "-pix_fmt", "yuv420p", str(clip),
        ],
        check=False,
        capture_output=True,
    )
    if made.returncode != 0 or not clip.is_file():
        pytest.skip(f"this ffmpeg cannot write to the test dir: {made.stderr[:200]!r}")
    info = probe_media(str(clip))
    assert info.is_known
    assert info.duration_seconds == pytest.approx(3.0, abs=0.15)
    assert info.fps == pytest.approx(25.0, abs=0.1)
    assert (info.width, info.height) == (320, 180)
    assert info.has_video is True
    assert info.has_audio is False
