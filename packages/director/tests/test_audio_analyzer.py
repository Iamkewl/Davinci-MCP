"""Offline tests for director.ingestion.audio_analyzer.

Generates a deterministic click track via the stdlib ``wave`` module and checks
that :func:`analyze_track` survives numpy 2.x / librosa 0.11 array-vs-scalar
drift (tempo arrives as a 1-element ndarray; plain ``float()`` on it raises
``TypeError: only 0-dimensional arrays can be converted to Python scalars``).
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import pytest
from director.ingestion.audio_analyzer import TrackAnalysis, analyze_track

SR = 44_100
CLICK_HZ = 440.0
BURST_SECONDS = 0.05
PERIOD_SECONDS = 0.5
DURATION_SECONDS = 20


def _write_click_wav(path: Path) -> None:
    """Deterministic mono 16-bit 44.1 kHz click track: 0.05 s 440 Hz bursts every 0.5 s."""
    burst_len = int(BURST_SECONDS * SR)
    period = int(PERIOD_SECONDS * SR)
    total_periods = DURATION_SECONDS * SR // period  # 20 s @ 0.5 s → 40 periods exactly

    one_period = bytearray()
    for i in range(period):
        if i < burst_len:
            value = math.sin(2.0 * math.pi * CLICK_HZ * (i / SR)) * 0.8
            one_period += int(value * 32767).to_bytes(2, "little", signed=True)
        else:
            one_period += b"\x00\x00"

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(bytes(one_period) * total_periods)


@pytest.fixture(scope="module")
def click_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("audio") / "click.wav"
    _write_click_wav(path)
    return path


def test_analyze_track_click_wav(click_wav: Path) -> None:
    analysis = analyze_track(str(click_wav))

    assert isinstance(analysis, TrackAnalysis)
    # bpm must be a plain Python float (not an ndarray / np.float64 wrapper).
    assert type(analysis.bpm) is float
    assert analysis.bpm > 0

    assert isinstance(analysis.beat_times, np.ndarray)
    # ~40 beats expected at 120 BPM over 20 s; be generous to stay robust offline.
    assert analysis.beat_times.ndim == 1
    assert analysis.beat_times.size >= 10
    assert np.all(np.isfinite(analysis.beat_times))
    assert np.all(analysis.beat_times >= 0.0)
    assert np.all(analysis.beat_times <= analysis.duration_seconds)

    # Onset arrays consistent: both non-empty 1-d, detections within the envelope bounds.
    assert isinstance(analysis.onset_times, np.ndarray)
    assert isinstance(analysis.onset_strength, np.ndarray)
    assert analysis.onset_times.ndim == 1
    assert analysis.onset_strength.ndim == 1
    assert analysis.onset_times.size > 0
    assert analysis.onset_strength.size > 0
    assert analysis.onset_times.size <= analysis.onset_strength.size
    assert np.all(analysis.onset_times >= 0.0)
    assert np.all(analysis.onset_times <= analysis.duration_seconds)

    assert analysis.duration_seconds > 0


def test_analyze_track_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        analyze_track(str(tmp_path / "nope.wav"))
