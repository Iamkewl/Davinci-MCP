"""Audio ingestion: librosa beat / BPM / onset detection.

We force the ``soundfile`` decode backend and pass raw bytes / path through
``librosa.load`` explicitly. This avoids the deprecated ``audioread`` / ``aifc``
shim that breaks on Python 3.13 — the exact footgun called out in the brief.

Public surface: :func:`analyze_track` returns a :class:`TrackAnalysis` with the
beat frames (in seconds), the BPM estimate, and an onset-strength envelope.
Everything is deterministic for fixed inputs.
"""

from __future__ import annotations

import numpy as np

from ..schemas import PerClipMap


class TrackAnalysis:
    """Plain result struct. Kept separate from PerClipMap so the audio module has
    no schema coupling."""

    __slots__ = ("beat_times", "bpm", "duration_seconds", "onset_strength", "onset_times")

    def __init__(
        self,
        bpm: float,
        beat_times: np.ndarray,
        onset_times: np.ndarray,
        onset_strength: np.ndarray,
        duration_seconds: float,
    ) -> None:
        self.bpm = float(bpm)
        self.beat_times = beat_times
        self.onset_times = onset_times
        self.onset_strength = onset_strength
        self.duration_seconds = float(duration_seconds)

    def to_per_clip_map(
        self,
        clip_id: str,
        source_path: str,
    ) -> PerClipMap:
        """Project this analysis onto a :class:`PerClipMap` for the Contextualizer."""
        return PerClipMap(
            clip_id=clip_id,
            source_path=source_path,
            duration_seconds=self.duration_seconds,
            tempo_bpm=self.bpm,
            beat_frames=[float(t) for t in self.beat_times.tolist()],
            onset_strength=[float(s) for s in self.onset_strength.tolist()],
            has_audio=True,
        )


# Force librosa to use soundfile. Imported eagerly so the choice is locked before
# any librosa.load() call.
try:

    # librosa 0.10 took a ``sr`` already-loaded numpy path; for path input it
    # uses audioread by default. We force soundfile inline below.
    _HAS_LIBROSA = True
except Exception:  # pragma: no cover — handled at call site
    _HAS_LIBROSA = False


def _load_audio_via_soundfile(path: str, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode audio to a mono float32 numpy array using ``soundfile`` directly,
    bypassing librosa's auto-backend selection."""
    import soundfile as sf

    audio, file_sr = sf.read(path, always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr is not None and sr != file_sr:
        # Resample through librosa if a target sr is requested; librosa will
        # re-pick the soundfile backend under the hood.
        import librosa as _lb

        audio = _lb.resample(audio, orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    return audio, int(file_sr)


def analyze_track(
    path: str,
    *,
    sr: int = 22_050,
    hop_length: int = 512,
) -> TrackAnalysis:
    """Run beat/onset analysis on an audio path.

    Args:
        path: filesystem path to a wav/mp3/etc.
        sr: target sample rate (default 22050 — fine for many beat trackers).
        hop_length: STFT hop used for onset envelope.

    Returns:
        :class:`TrackAnalysis` with the BPM estimate, beat times, onset times,
        onset strength envelope, and total decoded duration.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        RuntimeError: if librosa / soundfile are unavailable in this environment.
    """
    if not _HAS_LIBROSA:
        msg = "librosa is not installed; cannot analyze audio"
        raise RuntimeError(msg)
    import librosa as _lb

    audio, effective_sr = _load_audio_via_soundfile(path, sr=sr)
    duration_seconds = float(audio.shape[0] / effective_sr)

    # Onset strength envelope → peaks (onsets).
    onset_env = _lb.onset.onset_strength(y=audio, sr=effective_sr, hop_length=hop_length)
    onset_frames = _lb.onset.onset_detect(
        onset_envelope=onset_env,
        sr=effective_sr,
        hop_length=hop_length,
        units="time",
    )
    onset_strength = _lb.util.normalize(np.asarray(onset_env, dtype=np.float64))

    # Tempo + beats via librosa. ``beat_track`` returns (tempo, beat_frames).
    tempo, beat_frames = _lb.beat.beat_track(
        y=audio,
        sr=effective_sr,
        hop_length=hop_length,
        units="time",
        start_bpm=120.0,
    )
    return TrackAnalysis(
        bpm=float(tempo),
        beat_times=np.asarray(beat_frames, dtype=np.float64),
        onset_times=np.asarray(onset_frames, dtype=np.float64),
        onset_strength=onset_strength,
        duration_seconds=duration_seconds,
    )


# Pure-data helper used by tests + Planner (no I/O) ---------------------------------------------------------------------


def first_n_beats_close_to_times(
    beat_times: list[float],
    target_times: list[float],
    *,
    tolerance_seconds: float = 0.05,
) -> list[float]:
    """For each target time, return the nearest beat time within tolerance.

    Used by the planner to align cuts to downbeats. Pure, fully testable without
    any audio dependency loaded.
    """
    out: list[float] = []
    beats = sorted(beat_times)
    for target in target_times:
        # Linear scan is fine for the count of beats we generate for short reels.
        nearest = min(beats, key=lambda b: abs(b - target))
        out.append(nearest if abs(nearest - target) <= tolerance_seconds else target)
    return out


__all__ = ["TrackAnalysis", "analyze_track", "first_n_beats_close_to_times"]
