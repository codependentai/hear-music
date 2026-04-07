"""Tempo estimation tests."""
from __future__ import annotations

from hear_music.analysis import compute_onset_envelope, estimate_tempo


def _estimate(audio, sample_rate: int) -> float | None:
    envelope = compute_onset_envelope(audio, sample_rate)
    return estimate_tempo(envelope, sample_rate)


def test_120_bpm_click_track(click_track, sr):
    audio = click_track(bpm=120, duration_s=6.0)
    bpm = _estimate(audio, sr)
    assert bpm is not None
    assert abs(bpm - 120) <= 5, f"expected ~120 BPM, got {bpm}"


def test_90_bpm_click_track(click_track, sr):
    audio = click_track(bpm=90, duration_s=6.0)
    bpm = _estimate(audio, sr)
    assert bpm is not None
    assert abs(bpm - 90) <= 5, f"expected ~90 BPM, got {bpm}"


def test_140_bpm_click_track(click_track, sr):
    audio = click_track(bpm=140, duration_s=6.0)
    bpm = _estimate(audio, sr)
    assert bpm is not None
    assert abs(bpm - 140) <= 5, f"expected ~140 BPM, got {bpm}"


def test_silence_returns_none_or_safe(silence, sr):
    audio = silence(3.0)
    # Should not raise
    bpm = _estimate(audio, sr)
    assert bpm is None or isinstance(bpm, float)
