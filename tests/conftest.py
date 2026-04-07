"""Shared fixtures for hear-music tests.

All tests use synthetic in-memory audio so the suite is fast, deterministic,
and free of external sample files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile


SAMPLE_RATE = 22050


def _to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


@pytest.fixture
def sr() -> int:
    return SAMPLE_RATE


@pytest.fixture
def sine_wave():
    """Factory: pure sine tone of (freq_hz, duration_s)."""

    def _make(freq_hz: float, duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
        return (0.6 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)

    return _make


@pytest.fixture
def silence():
    def _make(duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        return np.zeros(int(sample_rate * duration_s), dtype=np.float32)

    return _make


@pytest.fixture
def note_sequence(sine_wave, silence):
    """Factory: list of (freq_hz, duration_s) -> single concatenated audio array,
    with brief silence between notes so onsets are well-defined.
    """

    def _make(notes: list[tuple[float, float]], gap_s: float = 0.05) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for freq, dur in notes:
            chunks.append(sine_wave(freq, dur))
            if gap_s > 0:
                chunks.append(silence(gap_s))
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)

    return _make


@pytest.fixture
def click_track():
    """Factory: short tone burst at a fixed BPM for `duration_s` seconds."""

    def _make(
        bpm: float,
        duration_s: float,
        sample_rate: int = SAMPLE_RATE,
        burst_ms: float = 30.0,
        burst_freq: float = 1000.0,
    ) -> np.ndarray:
        total_samples = int(sample_rate * duration_s)
        audio = np.zeros(total_samples, dtype=np.float32)
        beat_period = 60.0 / bpm
        burst_len = int(sample_rate * burst_ms / 1000.0)
        envelope = np.linspace(1.0, 0.0, burst_len, dtype=np.float32) ** 2
        t_burst = np.arange(burst_len) / sample_rate
        burst = (np.sin(2 * np.pi * burst_freq * t_burst).astype(np.float32) * envelope * 0.8)

        beat = 0
        while True:
            start = int(round(beat * beat_period * sample_rate))
            if start >= total_samples:
                break
            end = min(start + burst_len, total_samples)
            audio[start:end] += burst[: end - start]
            beat += 1
        return audio

    return _make


@pytest.fixture
def chord_audio(sine_wave):
    """Factory: sustained chord made of multiple sine tones."""

    def _make(freqs: list[float], duration_s: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        chunks = [sine_wave(f, duration_s, sample_rate) for f in freqs]
        if not chunks:
            return np.zeros(int(sample_rate * duration_s), dtype=np.float32)
        mix = np.sum(chunks, axis=0)
        peak = float(np.max(np.abs(mix)))
        if peak > 0:
            mix = mix / peak * 0.8
        return mix.astype(np.float32)

    return _make


@pytest.fixture
def write_wav(tmp_path: Path):
    """Factory: write a numpy audio array to a temp WAV file and return its path."""

    def _make(audio: np.ndarray, name: str = "test.wav", sample_rate: int = SAMPLE_RATE) -> Path:
        out = tmp_path / name
        wavfile.write(out, sample_rate, _to_int16(audio))
        return out

    return _make
