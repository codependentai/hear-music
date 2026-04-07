"""CLI smoke tests."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
LIBROSA_AVAILABLE = False
try:
    import librosa  # noqa: F401

    LIBROSA_AVAILABLE = True
except ImportError:
    pass


def _run_module(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "hear_music", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_exits_zero():
    result = _run_module("--help")
    assert result.returncode == 0
    assert "hear-music" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_doctor_returns_valid_json():
    result = _run_module("doctor")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "checks" in payload
    assert "warnings" in payload


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not available")
def test_analyze_produces_outputs(write_wav, sine_wave, tmp_path: Path):
    audio = sine_wave(440.0, 1.0)
    wav_path = write_wav(audio, name="tone.wav")
    out_dir = tmp_path / "analysis_out"

    result = _run_module("analyze", str(wav_path), "--out-dir", str(out_dir))
    assert result.returncode == 0, f"analyze failed: {result.stderr}"

    payload = json.loads(result.stdout)
    assert "tempo_bpm" in payload
    assert "key" in payload
    assert "onset_count" in payload
    assert "chroma" in payload
    assert (out_dir / "normalized.wav").exists()
    assert (out_dir / "spectrogram.png").exists()
    assert (out_dir / "transcription.mid").exists()
    assert (out_dir / "analysis.json").exists()


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffprobe not available")
def test_info_returns_ffprobe_json(write_wav, sine_wave):
    audio = sine_wave(440.0, 0.5)
    wav_path = write_wav(audio, name="info.wav")
    result = _run_module("info", str(wav_path))
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "streams" in payload or "format" in payload


@pytest.mark.skipif(not LIBROSA_AVAILABLE or not FFMPEG_AVAILABLE, reason="librosa or ffmpeg missing")
def test_analyze_includes_librosa_block(write_wav, sine_wave, tmp_path: Path):
    audio = sine_wave(440.0, 1.5)
    wav_path = write_wav(audio, name="lib.wav")
    out_dir = tmp_path / "lib_out"
    result = _run_module("analyze", str(wav_path), "--out-dir", str(out_dir))
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "librosa" in payload, "expected librosa enrichment block when librosa is installed"
