"""CLI smoke tests."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from argparse import Namespace
from importlib.metadata import version
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


def test_module_version_matches_distribution_metadata():
    import hear_music

    assert hear_music.__version__ == version("hear-music")


def test_help_exits_zero():
    result = _run_module("--help")
    assert result.returncode == 0
    assert "hear-music" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_help_lists_general_sound_classification():
    result = _run_module("--help")
    assert result.returncode == 0
    assert "classify" in result.stdout


def test_help_lists_local_speech_transcription():
    result = _run_module("--help")
    assert result.returncode == 0
    assert "transcribe" in result.stdout


def test_help_lists_unified_listen_command():
    result = _run_module("--help")
    assert result.returncode == 0
    assert "listen" in result.stdout


def test_classify_missing_input_returns_clear_error(tmp_path: Path):
    missing = tmp_path / "missing.ogg"
    result = _run_module("classify", str(missing))
    assert result.returncode == 1
    assert "Input file not found" in result.stderr


def test_transcribe_missing_input_returns_clear_error(tmp_path: Path):
    missing = tmp_path / "missing.ogg"
    result = _run_module("transcribe", str(missing))
    assert result.returncode == 1
    assert "Input file not found" in result.stderr


@pytest.mark.parametrize("command", ["classify", "transcribe"])
def test_json_output_cannot_overwrite_source_audio(command, tmp_path):
    source = tmp_path / "recording.ogg"
    original = b"irreplaceable audio"
    source.write_bytes(original)

    result = _run_module(command, str(source), "--out", str(source))

    assert result.returncode == 1
    assert "Output path must differ from input" in result.stderr
    assert source.read_bytes() == original


def test_cmd_transcribe_turns_model_value_errors_into_cli_errors(tmp_path, monkeypatch, capsys):
    from hear_music import cli

    source = tmp_path / "voice.ogg"
    source.write_bytes(b"audio")
    monkeypatch.setattr(
        cli,
        "transcribe_speech",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid model")),
    )

    result = cli.cmd_transcribe(
        Namespace(
            input=str(source),
            out=None,
            model="bad-model",
            language=None,
            device="cpu",
            compute_type="int8",
            no_vad=False,
        )
    )

    assert result == 1
    assert "invalid model" in capsys.readouterr().err


def test_cmd_listen_rejects_output_collision_with_source(tmp_path, monkeypatch, capsys):
    from hear_music import cli

    source = tmp_path / "hearing.json"
    original = b"audio"
    source.write_bytes(original)
    monkeypatch.setattr(cli, "classify_environment", lambda *_args, **_kwargs: {})

    result = cli.cmd_listen(
        Namespace(
            input=str(source),
            out_dir=str(tmp_path),
            sound_model=None,
            speech_model="tiny",
            language=None,
            no_speech=True,
            no_vad=False,
            max_results=8,
            score_threshold=0.05,
        )
    )

    assert result == 1
    assert "Output path must differ from input" in capsys.readouterr().err
    assert source.read_bytes() == original


def test_cmd_listen_no_speech_removes_stale_speech_artifact(tmp_path, monkeypatch):
    from hear_music import cli

    source = tmp_path / "courtyard.ogg"
    source.write_bytes(b"audio")
    out_dir = tmp_path / "hearing"
    out_dir.mkdir()
    stale = out_dir / "speech.json"
    stale.write_text("STALE", encoding="utf-8")
    monkeypatch.setattr(cli, "classify_environment", lambda *_args, **_kwargs: {"events": []})

    result = cli.cmd_listen(
        Namespace(
            input=str(source),
            out_dir=str(out_dir),
            sound_model=None,
            speech_model="tiny",
            language=None,
            no_speech=True,
            no_vad=False,
            max_results=8,
            score_threshold=0.05,
        )
    )

    assert result == 0
    assert not stale.exists()
    assert json.loads((out_dir / "hearing.json").read_text())["speech"] is None


def test_cmd_listen_writes_combined_local_hearing_payload(tmp_path, monkeypatch, capsys):
    from hear_music import cli

    source = tmp_path / "courtyard.ogg"
    source.write_bytes(b"audio")
    out_dir = tmp_path / "hearing"
    monkeypatch.setattr(
        cli,
        "classify_environment",
        lambda *_args, **_kwargs: {"summary": [{"name": "Bird", "score": 0.8}]},
    )
    monkeypatch.setattr(
        cli,
        "transcribe_speech",
        lambda *_args, **_kwargs: {"text": "thank you", "segments": []},
    )

    result = cli.cmd_listen(
        Namespace(
            input=str(source),
            out_dir=str(out_dir),
            sound_model=None,
            speech_model="tiny",
            language=None,
            no_speech=False,
            no_vad=False,
            max_results=8,
            score_threshold=0.05,
        )
    )

    assert result == 0
    payload = json.loads((out_dir / "hearing.json").read_text(encoding="utf-8"))
    assert payload["processing"] == "local"
    assert payload["environment"]["summary"][0]["name"] == "Bird"
    assert payload["speech"]["text"] == "thank you"
    assert json.loads(capsys.readouterr().out)["processing"] == "local"


def test_doctor_returns_valid_json():
    result = _run_module("doctor")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert "checks" in payload
    assert "warnings" in payload


def test_doctor_reports_local_perception_dependencies():
    result = _run_module("doctor")
    assert result.returncode == 0
    checks = json.loads(result.stdout)["checks"]
    assert "mediapipe" in checks
    assert "faster_whisper" in checks


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
