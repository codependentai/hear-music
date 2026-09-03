from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from hear_music.analysis import (
    analyze_audio,
    generate_spectrogram,
    parse_midi,
    run_ffprobe_json,
)
from hear_music.perception import classify_environment
from hear_music.speech import transcribe_speech
from hear_music.visualization import create_visualization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hear-music",
        description="Perceive music, speech, and environmental sound from local audio files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze an audio file and write spectrogram, MIDI, and JSON artifacts.",
    )
    analyze_parser.add_argument("input", help="Input audio path")
    analyze_parser.add_argument(
        "--out-dir",
        help="Output directory. Defaults to a sibling folder named <input-stem>_analysis.",
    )

    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify general environmental sounds locally with YAMNet.",
    )
    classify_parser.add_argument("input", help="Input audio path")
    classify_parser.add_argument("--out", help="Optional JSON output path")
    classify_parser.add_argument("--model", help="Optional path to a YAMNet TFLite model")
    classify_parser.add_argument("--max-results", type=int, default=8)
    classify_parser.add_argument("--score-threshold", type=float, default=0.05)

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe speech locally on CPU with faster-whisper.",
    )
    transcribe_parser.add_argument("input", help="Input audio path")
    transcribe_parser.add_argument("--out", help="Optional JSON output path")
    transcribe_parser.add_argument("--model", default="tiny", help="Whisper model name or local path")
    transcribe_parser.add_argument("--language", help="ISO language code; defaults to auto-detection")
    transcribe_parser.add_argument("--device", default="cpu")
    transcribe_parser.add_argument("--compute-type", default="int8")
    transcribe_parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable voice activity detection (not recommended for environmental recordings)",
    )

    listen_parser = subparsers.add_parser(
        "listen",
        help="Run unified local environmental classification and speech recognition.",
    )
    listen_parser.add_argument("input", help="Input audio path")
    listen_parser.add_argument("--out-dir", help="Output directory for hearing.json")
    listen_parser.add_argument("--sound-model", help="Optional path to a YAMNet TFLite model")
    listen_parser.add_argument("--speech-model", default="tiny")
    listen_parser.add_argument("--language", help="ISO language code; defaults to auto-detection")
    listen_parser.add_argument("--no-speech", action="store_true")
    listen_parser.add_argument("--no-vad", action="store_true")
    listen_parser.add_argument("--max-results", type=int, default=8)
    listen_parser.add_argument("--score-threshold", type=float, default=0.05)

    spectrogram_parser = subparsers.add_parser(
        "spectrogram",
        help="Generate a simple spectrogram PNG using ffmpeg.",
    )
    spectrogram_parser.add_argument("input", help="Input audio path")
    spectrogram_parser.add_argument("--out", help="Optional output PNG path")

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Generate a waveform, mel spectrogram, and chromagram PNG.",
    )
    visualize_parser.add_argument("input", help="Input audio path")
    visualize_parser.add_argument("--out", help="Optional output PNG path")

    info_parser = subparsers.add_parser(
        "info",
        help="Return ffprobe metadata for an audio file as JSON.",
    )
    info_parser.add_argument("input", help="Input audio path")
    info_parser.add_argument("--out", help="Optional JSON output path")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Inspect the local environment for launcher, Python, ffmpeg, and optional dependency issues.",
    )
    doctor_parser.add_argument("--out", help="Optional JSON output path")

    midi_parser = subparsers.add_parser(
        "midi-json",
        help="Parse a MIDI file into note timing JSON.",
    )
    midi_parser.add_argument("input", help="Input MIDI path")
    midi_parser.add_argument("--out", help="Optional JSON output path")

    return parser


def resolve_analysis_dir(input_path: Path, out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir).expanduser().resolve()
    return (input_path.parent / f"{input_path.stem}_analysis").resolve()


def resolve_output_file(input_path: Path, out: str | None, suffix: str) -> Path:
    if out:
        return Path(out).expanduser().resolve()
    return (input_path.parent / f"{input_path.stem}_{suffix}").resolve()


def validate_output_paths(input_path: Path, output_paths: list[Path]) -> None:
    source = input_path.resolve()
    if any(output.resolve() == source for output in output_paths):
        raise ValueError("Output path must differ from input path")


def write_text_atomic(path: Path, text: str) -> None:
    """Write text without exposing a partially written artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cmd_analyze(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    out_dir = resolve_analysis_dir(input_path, args.out_dir)
    analysis = analyze_audio(input_path, out_dir)
    print(json.dumps(analysis, indent=2))
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        out_path = Path(args.out).expanduser().resolve() if args.out else None
        if out_path is not None:
            validate_output_paths(input_path, [out_path])
        payload = classify_environment(
            input_path,
            model_path=Path(args.model) if args.model else None,
            max_results=args.max_results,
            score_threshold=args.score_threshold,
        )
        rendered = json.dumps(payload, indent=2)
        if out_path is not None:
            write_text_atomic(out_path, rendered)
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(rendered)
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        out_path = Path(args.out).expanduser().resolve() if args.out else None
        if out_path is not None:
            validate_output_paths(input_path, [out_path])
        payload = transcribe_speech(
            input_path,
            model_name=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            vad_filter=not args.no_vad,
        )
        rendered = json.dumps(payload, indent=2, ensure_ascii=False)
        if out_path is not None:
            write_text_atomic(out_path, rendered)
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(rendered)
    return 0


def cmd_listen(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    out_dir = resolve_analysis_dir(input_path, args.out_dir)
    environment_path = out_dir / "environment.json"
    speech_path = out_dir / "speech.json"
    hearing_path = out_dir / "hearing.json"
    try:
        validate_output_paths(
            input_path, [environment_path, speech_path, hearing_path]
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        environment = classify_environment(
            input_path,
            model_path=Path(args.sound_model) if args.sound_model else None,
            max_results=args.max_results,
            score_threshold=args.score_threshold,
        )
        speech = None
        if not args.no_speech:
            speech = transcribe_speech(
                input_path,
                model_name=args.speech_model,
                language=args.language,
                device="cpu",
                compute_type="int8",
                vad_filter=not args.no_vad,
            )
        payload = {
            "source": str(input_path),
            "processing": "local",
            "environment": environment,
            "speech": speech,
        }
        write_text_atomic(
            environment_path,
            json.dumps(environment, indent=2, ensure_ascii=False),
        )
        if speech is not None:
            write_text_atomic(
                speech_path, json.dumps(speech, indent=2, ensure_ascii=False)
            )
        else:
            speech_path.unlink(missing_ok=True)
        rendered = json.dumps(payload, indent=2, ensure_ascii=False)
        write_text_atomic(hearing_path, rendered)
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(rendered)
    return 0


def cmd_spectrogram(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    out_path = resolve_output_file(input_path, args.out, "spectrogram.png")
    generate_spectrogram(input_path, out_path)
    print(json.dumps({"source": str(input_path), "spectrogram": str(out_path)}, indent=2))
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    out_path = resolve_output_file(input_path, args.out, "visualization.png")
    try:
        create_visualization(input_path, out_path)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps({"source": str(input_path), "visualization": str(out_path)}, indent=2))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    payload = run_ffprobe_json(input_path)
    rendered = json.dumps(payload, indent=2)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    checks = {
        "python_executable": sys.executable,
        "hear_music_module": bool(importlib.util.find_spec("hear_music")),
        "hear_music_command": shutil.which("hear-music"),
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "librosa": bool(importlib.util.find_spec("librosa")),
        "matplotlib": bool(importlib.util.find_spec("matplotlib")),
        "mediapipe": bool(importlib.util.find_spec("mediapipe")),
        "faster_whisper": bool(importlib.util.find_spec("faster_whisper")),
    }

    warnings: list[str] = []
    if not checks["hear_music_command"]:
        warnings.append("The hear-music launcher is not on PATH. Open a new terminal or reinstall with install.cmd.")
    if not checks["ffmpeg"] or not checks["ffprobe"]:
        warnings.append("ffmpeg/ffprobe were not found on PATH. Install ffmpeg before using audio commands.")
    if not checks["librosa"] or not checks["matplotlib"]:
        warnings.append("Optional visualize dependencies are missing. Install with: python -m pip install -e .[visualize]")
    if not checks["mediapipe"] or not checks["faster_whisper"]:
        warnings.append("Optional local perception dependencies are missing. Install with: python -m pip install -e .[perception]")

    payload = {
        "ok": len(warnings) == 0,
        "checks": checks,
        "warnings": warnings,
        "fallback": {
            "module_invocation": f'"{sys.executable}" -m hear_music --help',
        },
    }
    rendered = json.dumps(payload, indent=2)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


def cmd_midi_json(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    payload = parse_midi(input_path)
    rendered = json.dumps(payload, indent=2)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "classify":
        return cmd_classify(args)
    if args.command == "transcribe":
        return cmd_transcribe(args)
    if args.command == "listen":
        return cmd_listen(args)
    if args.command == "spectrogram":
        return cmd_spectrogram(args)
    if args.command == "visualize":
        return cmd_visualize(args)
    if args.command == "info":
        return cmd_info(args)
    if args.command == "doctor":
        return cmd_doctor(args)
    if args.command == "midi-json":
        return cmd_midi_json(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
