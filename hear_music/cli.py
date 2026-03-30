from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

from hear_music.analysis import analyze_audio, generate_spectrogram, parse_midi, run_ffprobe_json
from hear_music.visualization import create_visualization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hear-music",
        description="Generate spectrograms, MIDI, and JSON from audio or MIDI files.",
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


def cmd_analyze(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    out_dir = resolve_analysis_dir(input_path, args.out_dir)
    analysis = analyze_audio(input_path, out_dir)
    print(json.dumps(analysis, indent=2))
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
    }

    warnings: list[str] = []
    if not checks["hear_music_command"]:
        warnings.append("The hear-music launcher is not on PATH. Open a new terminal or reinstall with install.cmd.")
    if not checks["ffmpeg"] or not checks["ffprobe"]:
        warnings.append("ffmpeg/ffprobe were not found on PATH. Install ffmpeg before using audio commands.")
    if not checks["librosa"] or not checks["matplotlib"]:
        warnings.append("Optional visualize dependencies are missing. Install with: python -m pip install -e .[visualize]")

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
