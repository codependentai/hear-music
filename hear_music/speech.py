from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


def _default_model_factory(model_name: str, *, device: str, compute_type: str) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Local transcription requires the perception extra: "
            "pip install 'hear-music[perception]'"
        ) from exc
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe_speech(
    input_path: Path,
    *,
    model_name: str = "tiny",
    language: str | None = None,
    device: str = "cpu",
    compute_type: str = "int8",
    vad_filter: bool = True,
    model_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Transcribe speech locally with faster-whisper and VAD by default."""
    source = input_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    factory = model_factory or _default_model_factory
    model = factory(model_name, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        str(source),
        beam_size=5,
        language=language,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
    )
    segments = [
        {
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "text": str(segment.text).strip(),
            "avg_logprob": round(float(segment.avg_logprob), 6),
            "no_speech_prob": round(float(segment.no_speech_prob), 6),
        }
        for segment in segments_iter
        if str(segment.text).strip()
    ]
    return {
        "source": str(source),
        "processing": "local",
        "model": f"faster-whisper/{model_name}",
        "device": device,
        "compute_type": compute_type,
        "vad_filter": vad_filter,
        "language": getattr(info, "language", None),
        "language_probability": round(
            float(getattr(info, "language_probability", 0.0)), 6
        ),
        "text": " ".join(segment["text"] for segment in segments),
        "segments": segments,
    }
