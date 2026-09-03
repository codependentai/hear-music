from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from hear_music.analysis import load_wav_mono, run_ffmpeg

UrlOpener = Callable[[str], BinaryIO]
YAMNET_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
YAMNET_SHA256 = "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf"


def _open_url(url: str) -> BinaryIO:
    return urllib.request.urlopen(url, timeout=60)


def default_model_cache() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "hear-music" / "models"


def resolve_yamnet_model(model_path: Path | None = None) -> Path:
    """Resolve an explicit YAMNet model or fetch the pinned default model."""
    if model_path is not None:
        resolved = model_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"YAMNet model not found: {resolved}")
        return resolved

    destination = default_model_cache() / "yamnet-float32-v1.tflite"
    if destination.is_file():
        try:
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        except FileNotFoundError:
            pass
        else:
            if digest == YAMNET_SHA256:
                return destination
    return download_model(YAMNET_URL, destination, YAMNET_SHA256)


def download_model(
    url: str,
    destination: Path,
    expected_sha256: str,
    *,
    opener: UrlOpener = _open_url,
    max_bytes: int = 16 * 1024 * 1024,
) -> Path:
    """Download, verify, and atomically install a local inference model."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, partial_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(descriptor)
    partial = Path(partial_name)
    digest = hashlib.sha256()
    downloaded_bytes = 0
    try:
        with opener(url) as source, partial.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                downloaded_bytes += len(chunk)
                if downloaded_bytes > max_bytes:
                    raise RuntimeError(
                        "YAMNet model download exceeds maximum size of "
                        f"{max_bytes} bytes"
                    )
                digest.update(chunk)
                target.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"Model checksum mismatch for {url}")
        os.replace(partial, destination)
    except OSError as exc:
        raise RuntimeError(f"Could not download YAMNet model from {url}: {exc}") from exc
    finally:
        partial.unlink(missing_ok=True)
    return destination


def _run_yamnet(
    samples: np.ndarray,
    sample_rate: int,
    model_path: Path,
    max_results: int,
    score_threshold: float,
) -> list[Any]:
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import audio
        from mediapipe.tasks.python.components import containers
    except ImportError as exc:
        raise RuntimeError(
            "Environmental classification requires the perception extra: "
            "pip install 'hear-music[perception]'"
        ) from exc

    clip = containers.AudioData.create_from_array(samples, sample_rate)
    options = audio.AudioClassifierOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        max_results=max_results,
        score_threshold=score_threshold,
    )
    with audio.AudioClassifier.create_from_options(options) as classifier:
        return classifier.classify(clip)


def classify_environment(
    input_path: Path,
    *,
    model_path: Path | None = None,
    max_results: int = 8,
    score_threshold: float = 0.05,
    runner: Callable[[np.ndarray, int, Path, int, float], list[Any]] | None = None,
) -> dict[str, Any]:
    """Classify an audio file into timestamped AudioSet events, locally."""
    source = input_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    resolved_model = resolve_yamnet_model(model_path)
    selected_runner = runner or _run_yamnet

    with tempfile.TemporaryDirectory(prefix="hear-music-") as temp_dir:
        normalized = Path(temp_dir) / "normalized-16k.wav"
        run_ffmpeg(
            [
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                "-vn",
                str(normalized),
            ]
        )
        sample_rate, samples = load_wav_mono(normalized)

    results = selected_runner(
        samples,
        sample_rate,
        resolved_model,
        max_results,
        score_threshold,
    )
    events = render_environment_results(results)
    return build_environment_payload(
        source=str(source),
        duration_seconds=(len(samples) / sample_rate if sample_rate else 0.0),
        events=events,
    )


def render_environment_results(results: list[Any]) -> list[dict[str, Any]]:
    """Convert MediaPipe classifier results into stable JSON-friendly events."""
    events: list[dict[str, Any]] = []
    for result in results:
        categories = result.classifications[0].categories
        events.append(
            {
                "timestamp_ms": int(result.timestamp_ms),
                "labels": [
                    {
                        "name": str(category.category_name),
                        "score": round(float(category.score), 6),
                    }
                    for category in categories
                ],
            }
        )
    return events


def build_environment_payload(
    *, source: str, duration_seconds: float, events: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "source": source,
        "processing": "local",
        "model": "MediaPipe YAMNet / AudioSet 521 classes",
        "duration_seconds": round(float(duration_seconds), 3),
        "summary": summarize_events(events),
        "events": events,
    }


def summarize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return each label's strongest score and timestamp, strongest first."""
    best: dict[str, dict[str, Any]] = {}
    for event in events:
        timestamp_ms = int(event["timestamp_ms"])
        for label in event.get("labels", []):
            name = str(label["name"])
            score = float(label["score"])
            if name not in best or score > float(best[name]["score"]):
                best[name] = {
                    "name": name,
                    "score": score,
                    "timestamp_ms": timestamp_ms,
                }
    return sorted(best.values(), key=lambda item: float(item["score"]), reverse=True)
