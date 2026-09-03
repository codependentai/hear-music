"""Tests for environmental and speech perception helpers."""
from __future__ import annotations

import hashlib
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest


def test_classify_environment_runs_local_pipeline(
    write_wav, sine_wave, tmp_path
):
    from hear_music.perception import classify_environment

    source = write_wav(sine_wave(440.0, 1.0), name="environment.wav")
    model = tmp_path / "yamnet.tflite"
    model.write_bytes(b"test model")

    def fake_runner(samples, sample_rate, resolved_model, max_results, score_threshold):
        assert len(samples) == 16000
        assert sample_rate == 16000
        assert resolved_model == model.resolve()
        assert max_results == 8
        assert score_threshold == 0.05
        return [
            SimpleNamespace(
                timestamp_ms=0,
                classifications=[
                    SimpleNamespace(
                        categories=[SimpleNamespace(category_name="Bird", score=0.75)]
                    )
                ],
            )
        ]

    payload = classify_environment(source, model_path=model, runner=fake_runner)

    assert payload["processing"] == "local"
    assert payload["duration_seconds"] == 1.0
    assert payload["summary"][0]["name"] == "Bird"


def test_resolve_yamnet_model_prefers_explicit_local_file(tmp_path):
    from hear_music.perception import resolve_yamnet_model

    supplied = tmp_path / "my-yamnet.tflite"
    supplied.write_bytes(b"user supplied model")

    assert resolve_yamnet_model(supplied) == supplied.resolve()


def test_render_environment_results_preserves_timestamps_and_ranked_labels():
    from hear_music.perception import render_environment_results

    result = SimpleNamespace(
        timestamp_ms=975,
        classifications=[
            SimpleNamespace(
                categories=[
                    SimpleNamespace(category_name="Bird", score=0.5859375),
                    SimpleNamespace(category_name="Chirp, tweet", score=0.33203125),
                ]
            )
        ],
    )

    assert render_environment_results([result]) == [
        {
            "timestamp_ms": 975,
            "labels": [
                {"name": "Bird", "score": 0.585938},
                {"name": "Chirp, tweet", "score": 0.332031},
            ],
        }
    ]


def test_download_model_verifies_hash_before_installing(tmp_path):
    from hear_music.perception import download_model

    payload = b"tiny local model"
    expected = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "models" / "model.tflite"

    result = download_model(
        "https://example.invalid/model.tflite",
        destination,
        expected,
        opener=lambda _url: io.BytesIO(payload),
    )

    assert result == destination
    assert destination.read_bytes() == payload


def test_download_model_wraps_network_errors_and_removes_partial_file(tmp_path):
    from hear_music.perception import download_model

    destination = tmp_path / "models" / "model.tflite"

    def fail(_url):
        raise OSError("network unavailable")

    with pytest.raises(RuntimeError, match="Could not download YAMNet model"):
        download_model(
            "https://example.invalid/model.tflite",
            destination,
            "0" * 64,
            opener=fail,
        )

    assert not list(destination.parent.glob("*.part"))


def test_download_model_rejects_checksum_mismatch_without_installing(tmp_path):
    from hear_music.perception import download_model

    destination = tmp_path / "models" / "model.tflite"
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download_model(
            "https://example.invalid/model.tflite",
            destination,
            "0" * 64,
            opener=lambda _url: io.BytesIO(b"wrong bytes"),
        )

    assert not destination.exists()
    assert not list(destination.parent.glob("*.part"))


def test_download_model_uses_bounded_timeout_for_default_fetch(tmp_path, monkeypatch):
    from hear_music import perception

    payload = b"tiny local model"
    expected = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "models" / "model.tflite"
    calls = []

    def fake_urlopen(url, *, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(perception.urllib.request, "urlopen", fake_urlopen)

    perception.download_model(
        "https://example.invalid/model.tflite",
        destination,
        expected,
    )

    assert calls == [("https://example.invalid/model.tflite", 60.0)]


def test_download_model_rejects_oversize_stream_and_removes_partial_file(tmp_path):
    from hear_music.perception import download_model

    payload = b"x" * (16 * 1024 * 1024 + 1)
    destination = tmp_path / "models" / "model.tflite"
    destination.parent.mkdir()
    destination.write_bytes(b"existing verified model")

    with pytest.raises(RuntimeError, match="exceeds maximum size"):
        download_model(
            "https://example.invalid/model.tflite",
            destination,
            hashlib.sha256(payload).hexdigest(),
            opener=lambda _url: io.BytesIO(payload),
        )

    assert destination.read_bytes() == b"existing verified model"
    assert not list(destination.parent.glob("*.part"))


def test_concurrent_model_downloads_use_independent_partial_files(tmp_path):
    from hear_music.perception import download_model

    payload = b"same verified model" * 100
    expected = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "models" / "model.tflite"
    barrier = threading.Barrier(2)

    class BlockingStream(io.BytesIO):
        def read(self, size: int | None = -1):
            if self.tell() == 0:
                barrier.wait(timeout=3)
            return super().read(size)

    def fetch():
        return download_model(
            "https://example.invalid/model.tflite",
            destination,
            expected,
            opener=lambda _url: BlockingStream(payload),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: fetch(), range(2)))

    assert results == [destination, destination]
    assert destination.read_bytes() == payload
    assert not list(destination.parent.glob("*.part"))


def test_concurrent_resolution_recovers_corrupt_cache_without_disappearing(
    tmp_path, monkeypatch
):
    from hear_music import perception

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    destination = model_dir / "yamnet-float32-v1.tflite"
    destination.write_bytes(b"corrupt")
    payload = b"verified replacement"
    expected = hashlib.sha256(payload).hexdigest()
    read_barrier = threading.Barrier(2)
    original_read_bytes = type(destination).read_bytes
    real_download = perception.download_model

    def synchronized_read(path):
        data = original_read_bytes(path)
        if path == destination:
            read_barrier.wait(timeout=3)
        return data

    def fake_download(url, path, checksum):
        return real_download(
            url,
            path,
            checksum,
            opener=lambda _url: io.BytesIO(payload),
        )

    monkeypatch.setattr(perception, "default_model_cache", lambda: model_dir)
    monkeypatch.setattr(perception, "YAMNET_SHA256", expected)
    monkeypatch.setattr(type(destination), "read_bytes", synchronized_read)
    monkeypatch.setattr(perception, "download_model", fake_download)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: perception.resolve_yamnet_model(), range(2)))

    assert results == [destination, destination]
    assert original_read_bytes(destination) == payload


def test_build_environment_payload_includes_summary_and_local_provenance():
    from hear_music.perception import build_environment_payload

    events = [
        {"timestamp_ms": 0, "labels": [{"name": "Bird", "score": 0.2}]},
        {"timestamp_ms": 975, "labels": [{"name": "Bird", "score": 0.8}]},
    ]

    payload = build_environment_payload(
        source="courtyard.ogg",
        duration_seconds=1.95,
        events=events,
    )

    assert payload["source"] == "courtyard.ogg"
    assert payload["processing"] == "local"
    assert payload["model"] == "MediaPipe YAMNet / AudioSet 521 classes"
    assert payload["summary"] == [
        {"name": "Bird", "score": 0.8, "timestamp_ms": 975}
    ]


def test_summarize_events_keeps_each_labels_strongest_moment():
    from hear_music.perception import summarize_events

    events = [
        {"timestamp_ms": 0, "labels": [{"name": "Bird", "score": 0.2}]},
        {
            "timestamp_ms": 975,
            "labels": [
                {"name": "Bird", "score": 0.8},
                {"name": "Water", "score": 0.4},
            ],
        },
        {"timestamp_ms": 1950, "labels": [{"name": "Water", "score": 0.3}]},
    ]

    assert summarize_events(events) == [
        {"name": "Bird", "score": 0.8, "timestamp_ms": 975},
        {"name": "Water", "score": 0.4, "timestamp_ms": 975},
    ]
