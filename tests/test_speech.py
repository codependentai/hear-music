"""Tests for local speech recognition."""
from __future__ import annotations

from types import SimpleNamespace


def test_transcribe_speech_uses_vad_and_returns_timestamped_text(tmp_path):
    from hear_music.speech import transcribe_speech

    source = tmp_path / "voice.ogg"
    source.write_bytes(b"audio placeholder")
    captured = {}

    class FakeModel:
        def transcribe(self, path, **kwargs):
            captured["path"] = path
            captured.update(kwargs)
            segments = [
                SimpleNamespace(
                    start=0.25,
                    end=1.5,
                    text=" The sparrows are loud.",
                    avg_logprob=-0.1,
                    no_speech_prob=0.02,
                )
            ]
            info = SimpleNamespace(language="en", language_probability=0.99)
            return iter(segments), info

    payload = transcribe_speech(
        source,
        model_name="tiny",
        model_factory=lambda *_args, **_kwargs: FakeModel(),
    )

    assert captured["vad_filter"] is True
    assert captured["path"] == str(source.resolve())
    assert payload["processing"] == "local"
    assert payload["text"] == "The sparrows are loud."
    assert payload["segments"] == [
        {
            "start": 0.25,
            "end": 1.5,
            "text": "The sparrows are loud.",
            "avg_logprob": -0.1,
            "no_speech_prob": 0.02,
        }
    ]
