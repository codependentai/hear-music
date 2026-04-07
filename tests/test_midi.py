"""MIDI encode/decode round-trip tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from hear_music.analysis import (
    NoteEvent,
    encode_var_len,
    midi_to_name,
    parse_midi,
    write_midi,
)


def test_encode_var_len_zero():
    assert encode_var_len(0) == b"\x00"


def test_encode_var_len_127():
    assert encode_var_len(127) == b"\x7f"


def test_encode_var_len_128():
    assert encode_var_len(128) == b"\x81\x00"


def test_encode_var_len_large():
    # 0x200000 (2097152) is a known boundary case
    assert encode_var_len(0x200000) == b"\x81\x80\x80\x00"


def test_midi_to_name_a4():
    assert midi_to_name(69) == "A4"
    assert midi_to_name(60) == "C4"
    assert midi_to_name(72) == "C5"


def test_round_trip_single_note(tmp_path: Path):
    note = NoteEvent(
        pitch=60,
        name="C4",
        frequency_hz=261.63,
        start=0.0,
        duration=0.5,
        end=0.5,
        velocity=80,
        confidence=0.9,
    )
    midi_path = tmp_path / "single.mid"
    write_midi(midi_path, [note], bpm=120)

    parsed = parse_midi(midi_path)
    assert parsed["note_count"] == 1
    assert parsed["tempo_bpm"] == 120.0
    parsed_note = parsed["notes"][0]
    assert parsed_note["pitch"] == 60
    assert parsed_note["name"] == "C4"
    assert abs(parsed_note["duration"] - 0.5) < 0.02


def test_round_trip_multiple_notes(tmp_path: Path):
    notes = [
        NoteEvent(60, "C4", 261.63, 0.0, 0.25, 0.25, 80, 0.9),
        NoteEvent(64, "E4", 329.63, 0.25, 0.25, 0.5, 80, 0.9),
        NoteEvent(67, "G4", 392.00, 0.5, 0.25, 0.75, 80, 0.9),
    ]
    midi_path = tmp_path / "triad.mid"
    write_midi(midi_path, notes, bpm=100)

    parsed = parse_midi(midi_path)
    assert parsed["note_count"] == 3
    assert parsed["tempo_bpm"] == 100.0
    pitches = sorted(n["pitch"] for n in parsed["notes"])
    assert pitches == [60, 64, 67]


def test_tempo_round_trip(tmp_path: Path):
    note = NoteEvent(60, "C4", 261.63, 0.0, 0.5, 0.5, 80, 0.9)
    for bpm in (60, 90, 140, 180):
        path = tmp_path / f"tempo_{bpm}.mid"
        write_midi(path, [note], bpm=bpm)
        parsed = parse_midi(path)
        assert parsed["tempo_bpm"] == float(bpm), f"failed at {bpm} BPM"


def test_parse_midi_rejects_non_midi(tmp_path: Path):
    bad = tmp_path / "fake.mid"
    bad.write_bytes(b"NOT_MIDI_DATA")
    with pytest.raises(ValueError):
        parse_midi(bad)
