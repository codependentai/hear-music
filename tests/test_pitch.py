"""Pitch detection tests — covers existing autocorrelation pitch tracker."""
from __future__ import annotations

import numpy as np

from hear_music.analysis import (
    estimate_pitch,
    extract_notes,
    frequency_to_midi,
)


def test_a4_sine_detected_as_midi_69(sine_wave, sr):
    audio = sine_wave(440.0, 1.0)
    notes = extract_notes(audio, sr)
    assert notes, "expected at least one note from a sustained 440 Hz tone"
    pitches = [n.pitch for n in notes]
    assert 69 in pitches, f"expected MIDI 69 (A4), got {pitches}"


def test_c4_sine_detected_as_midi_60(sine_wave, sr):
    audio = sine_wave(261.63, 1.0)
    notes = extract_notes(audio, sr)
    assert notes
    assert any(n.pitch == 60 for n in notes), f"expected C4 (60), got {[n.pitch for n in notes]}"


def test_silence_yields_no_notes(silence, sr):
    audio = silence(1.0)
    notes = extract_notes(audio, sr)
    assert notes == []


def test_below_min_freq_rejected(sine_wave, sr):
    # 40 Hz is below MIN_FREQ (55 Hz)
    audio = sine_wave(40.0, 1.0)
    notes = extract_notes(audio, sr)
    assert all(n.frequency_hz >= 55.0 for n in notes), "should not emit notes below MIN_FREQ"


def test_above_max_freq_rejected(sine_wave, sr):
    # 2000 Hz is above MAX_FREQ (1760 Hz)
    audio = sine_wave(2000.0, 1.0)
    notes = extract_notes(audio, sr)
    assert all(n.frequency_hz <= 1760.0 for n in notes), "should not emit notes above MAX_FREQ"


def test_pitch_confidence_high_on_clean_sine(sine_wave, sr):
    audio = sine_wave(440.0, 0.2)
    # Take the middle frame
    from hear_music.analysis import FRAME_SIZE
    mid = len(audio) // 2
    frame = audio[mid - FRAME_SIZE // 2 : mid + FRAME_SIZE // 2]
    freq, conf = estimate_pitch(frame, sr)
    assert freq is not None
    assert conf > 0.5, f"expected high confidence on clean sine, got {conf}"


def test_frequency_to_midi_a440():
    assert frequency_to_midi(440.0) == 69
    assert frequency_to_midi(261.63) == 60
