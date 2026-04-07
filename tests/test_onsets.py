"""Onset detection tests."""
from __future__ import annotations

import numpy as np

from hear_music.analysis import compute_onset_envelope, detect_onsets


def test_click_track_produces_expected_onsets(click_track, sr):
    audio = click_track(bpm=60, duration_s=5.0)  # one click per second
    envelope = compute_onset_envelope(audio, sr)
    onsets = detect_onsets(envelope, sr)
    # Expect roughly 5 onsets, allow ±1 for edge effects
    assert 4 <= len(onsets) <= 6, f"expected 4-6 onsets at 60 BPM over 5s, got {len(onsets)}"


def test_faster_click_track_produces_more_onsets(click_track, sr):
    slow = click_track(bpm=60, duration_s=4.0)
    fast = click_track(bpm=180, duration_s=4.0)
    slow_onsets = detect_onsets(compute_onset_envelope(slow, sr), sr)
    fast_onsets = detect_onsets(compute_onset_envelope(fast, sr), sr)
    assert len(fast_onsets) > len(slow_onsets), (
        f"180 BPM ({len(fast_onsets)}) should yield more onsets than 60 BPM ({len(slow_onsets)})"
    )


def test_sustained_tone_has_minimal_onsets(sine_wave, sr):
    audio = sine_wave(440.0, 2.0)
    envelope = compute_onset_envelope(audio, sr)
    onsets = detect_onsets(envelope, sr)
    # A sustained tone should produce at most one onset (the attack)
    assert len(onsets) <= 1, f"sustained tone should not produce many onsets, got {len(onsets)}"


def test_silence_produces_no_onsets(silence, sr):
    audio = silence(2.0)
    envelope = compute_onset_envelope(audio, sr)
    onsets = detect_onsets(envelope, sr)
    assert len(onsets) == 0


def test_separated_notes_produce_distinct_onsets(note_sequence, sr):
    audio = note_sequence(
        [(440.0, 0.3), (440.0, 0.3), (440.0, 0.3), (440.0, 0.3)],
        gap_s=0.1,
    )
    envelope = compute_onset_envelope(audio, sr)
    onsets = detect_onsets(envelope, sr)
    # Four notes with silence between → at least 3 onsets (the first onset
    # at sample 0 may or may not be picked depending on flux init)
    assert len(onsets) >= 3, f"expected at least 3 onsets for 4 separated notes, got {len(onsets)}"
