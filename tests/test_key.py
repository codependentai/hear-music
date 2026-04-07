"""Key estimation tests."""
from __future__ import annotations

from hear_music.analysis import compute_chroma_vector, estimate_key


def _key_of(audio, sample_rate: int):
    chroma = compute_chroma_vector(audio, sample_rate)
    return estimate_key(chroma)


# Note frequencies for clean tests (Hz)
# C major: C4, E4, G4
C4 = 261.63
E4 = 329.63
G4 = 392.00
# A minor: A4, C5, E5
A4 = 440.00
C5 = 523.25
E5 = 659.25
# F# major: F#4, A#4, C#5
FS4 = 369.99
AS4 = 466.16
CS5 = 554.37


def test_c_major_triad(chord_audio, sr):
    audio = chord_audio([C4, E4, G4], duration_s=2.0)
    key = _key_of(audio, sr)
    assert key is not None
    assert key["tonic"] == "C"
    assert key["mode"] == "major"
    assert key["confidence"] > 0.5


def test_a_minor_triad(chord_audio, sr):
    audio = chord_audio([A4, C5, E5], duration_s=2.0)
    key = _key_of(audio, sr)
    assert key is not None
    # A minor and C major share the same chroma — accept either tonic but
    # require the result to be one of the two relatives
    assert (key["tonic"], key["mode"]) in {("A", "minor"), ("C", "major")}


def test_fs_major_triad(chord_audio, sr):
    audio = chord_audio([FS4, AS4, CS5], duration_s=2.0)
    key = _key_of(audio, sr)
    assert key is not None
    assert key["tonic"] == "F#"
    assert key["mode"] == "major"


def test_silence_returns_none(silence, sr):
    audio = silence(1.0)
    key = _key_of(audio, sr)
    assert key is None
