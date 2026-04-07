---
name: hear-music
description: Use the hear-music CLI to perceive audio files. Trigger when the user wants to analyze, transcribe, visualize, or inspect music or audio — generating spectrograms, mel/chroma/waveform images, MIDI transcription, JSON note events with onset/tempo/key estimation, ffprobe metadata, or diagnose hear-music install issues. Examples — "what's the BPM of this track", "make a spectrogram of song.mp3", "transcribe this melody to MIDI", "what key is this in", "show me the structure of this audio file".
---

# hear-music

Use the `hear-music` CLI to give Claude Code a way to *hear* an audio file. It produces machine-readable artifacts that can be reasoned about: spectrograms, visualizations, ffprobe metadata, onset times, tempo, key, chroma, note events, and MIDI.

## When to use this skill

- The user references an audio file (`.mp3`, `.wav`, `.flac`, `.m4a`, etc.) and wants any kind of analysis
- The user asks about tempo, key, BPM, structure, onsets, or notes in a piece of audio
- The user wants a visual representation of audio (spectrogram, waveform, mel, chromagram)
- The user wants to transcribe a melody or extract MIDI
- The user is debugging audio metadata, codec, duration, or tags

## Quick start

1. Run `hear-music doctor` when command resolution, Python environment, or `ffmpeg` availability is uncertain — it returns JSON describing the install state.
2. Run `hear-music info <audio>` for a fast read on duration, streams, codec, and tags before heavier work.
3. Pick the narrowest command that fits the task:
   - `hear-music analyze <audio>` — full pass: normalized WAV, spectrogram, MIDI, and `analysis.json` (tempo, key, onsets, chroma, notes)
   - `hear-music visualize <audio>` — waveform + mel spectrogram + chromagram PNG (most reliable on dense audio)
   - `hear-music spectrogram <audio>` — fast `ffmpeg` spectrogram only
   - `hear-music midi-json <file.mid>` — turn MIDI into JSON note events
4. Always pass explicit output paths or `--out-dir` when the input path contains spaces or the user wants predictable artifacts.

## Command patterns

Quote absolute paths on Windows when they contain spaces:

```bash
hear-music doctor
hear-music info "C:/path/to/song.mp3"
hear-music analyze "C:/path/to/song.mp3" --out-dir "C:/path/to/song_analysis"
hear-music visualize "C:/path/to/song.mp3" --out "C:/path/to/song_analysis/visualization.png"
hear-music midi-json "C:/path/to/transcription.mid"
```

If the launcher is unavailable but the package is installed, use the fallback:

```bash
python -m hear_music --help
```

## Reading `analysis.json`

The `analyze` command writes a JSON document with these fields you can reason about:

- **`tempo_bpm`** — estimated tempo (BPM). Reliable on most material with a clear pulse.
- **`tempo_estimated`** — `false` means the estimator couldn't decide and fell back to 120.
- **`key`** — `{tonic, mode, confidence}` from Krumhansl-Schmuckler chroma matching. Confidence above ~0.6 is solid; below that, treat as a guess.
- **`onset_count`** / **`onset_times`** — onset positions in seconds, after adaptive thresholding. Useful for slicing audio or counting events.
- **`chroma`** — 12-bin pitch class energy distribution (C, C#, D, ..., B).
- **`notes`** — onset-aware monophonic note events. Reliable on solo voice / whistled melodies / lightly layered material; treat as a sketch on dense produced tracks.
- **`librosa`** — only present when `librosa` is installed. Adds `librosa_tempo_bpm`, `beat_times`, `spectral_centroid_mean_hz`, and a refined `key_librosa`. Use these in preference to the lean estimates when present.

## Output selection guidance

| User wants… | Command |
|---|---|
| Quick read on duration, streams, codec, tags | `info` |
| Structure of a complex track (vocals + drums + harmony) | `visualize` |
| Tempo, key, BPM, onsets, structured analysis | `analyze` |
| Just a spectrogram image, fast | `spectrogram` |
| Convert a `.mid` file to JSON | `midi-json` |
| Diagnose install / PATH problems | `doctor` |

## Caveats

- Treat note transcription and generated MIDI as approximate. The pitch tracker is monophonic. Best on solo voice, whistled melodies, lightly layered material. Will struggle on dense produced tracks.
- `tempo_bpm`, `key`, `onset_count`, and `chroma` are reliable enough on most material to use directly in reasoning.
- `visualize` requires the `[visualize]` extras (`librosa` + `matplotlib`). Run `doctor` if it fails — the JSON output will tell you what's missing.
- Run `doctor` whenever the launcher behaves oddly. The most common causes are stale `PATH`, multiple Python installs, or missing `ffmpeg`.

## Install

If hear-music is not installed:

```bash
pipx install "git+https://github.com/codependentai/hear-music.git"
```

With visualizer extras:

```bash
pipx install "hear-music[visualize] @ git+https://github.com/codependentai/hear-music.git"
```

Requires Python 3.11+ and `ffmpeg` on `PATH`.
