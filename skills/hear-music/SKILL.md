---
name: hear-music
description: Use when an agent needs to perceive speech, environmental sound, or music locally with the hear-music CLI.
---

# Hear Music

Use the `hear-music` CLI to perceive audio files locally and produce machine-readable artifacts. The `perception` extra adds YAMNet environmental classification and CPU-local faster-whisper speech recognition; no hosted transcription service is required.

## Quick Start

1. Run `hear-music doctor` if command resolution, Python environment, or `ffmpeg` availability is uncertain.
2. Run `hear-music info <audio>` to inspect duration, streams, codec, and tags before heavier work.
3. Choose the narrowest command that fits the task:
   - `hear-music listen <audio>` for unified environmental classification and speech recognition
   - `hear-music classify <audio>` for timestamped general environmental-sound labels
   - `hear-music transcribe <audio>` for timestamped CPU-local speech recognition
   - `hear-music analyze <audio>` for the full pass: normalized WAV, spectrogram, MIDI, and `analysis.json`
   - `hear-music visualize <audio>` for waveform + mel spectrogram + chromagram
   - `hear-music spectrogram <audio>` for a fast `ffmpeg` spectrogram only
   - `hear-music midi-json <file.mid>` to turn MIDI into JSON note events
4. Prefer explicit output paths or output directories when the input path contains spaces or the user wants predictable artifacts.

## Command Patterns

Use quoted absolute paths for Windows paths with spaces:

```powershell
hear-music doctor
hear-music listen "C:\path\to\recording.ogg" --out-dir "C:\path\to\hearing"
hear-music classify "C:\path\to\recording.ogg" --out "C:\path\to\environment.json"
hear-music transcribe "C:\path\to\recording.ogg" --out "C:\path\to\speech.json"
hear-music info "C:\path\to\song.mp3"
hear-music analyze "C:\path\to\song.mp3" --out-dir "C:\path\to\song_analysis"
hear-music visualize "C:\path\to\song.mp3" --out "C:\path\to\song_analysis\visualization.png"
hear-music midi-json "C:\path\to\transcription.mid"
```

If the launcher is unavailable but the package is installed, use the fallback:

```powershell
python -m hear_music --help
```

## Output Selection

- Use `listen` when the agent needs one local hearing pass across speech and the surrounding soundscape. It writes `hearing.json`, `environment.json`, and `speech.json`.
- Use `classify` when speech content is irrelevant or the user asks what sounds are present. It uses YAMNet / AudioSet labels and preserves timestamps and strongest moments.
- Use `transcribe` when the user wants words. It uses faster-whisper on CPU with VAD by default and preserves detected language, timestamps, and confidence-related fields.
- Use `info` first when the user wants a quick read on duration, streams, embedded cover art, tags, or bitrate.
- Use `visualize` when the user wants AI-readable images of song structure or timbral changes — most reliable on dense audio.
- Use `analyze` when the user wants structured tempo, key, onsets, chroma, note events, a MIDI approximation, and a spectrogram bundle.
- Use `spectrogram` when the user only needs a fast frequency-over-time image and not the heavier analysis.

## What `analyze` returns

`analysis.json` includes:

- `tempo_bpm` and `tempo_estimated` — tempo estimated from the onset envelope; the generated MIDI uses this tempo
- `key` — `{tonic, mode, confidence}` from a Krumhansl-Schmuckler match against the chroma vector, or `null`
- `onset_count` and `onset_times` — adaptive-thresholded spectral flux peaks (seconds)
- `chroma` — 12-bin pitch class energy distribution
- `notes` — onset-aware note events (repeated same-pitch notes are split when an onset fires)
- `librosa` block (only present when `librosa` is installed) — `librosa_tempo_bpm`, `beat_times`, `spectral_centroid_mean_hz`, refined `key_librosa`

## Caveats

- Environmental labels are model predictions, not direct identification. Treat weak, surprising, or contextually implausible labels as uncertain; use specialist models such as BirdNET when species-level recognition matters.
- The pinned YAMNet model and selected Whisper model download once and then run locally from cache. Install with `hear-music[perception]` before using `listen`, `classify`, or `transcribe`.
- Treat `notes`/MIDI as approximate. Onset-aware segmentation and tempo-aware MIDI help, but the autocorrelation pitch tracker is monophonic and best on solo voice / whistled melodies / lightly layered material.
- `tempo_bpm`, `key`, and `onset_count` are reliable enough on most material to feed into downstream reasoning.
- Expect better visual results from `visualize` than note extraction on dense produced tracks with vocals, drums, bass, and harmony combined.
- Run `doctor` when an agent can only invoke the tool with `python -m`; the most common causes are stale `PATH`, multiple Python installs, or missing `ffmpeg`.
- If working inside the repo before install, `hear-music.cmd` in the repo root is also a valid local entry point.
