---
name: hear-music
description: Inspect and transform audio files with the hear-music CLI. Use when Codex needs to analyze music or audio into spectrograms, waveform/mel/chroma visualizations, MIDI, JSON note events, ffprobe metadata, or diagnose hear-music installation and PATH issues.
---

# Hear Music

Use the `hear-music` CLI to inspect audio files and produce machine-readable artifacts.

## Quick Start

1. Run `hear-music doctor` if command resolution, Python environment, or `ffmpeg` availability is uncertain.
2. Run `hear-music info <audio>` to inspect duration, streams, codec, and tags before heavier work.
3. Choose the narrowest command that fits the task:
   - `hear-music analyze <audio>` for the full pass: normalized WAV, spectrogram, MIDI, and `analysis.json`
   - `hear-music visualize <audio>` for waveform + mel spectrogram + chromagram
   - `hear-music spectrogram <audio>` for a fast `ffmpeg` spectrogram only
   - `hear-music midi-json <file.mid>` to turn MIDI into JSON note events
4. Prefer explicit output paths or output directories when the input path contains spaces or the user wants predictable artifacts.

## Command Patterns

Use quoted absolute paths for Windows paths with spaces:

```powershell
hear-music doctor
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

- Use `info` first when the user wants a quick read on duration, streams, embedded cover art, tags, or bitrate.
- Use `visualize` when the user wants AI-readable images of song structure or timbral changes.
- Use `analyze` when the user wants structured note events, a MIDI approximation, and a spectrogram bundle.
- Use `spectrogram` when the user only needs a fast frequency-over-time image and not the heavier analysis.

## Caveats

- Treat `analysis.json` and generated MIDI as approximate for full mixed songs. The transcription is most reliable on monophonic or lightly layered material.
- Expect better results from `visualize` than from note extraction on dense produced tracks with vocals, drums, bass, and harmony combined.
- Run `doctor` when an agent can only invoke the tool with `python -m`; the most common causes are stale `PATH`, multiple Python installs, or missing `ffmpeg`.
- If working inside the repo before install, `hear-music.cmd` in the repo root is also a valid local entry point.
