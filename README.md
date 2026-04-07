# hear-music

`hear-music` is a local CLI that lets an agent *hear* an audio file. It produces machine-readable artifacts an LLM can reason about:

- audio metadata from `ffprobe`
- a spectrogram image
- a richer waveform + mel + chroma visualization
- onset times, estimated tempo, and estimated key in JSON
- a normalized WAV
- a rough note transcription in JSON
- a simple MIDI file (using the estimated tempo)
- a JSON parser for `.mid` files

It is intentionally lightweight and built around `ffmpeg`, `numpy`, and `scipy` so it can run locally before being wrapped in Modal or another service. Optional `librosa` extras unlock a richer enrichment block in `analyze`.

> **What it does well:** spectrograms, visualizations, ffprobe metadata, onset detection, tempo estimation, and key estimation on most material.
>
> **What it does roughly:** note-level MIDI transcription. The pitch tracker is an autocorrelation-based monophonic estimator — treat the MIDI output as a "first pass hearing" rather than high-accuracy transcription. It works well on solo voice, whistled melodies, and lightly layered material; it will struggle with dense produced tracks.

Source-available — free for personal and educational use, commercial use requires a license. See [LICENSE](./LICENSE).

## Install

After install, the goal is simple: you can open a new terminal and run:

```powershell
hear-music --help
```

### Local Clone

```powershell
cd C:\AI\tools\hear-music
.\install.cmd
```

Optional richer visualizer support:

```powershell
.\install.cmd -Visualize
```

That installer:

- installs the package
- adds the Python scripts directory to your user `PATH`
- makes `hear-music` available as a global command in new terminals

### GitHub Install

The easiest Windows install from GitHub is:

```powershell
git clone https://github.com/codependentai/hear-music.git
cd hear-music
.\install.cmd
```

Or, if they want the cleanest command-line app install, use `pipx`:

```powershell
pipx install "git+https://github.com/codependentai/hear-music.git"
```

With visualizer extras:

```powershell
pipx install "hear-music[visualize] @ git+https://github.com/codependentai/hear-music.git"
```

If they use `pipx` for the first time, they may need:

```powershell
pipx ensurepath
```

## Requirements

- Python 3.11+
- `ffmpeg` available on `PATH`

The richer `visualize` command also installs:

- `librosa`
- `matplotlib`

## Usage

Analyze audio:

```powershell
hear-music analyze .\input.mp3
```

From this folder, the checked-in wrapper also works:

```powershell
.\hear-music.cmd analyze .\input.mp3
```

Parse a MIDI file into JSON:

```powershell
hear-music midi-json .\transcription.mid
```

Generate a simple spectrogram with `ffmpeg`:

```powershell
hear-music spectrogram .\input.mp3
```

Generate the richer 3-panel visual summary:

```powershell
hear-music visualize .\input.mp3
```

Inspect audio metadata:

```powershell
hear-music info .\input.mp3
```

Diagnose installation and environment issues:

```powershell
hear-music doctor
```

Choose an output directory:

```powershell
hear-music analyze .\input.mp3 --out-dir .\analysis
```

## Outputs

`analyze` writes:

- `normalized.wav`
- `spectrogram.png`
- `transcription.mid` (uses the estimated tempo, not a hard-coded 120 BPM)
- `analysis.json`

`visualize` writes:

- `visualization.png`

### `analysis.json` fields

```jsonc
{
  "source": "...",
  "duration": 12.34,
  "sample_rate": 22050,
  "tempo_bpm": 128.5,            // estimated via onset envelope autocorrelation
  "tempo_estimated": true,       // false ⇒ fell back to default 120
  "key": {                       // Krumhansl-Schmuckler estimate, may be null
    "tonic": "D",
    "mode": "minor",
    "confidence": 0.74
  },
  "onset_count": 47,
  "onset_times": [0.12, 0.51, 0.89, ...],
  "chroma": [0.06, 0.04, ...],   // 12-bin pitch class energy distribution
  "note_count": 21,
  "notes": [ /* NoteEvent records */ ],
  "files": { "normalized_wav": "...", "spectrogram": "...", "midi": "..." },
  "librosa": {                   // present only if librosa is installed
    "librosa_tempo_bpm": 128.0,
    "beat_times": [...],
    "spectral_centroid_mean_hz": 1843.2,
    "key_librosa": { "tonic": "D", "mode": "minor", "confidence": 0.81 }
  }
}
```

The note extraction is intentionally simple. Onset-aware segmentation now splits repeated same-pitch notes, and the MIDI is written at the estimated tempo, but it should still be treated as a first-pass "hearing" layer rather than high-accuracy music transcription.

## Commands

- `info`: return `ffprobe` metadata as JSON — start here for a quick read on duration, codec, tags
- `visualize`: waveform + mel spectrogram + chromagram PNG (most reliable for dense audio)
- `analyze`: full local pass — WAV, spectrogram, MIDI, onsets, tempo, key, chroma, JSON
- `spectrogram`: fast `ffmpeg` spectrogram only
- `doctor`: inspect PATH, Python, ffmpeg, and optional dependency issues
- `midi-json`: parse MIDI note events into JSON

## Codex Skill

This repo also includes a Codex skill at `skills/hear-music/` so agentic workflows can discover and use the tool with a purpose-built workflow instead of treating it like an unknown CLI.

## Troubleshooting

If `hear-music` is not recognized as a command:

```powershell
hear-music doctor
```

If the launcher still is not available in the current terminal, open a new terminal and try again.

As a fallback, you can invoke the installed module directly:

```powershell
python -m hear_music --help
```

Common causes:

- the terminal was opened before install and has stale `PATH`
- multiple Python installs are present and the package was installed into a different interpreter
- `ffmpeg` is missing from `PATH`

## Development

Install with the dev extras and run the test suite:

```powershell
python -m pip install -e .[dev]
python -m pytest
```

The test suite uses synthetic in-memory audio (sine waves, click tracks, chord triads) so it is fast, deterministic, and free of external sample files. It runs in well under a minute.

## Publishing Notes

Before pushing to GitHub, I recommend adding:

- one or two example screenshots in a `docs/` folder
