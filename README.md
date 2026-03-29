# hear-music

`hear-music` is a local CLI that lets an agent turn an audio file into:

- a normalized WAV
- a spectrogram image
- a richer waveform + mel + chroma visualization
- a rough note transcription in JSON
- a simple MIDI file
- a JSON parser for `.mid` files
- audio metadata from `ffprobe`

It is intentionally lightweight and built around `ffmpeg`, `numpy`, and `scipy` so it can run locally before being wrapped in Modal or another service.

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

Choose an output directory:

```powershell
hear-music analyze .\input.mp3 --out-dir .\analysis
```

## Outputs

`analyze` writes:

- `normalized.wav`
- `spectrogram.png`
- `transcription.mid`
- `analysis.json`

`visualize` writes:

- `visualization.png`

The note extraction is intentionally simple. It works best for monophonic or lightly layered material and should be treated as a first-pass "hearing" layer rather than high-accuracy music transcription.

## Commands

- `analyze`: full local pass that writes WAV, spectrogram, MIDI, and JSON
- `spectrogram`: fast `ffmpeg` spectrogram only
- `visualize`: waveform + mel spectrogram + chromagram PNG
- `info`: return `ffprobe` metadata as JSON
- `midi-json`: parse MIDI note events into JSON

## Publishing Notes

Before pushing to GitHub, I recommend adding:

- one or two example screenshots in a `docs/` folder
