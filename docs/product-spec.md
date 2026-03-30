# hear-music Product Spec

## Positioning

`hear-music` should evolve from a useful local CLI into an audio-perception layer for agents and creative tools.

The core idea is not just "analyze a song." It is:

- turn audio into structured machine-readable understanding
- expose that understanding through one stable interface
- let better analyzers plug in behind the same schema over time

In short: `hear-music` becomes to audio what OCR/vision pipelines became to images and PDFs.

## Product Thesis

Most existing tools solve only part of the problem:

- feature extraction
- source separation
- audio-to-MIDI
- desktop visualization
- studio mix analysis

What is still fragmented is the agent-facing layer:

- one installable tool
- one JSON schema
- one workflow for local, API, and agent use
- one surface for "hear this file and give me artifacts I can reason over"

That is the opportunity.

## Vision

Build a platform that lets an AI system:

1. inspect audio quickly
2. transform it into useful visual artifacts
3. extract structural musical information
4. compare tracks and segments
5. reason over sound with a stable schema
6. use the same workflows locally, in MCP, and via API

## Users

Primary users:

- AI builders who want audio understanding inside agent workflows
- creative technologists building music-aware applications
- music producers who want machine-readable analysis and diagnostics
- researchers exploring computational listening and music representation

Secondary users:

- prompt engineers using audio to steer video/image/model pipelines
- archive/catalog teams analyzing large libraries of music or sound
- media tooling companies that need an installable audio-analysis backend

## Core Product Surfaces

### 1. Local CLI

This is the current foundation.

Purpose:

- easiest developer entry point
- scriptable
- reliable in local and CI environments
- testbed for new analyzers before service deployment

Current commands:

- `doctor`
- `info`
- `spectrogram`
- `visualize`
- `analyze`
- `midi-json`

Future CLI commands:

- `compare`
- `segment`
- `separate`
- `beats`
- `chords`
- `key`
- `lyrics-align`
- `embed`
- `summary`
- `export-schema`
- `self-test`

### 2. MCP Server

Purpose:

- make audio analysis available to agents as a first-class tool
- remove shell friction
- return structured artifacts directly to LLMs

Possible MCP tools:

- `audio_info(path_or_url)`
- `audio_visualize(path_or_url)`
- `audio_analyze(path_or_url)`
- `audio_compare(file_a, file_b)`
- `audio_separate(path_or_url)`
- `audio_schema(path_or_url)`

### 3. Python Library

Purpose:

- let developers import the schema and pipeline directly
- support notebooks, apps, and custom wrappers

Possible package surface:

```python
from hear_music import analyze_audio, visualize_audio, compare_audio
```

### 4. Remote Jobs / API

Purpose:

- heavy analyzers can run remotely
- batch processing and hosted usage
- foundation for monetization

Possible endpoints:

- upload + analyze
- async jobs
- retrieve artifacts
- compare tracks
- webhook completion

### 5. Web App

Purpose:

- easiest onboarding for non-technical users
- drag-and-drop analysis
- artifact browser
- upgrade path into paid plans

## Canonical Output Schema

The most important long-term asset is the schema.

Every backend should converge on one top-level structure like:

```json
{
  "source": {},
  "audio": {},
  "metadata": {},
  "artifacts": {},
  "structure": {},
  "music": {},
  "stems": {},
  "embeddings": {},
  "diagnostics": {}
}
```

Suggested sections:

- `source`: original path, hash, mime type, ingest method
- `audio`: duration, sample rate, channel count, loudness, peaks
- `metadata`: tags, cover art, codec, bitrate
- `artifacts`: spectrogram, waveform image, chroma image, normalized wav, midi
- `structure`: sections, boundaries, transitions, intro/verse/chorus guesses
- `music`: tempo, beat grid, bars, key, chords, note events, melody, harmony
- `stems`: separated stem references and metadata
- `embeddings`: vector representations for retrieval/comparison
- `diagnostics`: warnings, confidence, backend names, fallbacks used

This is what makes the platform durable. Backends can improve without changing how users consume results.

## Architecture

### Ingestion Layer

- local file paths
- uploaded files
- URLs
- folders / batches

### Preprocessing Layer

- `ffmpeg` normalization
- mono/stereo conversions
- sample-rate standardization
- trim / chunk / silence handling

### Analyzer Layer

Each analyzer should be pluggable and optional.

Examples:

- lightweight native analyzer for basic pitch estimation
- `librosa` for spectral/chroma features
- `basic-pitch` for stronger audio-to-MIDI
- `demucs` for source separation
- `essentia` for richer music descriptors
- Whisper-style model for lyrics/transcription alignment if needed

### Artifact Layer

- PNGs
- WAVs
- MIDI
- JSON
- compressed previews

### Interface Layer

- CLI
- Python API
- MCP
- HTTP API
- web frontend

## Differentiation

`hear-music` is most distinct if it is:

- agent-first
- schema-first
- installable locally
- useful without the cloud
- upgradeable to better models and paid hosted services

The key differentiator is not any one analyzer. It is the unification layer.

## Roadmap

### v0.2

Goal: make the current CLI sharper and more trustworthy.

- add `self-test`
- add `compare`
- improve `doctor`
- add stronger error messages
- add sample outputs in repo
- optionally replace basic transcription backend with `basic-pitch`

### v0.5

Goal: become a serious agent/developer toolkit.

- define canonical JSON schema
- add section/segment detection
- add beat tracking and tempo estimation
- add key/chord detection
- add source separation
- ship Python import API
- ship MCP server

### v1.0

Goal: become an audio intelligence platform.

- hosted async API
- dashboard/web app
- batch jobs
- account/project model
- artifact storage
- usage metering
- comparison/retrieval workflows
- commercial plans

## Monetization Paths

### 1. Hosted API

Charge for:

- audio minutes processed
- premium analyzers
- batch jobs
- retained artifacts / storage

Best for:

- AI startups
- internal tooling teams
- creative SaaS companies

### 2. Pro Desktop / Local App

Charge for:

- premium features in a polished desktop UI
- one-click workflows
- model bundles

Best for:

- solo creators
- producers
- researchers

### 3. Team Platform

Charge for:

- shared libraries
- audit trail
- collaboration
- API keys
- workspace permissions

Best for:

- studios
- media companies
- AI product teams

### 4. Open Core

Keep the CLI open source.
Monetize:

- hosted service
- premium analyzers
- enterprise features
- support / consulting / custom integrations

This is likely the strongest model.

## Recommended Monetization Strategy

Best path:

1. keep the CLI open source and excellent
2. make the schema the standard
3. launch paid hosted analysis and premium backends
4. add team workflows later

This preserves trust and adoption while creating a clear commercial layer.

## Immediate Next Moves

### Product

- define the canonical schema in a dedicated doc
- decide whether `basic-pitch` becomes the default MIDI backend
- decide whether source separation belongs in core or premium

### Repo

- add screenshots in `docs/`
- add example outputs
- add roadmap section to README
- add issues/milestones around v0.2 and v0.5

### Technical

- add `compare` command
- add `self-test`
- add schema versioning
- add analyzer backend abstraction

### Commercial

- identify the first paying user type
- decide whether the first paid offer is API or desktop
- define what remains open vs premium

## Strategic Summary

The big opportunity is not "another music analyzer."

The opportunity is:

- a standard way for agents and applications to perceive audio
- with local-first tooling
- a stable schema
- better analyzers behind the same interface
- and a natural path from open CLI to paid hosted intelligence
