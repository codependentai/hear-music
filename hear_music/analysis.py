from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, find_peaks


DEFAULT_SAMPLE_RATE = 22050
FRAME_SIZE = 2048
HOP_SIZE = 512
MIN_FREQ = 55.0
MAX_FREQ = 1760.0
DEFAULT_BPM = 120
TICKS_PER_BEAT = 480


@dataclass
class NoteEvent:
    pitch: int
    name: str
    frequency_hz: float
    start: float
    duration: float
    end: float
    velocity: int
    confidence: float


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-y", *args]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg failed")


def run_ffprobe_json(input_path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(input_path),
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffprobe failed")
    return json.loads(completed.stdout)


def ensure_audio_outputs(input_path: Path, out_dir: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> tuple[Path, Path]:
    wav_path = out_dir / "normalized.wav"
    spectrogram_path = out_dir / "spectrogram.png"

    run_ffmpeg(
        [
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-vn",
            str(wav_path),
        ]
    )

    run_ffmpeg(
        [
            "-i",
            str(wav_path),
            "-lavfi",
            "showspectrumpic=s=1600x900:legend=disabled:gain=2",
            str(spectrogram_path),
        ]
    )

    return wav_path, spectrogram_path


def generate_spectrogram(input_path: Path, out_path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_wav = out_path.parent / f"{out_path.stem}.normalized.wav"
    run_ffmpeg(
        [
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-vn",
            str(temp_wav),
        ]
    )
    try:
        run_ffmpeg(
            [
                "-i",
                str(temp_wav),
                "-lavfi",
                "showspectrumpic=s=1600x900:legend=disabled:gain=2",
                str(out_path),
            ]
        )
    finally:
        if temp_wav.exists():
            temp_wav.unlink()
    return out_path


def load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)

    peak = np.max(np.abs(data)) if data.size else 0.0
    if peak > 0:
        data = data / peak
    return int(sample_rate), data


def frame_audio(audio: np.ndarray, frame_size: int = FRAME_SIZE, hop_size: int = HOP_SIZE) -> np.ndarray:
    if audio.size < frame_size:
        padded = np.zeros(frame_size, dtype=np.float32)
        padded[: audio.size] = audio
        return padded[np.newaxis, :]

    frames = []
    for start in range(0, len(audio) - frame_size + 1, hop_size):
        frames.append(audio[start : start + frame_size])
    return np.stack(frames) if frames else np.zeros((0, frame_size), dtype=np.float32)


def midi_to_name(midi_note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_note // 12) - 1
    return f"{names[midi_note % 12]}{octave}"


def frequency_to_midi(frequency: float) -> int:
    return int(round(69 + 12 * math.log2(frequency / 440.0)))


def midi_to_frequency(midi_note: int) -> float:
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def estimate_pitch(frame: np.ndarray, sample_rate: int) -> tuple[float | None, float]:
    if not np.any(frame):
        return None, 0.0

    windowed = frame * np.hanning(len(frame))
    rms = float(np.sqrt(np.mean(windowed**2)))
    if rms < 0.015:
        return None, rms

    centered = windowed - np.mean(windowed)
    autocorr = correlate(centered, centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]

    min_lag = max(1, int(sample_rate / MAX_FREQ))
    max_lag = min(len(autocorr) - 1, int(sample_rate / MIN_FREQ))
    if max_lag <= min_lag:
        return None, rms

    section = autocorr[min_lag:max_lag]
    if section.size == 0:
        return None, rms

    peaks, _ = find_peaks(section)
    if peaks.size == 0:
        return None, rms

    peak_index = int(peaks[np.argmax(section[peaks])])
    lag = peak_index + min_lag
    if lag <= 0:
        return None, rms

    frequency = sample_rate / lag
    confidence = float(section[peak_index] / max(autocorr[0], 1e-6))
    if not (MIN_FREQ <= frequency <= MAX_FREQ):
        return None, rms
    return frequency, max(0.0, min(confidence, 1.0))


def extract_notes(audio: np.ndarray, sample_rate: int) -> list[NoteEvent]:
    frames = frame_audio(audio)
    if frames.size == 0:
        return []

    active_pitch: int | None = None
    active_start = 0.0
    active_confidences: list[float] = []
    active_velocities: list[int] = []
    notes: list[NoteEvent] = []

    for index, frame in enumerate(frames):
        timestamp = index * HOP_SIZE / sample_rate
        frequency, confidence = estimate_pitch(frame, sample_rate)
        rms = float(np.sqrt(np.mean(frame**2)))

        if frequency is None:
            if active_pitch is not None:
                notes.append(
                    finalize_note(
                        pitch=active_pitch,
                        start=active_start,
                        end=timestamp,
                        confidences=active_confidences,
                        velocities=active_velocities,
                    )
                )
                active_pitch = None
                active_confidences = []
                active_velocities = []
            continue

        midi_note = frequency_to_midi(frequency)
        velocity = int(max(25, min(127, round(rms * 220))))

        if active_pitch is None:
            active_pitch = midi_note
            active_start = timestamp
            active_confidences = [confidence]
            active_velocities = [velocity]
            continue

        if midi_note == active_pitch:
            active_confidences.append(confidence)
            active_velocities.append(velocity)
            continue

        notes.append(
            finalize_note(
                pitch=active_pitch,
                start=active_start,
                end=timestamp,
                confidences=active_confidences,
                velocities=active_velocities,
            )
        )
        active_pitch = midi_note
        active_start = timestamp
        active_confidences = [confidence]
        active_velocities = [velocity]

    if active_pitch is not None:
        final_end = ((len(frames) - 1) * HOP_SIZE + FRAME_SIZE) / sample_rate
        notes.append(
            finalize_note(
                pitch=active_pitch,
                start=active_start,
                end=final_end,
                confidences=active_confidences,
                velocities=active_velocities,
            )
        )

    merged: list[NoteEvent] = []
    for note in notes:
        if note.duration < 0.045:
            continue
        if merged and merged[-1].pitch == note.pitch and note.start - merged[-1].end < 0.03:
            previous = merged.pop()
            merged.append(
                NoteEvent(
                    pitch=previous.pitch,
                    name=previous.name,
                    frequency_hz=previous.frequency_hz,
                    start=previous.start,
                    duration=note.end - previous.start,
                    end=note.end,
                    velocity=round((previous.velocity + note.velocity) / 2),
                    confidence=round((previous.confidence + note.confidence) / 2, 3),
                )
            )
            continue
        merged.append(note)
    return merged


def finalize_note(
    pitch: int,
    start: float,
    end: float,
    confidences: list[float],
    velocities: list[int],
) -> NoteEvent:
    duration = max(0.0, end - start)
    avg_confidence = round(float(np.mean(confidences)) if confidences else 0.0, 3)
    avg_velocity = int(round(float(np.mean(velocities)) if velocities else 80))
    return NoteEvent(
        pitch=pitch,
        name=midi_to_name(pitch),
        frequency_hz=round(midi_to_frequency(pitch), 3),
        start=round(start, 4),
        duration=round(duration, 4),
        end=round(end, 4),
        velocity=max(1, min(127, avg_velocity)),
        confidence=avg_confidence,
    )


def encode_var_len(value: int) -> bytes:
    buffer = value & 0x7F
    output = bytearray([buffer])
    value >>= 7
    while value:
        output.insert(0, 0x80 | (value & 0x7F))
        value >>= 7
    return bytes(output)


def build_midi_track(notes: list[NoteEvent], bpm: int = DEFAULT_BPM) -> bytes:
    events: list[tuple[int, bytes]] = []
    tempo = int(60_000_000 / bpm)
    events.append((0, bytes([0xFF, 0x51, 0x03, (tempo >> 16) & 0xFF, (tempo >> 8) & 0xFF, tempo & 0xFF])))
    events.append((0, bytes([0xC0, 0x00])))

    for note in notes:
        start_tick = seconds_to_ticks(note.start, bpm)
        end_tick = max(start_tick + 1, seconds_to_ticks(note.end, bpm))
        events.append((start_tick, bytes([0x90, note.pitch & 0x7F, note.velocity & 0x7F])))
        events.append((end_tick, bytes([0x80, note.pitch & 0x7F, 0x00])))

    events.sort(key=lambda item: (item[0], item[1][0] == 0x80))

    track_data = bytearray()
    previous_tick = 0
    for absolute_tick, message in events:
        delta = absolute_tick - previous_tick
        track_data.extend(encode_var_len(max(0, delta)))
        track_data.extend(message)
        previous_tick = absolute_tick

    track_data.extend(encode_var_len(0))
    track_data.extend(b"\xFF\x2F\x00")
    return bytes(track_data)


def write_midi(path: Path, notes: list[NoteEvent], bpm: int = DEFAULT_BPM) -> None:
    track = build_midi_track(notes, bpm=bpm)
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + TICKS_PER_BEAT.to_bytes(2, "big")
    chunk = b"MTrk" + len(track).to_bytes(4, "big") + track
    path.write_bytes(header + chunk)


def seconds_to_ticks(seconds: float, bpm: int = DEFAULT_BPM) -> int:
    beats = seconds * bpm / 60.0
    return int(round(beats * TICKS_PER_BEAT))


def decode_var_len(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            break
    return value, offset


def ticks_to_seconds(tick: int, tempo_events: list[tuple[int, int]], division: int) -> float:
    if not tempo_events:
        tempo_events = [(0, 500000)]

    tempo_events = sorted(tempo_events, key=lambda item: item[0])
    total_seconds = 0.0
    previous_tick = 0
    previous_tempo = tempo_events[0][1]

    for change_tick, tempo in tempo_events[1:]:
        if tick <= change_tick:
            break
        total_seconds += ((change_tick - previous_tick) / division) * (previous_tempo / 1_000_000.0)
        previous_tick = change_tick
        previous_tempo = tempo

    total_seconds += ((tick - previous_tick) / division) * (previous_tempo / 1_000_000.0)
    return total_seconds


def parse_midi(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if data[:4] != b"MThd":
        raise ValueError(f"{path} is not a valid MIDI file")

    header_length = int.from_bytes(data[4:8], "big")
    header = data[8 : 8 + header_length]
    format_type = int.from_bytes(header[0:2], "big")
    num_tracks = int.from_bytes(header[2:4], "big")
    division = int.from_bytes(header[4:6], "big")

    offset = 8 + header_length
    tempo_events: list[tuple[int, int]] = [(0, 500000)]
    track_summaries: list[dict[str, Any]] = []
    note_records: list[dict[str, Any]] = []

    for track_index in range(num_tracks):
        if data[offset : offset + 4] != b"MTrk":
            raise ValueError(f"Malformed MIDI track header in {path}")
        track_length = int.from_bytes(data[offset + 4 : offset + 8], "big")
        track_data = data[offset + 8 : offset + 8 + track_length]
        offset += 8 + track_length

        absolute_tick = 0
        cursor = 0
        running_status: int | None = None
        open_notes: dict[int, tuple[int, int]] = {}
        track_name = f"Track {track_index + 1}"

        while cursor < len(track_data):
            delta, cursor = decode_var_len(track_data, cursor)
            absolute_tick += delta
            status = track_data[cursor]

            if status < 0x80:
                if running_status is None:
                    raise ValueError("Encountered running status without prior MIDI status byte")
                status = running_status
            else:
                cursor += 1
                running_status = status

            if status == 0xFF:
                meta_type = track_data[cursor]
                cursor += 1
                meta_length, cursor = decode_var_len(track_data, cursor)
                meta_data = track_data[cursor : cursor + meta_length]
                cursor += meta_length

                if meta_type == 0x51 and meta_length == 3:
                    tempo_value = int.from_bytes(meta_data, "big")
                    tempo_events.append((absolute_tick, tempo_value))
                elif meta_type == 0x03:
                    track_name = meta_data.decode("latin1", errors="replace")
                continue

            if status in (0xF0, 0xF7):
                sysex_length, cursor = decode_var_len(track_data, cursor)
                cursor += sysex_length
                continue

            message_type = status & 0xF0
            channel = status & 0x0F

            if message_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                first = track_data[cursor]
                second = track_data[cursor + 1]
                cursor += 2
            elif message_type in (0xC0, 0xD0):
                first = track_data[cursor]
                second = None
                cursor += 1
            else:
                raise ValueError(f"Unsupported MIDI status byte: 0x{status:02X}")

            if message_type == 0x90 and second and second > 0:
                open_notes[(channel << 8) | first] = (absolute_tick, second)
            elif message_type in (0x80, 0x90):
                key = (channel << 8) | first
                if key in open_notes:
                    start_tick, velocity = open_notes.pop(key)
                    note_records.append(
                        {
                            "track": track_name,
                            "channel": channel,
                            "pitch": first,
                            "name": midi_to_name(first),
                            "velocity": velocity,
                            "start_tick": start_tick,
                            "end_tick": absolute_tick,
                        }
                    )

        track_summaries.append({"index": track_index, "name": track_name})

    note_records.sort(key=lambda note: (note["start_tick"], note["pitch"]))

    notes: list[dict[str, Any]] = []
    for note in note_records:
        start_seconds = ticks_to_seconds(note["start_tick"], tempo_events, division)
        end_seconds = ticks_to_seconds(note["end_tick"], tempo_events, division)
        notes.append(
            {
                **note,
                "start": round(start_seconds, 4),
                "end": round(end_seconds, 4),
                "duration": round(max(0.0, end_seconds - start_seconds), 4),
                "frequency_hz": round(midi_to_frequency(note["pitch"]), 3),
            }
        )

    bpm = round(60_000_000 / tempo_events[0][1], 2) if tempo_events else None
    duration = max((note["end"] for note in notes), default=0.0)
    return {
        "source": str(path),
        "format": format_type,
        "tracks": track_summaries,
        "ticks_per_beat": division,
        "tempo_bpm": bpm,
        "note_count": len(notes),
        "duration": round(duration, 4),
        "notes": notes,
    }


def analyze_audio(input_path: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path, spectrogram_path = ensure_audio_outputs(input_path, out_dir)
    sample_rate, audio = load_wav_mono(wav_path)
    notes = extract_notes(audio, sample_rate)
    midi_path = out_dir / "transcription.mid"
    write_midi(midi_path, notes, bpm=DEFAULT_BPM)

    analysis = {
        "source": str(input_path),
        "duration": round(len(audio) / sample_rate, 4),
        "sample_rate": sample_rate,
        "tempo_bpm": DEFAULT_BPM,
        "note_count": len(notes),
        "notes": [asdict(note) for note in notes],
        "files": {
            "normalized_wav": str(wav_path),
            "spectrogram": str(spectrogram_path),
            "midi": str(midi_path),
        },
    }

    analysis_path = out_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return analysis
