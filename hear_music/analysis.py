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

# Onset detection
ONSET_THRESHOLD = 0.15
ONSET_MIN_SEPARATION_S = 0.05
ONSET_BREAK_TOLERANCE_FRAMES = 1

# Tempo estimation bounds (BPM)
TEMPO_MIN_BPM = 60
TEMPO_MAX_BPM = 200

# Key estimation: Krumhansl-Schmuckler key profiles
NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


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


def compute_onset_envelope(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int = FRAME_SIZE,
    hop_size: int = HOP_SIZE,
) -> np.ndarray:
    """Spectral flux onset envelope. Sum of positive log-magnitude differences across bins.

    Returns raw (un-normalized) flux. Callers (e.g. detect_onsets) decide how to
    threshold it — normalizing here would amplify trivial frame-boundary noise on
    sustained tones into apparent onsets.
    """
    frames = frame_audio(audio, frame_size, hop_size)
    if frames.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)

    window = np.hanning(frame_size).astype(np.float32)
    spectra = np.abs(np.fft.rfft(frames * window, axis=1))
    log_spectra = np.log1p(spectra)
    diff = np.diff(log_spectra, axis=0)
    diff[diff < 0] = 0.0
    flux = diff.sum(axis=1).astype(np.float32)
    return flux


def detect_onsets(
    envelope: np.ndarray,
    sample_rate: int,
    hop_size: int = HOP_SIZE,
    threshold: float = ONSET_THRESHOLD,
    min_separation_s: float = ONSET_MIN_SEPARATION_S,
    local_window_s: float = 0.1,
) -> np.ndarray:
    """Pick onset times (seconds) from a raw flux envelope via adaptive peak picking.

    Two layers of robustness:
      1. **Dynamics gate** — if the envelope's peak isn't substantially larger
         than its median, the signal has no real attacks (e.g. a sustained tone)
         and we return no onsets at all.
      2. **Adaptive thresholding** — subtract a local moving mean so that real
         attacks stand out as positive deviations regardless of absolute level.
    """
    if envelope.size == 0:
        return np.zeros(0, dtype=np.float64)

    peak = float(envelope.max())
    median = float(np.median(envelope))
    # Dynamics gate: a real onset signal has peaks at least ~4x the median level.
    # Sustained / stationary signals fail this and return zero onsets.
    if peak <= max(median * 4.0, 1e-6):
        return np.zeros(0, dtype=np.float64)

    normalized = envelope / peak

    frame_rate = sample_rate / hop_size
    window_frames = max(3, int(round(local_window_s * frame_rate)))
    if normalized.size > window_frames:
        kernel = np.ones(window_frames, dtype=np.float32) / window_frames
        local_mean = np.convolve(normalized, kernel, mode="same")
        detrended = normalized - local_mean
        detrended[detrended < 0] = 0.0
    else:
        detrended = normalized

    min_distance = max(1, int(round(min_separation_s * frame_rate)))
    peaks, _ = find_peaks(detrended, height=threshold, distance=min_distance)
    if peaks.size == 0:
        return np.zeros(0, dtype=np.float64)

    # Onset envelope is computed on diffs, so frame index i corresponds to
    # the boundary between input frame i and i+1. Add 1 to map back.
    times = (peaks + 1) * hop_size / sample_rate
    return times.astype(np.float64)


def estimate_tempo(
    envelope: np.ndarray,
    sample_rate: int,
    hop_size: int = HOP_SIZE,
    min_bpm: int = TEMPO_MIN_BPM,
    max_bpm: int = TEMPO_MAX_BPM,
) -> float | None:
    """Estimate global tempo by autocorrelating the onset envelope.

    Implements octave correction: if the strongest peak corresponds to a slow
    tempo but a peak at half that lag is nearly as strong, prefer the shorter
    lag (faster tempo). Avoids the classic "picks 60 BPM for a 120 BPM track"
    failure mode.
    """
    if envelope.size < 8:
        return None

    centered = envelope - float(np.mean(envelope))
    autocorr = np.correlate(centered, centered, mode="full")
    autocorr = autocorr[autocorr.size // 2 :]
    if autocorr.size <= 1 or autocorr[0] <= 0:
        return None

    frame_rate = sample_rate / hop_size
    min_lag = max(1, int(round(60.0 / max_bpm * frame_rate)))
    max_lag = min(autocorr.size - 1, int(round(60.0 / min_bpm * frame_rate)))
    if max_lag <= min_lag:
        return None

    section = autocorr[min_lag : max_lag + 1]
    if section.size == 0:
        return None
    section_max = float(section.max())
    if section_max <= 0:
        return None

    # Pick the SMALLEST lag whose autocorrelation is at least 0.85 of the
    # absolute max within the search window. This prefers the fundamental
    # period over its multiples (octave correction).
    threshold_value = section_max * 0.85
    candidates = np.where(section >= threshold_value)[0]
    if candidates.size == 0:
        return None
    best_lag = int(candidates[0]) + min_lag
    if best_lag <= 0:
        return None
    bpm = 60.0 * frame_rate / best_lag
    return round(float(bpm), 2)


def compute_chroma_vector(
    audio: np.ndarray,
    sample_rate: int,
    frame_size: int = 4096,
    hop_size: int = 2048,
) -> np.ndarray:
    """Aggregate energy into 12 pitch classes across the whole signal."""
    frames = frame_audio(audio, frame_size, hop_size)
    if frames.size == 0:
        return np.zeros(12, dtype=np.float64)

    window = np.hanning(frame_size).astype(np.float32)
    spectra = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)

    valid = (freqs >= 27.5) & (freqs <= 5000.0) & (freqs > 0)
    valid_freqs = freqs[valid]
    if valid_freqs.size == 0:
        return np.zeros(12, dtype=np.float64)

    pitch_classes = np.round(69 + 12 * np.log2(valid_freqs / 440.0)).astype(int) % 12
    summed = spectra[:, valid].sum(axis=0)

    chroma = np.zeros(12, dtype=np.float64)
    for pc, energy in zip(pitch_classes, summed):
        chroma[pc] += float(energy)

    total = chroma.sum()
    if total > 0:
        chroma = chroma / total
    return chroma


def estimate_key(chroma: np.ndarray) -> dict[str, Any] | None:
    """Match a chroma vector against rotated Krumhansl profiles. Returns tonic + mode + confidence."""
    if chroma.size != 12 or float(chroma.sum()) == 0.0:
        return None

    best_score = -2.0
    best_tonic = None
    best_mode = None

    for shift in range(12):
        rotated = np.roll(chroma, -shift)
        for mode_name, profile in (("major", KRUMHANSL_MAJOR), ("minor", KRUMHANSL_MINOR)):
            if float(np.std(rotated)) == 0.0:
                continue
            corr = float(np.corrcoef(rotated, profile)[0, 1])
            if not np.isfinite(corr):
                continue
            if corr > best_score:
                best_score = corr
                best_tonic = NOTE_NAMES_SHARP[shift]
                best_mode = mode_name

    if best_tonic is None:
        return None

    return {
        "tonic": best_tonic,
        "mode": best_mode,
        "confidence": round(best_score, 3),
    }


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


def extract_notes(
    audio: np.ndarray,
    sample_rate: int,
    onset_times: np.ndarray | None = None,
) -> list[NoteEvent]:
    frames = frame_audio(audio)
    if frames.size == 0:
        return []

    # Map onset times into frame indices for fast lookup. Allow a small tolerance
    # around each onset so we still split repeated same-pitch notes when peak
    # picking lands a frame early or late.
    onset_frame_set: set[int] = set()
    if onset_times is not None and len(onset_times) > 0:
        for t in onset_times:
            base = int(round(float(t) * sample_rate / HOP_SIZE))
            for offset in range(-ONSET_BREAK_TOLERANCE_FRAMES, ONSET_BREAK_TOLERANCE_FRAMES + 1):
                onset_frame_set.add(base + offset)

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

        # Force a note boundary if an onset has fired here, even if the pitch
        # is unchanged. This is what lets us split repeated same-pitch notes.
        force_break = (
            active_pitch is not None
            and index in onset_frame_set
            and timestamp - active_start > 0.04
        )

        if active_pitch is None:
            active_pitch = midi_note
            active_start = timestamp
            active_confidences = [confidence]
            active_velocities = [velocity]
            continue

        if midi_note == active_pitch and not force_break:
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
            # Don't merge across an onset boundary — those notes were
            # deliberately split because the energy envelope said so.
            gap_start_frame = int(round(merged[-1].end * sample_rate / HOP_SIZE))
            gap_end_frame = int(round(note.start * sample_rate / HOP_SIZE))
            crosses_onset = any(
                f in onset_frame_set for f in range(gap_start_frame, gap_end_frame + 1)
            )
            if not crosses_onset:
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

    # Pick the tempo in effect at tick 0. tempo_events is seeded with the MIDI
    # default (500_000 us/qn = 120 BPM) so playback works even when a track has
    # no tempo meta; real meta events parsed from the file are appended after
    # that seed. Take the LAST event at tick 0 so any real declaration wins.
    initial_tempo = 500000
    for tick, tempo in tempo_events:
        if tick == 0:
            initial_tempo = tempo
    bpm = round(60_000_000 / initial_tempo, 2)
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


def _try_librosa_enrichment(input_path: Path) -> dict[str, Any] | None:
    """Optional richer analysis when librosa is installed. Returns None if unavailable."""
    try:
        import librosa  # type: ignore
    except ImportError:
        return None

    try:
        y, sr = librosa.load(str(input_path), sr=None, mono=True)
        if y.size == 0:
            return None
        tempo_value, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        chroma_full = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = chroma_full.mean(axis=1)
        chroma_norm = chroma_avg / float(chroma_avg.sum()) if float(chroma_avg.sum()) > 0 else chroma_avg
        librosa_key = estimate_key(np.asarray(chroma_norm, dtype=np.float64))
        tempo_scalar = float(np.asarray(tempo_value).reshape(-1)[0])
        return {
            "librosa_tempo_bpm": round(tempo_scalar, 2),
            "beat_times": [round(float(t), 4) for t in beat_times],
            "spectral_centroid_mean_hz": round(spectral_centroid, 2),
            "key_librosa": librosa_key,
        }
    except Exception as exc:  # noqa: BLE001
        return {"librosa_error": str(exc)}


def analyze_audio(input_path: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path, spectrogram_path = ensure_audio_outputs(input_path, out_dir)
    sample_rate, audio = load_wav_mono(wav_path)

    onset_envelope = compute_onset_envelope(audio, sample_rate)
    onset_times = detect_onsets(onset_envelope, sample_rate)
    estimated_tempo = estimate_tempo(onset_envelope, sample_rate)
    chroma_vector = compute_chroma_vector(audio, sample_rate)
    key = estimate_key(chroma_vector)

    notes = extract_notes(audio, sample_rate, onset_times=onset_times)

    tempo_for_midi = int(round(estimated_tempo)) if estimated_tempo else DEFAULT_BPM
    midi_path = out_dir / "transcription.mid"
    write_midi(midi_path, notes, bpm=tempo_for_midi)

    analysis: dict[str, Any] = {
        "source": str(input_path),
        "duration": round(len(audio) / sample_rate, 4),
        "sample_rate": sample_rate,
        "tempo_bpm": estimated_tempo if estimated_tempo is not None else DEFAULT_BPM,
        "tempo_estimated": estimated_tempo is not None,
        "key": key,
        "onset_count": int(len(onset_times)),
        "onset_times": [round(float(t), 4) for t in onset_times],
        "chroma": [round(float(v), 4) for v in chroma_vector],
        "note_count": len(notes),
        "notes": [asdict(note) for note in notes],
        "files": {
            "normalized_wav": str(wav_path),
            "spectrogram": str(spectrogram_path),
            "midi": str(midi_path),
        },
    }

    enrichment = _try_librosa_enrichment(input_path)
    if enrichment is not None:
        analysis["librosa"] = enrichment

    analysis_path = out_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return analysis
