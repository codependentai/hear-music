from __future__ import annotations

from pathlib import Path

import numpy as np


def create_visualization(input_path: Path, out_path: Path) -> Path:
    try:
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "The visualize command requires optional dependencies. "
            "Install them with: python -m pip install -e .[visualize]"
        ) from exc

    y, sr = librosa.load(str(input_path), sr=None, mono=True)
    if y.size == 0:
        raise RuntimeError(f"No audio samples were loaded from {input_path}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"Audio View: {input_path.name}", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    librosa.display.waveshow(y, sr=sr, ax=ax1, color="#1f77b4")
    ax1.set_title("Waveform")
    ax1.set_ylabel("Amplitude")

    ax2 = axes[1]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=min(8000, sr // 2))
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax2, cmap="magma")
    ax2.set_title("Mel Spectrogram")
    fig.colorbar(mel_img, ax=ax2, format="%+2.0f dB", label="Intensity")

    ax3 = axes[2]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_img = librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", ax=ax3, cmap="coolwarm")
    ax3.set_title("Chromagram")
    ax3.set_xlabel("Time (seconds)")
    fig.colorbar(chroma_img, ax=ax3, label="Intensity")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
