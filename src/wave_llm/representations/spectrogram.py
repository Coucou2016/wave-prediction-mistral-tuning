from __future__ import annotations

from pathlib import Path

import numpy as np


def save_melspectrogram_png(series: np.ndarray, out_png: Path, sr: int = 8000) -> None:
    try:
        import librosa
        import librosa.display
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("librosa/matplotlib required for spectrogram export") from e
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 32:
        raise ValueError("series too short")
    m = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=512, hop_length=128)
    m_db = librosa.power_to_db(m, ref=np.max)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 3))
    librosa.display.specshow(m_db, sr=sr, x_axis="time", y_axis="mel")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
