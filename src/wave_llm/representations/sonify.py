from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def sonify_series(
    series: np.ndarray,
    sr: int = 8000,
    out_wav: Path | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Map scalar series to mono audio for exploratory listening (not physical acoustics)."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("empty series")
    if normalize:
        x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-6)
    x = np.clip(x, -3, 3) / 3.0
    # upsample linearly to fixed duration ~30s
    target_n = sr * 30
    t_old = np.linspace(0, 1, num=x.size)
    t_new = np.linspace(0, 1, num=min(target_n, max(sr, x.size * 8)))
    y = np.interp(t_new, t_old, x).astype(np.float32)
    if out_wav is not None:
        try:
            import soundfile as sf

            sf.write(str(out_wav), y, sr)
        except Exception:
            pass
    return y
