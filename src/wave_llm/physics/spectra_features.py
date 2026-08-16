from __future__ import annotations

import numpy as np
import pandas as pd


def welch_band_ratios(
    series: pd.Series,
    fs_hz: float,
    bands_hz: tuple[tuple[float, float], ...] = ((1 / (10 * 86400), 1 / (2 * 86400)), (1 / 48, 1 / 6)),
) -> dict[str, float]:
    """Very coarse frequency-band energy ratios on uniformly sampled series (optional)."""
    from scipy import signal

    x = series.dropna().to_numpy(dtype=float)
    if x.size < 64:
        return {f"band_{i}": float("nan") for i in range(len(bands_hz))}
    f, pxx = signal.welch(x, fs=fs_hz, nperseg=min(256, x.size // 2))
    out: dict[str, float] = {}
    _trapz = getattr(np, "trapezoid", np.trapz)
    total = float(_trapz(pxx, f)) + 1e-12
    for i, (lo, hi) in enumerate(bands_hz):
        m = (f >= lo) & (f <= hi)
        out[f"band_{i}_power_ratio"] = float(_trapz(pxx[m], f[m]) / total)
    return out
