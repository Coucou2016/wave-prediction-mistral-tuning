from __future__ import annotations

import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d**2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(a[m] - b[m])))


def skill_score(rmse_model: float, rmse_ref: float) -> float:
    if not np.isfinite(rmse_model) or not np.isfinite(rmse_ref) or rmse_ref <= 0:
        return float("nan")
    return float(1.0 - rmse_model / rmse_ref)
