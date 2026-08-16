from __future__ import annotations

import numpy as np
import pandas as pd

G = 9.81


def deepwater_wavelength_tp(tp_s: pd.Series) -> pd.Series:
    return (G * tp_s**2) / (2.0 * np.pi)


def wave_steepness(hs: pd.Series, tp_s: pd.Series) -> pd.Series:
    lp = deepwater_wavelength_tp(tp_s)
    return hs / lp.replace(0, np.nan)


def add_lag_features(df: pd.DataFrame, hours: tuple[int, ...] = (1, 3, 6, 12, 24, 48)) -> pd.DataFrame:
    out = df.sort_values(["station_id", "time_utc"]).copy()
    g = out.groupby("station_id", group_keys=False)
    for h in hours:
        out[f"Hs_lag_{h}h"] = g["Hs_m"].shift(h)
    if "Tp_s" in out.columns:
        for h in hours:
            out[f"Tp_lag_{h}h"] = g["Tp_s"].shift(h)
    return out


def wind_wave_delta_deg(wave_dir: pd.Series, wind_dir: pd.Series) -> pd.Series:
    d = (wave_dir - wind_dir + 180) % 360 - 180
    return d.abs()
