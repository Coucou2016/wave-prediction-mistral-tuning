from __future__ import annotations

import numpy as np
import pandas as pd


def growth_decay_features(df: pd.DataFrame, hours: int = 6, dt_hours: int = 1) -> pd.DataFrame:
    out = df.sort_values(["station_id", "time_utc"]).copy()
    steps = max(1, int(round(hours / dt_hours)))
    g = out.groupby("station_id", group_keys=False)
    out["dHs_dt"] = g["Hs_m"].diff(steps) / (steps * dt_hours)
    return out
