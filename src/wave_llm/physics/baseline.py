from __future__ import annotations

import numpy as np
import pandas as pd


def climatology_baseline_hourly(df: pd.DataFrame) -> pd.Series:
    """Per-station calendar hour-of-year mean Hs (simple seasonal cycle)."""
    x = df.copy()
    x["hod"] = x["time_utc"].dt.dayofyear * 24 + x["time_utc"].dt.hour
    mu = x.groupby(["station_id", "hod"])["Hs_m"].transform("mean")
    return mu


def persistence_baseline_at_issue(df: pd.DataFrame, issue_time: pd.Timestamp, col: str = "Hs_m") -> float:
    row = df[df["time_utc"] <= issue_time].sort_values("time_utc")
    if row.empty:
        return float("nan")
    return float(row.iloc[-1][col])
