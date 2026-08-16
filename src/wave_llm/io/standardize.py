from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

COL_MAP = {
    "WVHT": "Hs_m",
    "DPD": "Tp_s",
    "APD": "Tm_s",
    "MWD": "Dir_deg",
    "WSPD": "wind_speed_ms",
    "WDIR": "wind_dir_deg",
    "WTMP": "sea_surface_temp_c",
    "PRES": "pressure_hpa",
    "ATMP": "air_temp_c",
}


def ndbc_build_time_utc(df: pd.DataFrame) -> pd.Series:
    """NDBC stdmet: YY MM DD hh mm (minute)."""
    need = ["YY", "MM", "DD", "hh", "mm"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing NDBC time column {c!r}; columns={list(df.columns)[:40]}")
    return pd.to_datetime(
        {
            "year": pd.to_numeric(df["YY"], errors="coerce"),
            "month": pd.to_numeric(df["MM"], errors="coerce"),
            "day": pd.to_numeric(df["DD"], errors="coerce"),
            "hour": pd.to_numeric(df["hh"], errors="coerce"),
            "minute": pd.to_numeric(df["mm"], errors="coerce"),
        },
        utc=True,
    )


def standardize_ndbc_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time_utc"] = ndbc_build_time_utc(out)
    for src, dst in COL_MAP.items():
        if src in out.columns:
            out[dst] = pd.to_numeric(out[src], errors="coerce")
    if "station_id" not in out.columns:
        raise ValueError("station_id column required")
    out["station_id"] = out["station_id"].astype(str)
    out["source"] = out.get("source", "NDBC")
    return out


def to_canonical_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "station_id",
        "source",
        "time_utc",
        "Hs_m",
        "Tp_s",
        "Tm_s",
        "Dir_deg",
        "wind_speed_ms",
        "wind_dir_deg",
        "sea_surface_temp_c",
    ]
    for c in cols:
        if c not in df.columns:
            if c in ("station_id", "source", "time_utc", "Hs_m"):
                raise KeyError(f"Missing required column {c}")
            df = df.copy()
            df[c] = np.nan
    return df[cols].sort_values(["station_id", "time_utc"]).reset_index(drop=True)
