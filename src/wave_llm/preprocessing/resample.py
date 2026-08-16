from __future__ import annotations

import pandas as pd


def resample_station_hourly(df: pd.DataFrame, rule: str = "1h") -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for sid, g in df.groupby("station_id", sort=False):
        g = g.set_index("time_utc").sort_index()
        agg: dict[str, str] = {
            "Hs_m": "mean",
            "Tp_s": "mean",
            "Tm_s": "mean",
            "Dir_deg": "mean",
            "wind_speed_ms": "mean",
            "wind_dir_deg": "mean",
            "sea_surface_temp_c": "mean",
            "source": "first",
        }
        cols = {k: v for k, v in agg.items() if k in g.columns}
        r = g.resample(rule).agg(cols)
        r["station_id"] = sid
        r = r.rename_axis("time_utc").reset_index()
        parts.append(r)
    out = pd.concat(parts, ignore_index=True)
    return out.dropna(subset=["Hs_m"])
