from __future__ import annotations

import numpy as np
import pandas as pd


def basic_qc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["time_utc", "Hs_m"])
    out = out.drop_duplicates(subset=["station_id", "time_utc"], keep="last")
    out["qc_flag"] = 0
    bad = (out["Hs_m"] < 0) | (out["Hs_m"] > 25)
    out.loc[bad, "qc_flag"] = 1
    out.loc[bad, "Hs_m"] = np.nan
    return out.dropna(subset=["Hs_m"])
