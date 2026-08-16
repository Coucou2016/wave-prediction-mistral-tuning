from __future__ import annotations

import pandas as pd


def residual_vs_baseline(hs: pd.Series, base: pd.Series) -> pd.Series:
    return hs - base
