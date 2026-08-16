from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from wave_llm.labels import PRED_LEVELS

log = logging.getLogger(__name__)


def row_skill(y_true: float, y_model: float, y_persist: float) -> float:
    """Per-sample skill vs persistence (1 - |e_model| / |e_persist|)."""
    if not np.isfinite(y_true) or not np.isfinite(y_model) or not np.isfinite(y_persist):
        return float("nan")
    ep = abs(float(y_true) - float(y_persist))
    em = abs(float(y_true) - float(y_model))
    if ep < 1e-6:
        return 1.0 if em < 1e-6 else -1.0
    return float(1.0 - em / ep)


def skill_to_level(skill: float, q_low: float, q_high: float) -> str:
    if not np.isfinite(skill):
        return "medium"
    if skill >= q_high:
        return "high"
    if skill <= q_low:
        return "low"
    return "medium"


def build_predictability_lookup(
    pred_path: Path,
    lead_h: int = 24,
    *,
    low_q: float = 0.33,
    high_q: float = 0.67,
) -> dict[tuple[str, str], str]:
    """Map (station_id, issue_time_iso) -> high|medium|low from LGBM vs persistence skill."""
    if not pred_path.is_file():
        log.warning("Predictions not found: %s — all predictability labels will be medium", pred_path)
        return {}

    df = pd.read_parquet(pred_path)
    df = df.loc[df["lead_h"] == int(lead_h)].copy()
    if df.empty:
        log.warning("No rows for lead_h=%s in %s", lead_h, pred_path)
        return {}

    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df["skill"] = [
        row_skill(r["y_true"], r["y_pred_lgbm"], r["y_pred_persist"]) for _, r in df.iterrows()
    ]
    finite = df["skill"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        q_lo, q_hi = -0.05, 0.05
    else:
        q_lo = float(finite.quantile(low_q))
        q_hi = float(finite.quantile(high_q))
    log.info(
        "Predictability thresholds (lead=%sh): low<=%.3f high>=%.3f (n=%s)",
        lead_h,
        q_lo,
        q_hi,
        len(finite),
    )

    lookup: dict[tuple[str, str], str] = {}
    for _, r in df.iterrows():
        key = (str(r["station_id"]), pd.Timestamp(r["time_utc"]).isoformat())
        lookup[key] = skill_to_level(float(r["skill"]), q_lo, q_hi)
    return lookup


def lookup_predictability(
    lookup: dict[tuple[str, str], str],
    station_id: str,
    issue_time: str,
    *,
    default: str = "medium",
) -> str:
    key = (str(station_id), pd.Timestamp(issue_time).isoformat())
    level = lookup.get(key, default)
    return level if level in PRED_LEVELS else default
