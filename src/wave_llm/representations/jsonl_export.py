from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def rule_wave_regime(window: pd.DataFrame) -> str:
    hs = window["Hs_m"].to_numpy(float)
    if hs.size == 0:
        return "unknown"
    p95 = float(np.nanpercentile(hs, 95))
    std = float(np.nanstd(hs))
    tp = window["Tp_s"].to_numpy(float) if "Tp_s" in window.columns else np.array([np.nan])
    tp_mean = float(np.nanmean(tp))
    gr = float(np.nanmax(np.diff(hs))) if hs.size > 2 else 0.0
    decl = float(np.nanmin(np.diff(hs))) if hs.size > 2 else 0.0
    if p95 < 1.0 and std < 0.35:
        return "calm_stable"
    if gr > 0.35 and p95 > float(np.nanmedian(hs)) + 0.8:
        return "storm_growth"
    if decl < -0.35:
        return "storm_decay"
    if tp_mean > 12 and "Dir_deg" in window.columns:
        dstd = float(window["Dir_deg"].std())
        if dstd < 30:
            return "swell_dominated"
    if tp_mean < 9 and "wind_dir_deg" in window.columns and "Dir_deg" in window.columns:
        dd = (window["Dir_deg"] - window["wind_dir_deg"]).abs()
        dd = ((dd + 180) % 360) - 180
        if float(dd.abs().median()) < 45:
            return "windsea_dominated"
    return "mixed_sea"


def build_windows(
    df: pd.DataFrame,
    history_hours: int,
    lead_hours: int,
    stride_hours: int = 24,
) -> list[dict[str, Any]]:
    """Strict issue-time: only history with time <= issue_time; target is Hs at issue_time + lead."""
    rows: list[dict[str, Any]] = []
    for sid, g in df.groupby("station_id", sort=False):
        g = g.sort_values("time_utc").reset_index(drop=True)
        times = g["time_utc"]
        step = max(1, int(stride_hours))
        for i in range(0, len(g), step):
            issue = times.iloc[i]
            hist_end = issue
            hist_start = issue - pd.Timedelta(hours=history_hours)
            mask = (g["time_utc"] > hist_start) & (g["time_utc"] <= hist_end)
            hist = g.loc[mask]
            if hist.shape[0] < max(24, history_hours // 4):
                continue
            tgt_time = issue + pd.Timedelta(hours=lead_hours)
            fut = g[g["time_utc"] == tgt_time]
            if fut.empty:
                # nearest within half step
                idx = (g["time_utc"] - tgt_time).abs().idxmin()
                if abs((g.loc[idx, "time_utc"] - tgt_time).total_seconds()) > 3600 * 1.5:
                    continue
                hs_tgt = float(g.loc[idx, "Hs_m"])
            else:
                hs_tgt = float(fut.iloc[0]["Hs_m"])
            regime = rule_wave_regime(hist)
            rows.append(
                {
                    "station_id": sid,
                    "issue_time": issue.isoformat(),
                    "lead_hours": lead_hours,
                    "history_hours": history_hours,
                    "wave_regime": regime,
                    "hist_Hs": hist["Hs_m"].tail(168).tolist(),
                    "target_Hs": hs_tgt,
                }
            )
            # stride: only every stride_hours by stepping index - approximate by i % stride
            # caller can subsample; here we thin by stride_hours on grid
    return rows


def _hist_list(hist: Any) -> list[float]:
    if hist is None:
        return []
    if isinstance(hist, (list, tuple)):
        return [float(x) for x in hist]
    return [float(x) for x in list(hist)]


def _persist_error_quantiles(samples: list[dict[str, Any]], llm_lead_h: int) -> tuple[float, float]:
    errs: list[float] = []
    for s in samples:
        if int(s.get("lead_hours", llm_lead_h)) != int(llm_lead_h):
            continue
        hist = _hist_list(s.get("hist_Hs"))
        if not hist:
            continue
        errs.append(abs(float(s["target_Hs"]) - float(hist[-1])))
    if not errs:
        return 0.3, 0.8
    arr = np.asarray(errs, dtype=float)
    return float(np.quantile(arr, 0.33)), float(np.quantile(arr, 0.67))


def _persist_only_level(target_hs: float, hist_hs: list[float], q_lo: float, q_hi: float) -> str:
    if not hist_hs:
        return "medium"
    err = abs(float(target_hs) - float(hist_hs[-1]))
    if err <= q_hi:
        return "high"
    if err >= q_lo:
        return "low"
    return "medium"


def export_jsonl(
    samples: list[dict[str, Any]],
    path: Path,
    instruction: str,
    predictability_lookup: dict[tuple[str, str], str] | None = None,
    *,
    llm_lead_h: int = 24,
) -> None:
    from wave_llm.labels import PRED_LEVELS, WAVE_REGIMES
    from wave_llm.representations.predictability import lookup_predictability

    label_hint = (
        f" wave_regime 仅限: {', '.join(WAVE_REGIMES)}; "
        f"predictability_24h 仅限: {', '.join(PRED_LEVELS)}."
    )
    err_lo, err_hi = _persist_error_quantiles(samples, llm_lead_h)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            lead = int(s.get("lead_hours", llm_lead_h))
            if lead != int(llm_lead_h):
                continue
            pred_lvl = lookup_predictability(
                predictability_lookup or {},
                str(s["station_id"]),
                str(s["issue_time"]),
                default="",
            )
            if pred_lvl not in PRED_LEVELS:
                pred_lvl = _persist_only_level(
                    float(s["target_Hs"]),
                    _hist_list(s.get("hist_Hs")),
                    err_lo,
                    err_hi,
                )
            rec = {
                "instruction": instruction + label_hint,
                "input": {
                    "station_id": s["station_id"],
                    "issue_time": s["issue_time"],
                    "lead_hours": lead,
                    "window_summary": {
                        "Hs_mean": float(np.mean(_hist_list(s.get("hist_Hs")))),
                        "Hs_p95": float(np.percentile(_hist_list(s.get("hist_Hs")), 95)),
                        "Hs_std": float(np.std(_hist_list(s.get("hist_Hs")))),
                    },
                },
                "output": {
                    "wave_regime": s["wave_regime"],
                    "predictability_24h": pred_lvl,
                    "notes": f"海况节律规则标签；可预测性来自 LGBM vs persistence skill（{llm_lead_h}h）。",
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
