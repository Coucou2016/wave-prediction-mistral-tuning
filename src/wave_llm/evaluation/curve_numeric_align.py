"""Align tabular point forecasts (LGBM/Chronos) with Mistral curve JSONL samples."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wave_llm.evaluation.metrics_numeric import mae, rmse


def persistence_curve(hs_hist: list[float], horizon: int) -> list[float]:
    v = float(hs_hist[-1]) if hs_hist else float("nan")
    return [v] * int(horizon)


def lookup_point_forecasts(
    pred_df: pd.DataFrame,
    station_id: str,
    issue_time: pd.Timestamp,
) -> dict[int, dict[str, float]]:
    """Map lead_h -> {persist, lgbm, chronos, truth} for one issue time."""
    pr = pred_df
    pr = pr.loc[pr["station_id"].astype(str) == str(station_id)].copy()
    pr["time_utc"] = pd.to_datetime(pr["time_utc"], utc=True)
    issue_time = pd.Timestamp(issue_time, tz="UTC")
    g = pr.loc[pr["time_utc"] == issue_time]
    out: dict[int, dict[str, float]] = {}
    for _, row in g.iterrows():
        lh = int(row["lead_h"])
        out[lh] = {
            "y_true": float(row.get("y_true", np.nan)),
            "persist": float(row.get("y_pred_persist", np.nan)),
            "lgbm": float(row.get("y_pred_lgbm", np.nan)),
            "chronos": float(row.get("y_pred_chronos", np.nan)),
        }
    return out


def enrich_curve_results_with_baselines(
    test_rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    pred_df: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Add persistence curve + optional LGBM/Chronos point RMSE to each eval row."""
    if pred_df is None or pred_df.empty:
        for rec, res in zip(test_rows, results):
            inp = rec.get("input") or {}
            hs_hist = list(inp.get("history_hs_m", []))
            hz = len(res.get("hs_true", []))
            res["hs_persist"] = persistence_curve(hs_hist, hz)
            yt = np.asarray(res["hs_true"], dtype=float)
            yp = np.asarray(res["hs_persist"], dtype=float)
            m = min(len(yt), len(yp))
            if m:
                res["rmse_persist"] = rmse(yt[:m], yp[:m])
                res["mae_persist"] = mae(yt[:m], yp[:m])
        return results

    for rec, res in zip(test_rows, results):
        inp = rec.get("input") or {}
        sid = str(inp.get("station_id", ""))
        issue = inp.get("issue_time")
        hs_hist = list(inp.get("history_hs_m", []))
        hz = len(res.get("hs_true", []))
        res["hs_persist"] = persistence_curve(hs_hist, hz)
        yt = np.asarray(res["hs_true"], dtype=float)
        yp = np.asarray(res["hs_persist"], dtype=float)
        m = min(len(yt), len(yp))
        if m:
            res["rmse_persist"] = rmse(yt[:m], yp[:m])
            res["mae_persist"] = mae(yt[:m], yp[:m])

        pts = lookup_point_forecasts(pred_df, sid, issue)
        res["numeric_leads"] = sorted(pts.keys())
        lgbm_rmse, chronos_rmse = [], []
        for lh, vals in pts.items():
            k = lh - 1
            if 0 <= k < len(yt):
                if np.isfinite(vals.get("lgbm", np.nan)):
                    lgbm_rmse.append((yt[k] - vals["lgbm"]) ** 2)
                if np.isfinite(vals.get("chronos", np.nan)):
                    chronos_rmse.append((yt[k] - vals["chronos"]) ** 2)
        if lgbm_rmse:
            res["rmse_lgbm_at_numeric_leads"] = float(np.sqrt(np.mean(lgbm_rmse)))
        if chronos_rmse:
            res["rmse_chronos_at_numeric_leads"] = float(np.sqrt(np.mean(chronos_rmse)))
        res["point_forecasts"] = pts
    return results


def aggregate_baseline_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    def _mean(key: str) -> float | None:
        vals = [r[key] for r in results if np.isfinite(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else None

    return {
        "mean_rmse_persist": _mean("rmse_persist"),
        "mean_rmse_lgbm_at_numeric_leads": _mean("rmse_lgbm_at_numeric_leads"),
        "mean_rmse_chronos_at_numeric_leads": _mean("rmse_chronos_at_numeric_leads"),
    }
