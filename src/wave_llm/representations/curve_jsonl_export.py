"""Export Mistral curve-forecast JSONL from real buoy panels (strict issue-time)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CURVE_INSTRUCTION = (
    "You are a wave forecasting assistant. Given past buoy observations before the issue time, "
    "forecast the next hourly significant wave heights. Return strict JSON only."
)


def _series_at_times(g: pd.DataFrame, times: pd.DatetimeIndex, col: str) -> list[float] | None:
    vals: list[float] = []
    tmap = g.set_index("time_utc")
    for t in times:
        if t not in tmap.index:
            # nearest within 45 min
            deltas = np.abs((tmap.index - t).total_seconds())
            idx = int(np.argmin(deltas))
            if float(deltas[idx]) > 2700:
                return None
            v = float(tmap.iloc[idx][col])
        else:
            v = float(tmap.loc[t, col])
        if not np.isfinite(v):
            return None
        vals.append(round(v, 3))
    return vals


def build_curve_samples(
    df: pd.DataFrame,
    *,
    history_hours: int = 168,
    horizon_hours: int = 24,
    dt_hours: int = 1,
    stride_hours: int = 24,
    time_end: pd.Timestamp | None = None,
    time_start: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    """Build samples with real future Hs arrays (no synthetic waves)."""
    rows: list[dict[str, Any]] = []
    h = int(history_hours)
    hz = int(horizon_hours)
    dt = max(1, int(dt_hours))
    step = max(1, int(stride_hours))

    for sid, g in df.groupby("station_id", sort=False):
        g = g.sort_values("time_utc").reset_index(drop=True)
        if time_start is not None:
            g = g[g["time_utc"] >= time_start]
        if time_end is not None:
            g = g[g["time_utc"] <= time_end]
        if len(g) < h + hz + 4:
            continue

        times = g["time_utc"].to_numpy()
        for i in range(h, len(g) - hz, step):
            issue = pd.Timestamp(times[i])
            n_hist = max(24, h // dt)
            n_fut = hz // dt
            hist_times = pd.date_range(issue - pd.Timedelta(hours=(n_hist - 1) * dt), periods=n_hist, freq=f"{dt}h")
            fut_times = pd.date_range(issue + pd.Timedelta(hours=dt), periods=n_fut, freq=f"{dt}h")
            if len(fut_times) != n_fut:
                continue

            hs_hist = _series_at_times(g, hist_times, "Hs_m")
            hs_fut = _series_at_times(g, fut_times, "Hs_m")
            if hs_hist is None or hs_fut is None:
                continue

            tp_hist = _series_at_times(g, hist_times, "Tp_s") if "Tp_s" in g.columns else None
            wind_hist = (
                _series_at_times(g, hist_times, "wind_speed_ms") if "wind_speed_ms" in g.columns else None
            )

            hs_arr = np.asarray(hs_hist, dtype=float)
            feats: dict[str, Any] = {
                "hs_mean_72h": float(np.mean(hs_arr[-72:])) if len(hs_arr) >= 72 else float(np.mean(hs_arr)),
                "hs_trend_24h": float(hs_arr[-1] - hs_arr[-min(24, len(hs_arr))]),
                "hs_std_24h": float(np.std(hs_arr[-24:])) if len(hs_arr) >= 24 else float(np.std(hs_arr)),
            }
            if tp_hist:
                tp_arr = np.asarray(tp_hist, dtype=float)
                feats["tp_mean_72h"] = float(np.nanmean(tp_arr[-72:])) if len(tp_arr) >= 72 else float(np.nanmean(tp_arr))
                feats["tp_trend_24h"] = float(tp_arr[-1] - tp_arr[-min(24, len(tp_arr))])
                feats["tp_last_6"] = [round(float(x), 3) for x in tp_arr[-6:].tolist()]
            if wind_hist:
                w_arr = np.asarray(wind_hist, dtype=float)
                feats["wind_mean_72h"] = float(np.nanmean(w_arr[-72:])) if len(w_arr) >= 72 else float(np.nanmean(w_arr))
                feats["wind_trend_24h"] = float(w_arr[-1] - w_arr[-min(24, len(w_arr))])
                feats["wind_last_6"] = [round(float(x), 3) for x in w_arr[-6:].tolist()]

            rows.append(
                {
                    "station_id": str(sid),
                    "issue_time": issue.isoformat(),
                    "history_dt_h": dt,
                    "history_hs_m": hs_hist,
                    "history_tp_s": tp_hist,
                    "history_wind_ms": wind_hist,
                    "features": feats,
                    "forecast_horizon_h": hz,
                    "forecast_dt_h": dt,
                    "hs_forecast_m": hs_fut,
                }
            )
    return rows


def _round_features(feats: dict[str, Any]) -> dict[str, Any]:
    """Round scalars; keep short list summaries (e.g. last_6) as rounded floats."""
    out: dict[str, Any] = {}
    for k, v in (feats or {}).items():
        if isinstance(v, (list, tuple)):
            vals = []
            for x in v:
                try:
                    vals.append(round(float(x), 3))
                except (TypeError, ValueError):
                    continue
            if vals:
                out[str(k)] = vals
            continue
        try:
            out[str(k)] = round(float(v), 3)
        except (TypeError, ValueError):
            continue
    return out


def sample_to_record(sample: dict[str, Any]) -> dict[str, Any]:
    """Build JSONL record.

    Keep full hourly Hs history (needed for curve SFT) but omit long Tp/wind
    series (use mean/trend/last_N summaries instead) — those are summarized in ``features`` to keep token length tractable
    (~3k+ tokens with three 168-length arrays blew up GPU training wall-time).
    """
    inp: dict[str, Any] = {
        "station_id": sample["station_id"],
        "issue_time": sample["issue_time"],
        "history_dt_h": sample["history_dt_h"],
        "history_hs_m": sample["history_hs_m"],
        "features": _round_features(sample.get("features", {})),
        "forecast_horizon_h": sample["forecast_horizon_h"],
        "forecast_dt_h": sample.get("forecast_dt_h", 1),
    }

    out = {
        "forecast_horizon_h": sample["forecast_horizon_h"],
        "dt_h": sample.get("forecast_dt_h", 1),
        "hs_forecast_m": sample["hs_forecast_m"],
        "uncertainty_level": sample.get("uncertainty_level", "medium"),
        "reason": sample.get("reason", "Derived from real NDBC hold-out window."),
    }
    return {
        "instruction": CURVE_INSTRUCTION,
        "input": inp,
        "output": out,
    }


def export_curve_jsonl(samples: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(sample_to_record(s), ensure_ascii=False) + "\n")


def export_train_test_curve_jsonl(
    panel: pd.DataFrame,
    out_dir: Path,
    *,
    history_hours: int,
    horizon_hours: int,
    dt_hours: int,
    stride_hours: int,
    train_frac: float = 0.8,
) -> tuple[int, int]:
    """Temporal hold-out: first train_frac for train JSONL, tail for test JSONL."""
    panel = panel.copy()
    panel["time_utc"] = pd.to_datetime(panel["time_utc"], utc=True)
    t_cut = panel["time_utc"].quantile(train_frac)

    train_s = build_curve_samples(
        panel,
        history_hours=history_hours,
        horizon_hours=horizon_hours,
        dt_hours=dt_hours,
        stride_hours=stride_hours,
        time_end=t_cut,
    )
    test_s = build_curve_samples(
        panel,
        history_hours=history_hours,
        horizon_hours=horizon_hours,
        dt_hours=dt_hours,
        stride_hours=stride_hours,
        time_start=t_cut,
    )
    export_curve_jsonl(train_s, out_dir / "curve_train.jsonl")
    export_curve_jsonl(test_s, out_dir / "curve_test.jsonl")
    log.info("Curve JSONL train=%s test=%s (cut=%s)", len(train_s), len(test_s), t_cut)
    return len(train_s), len(test_s)
