"""Time-series visual comparisons: numeric Hs forecasts vs Mistral regime / predictability."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from wave_llm.evaluation.metrics_numeric import mae, rmse
from wave_llm.evaluation.plots import _fig_style
from wave_llm.labels import WAVE_REGIMES

log = logging.getLogger(__name__)

_REGIME_COLORS = {
    "calm_stable": "#8dd3c7",
    "windsea_dominated": "#bebada",
    "swell_dominated": "#fb8072",
    "mixed_sea": "#80b1d3",
    "storm_growth": "#fdb462",
    "storm_decay": "#b3de69",
    "unknown": "#cccccc",
    "parse_failed": "#ff6666",
}


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_eval_merged_frame(
    test_jsonl: Path,
    eval_base_jsonl: Path | None,
    eval_lora_jsonl: Path | None,
    pred_df: pd.DataFrame,
    lead_h: int = 24,
    *,
    windows_parquet: Path | None = None,
    predictability_lookup: dict[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    """
    Align numeric hold-out forecasts with rule labels and optional Mistral eval rows.

    Test JSONL (random split) often does not overlap the temporal hold-out used in script 05;
    when windows_parquet is given, rows are built from prediction issue times joined to windows.
  Mistral eval (sample_idx) is attached via nearest issue time on the same station (<= 36 h).
    """
    pr = pred_df.copy()
    pr["time_utc"] = pd.to_datetime(pr["time_utc"], utc=True)
    pr = pr.loc[pr["lead_h"] == int(lead_h)].copy()
    pr["station_id"] = pr["station_id"].astype(str)

    win = None
    if windows_parquet and windows_parquet.is_file():
        win = pd.read_parquet(windows_parquet)
        win = win.loc[win["lead_hours"] == int(lead_h)].copy()
        win["issue_time"] = pd.to_datetime(win["issue_time"], utc=True)
        win["station_id"] = win["station_id"].astype(str)

    if win is not None and not win.empty:
        base = pr.merge(
            win[["station_id", "issue_time", "wave_regime", "target_Hs"]],
            left_on=["station_id", "time_utc"],
            right_on=["station_id", "issue_time"],
            how="inner",
        )
        base = base.drop(columns=["issue_time"], errors="ignore").rename(
            columns={"time_utc": "issue_time", "wave_regime": "true_regime"}
        )
    else:
        base = pr.rename(columns={"time_utc": "issue_time"})
        base["true_regime"] = ""
        base["target_Hs"] = base["y_true"]

    if predictability_lookup:
        base["true_pred_level"] = [
            predictability_lookup.get((str(r.station_id), pd.Timestamp(r.issue_time).isoformat()), "medium")
            for _, r in base.iterrows()
        ]
    else:
        base["true_pred_level"] = "medium"

    base["y_pred_chronos"] = base["y_pred_chronos"] if "y_pred_chronos" in base.columns else np.nan
    base["err_persist"] = (base["y_true"] - base["y_pred_persist"]).abs()
    base["err_lgbm"] = (base["y_true"] - base["y_pred_lgbm"]).abs()
    base["err_chronos"] = (base["y_true"] - base["y_pred_chronos"]).abs()
    base["base_regime"] = ""
    base["lora_regime"] = ""
    base["base_pred_level"] = ""
    base["lora_pred_level"] = ""
    base["regime_ok_base"] = False
    base["regime_ok_lora"] = False
    base["pred_ok_base"] = False
    base["pred_ok_lora"] = False

    tests = load_jsonl_rows(test_jsonl)
    ev_base = {r["sample_idx"]: r for r in load_jsonl_rows(eval_base_jsonl)} if eval_base_jsonl else {}
    ev_lora = {r["sample_idx"]: r for r in load_jsonl_rows(eval_lora_jsonl)} if eval_lora_jsonl else {}

    if tests and (ev_base or ev_lora):
        ev_rows: list[dict[str, Any]] = []
        for i, rec in enumerate(tests):
            inp = rec.get("input") or {}
            out = rec.get("output") or {}
            ev_rows.append(
                {
                    "sample_idx": i,
                    "station_id": str(inp.get("station_id", "")),
                    "issue_time": pd.Timestamp(inp.get("issue_time")),
                    "true_regime_eval": str(out.get("wave_regime", "")),
                    "true_pred_eval": str(out.get("predictability_24h", "")).lower(),
                    "base_regime": str(ev_base.get(i, {}).get("pred_wave_regime", "")),
                    "lora_regime": str(ev_lora.get(i, {}).get("pred_wave_regime", "")),
                    "base_pred_level": str(ev_base.get(i, {}).get("pred_predictability_24h", "")).lower(),
                    "lora_pred_level": str(ev_lora.get(i, {}).get("pred_predictability_24h", "")).lower(),
                }
            )
        ev_df = pd.DataFrame(ev_rows)
        pred_keys = set(zip(base["station_id"], pd.to_datetime(base["issue_time"], utc=True)))
        test_keys = set(zip(ev_df["station_id"], pd.to_datetime(ev_df["issue_time"], utc=True)))
        overlap = len(pred_keys & test_keys) / max(1, len(test_keys))
        if overlap >= 0.9:
            drop_cols = [c for c in ("base_regime", "lora_regime", "base_pred_level", "lora_pred_level") if c in base.columns]
            base = base.drop(columns=drop_cols, errors="ignore").merge(
                ev_df[
                    [
                        "station_id",
                        "issue_time",
                        "base_regime",
                        "lora_regime",
                        "base_pred_level",
                        "lora_pred_level",
                    ]
                ],
                on=["station_id", "issue_time"],
                how="left",
            )
        else:
            merged_ev = []
            for sid, grp in base.groupby("station_id", sort=False):
                g = grp.sort_values("issue_time").reset_index(drop=True)
                e = ev_df[ev_df["station_id"] == sid].sort_values("issue_time")
                if e.empty:
                    merged_ev.append(g)
                    continue
                aligned = pd.merge_asof(
                    g,
                    e,
                    on="issue_time",
                    direction="nearest",
                    tolerance=pd.Timedelta(hours=36),
                )
                merged_ev.append(aligned)
            base = pd.concat(merged_ev, ignore_index=True)
        base["regime_ok_base"] = False
        m_b = base["base_regime"].astype(str).str.len() > 0
        base.loc[m_b, "regime_ok_base"] = base.loc[m_b, "base_regime"] == base.loc[m_b, "true_regime"]
        base["regime_ok_lora"] = False
        m_l = base["lora_regime"].astype(str).str.len() > 0
        base.loc[m_l, "regime_ok_lora"] = base.loc[m_l, "lora_regime"] == base.loc[m_l, "true_regime"]
        base["pred_ok_base"] = False
        m_pb = base["base_pred_level"].astype(str).str.len() > 0
        base.loc[m_pb, "pred_ok_base"] = base.loc[m_pb, "base_pred_level"] == base.loc[m_pb, "true_pred_level"]
        base["pred_ok_lora"] = False
        m_pl = base["lora_pred_level"].astype(str).str.len() > 0
        base.loc[m_pl, "pred_ok_lora"] = base.loc[m_pl, "lora_pred_level"] == base.loc[m_pl, "true_pred_level"]

    return base


def _method_metrics(pred: pd.DataFrame, lead_h: int) -> dict[str, float]:
    g = pred.loc[pred["lead_h"] == int(lead_h)].copy()
    yt = g["y_true"].to_numpy(dtype=float)
    out: dict[str, float] = {}
    for name, col in [
        ("Persistence", "y_pred_persist"),
        ("LightGBM", "y_pred_lgbm"),
        ("Chronos", "y_pred_chronos"),
    ]:
        if col not in g.columns:
            continue
        yp = g[col].to_numpy(dtype=float)
        m = np.isfinite(yt) & np.isfinite(yp)
        if m.any():
            out[f"rmse_{name}"] = rmse(yt[m], yp[m])
            out[f"mae_{name}"] = mae(yt[m], yp[m])
    return out


def plot_multimethod_forecast_timeseries(
    pred: pd.DataFrame,
    station_id: str,
    lead_h: int,
    out: Path,
    *,
    max_points: int = 1200,
    title_suffix: str = "",
) -> None:
    """Truth vs Persistence / LightGBM / Chronos on hold-out test (numeric $H_s$ forecast)."""
    _fig_style()
    g = pred[(pred["station_id"].astype(str) == str(station_id)) & (pred["lead_h"] == int(lead_h))].copy()
    g = g.sort_values("time_utc")
    if g.empty:
        log.warning("No predictions for station=%s lead=%s", station_id, lead_h)
        return
    if len(g) > max_points:
        g = g.iloc[-max_points:]
    g["time_utc"] = pd.to_datetime(g["time_utc"], utc=True)

    mets = _method_metrics(g, lead_h)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})

    ax1.plot(g["time_utc"], g["y_true"], color="k", lw=1.1, label="NDBC truth $H_s(t+{0}h)$".format(lead_h))
    ax1.plot(g["time_utc"], g["y_pred_persist"], color="#7f7f7f", lw=0.95, alpha=0.92, label="Persistence")
    leg_lgbm = "LightGBM"
    if f"rmse_{leg_lgbm}" in mets:
        leg_lgbm += f" (RMSE={mets[f'rmse_{leg_lgbm}']:.3f}m)"
    ax1.plot(g["time_utc"], g["y_pred_lgbm"], color="#1f77b4", lw=0.95, label=leg_lgbm)
    if "y_pred_chronos" in g.columns and g["y_pred_chronos"].notna().any():
        t = g["time_utc"].to_numpy()
        yc = g["y_pred_chronos"].to_numpy(dtype=float)
        m = np.isfinite(yc)
        leg_c = "Chronos-T5"
        if "rmse_Chronos" in mets:
            leg_c += f" (RMSE={mets['rmse_Chronos']:.3f}m)"
        ax1.plot(t[m], yc[m], color="#9467bd", lw=0, marker="o", markersize=2.5, label=leg_c)
    ax1.set_ylabel("$H_s$ at target time (m)")
    ax1.set_title(
        f"Numeric forecast comparison — station {station_id}, lead {lead_h}h{title_suffix}\n"
        "Mistral-7B does not output $H_s$; see regime overlay figures for LLM outputs."
    )
    ax1.legend(loc="upper left", ncol=2, fontsize=8)

    ax2.plot(g["time_utc"], g["y_true"] - g["y_pred_persist"], color="#7f7f7f", lw=0.8, alpha=0.85, label="Truth − Persist")
    ax2.plot(g["time_utc"], g["y_true"] - g["y_pred_lgbm"], color="#1f77b4", lw=0.85, label="Truth − LightGBM")
    if "y_pred_chronos" in g.columns and g["y_pred_chronos"].notna().any():
        err_c = g["y_true"] - g["y_pred_chronos"]
        m = err_c.notna() & np.isfinite(err_c)
        ax2.plot(g.loc[m, "time_utc"], err_c[m], color="#9467bd", lw=0, marker="o", markersize=2.0, label="Truth − Chronos")
    ax2.axhline(0, color="k", lw=0.5, alpha=0.35)
    ax2.set_ylabel("Error (m)")
    ax2.set_xlabel("Issue time (UTC)")
    ax2.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_mistral_regime_timeline(
    panel: pd.DataFrame,
    merged: pd.DataFrame,
    station_id: str,
    lead_h: int,
    out: Path,
    *,
    max_panel_points: int = 2500,
) -> None:
    """$H_s$ series with colored issue-time markers: true / Base-Mistral / LoRA-Mistral regime."""
    _fig_style()
    sid = str(station_id)
    g = panel[panel["station_id"].astype(str) == sid].sort_values("time_utc")
    if len(g) > max_panel_points:
        g = g.iloc[-max_panel_points:]
    ev = merged[merged["station_id"].astype(str) == sid].sort_values("issue_time")
    if g.empty or ev.empty:
        return
    ev_lora = ev[ev["lora_regime"].astype(str).str.len() > 0]

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 0.55, 0.55, 0.55]})

    ax0 = axes[0]
    ax0.plot(g["time_utc"], g["Hs_m"], color="#1f77b4", lw=0.85)
    ax0.set_ylabel("$H_s$ (m)")
    ax0.set_title(f"Station {sid} — wave height + Mistral regime at forecast issue times (lead {lead_h}h)")

    for ax, col, title, subset in [
        (axes[1], "true_regime", "Label regime (windows)", ev),
        (axes[2], "base_regime", "Mistral Base (eval points)", ev_lora),
        (axes[3], "lora_regime", "Mistral + LoRA (eval points)", ev_lora),
    ]:
        for _, row in subset.iterrows():
            t = row["issue_time"]
            reg = str(row.get(col, "unknown"))
            c = _REGIME_COLORS.get(reg, _REGIME_COLORS["unknown"])
            ax.axvspan(t, t + pd.Timedelta(hours=6), color=c, alpha=0.85, linewidth=0)
            ok_col = "regime_ok_base" if col == "base_regime" else "regime_ok_lora" if col == "lora_regime" else None
            if ok_col and col != "true_regime":
                edge = "#2ca02c" if row.get(ok_col) else "#d62728"
                ax.axvline(t, color=edge, lw=1.2, alpha=0.9)
        ax.set_yticks([])
        ax.set_ylabel(title, fontsize=8)
    axes[-1].set_xlabel("UTC")
    handles = [
        Patch(facecolor=_REGIME_COLORS.get(r, "#ccc"), label=r, edgecolor="k", linewidth=0.3) for r in WAVE_REGIMES
    ]
    axes[0].legend(handles=handles, loc="upper right", ncol=3, fontsize=7, title="Regime")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_forecast_scatter_methods(
    pred: pd.DataFrame,
    lead_h: int,
    out: Path,
    *,
    max_points: int = 4000,
) -> None:
    """Scatter truth vs prediction for each numeric method (lead fixed)."""
    _fig_style()
    g = pred.loc[pred["lead_h"] == int(lead_h)].copy()
    if len(g) > max_points:
        g = g.sample(n=max_points, random_state=42)
    methods = [
        ("Persistence", "y_pred_persist", "#7f7f7f"),
        ("LightGBM", "y_pred_lgbm", "#1f77b4"),
    ]
    if "y_pred_chronos" in g.columns and g["y_pred_chronos"].notna().any():
        methods.append(("Chronos", "y_pred_chronos", "#9467bd"))

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    lims: list[float] = []
    for name, col, _ in methods:
        m = g[col].notna() & np.isfinite(g[col])
        lims.extend(g.loc[m, "y_true"].tolist())
        lims.extend(g.loc[m, col].tolist())
    lo, hi = (min(lims), max(lims)) if lims else (0, 3)
    pad = 0.05 * (hi - lo + 1e-6)

    for ax, (name, col, color) in zip(axes[0], methods):
        m = g[col].notna() & np.isfinite(g[col])
        x = g.loc[m, "y_true"].to_numpy()
        y = g.loc[m, col].to_numpy()
        ax.scatter(x, y, s=6, alpha=0.35, color=color, edgecolors="none")
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Truth $H_s$ (m)")
        ax.set_ylabel(f"Predicted ({name})")
        r = rmse(x, y) if len(x) else float("nan")
        ax.set_title(f"{name}\nRMSE={r:.3f} m, n={len(x)}")
    fig.suptitle(f"Hold-out scatter — all stations, lead {lead_h}h", fontsize=12, y=1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_mistral_predictability_vs_lgbm_error(merged: pd.DataFrame, out: Path) -> None:
    """Boxplot: LightGBM |error| grouped by LoRA-predicted predictability level."""
    _fig_style()
    if merged.empty or "lora_pred_level" not in merged.columns:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    levels = ["high", "medium", "low"]
    data = [merged.loc[merged["lora_pred_level"] == lv, "err_lgbm"].dropna().to_numpy() for lv in levels]
    data = [d for d in data if len(d)]
    labels = [lv for lv, d in zip(levels, data) if len(d)]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=labels)
    ax.set_xlabel("Mistral LoRA — predictability_24h (predicted)")
    ax.set_ylabel("|Truth − LightGBM| at target (m)")
    ax.set_title("Numeric forecast error vs LLM predictability class\n(same issue times as Mistral eval)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_eval_snapshot_grid(
    merged: pd.DataFrame,
    out: Path,
    *,
    n_cols: int = 4,
    max_samples: int = 16,
) -> None:
    """Small multiples: at each eval issue time, bar compare truth vs numeric preds; title shows Mistral regime."""
    _fig_style()
    ev = merged[merged["lora_regime"].astype(str).str.len() > 0].sort_values("issue_time").head(max_samples)
    if ev.empty:
        return
    n = len(ev)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows), squeeze=False)
    methods = ["y_true", "y_pred_persist", "y_pred_lgbm"]
    labels = ["Truth", "Persist", "LGBM"]
    colors = ["k", "#7f7f7f", "#1f77b4"]
    if ev["y_pred_chronos"].notna().any():
        methods.append("y_pred_chronos")
        labels.append("Chronos")
        colors.append("#9467bd")

    for idx, (_, row) in enumerate(ev.iterrows()):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        vals = [float(row[m]) for m in methods]
        ax.bar(labels, vals, color=colors, edgecolor="k", linewidth=0.4)
        ok = "OK" if row.get("regime_ok_lora") else "X"
        ax.set_title(
            f"{row['station_id']} {pd.Timestamp(row['issue_time']).strftime('%m-%d %H:%M')}\n"
            f"LoRA regime: {row.get('lora_regime','?')} ({ok})",
            fontsize=8,
        )
        ax.set_ylabel("$H_s$ (m)", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)

    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.suptitle(f"$H_s$ at target (+{ev['lead_h'].iloc[0]}h) vs numeric models — Mistral eval windows", fontsize=11, y=1.01)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison_summary(
    pred: pd.DataFrame,
    merged: pd.DataFrame,
    lead_h: int,
    out: Path,
) -> None:
    """Bar chart: RMSE by method (full test) + Mistral regime accuracy on eval subset."""
    _fig_style()
    mets = _method_metrics(pred, lead_h)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    names, rmses = [], []
    for key, label in [("rmse_Persistence", "Persistence"), ("rmse_LightGBM", "LightGBM"), ("rmse_Chronos", "Chronos")]:
        if key in mets:
            names.append(label)
            rmses.append(mets[key])
    if names:
        ax1.bar(names, rmses, color=["#7f7f7f", "#1f77b4", "#9467bd"][: len(names)], edgecolor="k")
        ax1.set_ylabel("RMSE (m)")
        ax1.set_title(f"Numeric $H_s$ forecast RMSE (test holdout, lead {lead_h}h)")

    ev = merged[merged["lora_regime"].astype(str).str.len() > 0] if not merged.empty else merged
    if not ev.empty:
        acc_base = float(ev["regime_ok_base"].mean())
        acc_lora = float(ev["regime_ok_lora"].mean())
        acc_p_base = float(ev["pred_ok_base"].mean())
        acc_p_lora = float(ev["pred_ok_lora"].mean())
        x = np.arange(2)
        w = 0.35
        ax2.bar(x - w / 2, [acc_base, acc_p_base], width=w, label="Mistral Base", color="#aec7e8", edgecolor="k")
        ax2.bar(x + w / 2, [acc_lora, acc_p_lora], width=w, label="Mistral LoRA", color="#ff7f0e", edgecolor="k")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["wave_regime", "predictability_24h"])
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"Mistral classification (n={len(ev)} eval-aligned samples)")
        ax2.legend(fontsize=8)
    fig.suptitle("Numeric forecast vs Mistral LLM tasks (complementary, not same output)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_model_rmse_comparison_by_lead(
    base_metrics: dict[str, Any],
    lora_metrics: dict[str, Any] | None,
    out: Path,
    *,
    horizon_h: int = 24,
) -> None:
    """Bar chart: per-step RMSE for Mistral Base vs LoRA curve forecasts."""
    _fig_style()
    b_steps = base_metrics.get("rmse_by_forecast_step_h") or {}
    l_steps = (lora_metrics or {}).get("rmse_by_forecast_step_h") or {}
    if not b_steps and not l_steps:
        log.warning("No rmse_by_forecast_step_h in curve metrics — skip %s", out)
        return
    steps = sorted({int(k) for k in list(b_steps.keys()) + list(l_steps.keys())})
    if horizon_h > 0:
        steps = [s for s in steps if s <= horizon_h]
    if not steps:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(steps))
    w = 0.38
    b_vals = [float(b_steps.get(str(s), np.nan)) for s in steps]
    ax.bar(x - w / 2, b_vals, width=w, label="Mistral Base", color="#aec7e8", edgecolor="k")
    if l_steps:
        l_vals = [float(l_steps.get(str(s), np.nan)) for s in steps]
        ax.bar(x + w / 2, l_vals, width=w, label="Mistral LoRA", color="#ff7f0e", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels([f"+{s}h" for s in steps], rotation=45 if len(steps) > 12 else 0, ha="right")
    ax.set_ylabel("RMSE (m)")
    ax.set_xlabel("Forecast step from issue time")
    ax.set_title(
        f"Mistral JSON curve forecast — RMSE by lead step (horizon {horizon_h}h)\n"
        f"Base mean={base_metrics.get('mean_rmse')}  LoRA mean={(lora_metrics or {}).get('mean_rmse')}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_forecast_panel_mistral_lora(
    test_rows: list[dict[str, Any]],
    base_results: list[dict[str, Any]],
    lora_results: list[dict[str, Any]],
    station_id: str,
    out: Path,
    *,
    pred_df: pd.DataFrame | None = None,
    max_panels: int = 3,
) -> None:
    """Top: truth / persist / LGBM-Chronos points / Mistral curves; bottom: per-model errors."""
    _fig_style()
    sid = str(station_id)
    base_by = {r["sample_idx"]: r for r in base_results}
    lora_by = {r["sample_idx"]: r for r in lora_results}
    picks: list[tuple[int, dict]] = []
    for i, rec in enumerate(test_rows):
        inp = rec.get("input") or {}
        if str(inp.get("station_id", "")) != sid or i not in base_by:
            continue
        picks.append((i, rec))
    if not picks:
        return
    picks = picks[-max_panels:]
    n = len(picks)
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(13, 3.2 * n),
        squeeze=False,
        gridspec_kw={"width_ratios": [3.2, 1.0], "wspace": 0.28},
    )
    if pred_df is not None and not pred_df.empty:
        pred_df = pred_df.copy()
        pred_df["time_utc"] = pd.to_datetime(pred_df["time_utc"], utc=True)

    for row_i, (idx, rec) in enumerate(picks):
        ax_main, ax_skill = axes[row_i]
        inp = rec.get("input") or {}
        out_rec = rec.get("output") or {}
        br = base_by[idx]
        lr = lora_by.get(idx, {})
        hs_hist = list(inp.get("history_hs_m", []))
        hs_true = np.asarray(out_rec.get("hs_forecast_m", br.get("hs_true", [])), dtype=float)
        hs_base = np.asarray(br.get("hs_pred", []), dtype=float)
        hs_lora = np.asarray(lr.get("hs_pred", []), dtype=float)
        hs_persist = np.asarray(br.get("hs_persist", lr.get("hs_persist", [])), dtype=float)
        if not len(hs_persist) and hs_hist:
            hs_persist = np.full(len(hs_true), float(hs_hist[-1]))
        dt = int(inp.get("history_dt_h", 1))
        n_fut = len(hs_true)
        hist_t = np.arange(-len(hs_hist) * dt, 0, dt)
        fut_t = np.arange(dt, (n_fut + 1) * dt, dt)[:n_fut]

        ax_main.plot(hist_t, hs_hist, color="#1f77b4", lw=0.85, label="History")
        ax_main.plot(fut_t, hs_true, color="k", lw=1.15, label="Truth")
        if len(hs_persist):
            ax_main.plot(fut_t[: len(hs_persist)], hs_persist, color="#7f7f7f", lw=0.95, ls="--", label="Persistence")
        if len(hs_base):
            ax_main.plot(fut_t[: len(hs_base)], hs_base, color="#aec7e8", lw=0.95, ls="--", label="Mistral Base")
        if len(hs_lora):
            ax_main.plot(fut_t[: len(hs_lora)], hs_lora, color="#ff7f0e", lw=1.0, label="Mistral LoRA")
        if pred_df is not None and not pred_df.empty:
            issue = pd.Timestamp(inp.get("issue_time"), tz="UTC")
            g = pred_df.loc[
                (pred_df["station_id"].astype(str) == sid) & (pred_df["time_utc"] == issue)
            ]
            if not g.empty:
                gl = g.sort_values("lead_h")
                ax_main.plot(
                    gl["lead_h"],
                    gl["y_pred_lgbm"],
                    color="#1f77b4",
                    lw=0,
                    marker="s",
                    markersize=4,
                    label="LightGBM (tabular leads)",
                )
                if "y_pred_chronos" in gl.columns and gl["y_pred_chronos"].notna().any():
                    m = gl["y_pred_chronos"].notna()
                    ax_main.plot(
                        gl.loc[m, "lead_h"],
                        gl.loc[m, "y_pred_chronos"],
                        color="#9467bd",
                        lw=0,
                        marker="o",
                        markersize=4,
                        label="Chronos",
                    )
        ax_main.axvline(0, color="gray", ls=":", lw=0.7)
        issue = inp.get("issue_time", "")
        ax_main.set_title(f"{sid} @ {issue}", fontsize=9)
        ax_main.set_ylabel("$H_s$ (m)")
        ax_main.legend(loc="upper left", fontsize=6.5, ncol=2)

        methods: list[tuple[str, np.ndarray, str]] = [
            ("Persist", hs_persist, "#7f7f7f"),
            ("Mistral-B", hs_base, "#aec7e8"),
            ("Mistral-L", hs_lora, "#ff7f0e"),
        ]
        rmse_bars, labels, colors = [], [], []
        for name, yp, c in methods:
            m = min(len(hs_true), len(yp))
            if m and np.isfinite(yp[:m]).any():
                rmse_bars.append(rmse(hs_true[:m], yp[:m]))
                labels.append(name)
                colors.append(c)
        pts = br.get("point_forecasts") or lr.get("point_forecasts") or {}
        if pts:
            lgbm_e = [
                (hs_true[int(lh) - 1] - vals["lgbm"]) ** 2
                for lh, vals in pts.items()
                if int(lh) - 1 < len(hs_true) and np.isfinite(vals.get("lgbm", np.nan))
            ]
            if lgbm_e:
                rmse_bars.append(float(np.sqrt(np.mean(lgbm_e))))
                labels.append("LGBM†")
                colors.append("#1f77b4")
        if rmse_bars:
            ax_skill.bar(labels, rmse_bars, color=colors, edgecolor="k", linewidth=0.4)
            ax_skill.set_ylabel("RMSE (m)")
            ax_skill.set_title("Sample RMSE", fontsize=8)
            ax_skill.tick_params(axis="x", labelsize=7)

        mlen = min(len(hs_true), len(hs_lora) if len(hs_lora) else len(hs_base))
        if mlen:
            ax_err = ax_main.twinx()
            ax_err.set_ylabel("LoRA err", fontsize=7, color="#ff7f0e")
            if len(hs_lora):
                err = hs_true[:mlen] - hs_lora[:mlen]
                ax_err.bar(fut_t[:mlen], err, width=0.6, alpha=0.15, color="#ff7f0e", label="_err")

    axes[-1][0].set_xlabel("Hours from issue time")
    fig.suptitle(
        f"Multi-method $H_s$ curve — station {sid} (†LGBM at configured lead hours only)",
        fontsize=11,
        y=1.01,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_curve_method_summary_bars(
    base_metrics: dict[str, Any],
    lora_metrics: dict[str, Any] | None,
    out: Path,
    *,
    baseline_metrics: dict[str, Any] | None = None,
) -> None:
    """Compare mean RMSE: Persistence / LGBM / Chronos / Mistral Base / LoRA."""
    _fig_style()
    baseline_metrics = baseline_metrics or base_metrics.get("baselines") or {}
    names, vals, colors = [], [], []
    for key, label, color in [
        ("mean_rmse_persist", "Persistence", "#7f7f7f"),
        ("mean_rmse_lgbm_at_numeric_leads", "LightGBM†", "#1f77b4"),
        ("mean_rmse_chronos_at_numeric_leads", "Chronos†", "#9467bd"),
    ]:
        v = baseline_metrics.get(key)
        if v is not None and np.isfinite(v):
            names.append(label)
            vals.append(float(v))
            colors.append(color)
    for m, label, color in [
        (base_metrics, "Mistral Base", "#aec7e8"),
        (lora_metrics, "Mistral LoRA", "#ff7f0e"),
    ]:
        if not m:
            continue
        v = m.get("mean_rmse")
        if v is not None and np.isfinite(v):
            names.append(label)
            vals.append(float(v))
            colors.append(color)
    if not names:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, vals, color=colors, edgecolor="k")
    ax.set_ylabel("Mean RMSE (m)")
    ax.set_title("Curve forecast — method comparison on Mistral eval subset\n(†tabular models at standard lead hours only)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_curve_forecast_examples(
    test_rows: list[dict[str, Any]],
    base_results: list[dict[str, Any]],
    lora_results: list[dict[str, Any]],
    station_id: str,
    out: Path,
    *,
    max_panels: int = 4,
) -> None:
    """Multi-panel: one issue time each — history Hs + future truth vs Mistral Base/LoRA curves."""
    _fig_style()
    sid = str(station_id)
    base_by = {r["sample_idx"]: r for r in base_results}
    lora_by = {r["sample_idx"]: r for r in lora_results}
    picks: list[tuple[int, dict]] = []
    for i, rec in enumerate(test_rows):
        inp = rec.get("input") or {}
        if str(inp.get("station_id", "")) != sid:
            continue
        if i in base_by:
            picks.append((i, rec))
    if not picks:
        return
    picks = picks[-max_panels:]
    n = len(picks)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n), squeeze=False)
    for row_ax, (idx, rec) in zip(axes[0], picks):
        inp = rec.get("input") or {}
        out_rec = rec.get("output") or {}
        br = base_by[idx]
        lr = lora_by.get(idx, {})
        hs_hist = list(inp.get("history_hs_m", []))
        hs_true = list(out_rec.get("hs_forecast_m", br.get("hs_true", [])))
        hs_base = list(br.get("hs_pred", []))
        hs_lora = list(lr.get("hs_pred", []))
        dt = int(inp.get("history_dt_h", 1))
        n_fut = len(hs_true)
        hist_t = np.arange(-len(hs_hist) * dt, 0, dt)
        fut_t = np.arange(dt, (n_fut + 1) * dt, dt)[:n_fut]

        row_ax.plot(hist_t, hs_hist, color="#1f77b4", lw=0.9, label="History $H_s$")
        row_ax.plot(fut_t, hs_true, color="k", lw=1.2, label="Truth (future)")
        if hs_base:
            row_ax.plot(fut_t[: len(hs_base)], hs_base, color="#aec7e8", lw=1.0, ls="--", label="Mistral Base")
        if hs_lora:
            row_ax.plot(fut_t[: len(hs_lora)], hs_lora, color="#ff7f0e", lw=1.0, label="Mistral LoRA")
        row_ax.axvline(0, color="gray", ls=":", lw=0.8)
        issue = inp.get("issue_time", "")
        row_ax.set_title(
            f"{sid} issue {issue}  |  RMSE base={br.get('rmse', float('nan')):.3f}  "
            f"LoRA={lr.get('rmse', float('nan')):.3f}" if lr else f"{sid} issue {issue}",
            fontsize=9,
        )
        row_ax.set_ylabel("$H_s$ (m)")
        row_ax.legend(loc="upper left", fontsize=7, ncol=2)
    axes[-1][0].set_xlabel("Hours relative to issue time")
    fig.suptitle(f"Mistral curve forecast examples — station {sid}", fontsize=11, y=1.01)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_mistral_figures_index(fig_dir: Path, image_names: list[str]) -> None:
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Mistral + forecast figures</title>",
        "<style>body{font-family:system-ui;margin:24px;max-width:1200px}"
        "img{max-width:100%;border:1px solid #ddd;margin:12px 0}"
        "h2{margin-top:2em}</style></head><body>",
        "<h1>Mistral &amp; multi-method forecast figures</h1>",
        "<p><b>Numeric lines</b>: Persistence, LightGBM, Chronos predict $H_s$ at $t$+lead. "
        "<b>Mistral classification</b>: <code>wave_regime</code> / <code>predictability_24h</code>. "
        "<b>Mistral curve</b> (<code>forecast_panel_mistral_curve_*</code>, "
        "<code>model_rmse_comparison_by_lead.png</code>): JSON <code>hs_forecast_m</code> sequences.</p>",
    ]
    sections = {
        "Summary": ["mistral_methods_summary", "mistral_forecast_scatter", "model_rmse_comparison_by_lead"],
        "Curve forecast (Mistral JSON)": ["forecast_panel_mistral_lora_", "forecast_panel_mistral_curve_", "curve_method_rmse"],
        "Time series (per station)": ["mistral_forecast_ts_", "mistral_regime_timeline_"],
        "Eval snapshots": ["mistral_eval_snapshots", "mistral_predictability_vs_lgbm_error"],
        "Classification (confusion)": ["mistral_base_regime", "mistral_lora_regime", "mistral_lora_predictability"],
    }
    used = set()
    for section, prefixes in sections.items():
        lines.append(f"<h2>{section}</h2>")
        for name in sorted(image_names):
            if any(name.startswith(p) or p in name for p in prefixes):
                lines.append(f"<h3>{name}</h3><img src='{name}' alt='{name}'/>")
                used.add(name)
    for name in sorted(image_names):
        if name not in used and name.startswith("mistral_"):
            lines.append(f"<h3>{name}</h3><img src='{name}' alt='{name}'/>")
    lines.append("</body></html>")
    (fig_dir / "mistral_index.html").write_text("\n".join(lines), encoding="utf-8")
