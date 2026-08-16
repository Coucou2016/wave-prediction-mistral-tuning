from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fig_style() -> None:
    from wave_llm.evaluation.science_plots_style import apply_science_style

    apply_science_style(font_size=10, use_times=True)


def plot_station_series(df: pd.DataFrame, station_id: str, out: Path) -> None:
    _fig_style()
    g = df[df["station_id"] == station_id].sort_values("time_utc")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(g["time_utc"], g["Hs_m"], lw=0.85, color="#1f77b4")
    ax.set_ylabel("$H_s$ (m)")
    ax.set_title(f"Significant wave height — station {station_id}")
    ax.set_xlabel("UTC")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_station_multivar(
    df: pd.DataFrame,
    station_id: str,
    out: Path,
    max_points: int = 4000,
) -> None:
    """Hs + Tp + wind on twin axes (subsample for readability)."""
    _fig_style()
    g = df[df["station_id"] == station_id].sort_values("time_utc").copy()
    if len(g) > max_points:
        g = g.iloc[:: max(1, len(g) // max_points)]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(11, 3.8))
    ax1.plot(g["time_utc"], g["Hs_m"], lw=0.9, color="#1f77b4", label="$H_s$")
    ax1.set_ylabel("$H_s$ (m)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    if g["Tp_s"].notna().any():
        ax2.plot(g["time_utc"], g["Tp_s"], lw=0.75, color="#d62728", alpha=0.85, label="$T_p$")
    if g["wind_speed_ms"].notna().any():
        ax2.plot(g["time_utc"], g["wind_speed_ms"], lw=0.65, color="#2ca02c", alpha=0.75, label="Wind (m/s)")
    ax2.set_ylabel("$T_p$ (s) / wind (m/s)")
    ax1.set_title(f"Wave + wind — {station_id}")
    ax1.set_xlabel("UTC")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_baseline_metrics(metrics_path: Path, out: Path) -> None:
    _fig_style()
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    leads = [r["lead_h"] for r in rows]
    rmse_p = [r["rmse_persist"] for r in rows]
    rmse_l = [r["rmse_lgbm"] for r in rows]
    skill = [r["skill_vs_persist"] for r in rows]
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(leads))
    w = 0.35
    ax1.bar(x - w / 2, rmse_p, width=w, label="Persistence", color="#7f7f7f")
    ax1.bar(x + w / 2, rmse_l, width=w, label="LightGBM", color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{h}h" for h in leads])
    ax1.set_ylabel("RMSE ($H_s$, m)")
    ax1.set_title("Forecast horizon vs RMSE")
    ax1.legend()

    ax2.axhline(0, color="#999", lw=1)
    ax2.bar(x, skill, color="#2ca02c", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{h}h" for h in leads])
    ax2.set_ylabel(r"Skill $= 1 - \mathrm{RMSE}_{LGBM}/\mathrm{RMSE}_{persist}$")
    ax2.set_title("LightGBM vs persistence")

    fig.suptitle("Numeric baselines (test tail holdout)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_regime_distribution(windows: pd.DataFrame, out: Path) -> None:
    _fig_style()
    out.parent.mkdir(parents=True, exist_ok=True)
    vc = windows["wave_regime"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = plt.cm.viridis(np.linspace(0.25, 0.9, len(vc)))
    ax.barh(vc.index.astype(str), vc.values, color=colors)
    ax.set_xlabel("Window count")
    ax.set_title("Rule-based wave regime (all leads)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_regime_by_station(windows: pd.DataFrame, out: Path) -> None:
    _fig_style()
    out.parent.mkdir(parents=True, exist_ok=True)
    ct = pd.crosstab(windows["station_id"], windows["wave_regime"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.82)
    ax.set_ylabel("Windows")
    ax.set_xlabel("Station")
    ax.legend(title="Regime", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title("Regime mix by station")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_station_map_cartopy(meta: pd.DataFrame, out: Path, padding_deg: float = 6.0) -> None:
    """Map with land/ocean/coastlines (Cartopy). `meta` columns: station_id, lat, lon."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as e:
        raise RuntimeError("cartopy is required for coastline basemap figures") from e
    _fig_style()
    if meta.empty:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    lats = meta["lat"].to_numpy(dtype=float)
    lons = meta["lon"].to_numpy(dtype=float)
    lon0, lon1 = float(lons.min() - padding_deg), float(lons.max() + padding_deg)
    lat0, lat1 = float(lats.min() - padding_deg), float(lats.max() + padding_deg)

    fig = plt.figure(figsize=(10.5, 6.2))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8e4dc", edgecolor="#666", linewidth=0.35, zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#c6d9ec", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.55, edgecolor="#222", zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="#888", alpha=0.6, zorder=1)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.55, linestyle="--")
    ax.scatter(
        lons,
        lats,
        transform=ccrs.PlateCarree(),
        s=70,
        color="#c0392b",
        edgecolors="k",
        linewidths=0.45,
        zorder=4,
        label="NDBC buoys (panel)",
    )
    for _, r in meta.iterrows():
        ax.text(
            float(r["lon"]) + 0.15,
            float(r["lat"]) + 0.15,
            str(r["station_id"]),
            transform=ccrs.PlateCarree(),
            fontsize=8,
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.85),
        )
    ax.legend(loc="lower left")
    ax.set_title("Station map — NDBC official locations + Natural Earth coastlines (Cartopy)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_hs_distribution_by_station(df: pd.DataFrame, out: Path) -> None:
    _fig_style()
    out.parent.mkdir(parents=True, exist_ok=True)
    stations = sorted(df["station_id"].astype(str).unique())
    parts = [df.loc[df["station_id"].astype(str) == s, "Hs_m"].dropna().to_numpy() for s in stations]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(parts, tick_labels=stations, vert=True)
    ax.set_ylabel("$H_s$ (m)")
    ax.set_title("$H_s$ distribution by station (hourly panel)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_truth_pred_error_panel(
    pred: pd.DataFrame,
    station_id: str,
    lead_h: int,
    out: Path,
    max_points: int = 1500,
) -> None:
    """
    Truth vs model predictions + error subplot.
    `pred` columns: time_utc, y_true, y_pred_persist, y_pred_lgbm, optional y_pred_chronos.
    """
    _fig_style()
    g = pred[(pred["station_id"].astype(str) == str(station_id)) & (pred["lead_h"] == int(lead_h))].copy()
    g = g.sort_values("time_utc")
    if g.empty:
        return
    if len(g) > max_points:
        g = g.iloc[-max_points:]
    g["time_utc"] = pd.to_datetime(g["time_utc"], utc=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )
    ax1.plot(g["time_utc"], g["y_true"], color="k", lw=1.0, label="NDBC truth $H_s$")
    ax1.plot(g["time_utc"], g["y_pred_persist"], color="#7f7f7f", lw=0.9, alpha=0.9, label="Persistence")
    ax1.plot(g["time_utc"], g["y_pred_lgbm"], color="#1f77b4", lw=0.9, alpha=0.95, label="LightGBM (tabular)")
    if "y_pred_chronos" in g.columns and g["y_pred_chronos"].notna().any():
        t = g["time_utc"].to_numpy()
        yc = g["y_pred_chronos"].to_numpy(dtype=float)
        m = np.isfinite(yc)
        ax1.plot(
            t[m],
            yc[m],
            color="#9467bd",
            lw=0.95,
            alpha=0.95,
            marker="o",
            markersize=2.2,
            linestyle="None",
            label="Chronos-T5 (HF pretrained, subsampled issues)",
        )
    ax1.set_ylabel("$H_s$ (m)")
    ax1.set_title(
        f"Hold-out test — station {station_id}, lead {lead_h} h\n"
        "Truth = NDBC observations; curves = model forward passes on real past context (no synthetic Hs)."
    )
    ax1.legend(loc="upper left", ncol=2, fontsize=8)

    err_lgbm = g["y_true"] - g["y_pred_lgbm"]
    ax2.plot(g["time_utc"], err_lgbm, color="#1f77b4", lw=0.85, label="Error truth − LightGBM")
    if "y_pred_chronos" in g.columns and g["y_pred_chronos"].notna().any():
        t = g["time_utc"].to_numpy()
        yc = g["y_pred_chronos"].to_numpy(dtype=float)
        m = np.isfinite(yc)
        err_c = g["y_true"] - g["y_pred_chronos"]
        ax2.plot(t[m], err_c.to_numpy()[m], color="#9467bd", lw=0.85, alpha=0.9, marker="o", markersize=2.0, linestyle="None", label="Error truth − Chronos")
    ax2.axhline(0.0, color="k", lw=0.6, alpha=0.35)
    ax2.set_ylabel("Error (m)")
    ax2.set_xlabel("Issue time (UTC)")
    ax2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
