from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
import pandas as pd  # noqa: E402

from wave_llm.evaluation.mistral_plots import (  # noqa: E402
    plot_mistral_predictability_accuracy,
    plot_mistral_regime_confusion,
)
from wave_llm.evaluation.mistral_ts_plots import (  # noqa: E402
    build_eval_merged_frame,
    plot_curve_method_summary_bars,
    plot_forecast_panel_mistral_lora,
    plot_eval_snapshot_grid,
    plot_forecast_scatter_methods,
    plot_method_comparison_summary,
    plot_model_rmse_comparison_by_lead,
    plot_mistral_predictability_vs_lgbm_error,
    plot_mistral_regime_timeline,
    plot_multimethod_forecast_timeseries,
    write_mistral_figures_index,
)
from wave_llm.models.mistral_lora import load_jsonl  # noqa: E402
from wave_llm.evaluation.plots import (  # noqa: E402
    plot_baseline_metrics,
    plot_hs_distribution_by_station,
    plot_regime_by_station,
    plot_regime_distribution,
    plot_station_map_cartopy,
    plot_station_multivar,
    plot_station_series,
    plot_truth_pred_error_panel,
)
from wave_llm.io.ndbc_station_meta import station_meta_for_ids  # noqa: E402
from wave_llm.representations.predictability import build_predictability_lookup  # noqa: E402
from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    out = ensure_dir(ROOT / "data" / "processed" / "figures")
    panel = ROOT / "data" / "processed" / "panel_hourly.parquet"
    metrics = ROOT / "data" / "processed" / "metrics" / "numeric_baselines.json"
    windows_p = ROOT / "data" / "processed" / "windows" / "windows.parquet"
    pred_all = ROOT / "data" / "processed" / "predictions" / "test_predictions_all.parquet"
    pred_lgbm = ROOT / "data" / "processed" / "predictions" / "test_predictions_lgbm.parquet"

    df = pd.read_parquet(panel)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)

    # --- NDBC official station positions + coastline map ---
    sids = sorted(df["station_id"].astype(str).unique().tolist())
    cache = ROOT / "data" / "raw" / "ndbc" / "station_table.txt"
    meta = station_meta_for_ids(sids, cache)
    plot_station_map_cartopy(meta, out / "station_map.png")
    logging.info("Wrote station_map.png (Cartopy coastlines + NDBC official lat/lon)")

    # Representative stations for multivariate + forecast panels
    var_by = df.groupby("station_id")["Hs_m"].var().sort_values(ascending=False)
    pick: list[str] = []
    for s in [str(df["station_id"].iloc[0]), str(var_by.index[0]), str(var_by.index[min(3, len(var_by) - 1)])]:
        if s not in pick:
            pick.append(s)
    for sid in pick[:3]:
        plot_station_series(df, sid, out / f"series_{sid}.png")
        plot_station_multivar(df, sid, out / f"multivar_{sid}.png")
        logging.info("Series + multivar: %s", sid)

    if metrics.exists():
        plot_baseline_metrics(metrics, out / "baseline_rmse_skill.png")
        logging.info("Wrote baseline_rmse_skill.png")

    if windows_p.exists():
        w = pd.read_parquet(windows_p)
        plot_regime_distribution(w, out / "regime_counts.png")
        plot_regime_by_station(w, out / "regime_by_station.png")
        logging.info("Wrote regime figures")

    plot_hs_distribution_by_station(df, out / "hs_boxplot_by_station.png")

    mistral_dir = ROOT / "data" / "processed" / "mistral"
    for src, dst in [
        (mistral_dir / "metrics_lora.json", out / "mistral_lora_regime_confusion.png"),
        (mistral_dir / "metrics_base.json", out / "mistral_base_regime_confusion.png"),
    ]:
        if src.exists():
            plot_mistral_regime_confusion(src, dst)
    lora_m = mistral_dir / "metrics_lora.json"
    if lora_m.exists():
        plot_mistral_predictability_accuracy(lora_m, out / "mistral_lora_predictability_accuracy.png")
    zs = mistral_dir / "metrics.json"
    if zs.exists() and not lora_m.exists():
        plot_mistral_regime_confusion(zs, out / "mistral_regime_confusion.png")
        plot_mistral_predictability_accuracy(zs, out / "mistral_predictability_accuracy.png")
    if not any((mistral_dir / f).exists() for f in ("metrics_lora.json", "metrics.json", "metrics_base.json")):
        logging.info("Skip Mistral figures — run 08_eval_base_vs_lora.py or 07c")

    # --- Truth / predictions / errors (real hold-out; Chronos merged if 05b ran) ---
    pred_path = pred_all if pred_all.exists() else pred_lgbm
    mcfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    leads_cfg = [int(x) for x in mcfg.get("target_lead_hours", [24])]
    default_lead = int(mcfg.get("figures", {}).get("forecast_lead_h", 24))
    if default_lead not in leads_cfg:
        default_lead = int(leads_cfg[min(len(leads_cfg) // 2, len(leads_cfg) - 1)])

    merged_eval = pd.DataFrame()
    if pred_path.exists():
        pr = pd.read_parquet(pred_path)
        pr["time_utc"] = pd.to_datetime(pr["time_utc"], utc=True)
        for sid in pick[:3]:
            plot_truth_pred_error_panel(pr, sid, default_lead, out / f"forecast_panel_{sid}_lead{default_lead}h.png")
            plot_multimethod_forecast_timeseries(
                pr, sid, default_lead, out / f"mistral_forecast_ts_{sid}_lead{default_lead}h.png"
            )
            logging.info("Forecast panel: %s lead=%sh", sid, default_lead)

        plot_forecast_scatter_methods(pr, default_lead, out / f"mistral_forecast_scatter_lead{default_lead}h.png")
        logging.info("Wrote mistral_forecast_scatter_lead%sh.png", default_lead)

        holdout_jsonl = ROOT / "data" / "processed" / "llm" / "test_holdout.jsonl"
        test_jsonl = holdout_jsonl if holdout_jsonl.exists() else ROOT / "data" / "processed" / "llm" / "test.jsonl"
        llm_cfg = mcfg.get("llm", {}) or {}
        pred_lookup = build_predictability_lookup(
            ROOT / llm_cfg.get("predictability_pred_path", "data/processed/predictions/test_predictions_lgbm.parquet"),
            lead_h=int(llm_cfg.get("predictability_lead_h", default_lead)),
        )
        eval_lora_p = mistral_dir / "eval_lora_results.jsonl"
        eval_holdout_p = mistral_dir / "eval_holdout_lora_results.jsonl"
        if holdout_jsonl.exists() and eval_holdout_p.exists():
            eval_lora_p = eval_holdout_p
            eval_base_p = mistral_dir / "eval_holdout_base_results.jsonl"
        else:
            eval_base_p = mistral_dir / "eval_base_results.jsonl"
        if test_jsonl.exists() and eval_lora_p.exists():
            merged_eval = build_eval_merged_frame(
                test_jsonl,
                eval_base_p,
                eval_lora_p,
                pr,
                lead_h=default_lead,
                windows_parquet=windows_p if windows_p.exists() else None,
                predictability_lookup=pred_lookup,
            )
            if not merged_eval.empty:
                plot_method_comparison_summary(pr, merged_eval, default_lead, out / "mistral_methods_summary.png")
                plot_mistral_predictability_vs_lgbm_error(
                    merged_eval, out / "mistral_predictability_vs_lgbm_error.png"
                )
                plot_eval_snapshot_grid(merged_eval, out / "mistral_eval_snapshots.png")
                for sid in pick[:3]:
                    plot_mistral_regime_timeline(
                        df, merged_eval, sid, default_lead, out / f"mistral_regime_timeline_{sid}_lead{default_lead}h.png"
                    )
                logging.info("Wrote Mistral + numeric comparison figures (n_eval=%s)", len(merged_eval))
    else:
        logging.warning("Missing prediction parquet — run scripts/05_train_numeric_baselines.py (and optional 05b).")

    curve_metrics_base = mistral_dir / "curve_metrics_base.json"
    curve_metrics_lora = mistral_dir / "curve_metrics_lora.json"
    if curve_metrics_base.exists():
        import json

        bm = json.loads(curve_metrics_base.read_text(encoding="utf-8"))
        lm = json.loads(curve_metrics_lora.read_text(encoding="utf-8")) if curve_metrics_lora.exists() else None
        plot_model_rmse_comparison_by_lead(
            bm, lm, out / "model_rmse_comparison_by_lead.png", horizon_h=default_lead
        )
        curve_test = ROOT / "data" / "processed" / "llm" / "curve" / "curve_test.jsonl"
        base_eval = mistral_dir / "curve_eval_base.jsonl"
        lora_eval = mistral_dir / "curve_eval_lora.jsonl"
        if curve_test.exists() and base_eval.exists():
            import json as _json

            test_rows = load_jsonl(curve_test)
            base_res = [_json.loads(l) for l in base_eval.read_text(encoding="utf-8").splitlines() if l.strip()]
            lora_res = (
                [_json.loads(l) for l in lora_eval.read_text(encoding="utf-8").splitlines() if l.strip()]
                if lora_eval.exists()
                else []
            )
            pred_path = pred_all if pred_all.exists() else pred_lgbm
            pred_df = pd.read_parquet(pred_path) if pred_path.exists() else None
            bm = json.loads(curve_metrics_base.read_text(encoding="utf-8"))
            plot_curve_method_summary_bars(
                bm, lm, out / "curve_method_rmse_summary.png", baseline_metrics=bm.get("baselines")
            )
            for sid in pick[:3]:
                plot_forecast_panel_mistral_lora(
                    test_rows,
                    base_res,
                    lora_res,
                    sid,
                    out / f"forecast_panel_mistral_lora_{sid}.png",
                    pred_df=pred_df,
                )
            logging.info("Wrote Mistral curve forecast panels")

    imgs = sorted(out.glob("*.png"))
    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>wave_llm figures</title>",
        "<style>body{font-family:system-ui;margin:24px;}img{max-width:100%;border:1px solid #ddd;margin:12px 0}</style>",
        "</head><body><h1>wave_llm — figures</h1>",
        "<p>Station map uses NDBC official station_table.txt positions + Cartopy coastlines. "
        "Forecast panels: black = NDBC truth; grey = persistence; blue = LightGBM; purple = Chronos-T5 (HF) if present. "
        "Mistral time-series overlays: <code>mistral_forecast_ts_*</code>, <code>mistral_regime_timeline_*</code>, "
        "<code>mistral_methods_summary.png</code>. See <a href='mistral_index.html'>mistral_index.html</a>.</p>",
    ]
    for p in imgs:
        rel = p.name
        lines.append(f"<h2>{rel}</h2><img src='{rel}' alt='{rel}'/>")
    lines.append("</body></html>")
    (out / "index.html").write_text("\n".join(lines), encoding="utf-8")
    mistral_imgs = sorted(p.name for p in imgs if p.name.startswith("mistral_"))
    if mistral_imgs:
        write_mistral_figures_index(out, mistral_imgs)
        logging.info("Wrote mistral_index.html (%s Mistral-related images)", len(mistral_imgs))
    logging.info("Wrote index.html (%s images)", len(imgs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
