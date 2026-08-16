#!/usr/bin/env python3
"""Evaluate Mistral Base vs LoRA on curve-forecast JSONL; write metrics + parquet."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

import pandas as pd  # noqa: E402

from wave_llm.evaluation.curve_numeric_align import (  # noqa: E402
    aggregate_baseline_metrics,
    enrich_curve_results_with_baselines,
)
from wave_llm.evaluation.mistral_ts_plots import (  # noqa: E402
    plot_curve_method_summary_bars,
    plot_forecast_panel_mistral_lora,
    plot_model_rmse_comparison_by_lead,
)
from wave_llm.models.mistral_curve import (  # noqa: E402
    curve_results_to_parquet,
    infer_curve_batch,
    metrics_from_curve_results,
)
from wave_llm.models.mistral_lora import load_jsonl  # noqa: E402
from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    mcfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    cc = mcfg.get("mistral_curve", {}) or {}
    max_eval = int(cc.get("max_eval_samples", 50))

    test_p = ROOT / "data" / "processed" / "llm" / "curve" / "curve_test.jsonl"
    if not test_p.exists():
        logging.error("Missing %s — run 06d_export_mistral_curve_jsonl.py", test_p)
        return 1

    rows_all = load_jsonl(test_p)
    # Diversify stations so forecast panels are not all from one buoy.
    by_sid: dict[str, list[dict]] = {}
    for r in rows_all:
        sid = str((r.get("input") or {}).get("station_id", ""))
        by_sid.setdefault(sid, []).append(r)
    rows: list[dict] = []
    sids = sorted(by_sid.keys())
    if sids and max_eval > 0:
        i = 0
        while len(rows) < max_eval and any(by_sid[s] for s in sids):
            sid = sids[i % len(sids)]
            if by_sid[sid]:
                rows.append(by_sid[sid].pop())
            i += 1
    else:
        rows = rows_all[:max_eval] if max_eval > 0 else rows_all
    out_dir = ensure_dir(ROOT / "data" / "processed" / "mistral")
    adapter_dir = Path(cc.get("output_dir", str(ROOT / "data" / "processed" / "mistral" / "curve_lora_run"))) / "adapter"

    pred_p = ROOT / "data" / "processed" / "predictions" / "test_predictions_all.parquet"
    if not pred_p.exists():
        pred_p = ROOT / "data" / "processed" / "predictions" / "test_predictions_lgbm.parquet"
    pred_df = pd.read_parquet(pred_p) if pred_p.exists() else None
    if pred_df is None:
        logging.warning("No test_predictions_*.parquet — run scripts/05_train_numeric_baselines.py (optional 05b)")

    logging.info("Curve eval BASE on %s samples", len(rows))
    base_res = infer_curve_batch(rows, mcfg, adapter_dir=None, max_samples=len(rows))
    base_res = enrich_curve_results_with_baselines(rows, base_res, pred_df)
    base_m = metrics_from_curve_results(base_res, "mistral_base_curve")
    base_m["baselines"] = aggregate_baseline_metrics(base_res)
    (out_dir / "curve_eval_base.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in base_res) + "\n",
        encoding="utf-8",
    )
    (out_dir / "curve_metrics_base.json").write_text(json.dumps(base_m, indent=2), encoding="utf-8")

    lora_res: list[dict] = []
    lora_m: dict | None = None
    if adapter_dir.exists():
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Curve eval LoRA %s", adapter_dir)
        lora_res = infer_curve_batch(rows, mcfg, adapter_dir=adapter_dir, max_samples=len(rows))
        lora_res = enrich_curve_results_with_baselines(rows, lora_res, pred_df)
        lora_m = metrics_from_curve_results(lora_res, "mistral_lora_curve")
        lora_m["baselines"] = aggregate_baseline_metrics(lora_res)
        (out_dir / "curve_eval_lora.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in lora_res) + "\n",
            encoding="utf-8",
        )
        (out_dir / "curve_metrics_lora.json").write_text(json.dumps(lora_m, indent=2), encoding="utf-8")
    else:
        logging.warning("No curve adapter at %s — base-only eval", adapter_dir)

    pred_dir = ensure_dir(ROOT / "data" / "processed" / "predictions")
    pq = curve_results_to_parquet(base_res, lora_res if lora_res else None)
    pq_path = pred_dir / "test_predictions_mistral_curve.parquet"
    pq.to_parquet(pq_path, index=False)
    logging.info("Wrote %s (%s rows)", pq_path, len(pq))

    compare = {
        "base": {"mean_rmse": base_m.get("mean_rmse"), "json_valid_rate": base_m.get("json_valid_rate"), "n": base_m.get("n_samples")},
        "lora": {
            "mean_rmse": lora_m.get("mean_rmse") if lora_m else None,
            "json_valid_rate": lora_m.get("json_valid_rate") if lora_m else None,
            "n": lora_m.get("n_samples") if lora_m else 0,
        },
    }
    (out_dir / "curve_compare_base_lora.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")
    logging.info("Curve compare: %s", compare)

    fig_dir = ensure_dir(ROOT / "data" / "processed" / "figures")
    plot_model_rmse_comparison_by_lead(
        base_m,
        lora_m,
        fig_dir / "model_rmse_comparison_by_lead.png",
        horizon_h=int(cc.get("horizon_hours", 24)),
    )
    test_rows = rows
    plot_curve_method_summary_bars(
        base_m,
        lora_m,
        fig_dir / "curve_method_rmse_summary.png",
        baseline_metrics=base_m.get("baselines"),
    )
    for sid in sorted({str(r.get("input", {}).get("station_id", "")) for r in test_rows})[:5]:
        if sid:
            plot_forecast_panel_mistral_lora(
                test_rows,
                base_res,
                lora_res if lora_res else [],
                sid,
                fig_dir / f"forecast_panel_mistral_lora_{sid}.png",
                pred_df=pred_df,
                max_panels=3,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
