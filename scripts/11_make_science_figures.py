"""Regenerate SciencePlots paper/report figures from local metrics JSON (+ optional eval JSONL).

Outputs:
  - data/processed/figures_science/  (report gallery)
  - paper/figures/                   (paper-ready copies)
  - paper/metrics/                   (sanitized numeric snapshots for the public repo)
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wave_llm.evaluation.mistral_plots import (  # noqa: E402
    plot_mistral_predictability_accuracy,
    plot_mistral_regime_confusion,
)
from wave_llm.evaluation.mistral_ts_plots import (  # noqa: E402
    plot_curve_forecast_examples,
    plot_curve_method_summary_bars,
    plot_forecast_panel_mistral_lora,
    plot_model_rmse_comparison_by_lead,
)
from wave_llm.evaluation.plots import plot_baseline_metrics  # noqa: E402
from wave_llm.evaluation.science_plots_style import (  # noqa: E402
    apply_science_style,
    verify_times_new_roman,
)
from wave_llm.models.mistral_lora import load_jsonl  # noqa: E402
from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("science_figures")


def _copy_metrics_snapshot(dst: Path) -> None:
    ensure_dir(dst)
    mapping = {
        "curve_metrics_base.json": ROOT / "data/processed/mistral/curve_metrics_base.json",
        "curve_metrics_lora.json": ROOT / "data/processed/mistral/curve_metrics_lora.json",
        "curve_compare_base_lora.json": ROOT / "data/processed/mistral/curve_compare_base_lora.json",
        "metrics_base.json": ROOT / "data/processed/mistral/metrics_base.json",
        "metrics_lora.json": ROOT / "data/processed/mistral/metrics_lora.json",
        "compare_base_lora.json": ROOT / "data/processed/mistral/compare_base_lora.json",
        "numeric_baselines.json": ROOT / "data/processed/metrics/numeric_baselines.json",
        "curve_lora_meta_v2.json": ROOT / "data/processed/mistral/curve_lora_run_v2/curve_lora_meta.json",
    }
    for name, src in mapping.items():
        if src.is_file():
            shutil.copy2(src, dst / name)
            log.info("Copied metrics snapshot %s", name)


def _plot_classification_summary(out: Path) -> None:
    apply_science_style(font_size=10)
    base = json.loads((ROOT / "data/processed/mistral/metrics_base.json").read_text(encoding="utf-8"))
    lora = json.loads((ROOT / "data/processed/mistral/metrics_lora.json").read_text(encoding="utf-8"))
    labels = ["Regime acc.", "Predictability acc."]
    b = [float(base["regime_accuracy"]), float(base["predictability_accuracy"])]
    l = [float(lora["regime_accuracy"]), float(lora["predictability_accuracy"])]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(x - w / 2, b, width=w, label="Mistral Base", color="#aec7e8", edgecolor="k")
    ax.bar(x + w / 2, l, width=w, label="Mistral LoRA", color="#ff7f0e", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Classification LoRA (n={base.get('n_samples', '?')})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _plot_lead_lines(base: dict, lora: dict, out: Path) -> None:
    apply_science_style(font_size=10)
    b_steps = {int(k): float(v) for k, v in (base.get("rmse_by_forecast_step_h") or {}).items()}
    l_steps = {int(k): float(v) for k, v in (lora.get("rmse_by_forecast_step_h") or {}).items()}
    steps = sorted(set(b_steps) | set(l_steps))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(steps, [b_steps[s] for s in steps], "-o", ms=3.5, label="Mistral Base", color="#4c78a8")
    ax.plot(steps, [l_steps[s] for s in steps], "-s", ms=3.5, label="Mistral LoRA", color="#f58518")
    bas = lora.get("baselines") or base.get("baselines") or {}
    if bas.get("mean_rmse_persist") is not None:
        ax.axhline(float(bas["mean_rmse_persist"]), color="#7f7f7f", ls="--", lw=1.0, label="Persist mean")
    if bas.get("mean_rmse_lgbm_at_numeric_leads") is not None:
        ax.axhline(float(bas["mean_rmse_lgbm_at_numeric_leads"]), color="#1f77b4", ls=":", lw=1.0, label="LGBM† mean")
    ax.set_xlabel("Forecast lead (h)")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("JSON curve RMSE by lead hour")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    font_info = verify_times_new_roman()
    log.info("Font check: %s", font_info)

    out_sci = ensure_dir(ROOT / "data" / "processed" / "figures_science")
    out_paper = ensure_dir(ROOT / "paper" / "figures")
    metrics_dir = ensure_dir(ROOT / "paper" / "metrics")
    _copy_metrics_snapshot(metrics_dir)

    base_p = ROOT / "data/processed/mistral/curve_metrics_base.json"
    lora_p = ROOT / "data/processed/mistral/curve_metrics_lora.json"
    if not base_p.is_file() or not lora_p.is_file():
        log.error("Missing curve metrics JSON")
        return 1
    base = json.loads(base_p.read_text(encoding="utf-8"))
    lora = json.loads(lora_p.read_text(encoding="utf-8"))

    numeric = ROOT / "data/processed/metrics/numeric_baselines.json"
    if numeric.is_file():
        plot_baseline_metrics(numeric, out_sci / "baseline_rmse_skill.png")
        log.info("Wrote baseline_rmse_skill.png")

    plot_model_rmse_comparison_by_lead(base, lora, out_sci / "model_rmse_comparison_by_lead.png")
    plot_curve_method_summary_bars(base, lora, out_sci / "curve_method_rmse_summary.png")
    _plot_lead_lines(base, lora, out_sci / "curve_rmse_by_lead_lines.png")
    log.info("Wrote curve summary SciencePlots")

    # Classification metrics / confusion
    for tag, mp in [
        ("base", ROOT / "data/processed/mistral/metrics_base.json"),
        ("lora", ROOT / "data/processed/mistral/metrics_lora.json"),
    ]:
        if mp.is_file():
            plot_mistral_regime_confusion(mp, out_sci / f"mistral_{tag}_regime_confusion.png")
            plot_mistral_predictability_accuracy(mp, out_sci / f"mistral_{tag}_predictability_accuracy.png")
    _plot_classification_summary(out_sci / "classification_base_vs_lora.png")

    # Optional curve example panels from eval JSONL
    test_jsonl = ROOT / "data/processed/llm/curve/curve_test.jsonl"
    if not test_jsonl.is_file():
        for cand in [
            ROOT / "data/processed/llm/test_mistral_curve.jsonl",
            ROOT / "data/processed/llm/curve_test.jsonl",
            ROOT / "data/processed/llm/test_curve.jsonl",
        ]:
            if cand.is_file():
                test_jsonl = cand
                break
    base_eval = ROOT / "data/processed/mistral/curve_eval_base.jsonl"
    lora_eval = ROOT / "data/processed/mistral/curve_eval_lora.jsonl"
    if test_jsonl.is_file() and base_eval.is_file() and lora_eval.is_file():
        rows = load_jsonl(test_jsonl)
        base_res = load_jsonl(base_eval)
        lora_res = load_jsonl(lora_eval)
        stations = []
        for rec in rows:
            sid = str((rec.get("input") or {}).get("station_id", ""))
            if sid and sid not in stations:
                stations.append(sid)
        for sid in stations[:4]:
            plot_forecast_panel_mistral_lora(
                rows,
                base_res,
                lora_res,
                sid,
                out_sci / f"forecast_panel_mistral_lora_{sid}.png",
            )
            plot_curve_forecast_examples(
                rows,
                base_res,
                lora_res,
                sid,
                out_sci / f"curve_examples_{sid}.png",
                max_panels=3,
            )
            log.info("Wrote station panels for %s", sid)
    else:
        log.warning(
            "Skip forecast panels (missing test/eval JSONL). Have test=%s base=%s lora=%s",
            test_jsonl.is_file(),
            base_eval.is_file(),
            lora_eval.is_file(),
        )

    # Mirror into paper/figures
    for png in sorted(out_sci.glob("*.png")):
        shutil.copy2(png, out_paper / png.name)
    log.info("Mirrored %d PNGs -> paper/figures", len(list(out_paper.glob('*.png'))))

    # Small HTML index for local browsing
    index = out_sci / "index.html"
    cards = "\n".join(
        f'<div style="margin:12px"><h3>{p.name}</h3><img src="{p.name}" style="max-width:900px"/></div>'
        for p in sorted(out_sci.glob("*.png"))
    )
    index.write_text(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>SciencePlots figures</title></head>"
        f"<body><h1>SciencePlots figures</h1><pre>{json.dumps(font_info, indent=2)}</pre>{cards}</body></html>",
        encoding="utf-8",
    )
    (out_paper / "FONT_CHECK.json").write_text(json.dumps(font_info, indent=2), encoding="utf-8")
    log.info("Done. Science dir=%s paper dir=%s", out_sci, out_paper)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
