from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wave_llm.evaluation.mistral_plots import (  # noqa: E402
    plot_mistral_predictability_accuracy,
    plot_mistral_regime_confusion,
)
from wave_llm.models.mistral_lora import load_jsonl, metrics_from_results, run_batch_infer  # noqa: E402
from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    mcfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    lora = mcfg.get("lora", {}) or {}
    max_eval = int(lora.get("max_eval_samples", 24))
    adapter_dir = Path(lora.get("output_dir", str(ROOT / "data" / "processed" / "mistral" / "lora_run"))) / "adapter"

    holdout_p = ROOT / "data" / "processed" / "llm" / "test_holdout.jsonl"
    test_p = holdout_p if holdout_p.exists() else ROOT / "data" / "processed" / "llm" / "test.jsonl"
    if not test_p.exists():
        test_p = ROOT / "data" / "processed" / "llm" / "val.jsonl"
    if test_p == holdout_p:
        logging.info("Using hold-out-aligned test set: %s", holdout_p)
    if not test_p.exists():
        logging.error("No test.jsonl / val.jsonl — run 06b_split_llm_jsonl.py")
        return 1

    rows = load_jsonl(test_p)[:max_eval]
    out_dir = ensure_dir(ROOT / "data" / "processed" / "mistral")
    tag = "holdout_" if test_p == holdout_p else ""

    logging.info("Eval BASE (pretrained, no adapter) on %s samples", len(rows))
    base_res = run_batch_infer(rows, mcfg, adapter_dir=None, max_samples=len(rows))
    base_m = metrics_from_results(base_res, "base_pretrained")
    base_m["model_id"] = lora.get("base_model_id", "mistralai/Mistral-7B-Instruct-v0.3")
    base_m["backend"] = "hf_lora"

    lora_res: list[dict] = []
    lora_m: dict | None = None
    if adapter_dir.exists():
        logging.info("Eval LoRA adapter %s", adapter_dir)
        lora_res = run_batch_infer(rows, mcfg, adapter_dir=adapter_dir, max_samples=len(rows))
        lora_m = metrics_from_results(lora_res, "lora_finetuned")
        lora_m["model_id"] = str(adapter_dir)
        lora_m["backend"] = "hf_lora"
    else:
        logging.warning("No adapter at %s — only base metrics", adapter_dir)

    (out_dir / f"eval_{tag}base_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in base_res) + "\n",
        encoding="utf-8",
    )
    (out_dir / "metrics_base.json").write_text(json.dumps(base_m, ensure_ascii=False, indent=2), encoding="utf-8")

    if lora_m:
        (out_dir / f"eval_{tag}lora_results.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in lora_res) + "\n",
            encoding="utf-8",
        )
        (out_dir / "metrics_lora.json").write_text(json.dumps(lora_m, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "metrics.json").write_text(json.dumps(lora_m, ensure_ascii=False, indent=2), encoding="utf-8")

    compare = {
        "base": {
            "regime_accuracy": base_m.get("regime_accuracy"),
            "predictability_accuracy": base_m.get("predictability_accuracy"),
            "n": base_m.get("n_samples"),
        },
        "lora": {
            "regime_accuracy": lora_m.get("regime_accuracy") if lora_m else None,
            "predictability_accuracy": lora_m.get("predictability_accuracy") if lora_m else None,
            "n": lora_m.get("n_samples") if lora_m else 0,
        },
    }
    (out_dir / "compare_base_lora.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")
    logging.info("Compare: %s", compare)

    fig_dir = ensure_dir(ROOT / "data" / "processed" / "figures")
    plot_mistral_regime_confusion(out_dir / "metrics_base.json", fig_dir / "mistral_base_regime_confusion.png")
    if lora_m:
        plot_mistral_regime_confusion(out_dir / "metrics_lora.json", fig_dir / "mistral_lora_regime_confusion.png")
        plot_mistral_predictability_accuracy(out_dir / "metrics_lora.json", fig_dir / "mistral_lora_predictability_accuracy.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
