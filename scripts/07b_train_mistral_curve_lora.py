#!/usr/bin/env python3
"""Fine-tune Mistral LoRA on curve-forecast JSONL (hs_forecast_m sequences)."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wave_llm.models.mistral_curve import train_curve_lora  # noqa: E402
from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    mcfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    cc = mcfg.get("mistral_curve", {}) or {}

    curve_dir = ROOT / "data" / "processed" / "llm" / "curve"
    train_p = curve_dir / "curve_train.jsonl"
    val_p = curve_dir / "curve_val.jsonl"
    if not train_p.exists():
        logging.error("Missing %s — run scripts/06d_export_mistral_curve_jsonl.py first", train_p)
        return 1

    out_dir = ensure_dir(Path(cc.get("output_dir", str(ROOT / "data" / "processed" / "mistral" / "curve_lora_run"))))

    import torch

    if not torch.cuda.is_available():
        logging.warning("No CUDA — curve LoRA on CPU is very slow; reduce max_train_samples in yaml.")

    adapter = train_curve_lora(train_p, val_p, out_dir, mcfg)
    logging.info("Done. Curve LoRA adapter at %s", adapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
