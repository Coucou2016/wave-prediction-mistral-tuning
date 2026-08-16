from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")

"""
Fine-tune official Mistral (HF pretrained) with LoRA on buoy JSONL.

Local weights (no token):
  Set in configs/model_config.yaml:
    lora.local_model_path: "D:/path/to/Mistral-7B-Instruct-v0.3"
  Or env: MISTRAL_LOCAL_PATH

Gated model on Hub:
  Set HF_TOKEN or huggingface-cli login
"""

import logging
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wave_llm.models.mistral_lora import train_lora  # noqa: E402
from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    mcfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    lora = mcfg.get("lora", {}) or {}

    train_p = ROOT / "data" / "processed" / "llm" / "train.jsonl"
    val_p = ROOT / "data" / "processed" / "llm" / "val.jsonl"
    if not train_p.exists():
        logging.error("Missing %s — run 06_export_llm_jsonl.py then 06b_split_llm_jsonl.py", train_p)
        return 1

    out_dir = ensure_dir(Path(lora.get("output_dir", str(ROOT / "data" / "processed" / "mistral" / "lora_run"))))

    import torch

    if not torch.cuda.is_available():
        logging.warning(
            "No CUDA — LoRA on CPU is very slow and may need lots of RAM. "
            "Reduce lora.max_train_samples (e.g. 64) or run on a GPU machine with the same local_model_path."
        )

    train_lora(train_p, val_p, out_dir, mcfg)
    logging.info("Done. Adapter at %s/adapter", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
