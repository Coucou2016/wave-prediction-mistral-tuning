from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    mcfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    lora = mcfg.get("lora", {}) or {}
    seed = int(lora.get("split_seed", 42))
    val_frac = float(lora.get("val_frac", 0.1))
    test_frac = float(lora.get("test_frac", 0.1))

    src = ROOT / "data" / "processed" / "llm" / "train_mistral.jsonl"
    if not src.exists():
        logging.error("Run scripts/06_export_llm_jsonl.py first")
        return 1

    rows = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_rows = rows[:n_test]
    val_rows = rows[n_test : n_test + n_val]
    train_rows = rows[n_test + n_val :]

    out = ensure_dir(ROOT / "data" / "processed" / "llm")
    for name, part in [("train.jsonl", train_rows), ("val.jsonl", val_rows), ("test.jsonl", test_rows)]:
        p = out / name
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in part) + "\n", encoding="utf-8")
        logging.info("Wrote %s (%s lines)", p, len(part))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
