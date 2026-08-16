#!/usr/bin/env python3
"""Export Mistral curve-forecast JSONL (real Hs sequences, strict issue-time)."""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from wave_llm.models.mistral_lora import load_jsonl  # noqa: E402
from wave_llm.representations.curve_jsonl_export import export_train_test_curve_jsonl  # noqa: E402
from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Export curve_train/test JSONL from panel_hourly.parquet")
    p.add_argument("--horizon", type=int, default=None, help="Forecast horizon hours (default: yaml mistral_curve.horizon_hours)")
    p.add_argument("--history-hours", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    args = p.parse_args()

    mcfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    cc = mcfg.get("mistral_curve", {}) or {}
    hz = int(args.horizon or cc.get("horizon_hours", 24))
    hh = int(args.history_hours or cc.get("history_hours", 168))
    dt = int(cc.get("dt_hours", 1))
    stride = int(args.stride or cc.get("stride_hours", 24))

    panel_p = ROOT / "data" / "processed" / "panel_hourly.parquet"
    if not panel_p.exists():
        logging.error("Missing %s — run pipeline 01-03 first", panel_p)
        return 1

    df = pd.read_parquet(panel_p)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    out_dir = ensure_dir(ROOT / "data" / "processed" / "llm" / "curve")

    n_train, n_test = export_train_test_curve_jsonl(
        df,
        out_dir,
        history_hours=hh,
        horizon_hours=hz,
        dt_hours=dt,
        stride_hours=stride,
        train_frac=float(cc.get("train_frac", 0.8)),
    )
    if n_train == 0 or n_test == 0:
        logging.error("No curve samples exported (train=%s test=%s)", n_train, n_test)
        return 1

    train_rows = load_jsonl(out_dir / "curve_train.jsonl")
    seed = int(cc.get("split_seed", 42))
    val_frac = float(cc.get("val_frac", 0.1))
    random.Random(seed).shuffle(train_rows)
    n_val = max(1, int(len(train_rows) * val_frac))
    val_rows = train_rows[:n_val]
    train_rows = train_rows[n_val:]
    _write_jsonl(train_rows, out_dir / "curve_train.jsonl")
    _write_jsonl(val_rows, out_dir / "curve_val.jsonl")
    logging.info(
        "Curve JSONL horizon=%sh history=%sh — train=%s val=%s test=%s → %s",
        hz,
        hh,
        len(train_rows),
        len(val_rows),
        n_test,
        out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
