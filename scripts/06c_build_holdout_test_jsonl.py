#!/usr/bin/env python3
"""Build test_holdout.jsonl from windows that overlap numeric hold-out predictions (same time axis as script 05)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from wave_llm.representations.jsonl_export import export_jsonl  # noqa: E402
from wave_llm.representations.predictability import build_predictability_lookup  # noqa: E402
from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    mcfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    llm = mcfg.get("llm", {}) or {}
    lead_h = int(llm.get("predictability_lead_h", 24))
    instr = llm.get("instruction", "")

    pred_p = ROOT / "data" / "processed" / "predictions" / "test_predictions_lgbm.parquet"
    if not pred_p.exists():
        pred_p = ROOT / "data" / "processed" / "predictions" / "test_predictions_all.parquet"
    win_p = ROOT / "data" / "processed" / "windows" / "windows.parquet"
    if not pred_p.exists() or not win_p.exists():
        logging.error("Need windows.parquet and test_predictions_*.parquet")
        return 1

    pr = pd.read_parquet(pred_p)
    pr["time_utc"] = pd.to_datetime(pr["time_utc"], utc=True)
    pr = pr.loc[pr["lead_h"] == lead_h, ["station_id", "time_utc"]].drop_duplicates()

    w = pd.read_parquet(win_p)
    w = w.loc[w["lead_hours"] == lead_h].copy()
    w["issue_time"] = pd.to_datetime(w["issue_time"], utc=True)
    w["station_id"] = w["station_id"].astype(str)
    pr["station_id"] = pr["station_id"].astype(str)

    keys = pr.rename(columns={"time_utc": "issue_time"})
    hold = w.merge(keys, on=["station_id", "issue_time"], how="inner")
    logging.info("Hold-out windows overlapping predictions: %s", len(hold))

    pred_lookup = build_predictability_lookup(
        ROOT / llm.get("predictability_pred_path", "data/processed/predictions/test_predictions_lgbm.parquet"),
        lead_h=lead_h,
    )
    hold["issue_time"] = hold["issue_time"].apply(lambda t: pd.Timestamp(t).isoformat())
    samples = hold.to_dict(orient="records")
    out = ensure_dir(ROOT / "data" / "processed" / "llm") / "test_holdout.jsonl"
    export_jsonl(samples, out, instr, pred_lookup, llm_lead_h=lead_h)
    logging.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
