from __future__ import annotations

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
    llm_cfg = mcfg.get("llm", {}) or {}
    instr = llm_cfg.get("instruction", "")
    lead_h = int(llm_cfg.get("predictability_lead_h", 24))
    w = pd.read_parquet(ROOT / "data" / "processed" / "windows" / "windows.parquet")
    w = w.loc[w["lead_hours"] == lead_h].copy()
    samples = w.to_dict(orient="records")
    mx = int(llm_cfg.get("max_windows_per_station", 10_000))
    if mx and mx < len(samples):
        # stratified cap per station
        parts = []
        for sid, grp in w.groupby("station_id"):
            g = grp.head(mx)
            parts.append(g)
        samples = pd.concat(parts, ignore_index=True).to_dict(orient="records")
    pred_rel = llm_cfg.get(
        "predictability_pred_path",
        "data/processed/predictions/test_predictions_lgbm.parquet",
    )
    pred_path = ROOT / pred_rel
    pred_lookup = build_predictability_lookup(pred_path, lead_h=lead_h)

    out_dir = ensure_dir(ROOT / "data" / "processed" / "llm")
    export_jsonl(
        samples,
        out_dir / "train_mistral.jsonl",
        instr,
        pred_lookup,
        llm_lead_h=lead_h,
    )
    n_out = sum(1 for _ in (out_dir / "train_mistral.jsonl").open(encoding="utf-8"))
    logging.info("Exported %s JSONL lines (lead=%sh, capped from %s windows)", n_out, lead_h, len(samples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
