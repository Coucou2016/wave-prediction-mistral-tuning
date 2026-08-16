from __future__ import annotations

"""
Merge Amazon Chronos-T5 (Hugging Face pretrained) point forecasts into the hold-out table.

Design:
- One Chronos forward pass per (station_id, issue_time) with prediction_length = max(leads),
  then slice indices (lead-1) for each horizon. This avoids 5× redundant computation.

Notes:
- Continuous Hs from Mistral/Ministral LoRA is not wired here (those checkpoints target JSONL
  classification/explanation in this repo). Chronos is a pretrained sequence model on real context.
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    import torch
    from chronos import ChronosPipeline

    mcfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    ccfg = mcfg.get("chronos", {}) or {}
    model_id = str(ccfg.get("model_id", "amazon/chronos-t5-tiny"))
    ctx_len = int(ccfg.get("context_length", 256))
    num_samples = int(ccfg.get("num_samples", 12))
    max_unique_issues = int(ccfg.get("max_unique_issues_per_station", 18))

    pred_path = ROOT / "data" / "processed" / "predictions" / "test_predictions_lgbm.parquet"
    panel_path = ROOT / "data" / "processed" / "panel_hourly.parquet"
    if not pred_path.exists():
        logging.error("Missing %s — run scripts/05_train_numeric_baselines.py first", pred_path)
        return 1
    pred = pd.read_parquet(pred_path)
    panel = pd.read_parquet(panel_path)
    panel["time_utc"] = pd.to_datetime(panel["time_utc"], utc=True)
    pred["time_utc"] = pd.to_datetime(pred["time_utc"], utc=True)

    leads = sorted({int(x) for x in pred["lead_h"].unique().tolist()})
    max_lead = int(max(leads))
    pred_len = min(max_lead, 64)

    logging.info("Loading Chronos pipeline: %s", model_id)
    pipe = ChronosPipeline.from_pretrained(model_id, device_map="cpu")

    out_dir = ensure_dir(ROOT / "data" / "processed" / "predictions")
    ch_rows: list[dict] = []

    for sid, gpanel in panel.groupby("station_id", sort=False):
        gpanel = gpanel.sort_values("time_utc")
        issues = (
            pred.loc[(pred["station_id"].astype(str) == str(sid)) & (pred["lead_h"] == 24), ["time_utc"]]
            .drop_duplicates()
            .sort_values("time_utc")
        )
        if issues.empty:
            # fallback: any lead
            issues = pred.loc[pred["station_id"].astype(str) == str(sid), ["time_utc"]].drop_duplicates().sort_values("time_utc")
        if issues.empty:
            continue
        if len(issues) > max_unique_issues:
            issues = issues.iloc[:: max(1, len(issues) // max_unique_issues)]

        for t_issue in issues["time_utc"].tolist():
            t_issue = pd.Timestamp(t_issue)
            past = gpanel.loc[gpanel["time_utc"] <= t_issue, "Hs_m"].tail(ctx_len).to_numpy(dtype=np.float32)
            if past.size < 32 or np.any(~np.isfinite(past)):
                preds = {L: float("nan") for L in leads}
            else:
                tensor = torch.from_numpy(past)[None, :]
                fc = pipe.predict(
                    tensor,
                    prediction_length=pred_len,
                    num_samples=num_samples,
                    limit_prediction_length=True,
                )
                # fc: [1, samples, pred_len]
                med = fc[0].median(dim=0).values.detach().cpu().numpy().astype(float)
                preds = {}
                for L in leads:
                    if L <= pred_len:
                        preds[int(L)] = float(med[L - 1])
                    else:
                        preds[int(L)] = float("nan")

            for L in leads:
                ch_rows.append(
                    {
                        "time_utc": t_issue,
                        "station_id": str(sid),
                        "lead_h": int(L),
                        "y_pred_chronos": preds.get(int(L), float("nan")),
                    }
                )

    ch = pd.DataFrame(ch_rows)
    if ch.empty:
        logging.error("No Chronos rows produced")
        return 1
    ch["time_utc"] = pd.to_datetime(ch["time_utc"], utc=True)
    merged = pred.merge(ch, on=["time_utc", "station_id", "lead_h"], how="left")
    out = out_dir / "test_predictions_all.parquet"
    merged.to_parquet(out, index=False)
    logging.info("Wrote %s rows=%s", out, len(merged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
