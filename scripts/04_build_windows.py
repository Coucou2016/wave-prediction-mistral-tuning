from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from wave_llm.representations.jsonl_export import build_windows  # noqa: E402
from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    mcfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    hist_days = int(mcfg.get("history_days", 30))
    leads = mcfg.get("target_lead_hours", [24])
    df = pd.read_parquet(ROOT / "data" / "processed" / "panel_hourly.parquet")
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    all_rows: list[dict] = []
    for L in leads:
        w = build_windows(df, history_hours=hist_days * 24, lead_hours=int(L), stride_hours=24)
        for r in w:
            r["lead_hours"] = int(L)
        all_rows.extend(w)
    out_dir = ensure_dir(ROOT / "data" / "processed" / "windows")
    pq = out_dir / "windows.parquet"
    pd.DataFrame(all_rows).to_parquet(pq, index=False)
    logging.info("Wrote %s windows=%s", pq, len(all_rows))
    with (out_dir / "windows_meta.json").open("w", encoding="utf-8") as f:
        json.dump({"n": len(all_rows), "leads": leads, "history_days": hist_days}, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
