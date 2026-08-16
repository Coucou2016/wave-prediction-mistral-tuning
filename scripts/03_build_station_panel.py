from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from wave_llm.io.standardize import standardize_ndbc_frame, to_canonical_table  # noqa: E402
from wave_llm.preprocessing.qc import basic_qc  # noqa: E402
from wave_llm.preprocessing.resample import resample_station_hourly  # noqa: E402
from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "model_config.yaml").read_text(encoding="utf-8"))
    rule = cfg.get("resample_rule", "1h")
    interim = ROOT / "data" / "interim"
    raw = pd.read_parquet(interim / "ndbc_stdmet_raw.parquet")
    std = standardize_ndbc_frame(raw)
    can = to_canonical_table(std)
    qc = basic_qc(can)
    rs = resample_station_hourly(qc, rule=rule)
    out = ensure_dir(ROOT / "data" / "processed") / "panel_hourly.parquet"
    rs.to_parquet(out, index=False)
    logging.info("Wrote %s rows=%s", out, len(rs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
