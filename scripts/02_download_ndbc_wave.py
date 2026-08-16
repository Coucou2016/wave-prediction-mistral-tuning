from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from wave_llm.io.download_ndbc import station_years_to_parquet  # noqa: E402
from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    panel = yaml.safe_load((ROOT / "configs" / "station_panels.yaml").read_text(encoding="utf-8"))
    ndbc = panel["ndbc"]
    stations = ndbc["stations"]
    years = ndbc["years"]
    raw = ROOT / "data" / "raw"
    interim = ensure_dir(ROOT / "data" / "interim")
    pq = interim / "ndbc_stdmet_raw.parquet"
    station_years_to_parquet(stations, years, raw, pq)
    logging.info("Wrote %s", pq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
