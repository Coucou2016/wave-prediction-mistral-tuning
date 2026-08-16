from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wave_llm.io.download_cmems import download_cmems_hourly  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    ok = download_cmems_hourly(ROOT)
    return 0 if ok else 0  # non-fatal skip


if __name__ == "__main__":
    raise SystemExit(main())
