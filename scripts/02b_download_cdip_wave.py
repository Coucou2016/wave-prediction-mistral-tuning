from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from wave_llm.io.download_cdip import download_cdip_subset_to_netcdf  # noqa: E402
from wave_llm.util import ensure_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    panel = yaml.safe_load((ROOT / "configs" / "station_panels.yaml").read_text(encoding="utf-8"))
    deps = panel.get("cdip", {}).get("deployments", [])
    out_dir = ensure_dir(ROOT / "data" / "raw" / "cdip")
    for d in deps:
        url = d.get("opendap_url")
        if not url:
            continue
        name = d.get("id", "cdip").replace("/", "_")
        dest = out_dir / f"{name}.nc"
        try:
            download_cdip_subset_to_netcdf(url, dest)
            logging.info("Saved %s", dest)
        except Exception as e:
            logging.error("CDIP download failed (%s): %s", url, e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
