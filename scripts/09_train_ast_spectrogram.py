from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    if os.environ.get("WAVE_LLM_RUN_AST") != "1":
        logging.info("Set WAVE_LLM_RUN_AST=1 to train AST on exported spectrograms. Skipping.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
