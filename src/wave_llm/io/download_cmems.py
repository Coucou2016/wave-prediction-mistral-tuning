from __future__ import annotations

import logging
import os
from pathlib import Path

from wave_llm.util import ensure_dir, load_yaml

log = logging.getLogger(__name__)


def download_cmems_hourly(
    project_root: Path,
    dataset_id: str | None = None,
    output_directory: str | None = None,
) -> bool:
    """
    Download CMEMS insitu wave hourly product if credentials are available.
    Uses COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD.
    """
    user = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
    pwd = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
    if not user or not pwd:
        log.warning(
            "CMEMS credentials not set (COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD). Skipping CMEMS download."
        )
        return False
    try:
        import copernicusmarine as cm  # type: ignore
    except ImportError:
        log.error("copernicusmarine not installed; pip install copernicusmarine")
        return False
    cfg = load_yaml(project_root / "configs" / "data_sources.yaml")
    ds = dataset_id or cfg.get("cmems", {}).get("dataset_id", "cmems_obs-ins_glo_wav_my_na_PT1H")
    out = output_directory or str(project_root / "data" / "raw" / cfg.get("cmems", {}).get("output_subdir", "cmems"))
    ensure_dir(Path(out))
    log.info("CMEMS download dataset_id=%s -> %s", ds, out)
    get_fn = getattr(cm, "get", None)
    if get_fn is not None:
        get_fn(dataset_id=ds, output_directory=out)
    else:
        subset_fn = getattr(cm, "subset", None)
        if subset_fn is None:
            log.error("copernicusmarine has no get/subset entrypoint")
            return False
        subset_fn(dataset_id=ds, output_filename=str(Path(out) / f"{ds}.nc"))
    return True
