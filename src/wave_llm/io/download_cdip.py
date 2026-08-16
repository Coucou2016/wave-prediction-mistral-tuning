from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

log = logging.getLogger(__name__)


def open_cdip_opendap(url: str) -> xr.Dataset:
    return xr.open_dataset(url, decode_times=True)


def download_cdip_subset_to_netcdf(url: str, dest: Path, max_bytes: int | None = None) -> Path:
    """
    Load via OPeNDAP and save locally. CDIP may block some IPs; caller handles errors.
    """
    ds = open_cdip_opendap(url)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(dest)
    finally:
        ds.close()
    return dest
