from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from wave_llm.util import download_url, ensure_dir

log = logging.getLogger(__name__)

STATION_TABLE_URL = "https://www.ndbc.noaa.gov/data/stations/station_table.txt"

# Decimal degrees inside LOCATION field, e.g. "36.787 N 122.408 W"
_LOC_RE = re.compile(
    r"(?P<lat>[0-9]+\.[0-9]+)\s*(?P<lat_h>[NS])\s+(?P<lon>[0-9]+\.[0-9]+)\s*(?P<lon_h>[EW])",
    re.IGNORECASE,
)


def parse_lat_lon(location_field: str) -> tuple[float, float] | None:
    if not location_field:
        return None
    m = _LOC_RE.search(location_field)
    if not m:
        return None
    lat = float(m.group("lat"))
    lon = float(m.group("lon"))
    if m.group("lat_h").upper() == "S":
        lat = -lat
    if m.group("lon_h").upper() == "W":
        lon = -lon
    return lat, lon


def download_ndbc_station_table(dest: Path) -> Path:
    ensure_dir(dest.parent)
    download_url(STATION_TABLE_URL, dest, retries=6, timeout=120)
    return dest


def load_station_table(path: Path) -> pd.DataFrame:
    """Parse NDBC station_table.txt (pipe-delimited, '#' comment/header lines)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") and "STATION_ID" in line:
            header = [h.strip() for h in line.lstrip("#").strip().split("|")]
            continue
        if line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if header is None:
            continue
        # pad missing trailing columns
        if len(parts) < len(header):
            parts.extend([""] * (len(header) - len(parts)))
        row = {header[i]: parts[i] for i in range(min(len(header), len(parts)))}
        rows.append(row)
    if not rows:
        raise ValueError(f"No rows parsed from {path}")
    return pd.DataFrame(rows)


def station_meta_for_ids(station_ids: list[str], cache_path: Path) -> pd.DataFrame:
    """
    Return real lat/lon/name from NDBC station_table for given STATION_IDs (case-insensitive).
    """
    download_ndbc_station_table(cache_path)
    tbl = load_station_table(cache_path)
    if "STATION_ID" not in tbl.columns or "LOCATION" not in tbl.columns:
        raise KeyError(f"Unexpected station table columns: {tbl.columns.tolist()}")
    want = {s.upper() for s in station_ids}
    sub = tbl[tbl["STATION_ID"].str.upper().isin(want)].copy()
    coords = []
    for _, r in sub.iterrows():
        loc = r.get("LOCATION", "")
        ll = parse_lat_lon(str(loc))
        coords.append(ll)
    sub["lat"] = [c[0] if c else float("nan") for c in coords]
    sub["lon"] = [c[1] if c else float("nan") for c in coords]
    sub["station_id"] = sub["STATION_ID"].str.upper()
    name_col = "NAME" if "NAME" in sub.columns else None
    out = pd.DataFrame(
        {
            "station_id": sub["station_id"],
            "lat": sub["lat"],
            "lon": sub["lon"],
            "name": sub[name_col] if name_col else "",
            "location_raw": sub["LOCATION"].astype(str),
        }
    )
    missing = want - set(out["station_id"].tolist())
    if missing:
        log.warning("Stations not found in NDBC station_table: %s", sorted(missing))
    return out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
