from __future__ import annotations

import gzip
import logging
from io import StringIO
from pathlib import Path

import pandas as pd

from wave_llm.util import download_url, ensure_dir

log = logging.getLogger(__name__)

NDBC_HIST_BASE = "https://www.ndbc.noaa.gov/data/historical/stdmet"


def ndbc_stdmet_url(station: str, year: int) -> str:
    st = station.lower().strip()
    return f"{NDBC_HIST_BASE}/{st}h{year}.txt.gz"


def parse_ndbc_stdmet_text(text: str) -> pd.DataFrame:
    """Parse NDBC historical stdmet: first # line has column names; next # line units; then data."""
    raw_lines = text.splitlines()
    lines = [ln.rstrip() for ln in raw_lines]
    header_i = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("#") and "YY" in s:
            header_i = i
            break
    if header_i is None:
        raise ValueError("Could not find NDBC #YY header line")
    names = lines[header_i].lstrip("#").strip().split()
    j = header_i + 1
    while j < len(lines) and lines[j].strip().startswith("#"):
        j += 1
    body = "\n".join(lines[j:])
    df = pd.read_csv(
        StringIO(body),
        sep=r"\s+",
        names=names,
        na_values=["MM", "99.00", "999.0", "9999.0", "999", "99.0", "NaN"],
        low_memory=False,
    )
    return df


def read_ndbc_stdmet_gz(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        return parse_ndbc_stdmet_text(f.read())


def download_ndbc_year(station: str, year: int, out_dir: Path) -> Path | None:
    url = ndbc_stdmet_url(station, year)
    dest = ensure_dir(out_dir) / f"{station.lower()}h{year}.txt.gz"
    if dest.exists() and dest.stat().st_size > 1000:
        log.info("Reuse existing %s", dest)
        return dest
    try:
        download_url(url, dest, retries=8, timeout=240)
        return dest
    except Exception as e:
        log.error("Failed NDBC %s %s: %s", station, year, e)
        return None


def station_years_to_parquet(stations: list[str], years: list[int], raw_dir: Path, parquet_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for st in stations:
        for y in years:
            gz = download_ndbc_year(st, y, raw_dir / "ndbc")
            if gz is None:
                continue
            df = read_ndbc_stdmet_gz(gz)
            df["station_id"] = st.upper()
            df["source"] = "NDBC"
            df["file_year"] = y
            frames.append(df)
    if not frames:
        raise RuntimeError("No NDBC data downloaded — check stations, years, and connectivity.")
    out = pd.concat(frames, ignore_index=True)
    ensure_dir(parquet_path.parent)
    out.to_parquet(parquet_path, index=False)
    return out
