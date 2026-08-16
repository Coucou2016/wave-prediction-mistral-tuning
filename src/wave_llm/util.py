from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests
import yaml

log = logging.getLogger(__name__)


def load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root_from_config(cfg: dict[str, Any], default: Path | None = None) -> Path:
    pr = cfg.get("project_root", ".")
    root = Path(pr).resolve()
    if default is not None and not root.exists():
        return default.resolve()
    return root


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_url(
    url: str,
    dest: Path,
    chunk: int = 1 << 20,
    timeout: int = 120,
    retries: int = 5,
    headers: dict[str, str] | None = None,
) -> Path:
    """Stream download with retries; raises last error if all fail."""
    ensure_dir(dest.parent)
    hdrs = {
        "User-Agent": "wave-llm-pipeline/0.1 (+https://github.com/)",
        "Accept": "*/*",
    }
    if headers:
        hdrs.update(headers)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=hdrs) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as out:
                    for c in r.iter_content(chunk_size=chunk):
                        if c:
                            out.write(c)
                tmp.replace(dest)
            log.info("Downloaded %s -> %s", url, dest)
            return dest
        except Exception as e:
            last_err = e
            log.warning("Attempt %s/%s failed for %s: %s", attempt, retries, url, e)
    assert last_err is not None
    raise last_err
