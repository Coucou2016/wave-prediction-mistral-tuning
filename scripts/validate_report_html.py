#!/usr/bin/env python3
"""Validate self-contained paper.html / report.html under docs/output/."""
from __future__ import annotations

import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "output"


def check_html(path: Path, expect_min_imgs: int = 1) -> list[str]:
    errs: list[str] = []
    if not path.exists():
        return [f"MISSING {path}"]
    text = path.read_text(encoding="utf-8")
    size = path.stat().st_size
    if size < 10_000:
        errs.append(f"{path.name}: too small ({size} bytes)")
    if "<!DOCTYPE html>" not in text and "<!doctype html>" not in text.lower():
        errs.append(f"{path.name}: missing DOCTYPE")
    # external css/js
    for m in re.finditer(r'<link\b[^>]*href=["\']([^"\']+)["\']', text, re.I):
        href = m.group(1)
        if not href.startswith("data:"):
            errs.append(f"{path.name}: external link href={href}")
    for m in re.finditer(r'<script\b[^>]*src=["\']([^"\']+)["\']', text, re.I):
        errs.append(f"{path.name}: external script src={m.group(1)}")
    # images must be data:
    imgs = re.findall(r'<img\b[^>]*src=["\']([^"\']+)["\']', text, re.I)
    if len(imgs) < expect_min_imgs:
        errs.append(f"{path.name}: expected >= {expect_min_imgs} imgs, found {len(imgs)}")
    for src in imgs:
        if not src.startswith("data:image/"):
            errs.append(f"{path.name}: non-data img src={src[:80]}")
    # anchors
    ids = set(re.findall(r'\bid=["\']([^"\']+)["\']', text))
    hrefs = re.findall(r'href=["\']#([^"\']+)["\']', text)
    missing_anchors = sorted({h for h in hrefs if h not in ids})
    if missing_anchors:
        errs.append(f"{path.name}: broken TOC anchors: {missing_anchors[:12]}")
    print(f"OK-STATS {path.name}: size={size}, imgs={len(imgs)}, ids={len(ids)}, toc_links={len(hrefs)}")
    return errs


def main() -> int:
    errs: list[str] = []
    errs += check_html(OUT / "report.html", expect_min_imgs=10)
    errs += check_html(OUT / "paper.html", expect_min_imgs=5)
    for pdf in ("report.pdf", "paper.pdf"):
        p = OUT / pdf
        if not p.exists():
            errs.append(f"MISSING {pdf}")
        else:
            print(f"OK-STATS {pdf}: size={p.stat().st_size}")
    md = OUT / "research_report.md"
    if not md.exists() or md.stat().st_size < 500:
        errs.append("research_report.md missing or too small")
    else:
        print(f"OK-STATS research_report.md: size={md.stat().st_size}")
    if errs:
        print("FAIL")
        for e in errs:
            print(" -", e)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
