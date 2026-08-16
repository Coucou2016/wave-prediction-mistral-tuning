#!/usr/bin/env python3
"""Screenshot key sections of report.html for visual QA."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
html = (ROOT / "docs" / "output" / "report.html").resolve().as_uri()
shot_dir = ROOT / "docs" / "output" / "_shots"
shot_dir.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1200, "height": 900})
    page.goto(html, wait_until="networkidle", timeout=180000)
    page.screenshot(path=str(shot_dir / "cover.png"), full_page=False)
    for name, sel in [
        ("toc.png", "#toc"),
        ("fig9.png", "#fig-r9"),
        ("table_curve.png", "#tab-curve"),
        ("fig10.png", "#fig-r10"),
        ("fig13.png", "#fig-r13"),
    ]:
        page.evaluate(
            "(sel) => { const el = document.querySelector(sel); if (el) window.scrollTo(0, el.offsetTop - 40); }",
            sel,
        )
        page.wait_for_timeout(300)
        page.screenshot(path=str(shot_dir / name), full_page=False)
    browser.close()

print("wrote", sorted(p.name for p in shot_dir.glob("*.png")))
