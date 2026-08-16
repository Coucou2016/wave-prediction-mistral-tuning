"""SciencePlots + Times New Roman styling for report/paper figures."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

# Prefer SciencePlots styles; fall back gracefully if package/fonts missing.
_STYLE_CANDIDATES = (
    ["science", "no-latex"],
    ["science"],
    [],
)


def _available_styles(requested: list[str]) -> list[str]:
    have = set(plt.style.available)
    ok = [s for s in requested if s in have]
    if requested and not ok:
        log.warning("SciencePlots styles %s not in matplotlib; available sample: %s", requested, sorted(have)[:12])
    return ok


def apply_science_style(*, font_size: int = 10, use_times: bool = True) -> None:
    """Apply SciencePlots (if installed) and Times New Roman when available."""
    try:
        import scienceplots  # noqa: F401
    except ImportError:
        log.warning("SciencePlots not importable; using plain matplotlib rcParams")

    applied = False
    for cand in _STYLE_CANDIDATES:
        styles = _available_styles(cand)
        if cand and not styles:
            continue
        try:
            if styles:
                plt.style.use(styles)
            applied = True
            break
        except OSError as exc:
            log.warning("plt.style.use(%s) failed: %s", styles, exc)

    if not applied:
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.grid": True,
                "grid.alpha": 0.25,
            }
        )

    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    if use_times:
        # Windows usually ships Times New Roman; keep DejaVu as silent fallback.
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
            "Liberation Serif",
            "serif",
        ]
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams["axes.unicode_minus"] = False


@contextmanager
def science_style(*, font_size: int = 10, use_times: bool = True) -> Iterator[None]:
    """Temporary SciencePlots context that restores prior rcParams on exit."""
    with plt.rc_context():
        apply_science_style(font_size=font_size, use_times=use_times)
        yield


def verify_times_new_roman() -> dict[str, object]:
    """Return font availability info for acceptance checks."""
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    hit = "Times New Roman" in names
    return {
        "times_new_roman_available": hit,
        "serif_rc": list(plt.rcParams.get("font.serif", [])),
        "sample_serif_fonts": sorted(n for n in names if "Times" in n or "Serif" in n)[:20],
    }
