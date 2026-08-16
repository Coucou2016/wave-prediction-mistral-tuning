from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _fig_style() -> None:
    from wave_llm.evaluation.science_plots_style import apply_science_style

    apply_science_style(font_size=10, use_times=True)


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    s = text.strip()
    if "{" in s and "}" in s:
        chunk = s[s.find("{") : s.rfind("}") + 1]
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass
    return None


def plot_mistral_regime_confusion(metrics_path: Path, out: Path) -> None:
    """Expect metrics.json with keys y_true_regime, y_pred_regime (parallel lists)."""
    _fig_style()
    if not metrics_path.exists():
        return
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    yt = m.get("y_true_regime") or []
    yp = m.get("y_pred_regime") or []
    if not yt or not yp or len(yt) != len(yp):
        log.warning("metrics.json missing regime lists")
        return
    labels = sorted(set(yt) | set(yp))
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=float)
    for t, p in zip(yt, yp):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1.0
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("LLM predicted regime")
    ax.set_ylabel("Label (rule / JSONL output)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    acc = m.get("regime_accuracy")
    mid = m.get("model_id", "")
    backend = m.get("backend", "")
    title = f"LLM zero-shot — wave_regime ({backend} / {mid})"
    if acc is not None:
        title += f" acc={float(acc):.3f}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_mistral_predictability_accuracy(metrics_path: Path, out: Path) -> None:
    _fig_style()
    if not metrics_path.exists():
        return
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    acc = m.get("predictability_accuracy")
    n = m.get("n_samples")
    if acc is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(["predictability_24h"], [float(acc)], color="#6baed6", edgecolor="k")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"LLM zero-shot — predictability_24h (model: {m.get('model_id', '')})")
    if n is not None:
        ax.text(0, float(acc) + 0.03, f"n={int(n)}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
