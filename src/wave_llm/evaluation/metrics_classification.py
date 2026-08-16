from __future__ import annotations

from typing import Sequence

import numpy as np


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    t = np.asarray(y_true)
    p = np.asarray(y_pred)
    return float((t == p).mean()) if t.size else float("nan")
