"""Trivial baselines the twin must beat to justify its complexity.

Per the spec, every backtest compares the twin against at least two trivial
predictors. If the twin can't beat "BG stays flat" and "linear extrapolation",
there is no point fitting ISF/CR off it.
"""

from __future__ import annotations

import numpy as np


def persistence(bg_history: np.ndarray, horizon_steps: int) -> np.ndarray:
    """'BG stays where it is': repeat the last observed value forward."""
    last = float(bg_history[-1])
    return np.full(horizon_steps, last, dtype=float)


def linear_extrapolation(
    bg_history: np.ndarray, horizon_steps: int, trend_window: int = 6
) -> np.ndarray:
    """Extrapolate the recent linear trend (default last 30 min at 5-min grid)."""
    hist = np.asarray(bg_history, dtype=float)
    w = min(trend_window, hist.size)
    y = hist[-w:]
    x = np.arange(w, dtype=float)
    if w < 2:
        slope = 0.0
    else:
        slope = float(np.polyfit(x, y, 1)[0])
    last = float(hist[-1])
    steps = np.arange(1, horizon_steps + 1, dtype=float)
    return last + slope * steps
