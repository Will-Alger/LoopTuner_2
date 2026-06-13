"""Accuracy and clinical metrics for BG predictions.

All functions take 1-D numpy arrays of predicted and actual mg/dL and ignore
NaNs pairwise. Clinical metrics matter more than raw accuracy here: a model that
is RMSE-accurate but misses hypos is dangerous decision support.
"""

from __future__ import annotations

import numpy as np

HYPO_THRESHOLD = 70.0
TIR_LOW = 70.0
TIR_HIGH = 180.0


def _paired(pred: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(actual)
    return pred[mask], actual[mask]


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    p, a = _paired(pred, actual)
    if p.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - a) ** 2)))


def mape(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute percentage error (%), guarding against tiny denominators."""
    p, a = _paired(pred, actual)
    if p.size == 0:
        return float("nan")
    denom = np.clip(np.abs(a), 1.0, None)
    return float(np.mean(np.abs(p - a) / denom) * 100.0)


def mard(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute relative difference (%) — the CGM-standard accuracy metric."""
    return mape(pred, actual)


def time_in_range(bg: np.ndarray, low: float = TIR_LOW, high: float = TIR_HIGH) -> float:
    b = np.asarray(bg, dtype=float)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return float("nan")
    return float(np.mean((b >= low) & (b <= high)))


def time_in_range_error(pred: np.ndarray, actual: np.ndarray) -> float:
    """Signed error in predicted vs actual time-in-range fraction."""
    p, a = _paired(pred, actual)
    if p.size == 0:
        return float("nan")
    return time_in_range(p) - time_in_range(a)


def hypo_precision_recall(
    pred: np.ndarray, actual: np.ndarray, threshold: float = HYPO_THRESHOLD
) -> dict[str, float]:
    """Point-wise hypo (BG < threshold) precision/recall/F1."""
    p, a = _paired(pred, actual)
    if p.size == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    pred_hypo = p < threshold
    act_hypo = a < threshold
    tp = float(np.sum(pred_hypo & act_hypo))
    fp = float(np.sum(pred_hypo & ~act_hypo))
    fn = float(np.sum(~pred_hypo & act_hypo))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_actual_hypo": float(np.sum(act_hypo)),
    }


def clinical_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Bundle of accuracy + clinical metrics for a set of paired predictions."""
    p, a = _paired(pred, actual)
    out = {
        "n": float(p.size),
        "rmse": rmse(p, a),
        "mape": mape(p, a),
        "bias": float(np.mean(p - a)) if p.size else float("nan"),
        "tir_error": time_in_range_error(p, a),
    }
    out.update({f"hypo_{k}": v for k, v in hypo_precision_recall(p, a).items()})
    return out
