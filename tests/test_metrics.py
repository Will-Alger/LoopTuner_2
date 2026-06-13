import numpy as np

from looptuner.eval.baselines import linear_extrapolation, persistence
from looptuner.eval.metrics import (
    clinical_metrics,
    hypo_precision_recall,
    mape,
    rmse,
    time_in_range,
)


def test_rmse_and_mape_ignore_nans():
    pred = np.array([100.0, np.nan, 120.0])
    actual = np.array([110.0, 90.0, 120.0])
    # Only idx 0 (err 10) and idx 2 (err 0) are paired; idx 1 dropped via NaN.
    assert abs(rmse(pred, actual) - np.sqrt((10.0 ** 2 + 0.0) / 2)) < 1e-9
    assert abs(mape(pred, actual) - (10.0 / 110.0 * 100) / 2) < 1e-9


def test_hypo_precision_recall():
    pred = np.array([65.0, 80.0, 60.0, 100.0])
    actual = np.array([68.0, 90.0, 75.0, 100.0])
    out = hypo_precision_recall(pred, actual, threshold=70.0)
    # pred hypo at idx 0,2; actual hypo at idx 0 -> tp=1, fp=1, fn=0
    assert out["precision"] == 0.5
    assert out["recall"] == 1.0


def test_time_in_range():
    bg = np.array([60.0, 100.0, 150.0, 200.0])  # 2 of 4 in [70,180]
    assert abs(time_in_range(bg) - 0.5) < 1e-9


def test_clinical_metrics_bundle():
    rng = np.random.default_rng(0)
    actual = rng.uniform(70, 180, 200)
    pred = actual + rng.normal(0, 10, 200)
    m = clinical_metrics(pred, actual)
    assert m["n"] == 200
    assert m["rmse"] > 0
    assert "hypo_precision" in m


def test_baselines():
    hist = np.array([100.0, 105.0, 110.0, 115.0, 120.0, 125.0])
    assert persistence(hist, 6)[-1] == 125.0
    lin = linear_extrapolation(hist, 6)
    # slope is +5/step, 6 steps ahead from 125 -> 155
    assert abs(lin[-1] - 155.0) < 1e-6
