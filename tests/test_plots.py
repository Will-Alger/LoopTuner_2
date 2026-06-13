import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from looptuner import plots  # noqa: E402
from looptuner.drift import DriftResult  # noqa: E402
from looptuner.model.inverse import InverseResult  # noqa: E402


def _backtest_df():
    rng = np.random.default_rng(0)
    rows = []
    base = pd.Timestamp("2026-01-01", tz="UTC")
    for h in (30, 60, 120):
        for i in range(60):
            actual = float(rng.uniform(70, 200))
            pred = actual + float(rng.normal(0, 10 + h / 6))
            rows.append({
                "timestamp": base + pd.Timedelta(minutes=5 * i),
                "horizon_min": h,
                "pred": pred,
                "actual": actual,
                "signed_err": pred - actual,
                "abs_err": abs(pred - actual),
                "pct_err": abs(pred - actual) / actual * 100,
                "persist_pred": actual + float(rng.normal(0, 15)),
                "linear_pred": actual + float(rng.normal(0, 20)),
                "in50": bool(rng.random() < 0.5),
                "in90": bool(rng.random() < 0.9),
                "day": str((base + pd.Timedelta(minutes=5 * i)).date()),
            })
    return pd.DataFrame(rows)


def test_df_based_figures():
    df = _backtest_df()
    for fn in (plots.fig_accuracy_by_horizon, plots.fig_calibration, plots.fig_twin_quality):
        fig = fn(df)
        assert isinstance(fig, Figure)


def test_forecast_and_counterfactual_figures():
    times = [pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=5 * j) for j in range(13)]
    point = np.linspace(150, 110, 13)
    res = {
        "times": times,
        "point": point,
        "baseline": np.linspace(150, 130, 13),
        "anchor_bg": 150.0,
        "bands": {
            0.9: {"lo": point - 30, "hi": point + 30},
            0.5: {"lo": point - 15, "hi": point + 15},
        },
    }
    assert isinstance(plots.fig_forecast(res), Figure)
    cf = {"times": times, "actual": point + 5, "counterfactual": point - 10, "day": "2026-01-01"}
    assert isinstance(plots.fig_counterfactual(cf), Figure)


def test_isf_cr_and_drift_figures():
    rng = np.random.default_rng(1)
    isf_samples = 50 + rng.normal(0, 3, (6, 24))
    cr_samples = 10 + rng.normal(0, 1, (6, 24))
    inv = InverseResult(
        hours=np.arange(24),
        isf_samples=isf_samples,
        cr_samples=cr_samples,
        current_isf=np.full(24, 50.0),
        current_cr=np.full(24, 10.0),
        carb_support_by_hour=np.where(np.arange(24) % 6 == 0, 60.0, 5.0),
        coverage_levels=(0.5, 0.9),
        n_models=6,
    )
    assert isinstance(plots.fig_isf_cr(inv), Figure)

    dr = DriftResult(
        horizon_min=60,
        days=["2026-01-01", "2026-01-02"],
        per_hour_mape=rng.uniform(10, 30, 24),
        recent_hour_mape=rng.uniform(10, 40, 24),
        baseline_hour_mape=rng.uniform(10, 25, 24),
        flags=[{"hour": 12, "recent_mape": 40.0, "baseline_mape": 18.0}],
        n_predictions=200,
    )
    assert isinstance(plots.fig_drift(dr), Figure)
