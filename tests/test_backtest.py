import copy

import numpy as np

from looptuner.backtest.engine import BacktestArrays, run_backtest
from looptuner.backtest.report import (
    append_benchmark_log,
    render_markdown_report,
    twin_quality_by_day,
)
from looptuner.eval.metrics import rmse
from looptuner.ingest.synthetic import generate_synthetic_dataset
from looptuner.model.uncertainty import ConformalCalibrator


def test_conformal_coverage_is_calibrated():
    rng = np.random.default_rng(0)
    cal = rng.normal(0, 20, 2000)
    conf = ConformalCalibrator(coverage_levels=(0.5, 0.9)).fit({60: np.abs(cal)})
    test = rng.normal(0, 20, 5000)
    cov90 = np.mean([conf.covers(0.0, a, 60, 0.9) for a in test])
    cov50 = np.mean([conf.covers(0.0, a, 60, 0.5) for a in test])
    assert 0.86 < cov90 < 0.94
    assert 0.44 < cov50 < 0.56


def test_anchored_window_has_no_future_leakage():
    ds, _ = generate_synthetic_dataset(n_days=4, seed=1)
    arr = BacktestArrays.from_dataset(ds)
    a = arr.n // 2
    i_base, c_base = arr.anchored_window(a, horizon=24)

    # Inject a huge bolus and a meal AFTER the anchor; the anchored window must
    # be identical (future discretionary inputs are not visible at prediction time).
    arr2 = copy.deepcopy(arr)
    arr2.delivery[a + 5] += 10.0
    arr2.carbs[a + 8] += 80.0
    i_leak, c_leak = arr2.anchored_window(a, horizon=24)
    assert np.allclose(i_base, i_leak)
    assert np.allclose(c_base, c_leak)

    # But a PAST bolus does change the on-board insulin within the window.
    arr3 = copy.deepcopy(arr)
    arr3.delivery[a - 2] += 10.0
    i_past, _ = arr3.anchored_window(a, horizon=24)
    assert not np.allclose(i_base, i_past)


def test_run_backtest_produces_calibrated_records():
    ds, _ = generate_synthetic_dataset(n_days=7, seed=2)
    df, meta = run_backtest(
        ds, horizons_min=(60, 120), test_days=1, epochs=40, anchor_stride=4, seed=0
    )
    assert len(df) > 0
    assert meta["mode"] == "walk_forward"
    # On well-specified synthetic data the twin beats persistence at 2h.
    sub = df[df["horizon_min"] == 120].dropna(subset=["persist_pred"])
    twin = rmse(sub["pred"].to_numpy(), sub["actual"].to_numpy())
    persist = rmse(sub["persist_pred"].to_numpy(), sub["actual"].to_numpy())
    assert twin < persist
    # 90% conformal band should roughly cover.
    assert 0.75 < df["in90"].mean() < 1.0


def test_report_and_benchmark_log(tmp_path):
    ds, _ = generate_synthetic_dataset(n_days=6, seed=3)
    df, meta = run_backtest(
        ds, horizons_min=(60, 120), test_days=1, epochs=30, anchor_stride=6, seed=0
    )
    md = render_markdown_report(df, meta)
    assert "Headline accuracy" in md
    assert "Per-day twin quality" in md
    q = twin_quality_by_day(df, 120)
    assert not q.empty

    log_path = tmp_path / "bench.parquet"
    append_benchmark_log(log_path, df, meta)
    append_benchmark_log(log_path, df, meta)  # append twice
    import pandas as pd

    log = pd.read_parquet(log_path)
    assert len(log) == 2 * df["horizon_min"].nunique()


def test_shadow_mode_runs():
    ds, _ = generate_synthetic_dataset(n_days=5, seed=4)
    df, meta = run_backtest(ds, horizons_min=(30, 60), epochs=30, last_hours=24.0)
    assert meta["mode"] == "shadow"
    assert len(df) > 0
