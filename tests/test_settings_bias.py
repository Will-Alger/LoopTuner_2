import pandas as pd

from looptuner.ingest.schema import (
    BasalSegment,
    BolusEvent,
    CarbEvent,
    Profile,
    ScheduleEntry,
    build_tidy_dataset,
)
from looptuner.ingest.synthetic import generate_synthetic_dataset
from looptuner.settings_bias import compute_settings_bias, render_settings_bias_markdown


def _engineered_dataset(n_days: int = 6):
    """Dataset where every noon meal bolus is followed by a hypo, and fasting nights
    drift steadily downward — so the report should flag both."""
    start = pd.Timestamp("2026-02-01 00:00", tz="UTC")
    steps = n_days * 24 * 12
    grid = pd.date_range(start, periods=steps, freq="5min", tz="UTC")

    bg_samples, boluses, carbs = [], [], []
    for ts in grid:
        hour = ts.hour
        minute = ts.hour * 60 + ts.minute
        if hour < 6:  # fasting night: decline 130 -> 90 over 6h
            val = 130 - (minute / 360.0) * 40
        elif 12 * 60 <= minute <= 13 * 60:  # post-meal dip to ~58
            val = 58.0
        else:
            val = 120.0
        bg_samples.append((ts, float(val)))

    for d in range(n_days + 1):
        meal = start + pd.Timedelta(days=d, hours=12)
        if meal <= grid[-1]:
            boluses.append(BolusEvent(meal, 5.0))
            carbs.append(CarbEvent(meal, 45.0, 180.0))

    basal = [BasalSegment(grid[0], grid[-1], 0.8, False)]
    profile = Profile(
        timezone="UTC",
        isf=(ScheduleEntry(0, 50.0),),
        cr=(ScheduleEntry(0, 10.0),),
        basal=(ScheduleEntry(0, 0.8),),
        target=(ScheduleEntry(0, 105.0),),
    )
    return build_tidy_dataset(
        bg_samples=bg_samples,
        boluses=boluses,
        carbs=carbs,
        basal_segments=basal,
        profile=profile,
        timezone_name="UTC",
        source="engineered",
    )


def test_detects_aggressive_meal_dosing_and_overnight_drift():
    ds = _engineered_dataset(n_days=6)
    res = compute_settings_bias(ds)
    # Every meal bolus is followed by a hypo within 3h.
    assert res.meal_bolus.n >= 5
    assert res.meal_bolus.frac_to_hypo > 0.8
    # Fasting nights drift down steadily (~-6.7 mg/dL/h engineered).
    assert res.n_fasting_nights >= 5
    assert res.overnight_drift_mgdl_per_h < -4
    # Most low time is attributable to meals or overnight, not corrections.
    assert res.hypo_attribution["post_correction"] == 0.0
    assert res.n_hypo_episodes > 0

    md = render_settings_bias_markdown(res)
    assert "carb ratio possibly too aggressive" in md
    assert "overnight over-delivery" in md


def test_runs_on_synthetic_without_error():
    ds, _ = generate_synthetic_dataset(n_days=7, seed=1)
    res = compute_settings_bias(ds)
    assert res.tbr_by_hour.shape == (24,)
    assert 0.0 <= res.tir <= 1.0
    md = render_settings_bias_markdown(res)
    assert "Settings-bias report" in md
    assert "Decision support only" in md
    assert "dose recommendation" in md

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from looptuner import plots

    assert isinstance(plots.fig_settings_bias(res), Figure)
