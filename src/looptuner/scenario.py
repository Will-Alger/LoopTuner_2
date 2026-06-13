"""Counterfactual replays and interactive scenario forecasts.

Two capabilities, both driven by the SAME trained forward model (no separate code
path — a bug in one shows up in the other):

* ``counterfactual_replay`` — take a real day with its actual insulin/carb inputs and
  re-simulate it under proposed time-varying ISF(t)/CR(t)/basal(t): "what would my BG
  have looked like with ISF=42 instead of 50 from 4-8am on this day?"

* ``scenario_forecast`` — from the current state (last CGM + insulin/carbs on board),
  take a hypothetical input (a bolus, a meal, an ISF/CR/basal override) and predict
  the next 1-6h with conformal credible intervals. No future-leakage; designed to run
  well under the latency targets (<2s, <200ms for the batched point path).

Overrides are expressed as multipliers on the model's learned ISF(t)/CR(t), so they
compose with whatever diurnal pattern the twin has learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from looptuner.backtest.engine import BacktestArrays
from looptuner.ingest.schema import GRID_MINUTES, TidyDataset, safe_zoneinfo
from looptuner.model.kernels import carb_activity_grid, insulin_activity_grid
from looptuner.model.twin import ForwardSimulator
from looptuner.model.uncertainty import ConformalCalibrator


def _padded_arrays(dataset: TidyDataset, extra_steps: int) -> BacktestArrays:
    """BacktestArrays extended ``extra_steps`` into the future for forecasting.

    Future steps carry the profile's scheduled basal, no carbs/boluses, and NaN CGM —
    so a forecast can run past the end of the data while past insulin/carbs on board
    keep decaying through the kernels.
    """
    arr = BacktestArrays.from_dataset(dataset)
    if extra_steps <= 0:
        return arr
    prof = dataset.profile
    tz = safe_zoneinfo(dataset.timezone)
    last = arr.timestamps[-1]
    future_ts = pd.DatetimeIndex(
        [last + pd.Timedelta(minutes=GRID_MINUTES * (i + 1)) for i in range(extra_steps)]
    )
    local = future_ts.tz_convert(tz)
    fut_minute = (local.hour * 60 + local.minute).to_numpy().astype(float)
    fut_sched = np.array([prof.basal_at(ts) for ts in future_ts]) * (GRID_MINUTES / 60.0)
    z = np.zeros(extra_steps)
    return BacktestArrays(
        bg=np.concatenate([arr.bg, np.full(extra_steps, np.nan)]),
        delivery=np.concatenate([arr.delivery, fut_sched]),
        bolus=np.concatenate([arr.bolus, z]),
        scheduled_basal=np.concatenate([arr.scheduled_basal, fut_sched]),
        carbs=np.concatenate([arr.carbs, z]),
        absorptions=np.concatenate([arr.absorptions, np.full(extra_steps, 180.0)]),
        minute_of_day=np.concatenate([arr.minute_of_day, fut_minute]),
        day_codes=np.concatenate([arr.day_codes, np.full(extra_steps, arr.day_codes[-1])]),
        days=arr.days,
        timestamps=arr.timestamps.append(future_ts),
        dia_minutes=arr.dia_minutes,
        near_carb=np.concatenate([arr.near_carb, np.zeros(extra_steps, bool)]),
        near_bolus=np.concatenate([arr.near_bolus, np.zeros(extra_steps, bool)]),
        n=arr.n + extra_steps,
    )


@dataclass
class CounterfactualSpec:
    """A proposed change to settings and/or inputs, relative to current behavior."""

    isf_scale_by_hour: np.ndarray = field(default_factory=lambda: np.ones(24))
    cr_scale_by_hour: np.ndarray = field(default_factory=lambda: np.ones(24))
    basal_rate_by_hour: np.ndarray | None = None  # absolute U/hr forward override
    added_boluses: list[tuple[float, float]] = field(default_factory=list)  # (offset_min, U)
    added_carbs: list[tuple[float, float, float]] = field(default_factory=list)  # (off, g, absorb)

    @classmethod
    def flat(cls) -> CounterfactualSpec:
        return cls()

    def with_isf_scale(self, hour_start: int, hour_end: int, scale: float) -> CounterfactualSpec:
        arr = self.isf_scale_by_hour.copy()
        for h in range(hour_start, hour_end):
            arr[h % 24] = scale
        self.isf_scale_by_hour = arr
        return self

    def with_cr_scale(self, hour_start: int, hour_end: int, scale: float) -> CounterfactualSpec:
        arr = self.cr_scale_by_hour.copy()
        for h in range(hour_start, hour_end):
            arr[h % 24] = scale
        self.cr_scale_by_hour = arr
        return self


def isf_scale_from_absolute(profile, hour_start: int, hour_end: int, proposed: float) -> np.ndarray:
    """Build an ISF *multiplier* schedule from a proposed absolute ISF over a window.

    multiplier(h) = proposed / current_profile_ISF(h), so it expresses "set ISF to
    ``proposed``" relative to the user's current setting (which the model is anchored
    to).
    """
    scale = np.ones(24)
    for h in range(hour_start, hour_end):
        ts = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=h % 24)
        cur = profile.isf_at(ts)
        scale[h % 24] = proposed / cur if cur else 1.0
    return scale


def _scale_steps(
    scale_by_hour: np.ndarray, start_minute: float, horizon: int, dt: float
) -> np.ndarray:
    minutes = start_minute + np.arange(horizon + 1) * dt
    hours = (minutes // 60).astype(int) % 24
    return scale_by_hour[hours]


def _build_window(
    arr: BacktestArrays,
    anchor: int,
    horizon: int,
    spec: CounterfactualSpec,
    *,
    replay: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Construct forcing + ISF/CR scale series for a window.

    replay=True uses the day's ACTUAL future inputs (a what-if on settings); replay=
    False is an honest forecast (future discretionary inputs unknown → scheduled basal
    only) plus any hypothetical added bolus/carb the user is proposing.
    """
    kernel = int(np.ceil(arr.dia_minutes / GRID_MINUTES)) + 1
    lo = max(0, anchor - kernel)
    hi = min(arr.n - 1, anchor + horizon)
    idxs = np.arange(lo, hi + 1)
    past = idxs <= anchor

    if replay:
        basal_part = arr.delivery[lo : hi + 1] - arr.bolus[lo : hi + 1]
        bolus_part = arr.bolus[lo : hi + 1].copy()
        local_carbs = arr.carbs[lo : hi + 1].copy()
    else:
        # Forecast: past actual, future scheduled basal only, no future carbs/boluses.
        basal_part = np.where(past, arr.delivery[lo : hi + 1] - arr.bolus[lo : hi + 1],
                              arr.scheduled_basal[lo : hi + 1])
        bolus_part = np.where(past, arr.bolus[lo : hi + 1], 0.0)
        local_carbs = np.where(past, arr.carbs[lo : hi + 1], 0.0)
    local_absorb = arr.absorptions[lo : hi + 1].copy()

    # Basal override (absolute U/hr by hour) applied to the forward portion.
    if spec.basal_rate_by_hour is not None:
        minutes = arr.minute_of_day[lo : hi + 1]
        hours = (minutes // 60).astype(int) % 24
        override = spec.basal_rate_by_hour[hours] * (GRID_MINUTES / 60.0)
        forward = idxs > anchor
        basal_part = np.where(forward, override, basal_part)

    # Hypothetical added inputs (offsets in minutes from the anchor).
    for off, units in spec.added_boluses:
        k = int(round(off / GRID_MINUTES)) + (anchor - lo)
        if 0 <= k < len(bolus_part):
            bolus_part[k] += units
    for off, grams, absorb in spec.added_carbs:
        k = int(round(off / GRID_MINUTES)) + (anchor - lo)
        if 0 <= k < len(local_carbs):
            local_carbs[k] += grams
            local_absorb[k] = absorb

    local_delivery = bolus_part + basal_part
    i_act = insulin_activity_grid(local_delivery, GRID_MINUTES, dia_minutes=arr.dia_minutes)
    c_app = carb_activity_grid(local_carbs, local_absorb, GRID_MINUTES)
    offset = anchor - lo
    i_win = i_act[offset : offset + horizon + 1]
    c_win = c_app[offset : offset + horizon + 1]

    start_minute = float(arr.minute_of_day[anchor])
    isf_steps = _scale_steps(spec.isf_scale_by_hour, start_minute, horizon, GRID_MINUTES)
    cr_steps = _scale_steps(spec.cr_scale_by_hour, start_minute, horizon, GRID_MINUTES)
    return i_win, c_win, isf_steps, cr_steps, start_minute


def _default_anchor(arr: BacktestArrays) -> int:
    finite = np.where(np.isfinite(arr.bg))[0]
    if finite.size == 0:
        raise ValueError("No CGM data to anchor a scenario on.")
    return int(finite[-1])


def calibrate_conformal(
    sim: ForwardSimulator,
    dataset: TidyDataset,
    horizons_min: tuple[int, ...] = (30, 60, 90, 120, 180, 240, 300, 360),
    coverage_levels: tuple[float, ...] = (0.5, 0.9),
    calib_days: int = 1,
) -> ConformalCalibrator:
    """Calibrate conformal intervals for an already-trained model on recent days.

    Uses anchored (no-leakage) residuals over the most recent ``calib_days`` days so a
    scenario forecast can carry calibrated bands without retraining.
    """
    arr = BacktestArrays.from_dataset(dataset)
    n_days = len(arr.days)
    codes = set(range(max(0, n_days - calib_days), n_days))
    max_h = max(horizons_min) // GRID_MINUTES
    residuals: dict[int, list[float]] = {h: [] for h in horizons_min}
    anchors = np.where(np.isin(arr.day_codes, list(codes)) & np.isfinite(arr.bg))[0]
    for a in anchors:
        if a + max_h >= arr.n:
            continue
        i_win, c_win = arr.anchored_window(a, max_h)
        traj = sim.roll(i_win, c_win, arr.minute_of_day[a], arr.bg[a])
        for h_min in horizons_min:
            h = h_min // GRID_MINUTES
            actual = arr.bg[min(arr.n - 1, a + h)]
            if np.isfinite(actual):
                residuals[h_min].append(abs(traj[h] - actual))
    conf = ConformalCalibrator(coverage_levels=coverage_levels)
    conf.fit({h: np.array(v) for h, v in residuals.items() if v})
    return conf


def scenario_forecast(
    sim: ForwardSimulator,
    dataset: TidyDataset,
    spec: CounterfactualSpec,
    conformal: ConformalCalibrator | None = None,
    anchor: int | None = None,
    horizon_min: int = 360,
    coverage_levels: tuple[float, ...] = (0.5, 0.9),
) -> dict:
    """Forecast BG forward from the current state under ``spec``, with conformal bands.

    Returns timestamps, the scenario point trajectory, the no-intervention baseline
    (so you can read off the delta), and lo/hi bands per coverage level.
    """
    horizon = horizon_min // GRID_MINUTES
    base_arr = BacktestArrays.from_dataset(dataset)
    anchor = _default_anchor(base_arr) if anchor is None else anchor
    # Pad enough future grid so the window from the anchor fits.
    extra = max(0, anchor + horizon - (base_arr.n - 1))
    arr = _padded_arrays(dataset, extra)
    g0 = float(arr.bg[anchor])

    i_s, c_s, isf_s, cr_s, start_min = _build_window(arr, anchor, horizon, spec, replay=False)
    point = sim.roll(i_s, c_s, start_min, g0, isf_scale=isf_s, cr_scale=cr_s)

    i_b, c_b, _, _, _ = _build_window(arr, anchor, horizon, CounterfactualSpec.flat(), replay=False)
    baseline = sim.roll(i_b, c_b, start_min, g0)

    times = [
        arr.timestamps[anchor] + pd.Timedelta(minutes=j * GRID_MINUTES)
        for j in range(horizon + 1)
    ]
    bands: dict[float, dict[str, np.ndarray]] = {}
    if conformal is not None and conformal.quantiles:
        step_min = np.arange(horizon + 1) * GRID_MINUTES
        for cov in coverage_levels:
            w = np.array([conformal.half_width_at(m, cov) for m in step_min])
            bands[cov] = {"lo": point - w, "hi": point + w}

    return {
        "anchor_time": arr.timestamps[anchor],
        "anchor_bg": g0,
        "times": times,
        "point": point,
        "baseline": baseline,
        "bands": bands,
        "horizon_min": horizon_min,
    }


def counterfactual_replay(
    sim: ForwardSimulator,
    dataset: TidyDataset,
    spec: CounterfactualSpec,
    day: str | None = None,
) -> dict:
    """Re-simulate a real day under proposed settings, alongside what actually happened."""
    arr = BacktestArrays.from_dataset(dataset)
    if day is None:
        target_code = int(arr.day_codes[_default_anchor(arr)])
    else:
        target = pd.Timestamp(day).date()
        if target not in arr.days:
            raise ValueError(f"Day {day} not in dataset (have {arr.days[0]}..{arr.days[-1]}).")
        target_code = arr.days.index(target)

    day_idx = np.where(arr.day_codes == target_code)[0]
    finite = day_idx[np.isfinite(arr.bg[day_idx])]
    if finite.size == 0:
        raise ValueError("No CGM on the requested day.")
    anchor = int(finite[0])
    horizon = int(day_idx[-1] - anchor)
    g0 = float(arr.bg[anchor])

    i_s, c_s, isf_s, cr_s, start_min = _build_window(arr, anchor, horizon, spec, replay=True)
    counterfactual = sim.roll(i_s, c_s, start_min, g0, isf_scale=isf_s, cr_scale=cr_s)

    times = [
        arr.timestamps[anchor] + pd.Timedelta(minutes=j * GRID_MINUTES)
        for j in range(horizon + 1)
    ]
    actual = arr.bg[anchor : anchor + horizon + 1]
    return {
        "day": str(arr.timestamps[anchor].date()),
        "times": times,
        "counterfactual": counterfactual,
        "actual": actual,
    }


def summarize_forecast(result: dict) -> dict:
    """Headline numbers for a forecast: end BG, min/max, time-below-70, delta vs baseline."""
    point = result["point"]
    base = result["baseline"]
    return {
        "anchor_bg": round(result["anchor_bg"], 0),
        "end_bg": round(float(point[-1]), 0),
        "min_bg": round(float(np.min(point)), 0),
        "max_bg": round(float(np.max(point)), 0),
        "frac_below_70": round(float(np.mean(point < 70)), 2),
        "delta_vs_baseline_end": round(float(point[-1] - base[-1]), 0),
    }
