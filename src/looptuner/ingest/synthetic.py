"""Synthetic T1D data generator.

Produces a ``TidyDataset`` with realistic CGM/insulin/carb/basal signals from a
known physiological model, and exposes the *ground-truth* time-of-day ISF(t)/CR(t)
so we can validate (a) forward-simulation accuracy and (b) inverse recovery of
ISF/CR — neither of which we can check against real data where the truth is unknown.

The generator uses the SAME mechanistic kernels (Loop insulin curve, carb
absorption) as the twin, with a homeostatic pull toward a baseline and an
endogenous-production term that balances scheduled basal. ISF/CR vary across the
day (dawn resistance), which is exactly the structure the inverse fit must recover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from looptuner.ingest.schema import (
    GRID_MINUTES,
    BasalSegment,
    BolusEvent,
    CarbEvent,
    Profile,
    ScheduleEntry,
    TidyDataset,
    build_tidy_dataset,
)
from looptuner.model.kernels import carb_activity_grid, insulin_activity_grid


@dataclass
class SyntheticTruth:
    """Ground-truth dynamics behind a synthetic dataset (for validation only)."""

    isf_by_hour: np.ndarray  # length-24 mg/dL per U
    cr_by_hour: np.ndarray  # length-24 g per U
    basal_u_hr: float
    glucose_baseline: float
    clearance_per_min: float
    noiseless_bg: pd.Series  # the true latent BG on the grid (no sensor noise)

    def isf_at_hour(self, hour: int) -> float:
        return float(self.isf_by_hour[hour % 24])

    def cr_at_hour(self, hour: int) -> float:
        return float(self.cr_by_hour[hour % 24])


def _diurnal_isf(hours: np.ndarray, base: float, dawn_drop: float) -> np.ndarray:
    """ISF lower (more resistant) around the dawn window (~3-8am), higher midday."""
    # Single cosine dip centered at 5am.
    phase = np.cos((hours - 5.0) / 24.0 * 2 * np.pi)
    return base - dawn_drop * np.clip(phase, 0, None)


def generate_synthetic_dataset(
    n_days: int = 14,
    seed: int = 0,
    timezone_name: str = "America/New_York",
    start: pd.Timestamp | None = None,
) -> tuple[TidyDataset, SyntheticTruth]:
    """Generate ``n_days`` of synthetic data ending at a round local midnight."""
    rng = np.random.default_rng(seed)

    if start is None:
        start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    n_steps = int(n_days * 24 * 60 / GRID_MINUTES)
    grid = pd.date_range(start, periods=n_steps, freq=f"{GRID_MINUTES}min", tz="UTC")

    # --- Ground-truth time-of-day ISF/CR ---------------------------------- #
    hours = np.arange(24)
    isf_by_hour = _diurnal_isf(hours, base=52.0, dawn_drop=16.0)  # 36..52 mg/dL/U
    cr_by_hour = _diurnal_isf(hours, base=11.0, dawn_drop=3.5)  # tighter ratio at dawn
    basal_u_hr = 0.8
    g_baseline = 120.0
    k_clear = 0.012  # per minute homeostatic pull toward baseline

    # Local hour per grid step (for circadian lookups).
    from looptuner.ingest.schema import safe_zoneinfo

    local = grid.tz_convert(safe_zoneinfo(timezone_name))
    hour_of = local.hour.to_numpy()
    isf_t = isf_by_hour[hour_of]
    cr_t = cr_by_hour[hour_of]

    # --- Generate treatments ---------------------------------------------- #
    boluses: list[BolusEvent] = []
    carbs: list[CarbEvent] = []
    bolus_series = np.zeros(n_steps)
    carb_series = np.zeros(n_steps)
    absorb_series = np.full(n_steps, 180.0)

    def nearest_step(ts: pd.Timestamp) -> int:
        return int(round((ts - grid[0]) / pd.Timedelta(minutes=GRID_MINUTES)))

    meals = [(7.5, 45.0, 180.0), (12.5, 65.0, 180.0), (18.5, 75.0, 240.0)]
    days = pd.date_range(local[0].normalize(), periods=n_days + 1, freq="D")
    for day in days:
        for hour, base_g, absorb in meals:
            jitter = rng.normal(0, 0.4)
            meal_local = day + pd.Timedelta(hours=hour + jitter)
            meal_utc = meal_local.tz_convert("UTC")
            k = nearest_step(meal_utc)
            if not (0 <= k < n_steps):
                continue
            grams = max(5.0, base_g + rng.normal(0, 8.0))
            # Carb counting is imperfect: bolus uses CR with multiplicative error.
            cr_here = cr_by_hour[local[k].hour]
            count_err = rng.normal(1.0, 0.12)
            dose = grams * count_err / cr_here
            carbs.append(CarbEvent(time=grid[k], grams=grams, absorption_minutes=absorb))
            boluses.append(BolusEvent(time=grid[k], units=round(dose, 2)))
            carb_series[k] += grams
            absorb_series[k] = absorb
            bolus_series[k] += round(dose, 2)
        # Occasional correction bolus mid-afternoon.
        if rng.random() < 0.4:
            corr_local = day + pd.Timedelta(hours=15 + rng.normal(0, 1.0))
            k = nearest_step(corr_local.tz_convert("UTC"))
            if 0 <= k < n_steps:
                u = round(float(rng.uniform(0.5, 1.5)), 2)
                boluses.append(BolusEvent(time=grid[k], units=u))
                bolus_series[k] += u

    # --- Basal: scheduled constant, with a few temp basals ---------------- #
    basal_segments = [
        BasalSegment(start=grid[0], end=grid[-1], rate=basal_u_hr, is_temp=False)
    ]
    basal_insulin = np.full(n_steps, basal_u_hr * GRID_MINUTES / 60.0)
    for _ in range(n_days):  # a handful of temp basals
        if rng.random() < 0.5:
            k = int(rng.integers(0, n_steps - 12))
            dur = int(rng.integers(6, 18))
            rate = float(rng.uniform(0.0, 1.6))
            basal_segments.append(
                BasalSegment(grid[k], grid[min(n_steps - 1, k + dur)], rate, is_temp=True)
            )
            basal_insulin[k : k + dur] = rate * GRID_MINUTES / 60.0

    # --- Integrate the latent glucose ODE (Euler, 5-min) ------------------ #
    insulin_delivery = bolus_series + basal_insulin
    i_act = insulin_activity_grid(insulin_delivery, GRID_MINUTES)
    c_app = carb_activity_grid(carb_series, absorb_series, GRID_MINUTES)

    dt = float(GRID_MINUTES)
    g = np.zeros(n_steps)
    g[0] = g_baseline
    proc_sigma = 0.6
    for k in range(n_steps - 1):
        egp = isf_t[k] * (basal_u_hr / 60.0)  # balances scheduled basal
        dgdt = (
            egp
            - isf_t[k] * i_act[k]
            + (isf_t[k] / cr_t[k]) * c_app[k]
            - k_clear * (g[k] - g_baseline)
        )
        g[k + 1] = g[k] + dt * dgdt + rng.normal(0, proc_sigma)
        g[k + 1] = float(np.clip(g[k + 1], 40.0, 400.0))

    noiseless = pd.Series(g, index=grid, name="bg_true")

    # --- Sensor model: AR(1) noise + occasional gaps ---------------------- #
    sensor = np.zeros(n_steps)
    e = 0.0
    for k in range(n_steps):
        e = 0.6 * e + rng.normal(0, 6.0)
        sensor[k] = e
    bg_obs = g + sensor

    bg_samples: list[tuple[pd.Timestamp, float]] = []
    gap = np.zeros(n_steps, dtype=bool)
    # A couple of sensor dropouts per few days.
    n_gaps = max(0, n_days // 3)
    for _ in range(n_gaps):
        k = int(rng.integers(0, n_steps - 24))
        gap[k : k + int(rng.integers(4, 20))] = True
    for k in range(n_steps):
        if not gap[k]:
            bg_samples.append((grid[k], float(np.clip(bg_obs[k], 40.0, 400.0))))

    # --- Profile reflecting the (rounded) ground truth -------------------- #
    isf_sched = tuple(ScheduleEntry(h * 3600, round(float(isf_by_hour[h]), 1)) for h in range(24))
    cr_sched = tuple(ScheduleEntry(h * 3600, round(float(cr_by_hour[h]), 1)) for h in range(24))
    profile = Profile(
        timezone=timezone_name,
        dia_hours=6.0,
        isf=isf_sched,
        cr=cr_sched,
        basal=(ScheduleEntry(0, basal_u_hr),),
        target=(ScheduleEntry(0, 105.0),),
    )

    ds = build_tidy_dataset(
        bg_samples=bg_samples,
        boluses=boluses,
        carbs=carbs,
        basal_segments=basal_segments,
        profile=profile,
        timezone_name=timezone_name,
        source=f"synthetic(seed={seed},days={n_days})",
    )
    truth = SyntheticTruth(
        isf_by_hour=isf_by_hour,
        cr_by_hour=cr_by_hour,
        basal_u_hr=basal_u_hr,
        glucose_baseline=g_baseline,
        clearance_per_min=k_clear,
        noiseless_bg=noiseless,
    )
    return ds, truth
