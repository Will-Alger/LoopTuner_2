"""Walk-forward backtest engine with no-future-leakage anchored prediction.

At each anchor time ``a`` we build forcing that uses ONLY information available at
``a``: insulin already delivered (its on-board effect decays forward via Loop's
curve), carbs already entered, and the assumption that *scheduled* basal continues
with no future boluses/carbs — exactly Loop's own forward-prediction semantics. The
twin then predicts the trajectory and we compare to what actually happened.

The train/predict cursor moves strictly forward: for each test day the model is
trained only on prior days (expanding window), and conformal intervals are calibrated
only on prior-day residuals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from looptuner.eval.baselines import linear_extrapolation, persistence
from looptuner.ingest.schema import GRID_MINUTES, TidyDataset
from looptuner.model.kernels import carb_activity_grid, insulin_activity_grid
from looptuner.model.twin import ForwardSimulator
from looptuner.model.uncertainty import ConformalCalibrator

DEFAULT_HORIZONS_MIN = (30, 60, 120, 240)


@dataclass
class BacktestArrays:
    """Precomputed numpy arrays for anchored prediction."""

    bg: np.ndarray
    delivery: np.ndarray  # actual insulin delivered per step (bolus + basal)
    bolus: np.ndarray  # actual bolus insulin per step (for basal-override replays)
    scheduled_basal: np.ndarray  # scheduled basal insulin per step (for forward assumption)
    carbs: np.ndarray
    absorptions: np.ndarray
    minute_of_day: np.ndarray
    day_codes: np.ndarray
    days: list
    timestamps: pd.DatetimeIndex
    dia_minutes: float
    near_carb: np.ndarray  # a carb entry within +-15min of this step
    near_bolus: np.ndarray  # a bolus within +-15min of this step
    n: int

    @classmethod
    def from_dataset(cls, dataset: TidyDataset) -> BacktestArrays:
        frame = dataset.frame
        n = len(frame)
        bolus = frame["bolus"].to_numpy().astype(float)
        delivery = bolus + frame["basal_insulin"].to_numpy()
        carbs = frame["carbs"].to_numpy().astype(float)
        absorptions = np.full(n, 180.0)
        for c in dataset.carbs:
            k = frame.index.get_indexer([c.time.floor(f"{GRID_MINUTES}min")])
            if k[0] >= 0:
                absorptions[k[0]] = c.absorption_minutes

        # Scheduled basal insulin per step from the profile schedule.
        prof = dataset.profile
        sched_rate = np.array([prof.basal_at(ts) for ts in frame.index], dtype=float)
        scheduled_basal = sched_rate * (GRID_MINUTES / 60.0)

        day_series = dataset.day_index()
        days = sorted(day_series.unique().tolist())
        code = {d: i for i, d in enumerate(days)}
        day_codes = np.array([code[d] for d in day_series.to_numpy()], dtype=int)

        near_carb = _within(carbs > 0, radius=3)
        near_bolus = _within(frame["bolus"].to_numpy() > 0, radius=3)

        return cls(
            bg=frame["bg"].to_numpy().astype(float),
            delivery=delivery,
            bolus=bolus,
            scheduled_basal=scheduled_basal,
            carbs=carbs,
            absorptions=absorptions,
            minute_of_day=frame["minute_of_day"].to_numpy().astype(float),
            day_codes=day_codes,
            days=days,
            timestamps=frame.index,
            dia_minutes=prof.dia_hours * 60.0,
            near_carb=near_carb,
            near_bolus=near_bolus,
            n=n,
        )

    def anchored_window(self, anchor: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Forcing (i_act, c_app) over [anchor, anchor+horizon] with no future leakage."""
        kernel = int(np.ceil(self.dia_minutes / GRID_MINUTES)) + 1
        lo = max(0, anchor - kernel)
        hi = min(self.n - 1, anchor + horizon)
        idxs = np.arange(lo, hi + 1)
        past = idxs <= anchor
        # Past: actual delivery; future: scheduled basal only (no new boluses/temps).
        local_delivery = np.where(
            past, self.delivery[lo : hi + 1], self.scheduled_basal[lo : hi + 1]
        )
        local_carbs = np.where(past, self.carbs[lo : hi + 1], 0.0)
        local_absorb = self.absorptions[lo : hi + 1]
        i_act = insulin_activity_grid(local_delivery, GRID_MINUTES, dia_minutes=self.dia_minutes)
        c_app = carb_activity_grid(local_carbs, local_absorb, GRID_MINUTES)
        offset = anchor - lo
        return i_act[offset : offset + horizon + 1], c_app[offset : offset + horizon + 1]


def _within(flags: np.ndarray, radius: int) -> np.ndarray:
    """True where any flag is within +-radius steps."""
    out = np.zeros_like(flags, dtype=bool)
    idx = np.where(flags)[0]
    for i in idx:
        out[max(0, i - radius) : i + radius + 1] = True
    return out


def run_backtest(
    dataset: TidyDataset,
    horizons_min: tuple[int, ...] = DEFAULT_HORIZONS_MIN,
    test_days: int = 3,
    epochs: int = 150,
    coverage_levels: tuple[float, ...] = (0.5, 0.9),
    anchor_stride: int = 1,
    device: str = "cpu",
    seed: int = 0,
    last_hours: float | None = None,
    progress=None,
) -> tuple[pd.DataFrame, dict]:
    """Walk forward through history producing one record per (anchor, horizon).

    If ``last_hours`` is set (shadow mode), only anchors in that trailing window are
    evaluated, using a single model trained on everything before the window.
    """
    arr = BacktestArrays.from_dataset(dataset)
    n_days = len(arr.days)
    records: list[dict] = []

    if n_days < 2:
        raise ValueError("Backtest needs at least 2 distinct days of data.")

    if last_hours is not None:
        # Shadow mode: one window, train strictly on days before it.
        cutoff_ts = arr.timestamps[-1] - pd.Timedelta(hours=last_hours)
        window_mask = arr.timestamps >= cutoff_ts
        sim, conformal = _train_and_calibrate(
            dataset, arr, train_upto_day=int(arr.day_codes[window_mask].min()),
            horizons_min=horizons_min, epochs=epochs, coverage_levels=coverage_levels,
            device=device, seed=seed,
        )
        anchors = np.where(window_mask)[0]
        _emit(records, arr, sim, conformal, anchors, horizons_min, anchor_stride)
    else:
        test_codes = list(range(max(1, n_days - test_days), n_days))
        for d in test_codes:
            if progress:
                progress(d, test_codes)
            sim, conformal = _train_and_calibrate(
                dataset, arr, train_upto_day=d, horizons_min=horizons_min, epochs=epochs,
                coverage_levels=coverage_levels, device=device, seed=seed,
            )
            anchors = np.where(arr.day_codes == d)[0]
            _emit(records, arr, sim, conformal, anchors, horizons_min, anchor_stride)

    df = pd.DataFrame.from_records(records)
    meta = {
        "source": dataset.source,
        "coverage_days": round(dataset.coverage_days(), 2),
        "horizons_min": list(horizons_min),
        "coverage_levels": list(coverage_levels),
        "n_predictions": len(df),
        "mode": "shadow" if last_hours is not None else "walk_forward",
    }
    return df, meta


def _train_and_calibrate(
    dataset, arr, train_upto_day, horizons_min, epochs, coverage_levels, device, seed
):
    """Train the twin on days < ``train_upto_day`` and conformal-calibrate on the
    most recent training day (held out from the fit's own training pool isn't
    necessary for split conformal; we calibrate on anchored residuals there)."""
    sim = ForwardSimulator.from_dataset(dataset, device=device, seed=seed)
    # Train using only prior days: temporarily restrict by val_days so the last
    # training day acts as validation, and the fit never sees test days.
    sim.fit_until_day(dataset, train_upto_day=train_upto_day, epochs=epochs)

    calib_day = train_upto_day - 1
    calib_anchors = np.where(arr.day_codes == calib_day)[0]
    residuals: dict[int, list[float]] = {h: [] for h in horizons_min}
    max_h = max(horizons_min) // GRID_MINUTES
    for a in calib_anchors:
        if a + max_h >= arr.n or not np.isfinite(arr.bg[a]):
            continue
        i_act, c_app = arr.anchored_window(a, max_h)
        traj = sim.roll(i_act, c_app, arr.minute_of_day[a], arr.bg[a])
        for h_min in horizons_min:
            h = h_min // GRID_MINUTES
            actual = arr.bg[min(arr.n - 1, a + h)]
            if np.isfinite(actual):
                residuals[h_min].append(abs(traj[h] - actual))
    conformal = ConformalCalibrator(coverage_levels=coverage_levels)
    conformal.fit({h: np.array(v) for h, v in residuals.items() if v})
    return sim, conformal


def _emit(records, arr, sim, conformal, anchors, horizons_min, stride):
    max_h = max(horizons_min) // GRID_MINUTES
    for a in anchors[::stride]:
        if a + max_h >= arr.n or not np.isfinite(arr.bg[a]):
            continue
        i_act, c_app = arr.anchored_window(a, max_h)
        traj = sim.roll(i_act, c_app, arr.minute_of_day[a], arr.bg[a])
        hist = arr.bg[max(0, a - 6) : a + 1]
        hist = hist[np.isfinite(hist)]
        hour = int(arr.minute_of_day[a] // 60)
        for h_min in horizons_min:
            h = h_min // GRID_MINUTES
            actual = arr.bg[min(arr.n - 1, a + h)]
            if not np.isfinite(actual):
                continue
            pred = float(traj[h])
            rec = {
                "timestamp": arr.timestamps[a],
                "anchor": int(a),
                "horizon_min": h_min,
                "pred": pred,
                "actual": float(actual),
                "signed_err": pred - float(actual),
                "abs_err": abs(pred - float(actual)),
                "pct_err": abs(pred - float(actual)) / max(1.0, abs(actual)) * 100.0,
                "hour": hour,
                "overnight": hour < 6,
                "near_carb": bool(arr.near_carb[a]),
                "near_bolus": bool(arr.near_bolus[a]),
                "persist_pred": float(persistence(hist, h)[-1]) if hist.size else np.nan,
                "linear_pred": (
                    float(linear_extrapolation(hist, h)[-1]) if hist.size >= 2 else np.nan
                ),
                "day": str(arr.timestamps[a].date()),
            }
            for cov in conformal.coverage_levels:
                rec[f"in{int(cov*100)}"] = conformal.covers(pred, actual, h_min, cov)
                lo, hi = conformal.interval(pred, h_min, cov)
                rec[f"lo{int(cov*100)}"] = lo
                rec[f"hi{int(cov*100)}"] = hi
            records.append(rec)
