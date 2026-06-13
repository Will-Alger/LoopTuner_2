"""Mechanistic kernels shared by the synthetic generator and the hybrid twin.

These are deliberately backend-agnostic (plain numpy float math) so the exact same
insulin/carb activity curves drive (a) synthetic data generation, (b) the forward
simulator's mechanistic backbone, and (c) the scenario simulator. One source of
truth for the physics means a bug shows up everywhere at once, not in one path.

We reuse Loop's exponential insulin model rather than relearning insulin
pharmacokinetics — the neural part of the twin only has to learn what Loop does
*not* model (time-varying sensitivity + residuals).
"""

from __future__ import annotations

import numpy as np


def exponential_insulin_activity(
    t_minutes: np.ndarray,
    dia_minutes: float = 360.0,
    peak_minutes: float = 75.0,
) -> np.ndarray:
    """Loop's exponential insulin *activity* curve (fraction of a unit acting per minute).

    Integrates to 1 over [0, dia_minutes], so a bolus of ``u`` units delivers a
    total BG effect of ``u * ISF`` when multiplied by ISF. Reference: the
    exponential insulin model used by Loop / OpenAPS (Dragan Maksimovic).

    Parameters
    ----------
    t_minutes : minutes since delivery (array). Activity is 0 for t<0 and t>dia.
    dia_minutes : duration of insulin action.
    peak_minutes : time of peak activity.
    """
    t = np.asarray(t_minutes, dtype=float)
    td = float(dia_minutes)
    tp = float(peak_minutes)
    tau = tp * (1 - tp / td) / (1 - 2 * tp / td)
    a = 2 * tau / td
    s = 1.0 / (1 - a + (1 + a) * np.exp(-td / tau))
    activity = (s / tau**2) * t * (1 - t / td) * np.exp(-t / tau)
    activity = np.where((t >= 0) & (t <= td), activity, 0.0)
    return activity


def carb_activity(t_minutes: np.ndarray, absorption_minutes: float = 180.0) -> np.ndarray:
    """Rate of carb appearance per gram per minute (gamma-shaped, integrates to ~1).

    A single-peak absorption curve scaled so the bulk of a meal is absorbed within
    roughly ``absorption_minutes``. Total appearance per gram -> 1, so a meal of
    ``g`` grams contributes ``g * (ISF/CR)`` mg/dL of total rise.
    """
    t = np.asarray(t_minutes, dtype=float)
    # tau chosen so the curve's mass is concentrated within absorption_minutes.
    tau = max(absorption_minutes, 1.0) / 3.0
    activity = (t / tau**2) * np.exp(-t / tau)
    return np.where(t >= 0, activity, 0.0)


def insulin_activity_grid(
    deliveries: np.ndarray,
    grid_minutes: float,
    dia_minutes: float = 360.0,
    peak_minutes: float = 75.0,
) -> np.ndarray:
    """Convolve a per-bin insulin delivery series with the activity curve.

    Returns the insulin *activity* per bin: sum over past deliveries of
    ``units * activity(elapsed)``. Units are (activity fraction)/min * units,
    i.e. when multiplied by ISF (mg/dL per U) and dt (min) it gives mg/dL.
    """
    deliveries = np.asarray(deliveries, dtype=float)
    n = len(deliveries)
    horizon = int(np.ceil(dia_minutes / grid_minutes)) + 1
    lag_min = np.arange(horizon) * grid_minutes
    kernel = exponential_insulin_activity(lag_min, dia_minutes, peak_minutes)
    out = np.zeros(n, dtype=float)
    nz = np.nonzero(deliveries)[0]
    for i in nz:
        end = min(n, i + horizon)
        out[i:end] += deliveries[i] * kernel[: end - i]
    return out


def carb_activity_grid(
    entries: np.ndarray,
    absorptions: np.ndarray,
    grid_minutes: float,
    max_minutes: float = 480.0,
) -> np.ndarray:
    """Convolve a per-bin carb entry series with per-entry absorption curves.

    Returns carb rate-of-appearance per bin (grams/min, per-minute rate), to be
    scaled by (ISF/CR) and the integrator's dt downstream — mirroring how insulin
    activity is handled, so the solver's dt is the single place time-scaling lives.
    Each entry may have its own absorption window.
    """
    entries = np.asarray(entries, dtype=float)
    absorptions = np.asarray(absorptions, dtype=float)
    n = len(entries)
    horizon = int(np.ceil(max_minutes / grid_minutes)) + 1
    lag_min = np.arange(horizon) * grid_minutes
    out = np.zeros(n, dtype=float)
    nz = np.nonzero(entries)[0]
    for i in nz:
        end = min(n, i + horizon)
        kernel = carb_activity(lag_min, float(absorptions[i]))
        out[i:end] += entries[i] * kernel[: end - i]
    return out
