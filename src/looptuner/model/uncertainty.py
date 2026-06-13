"""Conformal prediction for calibrated BG trajectory intervals.

Per the Phase 1 decision, trajectory bands use split-conformal prediction (not BNNs
inside the ODE): it wraps the trained twin, needs only a calibration set of past
residuals, and gives finite-sample marginal coverage on exchangeable data. The
*interesting* question is conditional coverage (per hour-of-day), which the backtest
reports — conformal guarantees the marginal, and we surface where the conditional
guarantee breaks.

ISF/CR *parameter* uncertainty is handled separately (ensemble/Laplace), not here —
conformal gives prediction intervals, not parameter posteriors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _conformal_quantile(abs_residuals: np.ndarray, coverage: float) -> float:
    """Finite-sample-corrected quantile of absolute residuals for split conformal."""
    r = np.asarray(abs_residuals, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return float("nan")
    # Rank for (1 - alpha) coverage with the (n+1) correction.
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(r, level, method="higher"))


@dataclass
class ConformalCalibrator:
    """Per-horizon symmetric conformal intervals at one or more coverage levels."""

    coverage_levels: tuple[float, ...] = (0.5, 0.9)
    # quantiles[horizon_min][coverage] -> half-width in mg/dL
    quantiles: dict[int, dict[float, float]] = field(default_factory=dict)

    def fit(self, residuals_by_horizon: dict[int, np.ndarray]) -> ConformalCalibrator:
        """Calibrate from absolute residuals grouped by horizon (minutes)."""
        self.quantiles = {}
        for h, res in residuals_by_horizon.items():
            self.quantiles[h] = {
                cov: _conformal_quantile(np.abs(res), cov) for cov in self.coverage_levels
            }
        return self

    def half_width(self, horizon_min: int, coverage: float) -> float:
        if horizon_min in self.quantiles and coverage in self.quantiles[horizon_min]:
            return self.quantiles[horizon_min][coverage]
        # Fall back to the nearest calibrated horizon.
        if not self.quantiles:
            return float("nan")
        nearest = min(self.quantiles, key=lambda h: abs(h - horizon_min))
        return self.quantiles[nearest].get(coverage, float("nan"))

    def half_width_at(self, minute: float, coverage: float) -> float:
        """Half-width at an arbitrary horizon (minutes), linearly interpolated across
        calibrated horizons — used to band a full scenario trajectory, not just the
        calibrated grid points."""
        if not self.quantiles:
            return float("nan")
        hs = sorted(self.quantiles)
        ws = [self.quantiles[h].get(coverage, float("nan")) for h in hs]
        return float(np.interp(minute, hs, ws, left=ws[0], right=ws[-1]))

    def interval(self, pred: float, horizon_min: int, coverage: float) -> tuple[float, float]:
        w = self.half_width(horizon_min, coverage)
        return (pred - w, pred + w)

    def covers(self, pred: float, actual: float, horizon_min: int, coverage: float) -> bool:
        lo, hi = self.interval(pred, horizon_min, coverage)
        return bool(lo <= actual <= hi)
