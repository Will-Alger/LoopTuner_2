"""Inverse fit: extract per-hour ISF(t) and CR(t) with credible intervals.

The twin already parameterizes ISF(t)/CR(t) as functions inside the model, so a
single fit gives point estimates. The value here is honest *uncertainty*: we train a
small deep ensemble (Phase 1's choice for parameter intervals — not BNNs), each
member holding out a different day, and read the spread of ISF(h)/CR(h) across
members. Where the data identifies a parameter (ISF, given lots of insulin/CGM
response) the ensemble agrees and intervals are tight; where it doesn't (CR, given
this user's sparse announced carbs) the ensemble disagrees and intervals are wide —
which is exactly the truth the user needs before changing a setting.

Nothing here recommends a dose. It surfaces settings with credible intervals; the
user reviews and enters changes manually.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from looptuner.ingest.schema import TidyDataset
from looptuner.model.twin import ForwardSimulator


@dataclass
class InverseResult:
    hours: np.ndarray
    isf_samples: np.ndarray  # (n_models, 24)
    cr_samples: np.ndarray  # (n_models, 24)
    current_isf: np.ndarray  # (24,) from profile
    current_cr: np.ndarray  # (24,)
    carb_support_by_hour: np.ndarray  # grams of announced carbs per hour (CR identifiability)
    coverage_levels: tuple[float, ...]
    n_models: int

    def _ci(self, samples: np.ndarray, coverage: float) -> tuple[np.ndarray, np.ndarray]:
        lo_q = (1 - coverage) / 2 * 100
        hi_q = (1 + coverage) / 2 * 100
        return np.percentile(samples, lo_q, axis=0), np.percentile(samples, hi_q, axis=0)

    def isf_summary(self, coverage: float = 0.9) -> dict[str, np.ndarray]:
        lo, hi = self._ci(self.isf_samples, coverage)
        return {"median": np.median(self.isf_samples, axis=0), "lo": lo, "hi": hi}

    def cr_summary(self, coverage: float = 0.9) -> dict[str, np.ndarray]:
        lo, hi = self._ci(self.cr_samples, coverage)
        return {"median": np.median(self.cr_samples, axis=0), "lo": lo, "hi": hi}


def _carb_support_by_hour(dataset: TidyDataset) -> np.ndarray:
    frame = dataset.frame
    hours = (frame["minute_of_day"].to_numpy() // 60).astype(int) % 24
    grams = frame["carbs"].to_numpy().astype(float)
    out = np.zeros(24)
    for h, g in zip(hours, grams, strict=False):
        out[h] += g
    return out


def run_inverse(
    dataset: TidyDataset,
    n_models: int = 8,
    epochs: int = 200,
    base_seed: int = 0,
    coverage_levels: tuple[float, ...] = (0.5, 0.9),
    device: str = "cpu",
    progress=None,
) -> InverseResult:
    """Train a leave-one-day-out ensemble and extract ISF(h)/CR(h) distributions."""
    forcing_days = sorted(dataset.day_index().unique().tolist())
    n_days = len(forcing_days)
    if n_days < 2:
        raise ValueError("Inverse fit needs at least 2 days of data.")

    prof = dataset.profile
    import pandas as pd

    base = pd.Timestamp("2026-01-01", tz="UTC")
    current_isf = np.array([prof.isf_at(base + pd.Timedelta(hours=h)) for h in range(24)])
    current_cr = np.array([prof.cr_at(base + pd.Timedelta(hours=h)) for h in range(24)])
    isf_level = float(current_isf.mean())
    cr_level = float(current_cr.mean())

    isf_samples, cr_samples = [], []
    for i in range(n_models):
        if progress:
            progress(i, n_models)
        sim = ForwardSimulator.from_dataset(dataset, device=device, seed=base_seed + i)
        held = i % n_days
        train_codes = set(range(n_days)) - {held}
        sim.fit_days(dataset, train_codes, {held}, epochs=epochs)
        isf, cr = sim.model.sens.schedule()
        isf = isf.cpu().numpy()
        cr = cr.cpu().numpy()
        # Absolute ISF/CR level is confounded with EGP and not identifiable from CGM;
        # only the diurnal SHAPE is. Pin each member's 24h-average to the clinically
        # tuned profile level and report the (identifiable) shape with its uncertainty.
        isf_samples.append(isf * (isf_level / isf.mean()))
        cr_samples.append(cr * (cr_level / cr.mean()))

    return InverseResult(
        hours=np.arange(24),
        isf_samples=np.array(isf_samples),
        cr_samples=np.array(cr_samples),
        current_isf=current_isf,
        current_cr=current_cr,
        carb_support_by_hour=_carb_support_by_hour(dataset),
        coverage_levels=coverage_levels,
        n_models=n_models,
    )


@dataclass
class Recommendation:
    hour: int
    param: str  # "ISF" or "CR"
    current: float
    proposed: float
    lo: float
    hi: float
    action: str  # "raise" / "lower" / "keep" / "insufficient-data"
    confidence: str  # "high" / "medium" / "low"


def recommendations(
    result: InverseResult,
    coverage: float = 0.9,
    rel_threshold: float = 0.1,
    min_carb_grams_for_cr: float = 30.0,
) -> list[Recommendation]:
    """Turn the ensemble into per-hour, per-parameter suggestions with intervals.

    A change is suggested only when the credible interval EXCLUDES the current value
    and the median differs by more than ``rel_threshold``. CR is marked
    insufficient-data for hours with little announced carb support.
    """
    recs: list[Recommendation] = []
    isf = result.isf_summary(coverage)
    cr = result.cr_summary(coverage)

    def confidence(lo: float, hi: float, median: float) -> str:
        if median <= 0:
            return "low"
        width = (hi - lo) / median
        return "high" if width < 0.25 else "medium" if width < 0.6 else "low"

    def rnd(v) -> float:
        return round(float(v), 1)

    for h in range(24):
        # ISF
        cur, med = float(result.current_isf[h]), float(isf["median"][h])
        lo, hi = float(isf["lo"][h]), float(isf["hi"][h])
        excludes = cur < lo or cur > hi
        big = abs(med - cur) / max(cur, 1e-6) > rel_threshold
        action = ("raise" if med > cur else "lower") if (excludes and big) else "keep"
        recs.append(
            Recommendation(h, "ISF", rnd(cur), rnd(med), rnd(lo), rnd(hi),
                           action, confidence(lo, hi, med))
        )
        # CR — gated on carb support
        cur_c, med_c = float(result.current_cr[h]), float(cr["median"][h])
        lo_c, hi_c = float(cr["lo"][h]), float(cr["hi"][h])
        if result.carb_support_by_hour[h] < min_carb_grams_for_cr:
            recs.append(
                Recommendation(h, "CR", rnd(cur_c), rnd(med_c), rnd(lo_c),
                               rnd(hi_c), "insufficient-data", "low")
            )
        else:
            excludes_c = cur_c < lo_c or cur_c > hi_c
            big_c = abs(med_c - cur_c) / max(cur_c, 1e-6) > rel_threshold
            act_c = ("raise" if med_c > cur_c else "lower") if (excludes_c and big_c) else "keep"
            recs.append(
                Recommendation(h, "CR", rnd(cur_c), rnd(med_c), rnd(lo_c),
                               rnd(hi_c), act_c, confidence(lo_c, hi_c, med_c))
            )
    return recs
