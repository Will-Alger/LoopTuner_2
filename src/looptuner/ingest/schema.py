"""Canonical, tidy time-series representation of a patient's glucose dynamics.

Everything downstream (forward simulator, backtest, inverse fit) consumes a
``TidyDataset``: a regular 5-minute grid of CGM + insulin + carb + basal signals,
plus the patient's current Loop profile (ISF/CR/basal/target schedules).

Design notes
------------
* Boluses and carb entries are *impulses* (Dirac-like), not smooth signals. We
  keep them as event lists with exact masses AND bin them onto the grid; the
  forward model injects them as discrete state increments at step boundaries so
  the ODE solver never has to integrate through a delta. (This is the single
  most important numerical pitfall flagged in Phase 1.)
* Basal is a piecewise-constant *rate* (U/hr): scheduled basal, overridden by
  temp basals. We resolve it to a net rate per grid bin.
* All timestamps are tz-aware UTC internally. Local time-of-day (for circadian
  features and schedule lookup) is derived via the dataset timezone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd


def safe_zoneinfo(name: str) -> ZoneInfo:
    """Resolve an IANA timezone tolerantly.

    Nightscout/Loop profiles sometimes store non-canonical casing (e.g.
    ``ETC/GMT+4`` instead of ``Etc/GMT+4``). Try the name as-is, then a
    title-cased area, then fall back to UTC.
    """
    for candidate in (name, _retitle_tz(name)):
        try:
            return ZoneInfo(candidate)
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            continue
    return ZoneInfo("UTC")


def _retitle_tz(name: str) -> str:
    # "ETC/GMT+4" -> "Etc/GMT+4", "america/new_york" -> "America/New_York"
    parts = name.split("/")
    fixed = []
    for p in parts:
        if p.upper() in {"GMT", "UTC", "UCT"} or (p and p[0] in "+-") or p[:3].upper() == "GMT":
            fixed.append(p.upper() if p.upper().startswith("GMT") else p)
        else:
            fixed.append("_".join(w.capitalize() for w in p.split("_")))
    return "/".join(fixed)

# Fixed analysis grid. Dexcom G7 reports ~every 5 minutes; everything is resampled
# onto this cadence.
GRID_MINUTES = 5
SECONDS_PER_DAY = 86_400


# --------------------------------------------------------------------------- #
# Events                                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BolusEvent:
    """A discrete insulin bolus (units delivered at an instant)."""

    time: pd.Timestamp
    units: float


@dataclass(frozen=True)
class CarbEvent:
    """A carb entry. ``absorption_minutes`` is the announced/estimated absorption window."""

    time: pd.Timestamp
    grams: float
    absorption_minutes: float = 180.0


@dataclass(frozen=True)
class BasalSegment:
    """A constant basal rate (U/hr) active over [start, end)."""

    start: pd.Timestamp
    end: pd.Timestamp
    rate: float
    is_temp: bool = False


# --------------------------------------------------------------------------- #
# Profile (time-of-day schedules)                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScheduleEntry:
    """One step of a time-of-day schedule: value applies from ``seconds`` until the next entry."""

    seconds: int  # seconds since local midnight
    value: float


def _eval_schedule(entries: tuple[ScheduleEntry, ...], second_of_day: float) -> float:
    """Step-function lookup: value of the last entry whose start <= second_of_day."""
    val = entries[0].value
    for e in entries:
        if e.seconds <= second_of_day:
            val = e.value
        else:
            break
    return val


@dataclass(frozen=True)
class Profile:
    """The patient's *current* Loop settings — the baseline the twin suggests changes to.

    Schedules are tuples of ScheduleEntry sorted by ``seconds``. ISF in mg/dL per U,
    CR in g per U, basal in U/hr, target in mg/dL.
    """

    timezone: str = "UTC"
    dia_hours: float = 6.0
    isf: tuple[ScheduleEntry, ...] = (ScheduleEntry(0, 50.0),)
    cr: tuple[ScheduleEntry, ...] = (ScheduleEntry(0, 10.0),)
    basal: tuple[ScheduleEntry, ...] = (ScheduleEntry(0, 0.8),)
    target: tuple[ScheduleEntry, ...] = (ScheduleEntry(0, 105.0),)

    def _sec_of_day(self, ts: pd.Timestamp) -> float:
        local = ts.tz_convert(safe_zoneinfo(self.timezone))
        return local.hour * 3600 + local.minute * 60 + local.second

    def isf_at(self, ts: pd.Timestamp) -> float:
        return _eval_schedule(self.isf, self._sec_of_day(ts))

    def cr_at(self, ts: pd.Timestamp) -> float:
        return _eval_schedule(self.cr, self._sec_of_day(ts))

    def basal_at(self, ts: pd.Timestamp) -> float:
        return _eval_schedule(self.basal, self._sec_of_day(ts))

    def target_at(self, ts: pd.Timestamp) -> float:
        return _eval_schedule(self.target, self._sec_of_day(ts))


# --------------------------------------------------------------------------- #
# Tidy dataset                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class TidyDataset:
    """Regular-grid signals plus exact event lists and the profile.

    ``frame`` columns (DatetimeIndex, UTC, 5-min freq):
        bg            CGM in mg/dL, NaN where the sensor had a gap.
        bolus         units delivered in this bin (binned from events).
        carbs         grams entered in this bin (binned from events).
        basal_rate    net basal rate (U/hr) active during this bin.
        basal_insulin units delivered via basal in this bin.
        minute_of_day local minute since midnight (for circadian features).
    """

    frame: pd.DataFrame
    profile: Profile
    boluses: list[BolusEvent] = field(default_factory=list)
    carbs: list[CarbEvent] = field(default_factory=list)
    basal_segments: list[BasalSegment] = field(default_factory=list)
    timezone: str = "UTC"
    source: str = "unknown"

    @property
    def n_steps(self) -> int:
        return len(self.frame)

    @property
    def grid_minutes(self) -> int:
        return GRID_MINUTES

    def coverage_days(self) -> float:
        return self.n_steps * GRID_MINUTES / (60 * 24)

    def day_index(self) -> pd.Series:
        """Local calendar date per grid row — used for held-out day splits."""
        local = self.frame.index.tz_convert(safe_zoneinfo(self.timezone))
        return pd.Series(local.date, index=self.frame.index, name="day")

    def summary(self) -> dict:
        bg = self.frame["bg"]
        return {
            "source": self.source,
            "timezone": self.timezone,
            "n_steps": self.n_steps,
            "coverage_days": round(self.coverage_days(), 2),
            "start": str(self.frame.index[0]) if self.n_steps else None,
            "end": str(self.frame.index[-1]) if self.n_steps else None,
            "bg_present_frac": round(float(bg.notna().mean()), 3) if self.n_steps else 0.0,
            "n_boluses": len(self.boluses),
            "n_carb_entries": len(self.carbs),
            "total_carbs_g": round(sum(c.grams for c in self.carbs), 1),
            "total_bolus_u": round(sum(b.units for b in self.boluses), 2),
        }


def _to_utc(ts: datetime | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize(UTC) if t.tzinfo is None else t.tz_convert(UTC)


def _net_basal_rate(grid: pd.DatetimeIndex, segments: list[BasalSegment]) -> np.ndarray:
    """Resolve net basal rate (U/hr) per grid point.

    Scheduled segments form the baseline; temp-basal segments override any
    scheduled rate during their window. Later temp segments win on overlap.
    """
    rate = np.zeros(len(grid), dtype=float)
    ordered = sorted(segments, key=lambda s: (s.is_temp, s.start))
    for seg in ordered:
        s = _to_utc(seg.start)
        e = _to_utc(seg.end)
        mask = np.asarray((grid >= s) & (grid < e))
        rate[mask] = seg.rate
    return rate


def build_tidy_dataset(
    *,
    bg_samples: list[tuple[pd.Timestamp, float]],
    boluses: list[BolusEvent],
    carbs: list[CarbEvent],
    basal_segments: list[BasalSegment],
    profile: Profile,
    timezone_name: str = "UTC",
    source: str = "unknown",
    gap_tolerance_minutes: float = 7.5,
) -> TidyDataset:
    """Assemble raw events into a regular 5-minute grid.

    BG is snapped to the nearest grid point within ``gap_tolerance_minutes``;
    grid points with no nearby reading are NaN (real sensor gaps stay gaps).
    Boluses and carbs are summed into their containing bin.
    """
    if not bg_samples:
        raise ValueError("No CGM samples — cannot build a tidy dataset.")

    bg_times = pd.DatetimeIndex([_to_utc(t) for t, _ in bg_samples])
    bg_vals = np.array([v for _, v in bg_samples], dtype=float)
    order = np.argsort(bg_times.values)
    bg_times, bg_vals = bg_times[order], bg_vals[order]

    all_times = list(bg_times)
    all_times += [_to_utc(b.time) for b in boluses]
    all_times += [_to_utc(c.time) for c in carbs]
    for seg in basal_segments:
        all_times.append(_to_utc(seg.start))
    t0 = min(all_times).floor(f"{GRID_MINUTES}min")
    t1 = max(all_times).ceil(f"{GRID_MINUTES}min")
    grid = pd.date_range(t0, t1, freq=f"{GRID_MINUTES}min", tz="UTC")

    # Snap each BG sample to its single nearest grid point within tolerance.
    # (Map samples->grid, not grid->samples, so one reading can't fill two bins.)
    bg_array = np.full(len(grid), np.nan, dtype=float)
    if len(bg_times):
        idx = grid.get_indexer(bg_times, method="nearest")
        delta = grid[idx] - bg_times  # TimedeltaIndex, unit-safe
        tol = pd.Timedelta(minutes=gap_tolerance_minutes)
        ok = np.asarray((delta >= -tol) & (delta <= tol))
        sums = np.zeros(len(grid), dtype=float)
        counts = np.zeros(len(grid), dtype=float)
        np.add.at(sums, idx[ok], bg_vals[ok])
        np.add.at(counts, idx[ok], 1.0)
        nz = counts > 0
        bg_array[nz] = sums[nz] / counts[nz]
    bg_series = pd.Series(bg_array, index=grid, dtype=float)

    bolus_series = pd.Series(0.0, index=grid, dtype=float)
    for b in boluses:
        idx = _to_utc(b.time).floor(f"{GRID_MINUTES}min")
        if idx in bolus_series.index:
            bolus_series.loc[idx] += b.units

    carb_series = pd.Series(0.0, index=grid, dtype=float)
    for c in carbs:
        idx = _to_utc(c.time).floor(f"{GRID_MINUTES}min")
        if idx in carb_series.index:
            carb_series.loc[idx] += c.grams

    basal_rate = _net_basal_rate(grid, basal_segments)
    basal_insulin = basal_rate * (GRID_MINUTES / 60.0)

    local = grid.tz_convert(safe_zoneinfo(timezone_name))
    minute_of_day = local.hour * 60 + local.minute

    frame = pd.DataFrame(
        {
            "bg": bg_series.to_numpy(),
            "bolus": bolus_series.to_numpy(),
            "carbs": carb_series.to_numpy(),
            "basal_rate": basal_rate,
            "basal_insulin": basal_insulin,
            "minute_of_day": minute_of_day,
        },
        index=grid,
    )

    return TidyDataset(
        frame=frame,
        profile=profile,
        boluses=sorted(boluses, key=lambda b: b.time),
        carbs=sorted(carbs, key=lambda c: c.time),
        basal_segments=sorted(basal_segments, key=lambda s: s.start),
        timezone=timezone_name,
        source=source,
    )
