"""Drift monitor: track the current twin's predicted-vs-actual error over time.

A health-check that runs the *saved* model (no retraining) forward over recent days
and reports per-hour-of-day error, flagging hours where accuracy has suddenly
worsened. A sudden jump means either the twin needs retraining or your physiology
actually changed (new infusion site, illness, stress) — both worth knowing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from looptuner.backtest.engine import BacktestArrays
from looptuner.ingest.schema import GRID_MINUTES, TidyDataset
from looptuner.model.twin import ForwardSimulator


@dataclass
class DriftResult:
    horizon_min: int
    days: list[str]
    per_hour_mape: np.ndarray  # (24,) MAPE over the whole window
    recent_hour_mape: np.ndarray  # (24,) MAPE over the most recent day
    baseline_hour_mape: np.ndarray  # (24,) MAPE over the earlier days
    flags: list[dict]  # hours whose recent error jumped vs baseline
    n_predictions: int


def compute_drift(
    dataset: TidyDataset,
    sim: ForwardSimulator,
    horizon_min: int = 60,
    days: int = 7,
    jump_ratio: float = 1.5,
    jump_abs_pp: float = 8.0,
    anchor_stride: int = 2,
) -> DriftResult:
    """Run the current model over the last ``days`` days and detect per-hour drift.

    A flag fires when the most recent day's MAPE for an hour exceeds the earlier-days
    baseline by both a relative factor (``jump_ratio``) and an absolute margin
    (``jump_abs_pp`` percentage points) — so noise alone won't trip it.
    """
    arr = BacktestArrays.from_dataset(dataset)
    h = horizon_min // GRID_MINUTES
    all_days = arr.days
    window_days = all_days[-days:]
    window_codes = [all_days.index(d) for d in window_days]
    recent_code = window_codes[-1]

    rows = []
    anchors = np.where(np.isin(arr.day_codes, window_codes) & np.isfinite(arr.bg))[0]
    for a in anchors[::anchor_stride]:
        if a + h >= arr.n:
            continue
        actual = arr.bg[a + h]
        if not np.isfinite(actual):
            continue
        i_win, c_win = arr.anchored_window(a, h)
        traj = sim.roll(i_win, c_win, arr.minute_of_day[a], arr.bg[a])
        pct = abs(traj[h] - actual) / max(1.0, abs(actual)) * 100.0
        rows.append(
            {"code": int(arr.day_codes[a]), "hour": int(arr.minute_of_day[a] // 60), "pct": pct}
        )

    df = pd.DataFrame(rows)
    per_hour = np.full(24, np.nan)
    recent = np.full(24, np.nan)
    baseline = np.full(24, np.nan)
    if not df.empty:
        for hr, g in df.groupby("hour"):
            per_hour[hr] = g["pct"].mean()
            recent[hr] = g[g["code"] == recent_code]["pct"].mean()
            baseline[hr] = g[g["code"] != recent_code]["pct"].mean()

    flags = []
    for hr in range(24):
        r, b = recent[hr], baseline[hr]
        if np.isfinite(r) and np.isfinite(b) and r > b * jump_ratio and r - b > jump_abs_pp:
            flags.append(
                {"hour": hr, "recent_mape": round(float(r), 1), "baseline_mape": round(float(b), 1)}
            )

    return DriftResult(
        horizon_min=horizon_min,
        days=[str(d) for d in window_days],
        per_hour_mape=per_hour,
        recent_hour_mape=recent,
        baseline_hour_mape=baseline,
        flags=flags,
        n_predictions=len(df),
    )


def render_drift_markdown(res: DriftResult) -> str:
    parts = [
        f"# Drift report — {res.horizon_min}min predictions",
        "",
        f"- Window: {res.days[0]} .. {res.days[-1]} ({len(res.days)} days, "
        f"{res.n_predictions} predictions)",
        "",
    ]
    if res.flags:
        parts.append("## ⚠ Hours with a sudden accuracy drop (retrain or physiology change?)")
        parts.append("")
        parts.append("| Hour | Recent MAPE% | Baseline MAPE% |")
        parts.append("|---|---|---|")
        for f in res.flags:
            parts.append(f"| {f['hour']:02d}:00 | {f['recent_mape']} | {f['baseline_mape']} |")
    else:
        parts.append("No sudden per-hour accuracy drops detected — the twin is tracking.")
    parts += [
        "",
        "## Per-hour error (whole window)",
        "",
        "| Hour | MAPE% | Recent | Baseline |",
        "|---|---|---|---|",
    ]
    for hr in range(24):
        if np.isfinite(res.per_hour_mape[hr]):
            rec = (
                f"{res.recent_hour_mape[hr]:.1f}"
                if np.isfinite(res.recent_hour_mape[hr])
                else "—"
            )
            bas = (
                f"{res.baseline_hour_mape[hr]:.1f}"
                if np.isfinite(res.baseline_hour_mape[hr])
                else "—"
            )
            parts.append(f"| {hr:02d}:00 | {res.per_hour_mape[hr]:.1f} | {rec} | {bas} |")
    return "\n".join(parts)
