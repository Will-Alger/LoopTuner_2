"""Incremental ('nightly') retraining with checkpoint versioning.

Retrains on the full accumulated history, validates on the most recent day, and only
*promotes* the new model to "current" if it scores at least as well as the existing
current model on that held-out day — otherwise the new checkpoint is kept alongside
but the previous model stays current. Every checkpoint is tracked with its validation
score, data hash, and timestamp so quality can be traced over time.

This is the safe way to "learn as it goes": no online SGD foot-guns, just a gated,
reproducible retrain you schedule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from looptuner.backtest.engine import BacktestArrays
from looptuner.config import dataframe_hash
from looptuner.ingest.schema import GRID_MINUTES, TidyDataset
from looptuner.model.twin import ForwardSimulator


def score_on_day(
    sim: ForwardSimulator, arr: BacktestArrays, day_code: int, horizon_min: int = 60
) -> float:
    """Anchored (no-leakage) MAPE of ``sim`` on one day at ``horizon_min``."""
    h = horizon_min // GRID_MINUTES
    anchors = np.where((arr.day_codes == day_code) & np.isfinite(arr.bg))[0]
    errs = []
    for a in anchors:
        if a + h >= arr.n:
            continue
        actual = arr.bg[a + h]
        if not np.isfinite(actual):
            continue
        i_win, c_win = arr.anchored_window(a, h)
        traj = sim.roll(i_win, c_win, arr.minute_of_day[a], arr.bg[a])
        errs.append(abs(traj[h] - actual) / max(1.0, abs(actual)) * 100.0)
    return float(np.mean(errs)) if errs else float("nan")


@dataclass
class IncrementalResult:
    new_checkpoint: str
    promoted: bool
    new_score: float
    previous_score: float
    horizon_min: int
    data_hash: str


def train_incremental(
    dataset: TidyDataset,
    runs_dir: str | Path,
    epochs: int = 300,
    horizon_min: int = 60,
    device: str = "cpu",
    seed: int = 0,
) -> IncrementalResult:
    """Retrain on full history, validate on the latest day, gate promotion on it."""
    runs_dir = Path(runs_dir)
    ckpt_dir = runs_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    current_path = runs_dir / "twin.pt"
    registry_path = ckpt_dir / "registry.json"

    arr = BacktestArrays.from_dataset(dataset)
    n_days = len(arr.days)
    if n_days < 2:
        raise ValueError("Incremental training needs at least 2 days.")
    latest_code = n_days - 1

    # Train a fresh model on all but the latest day; validate on the latest.
    new_sim = ForwardSimulator.from_dataset(dataset, device=device, seed=seed)
    new_sim.fit_days(dataset, set(range(n_days - 1)), {latest_code}, epochs=epochs)
    new_score = score_on_day(new_sim, arr, latest_code, horizon_min)

    prev_score = float("nan")
    if current_path.exists():
        cur_sim = ForwardSimulator.load(str(current_path), device=device)
        prev_score = score_on_day(cur_sim, arr, latest_code, horizon_min)

    ts = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S")
    new_ckpt = ckpt_dir / f"twin_{ts}.pt"
    new_sim.save(str(new_ckpt))

    # Promote if there's no current model or the new one is at least as good.
    promote = np.isnan(prev_score) or (new_score <= prev_score)
    if promote:
        new_sim.save(str(current_path))

    entry = {
        "timestamp": ts,
        "checkpoint": str(new_ckpt),
        "val_score_mape": round(new_score, 2),
        "previous_score_mape": None if np.isnan(prev_score) else round(prev_score, 2),
        "promoted": bool(promote),
        "horizon_min": horizon_min,
        "coverage_days": round(dataset.coverage_days(), 2),
        "data_hash": dataframe_hash(dataset.frame),
        "n_days": n_days,
    }
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append(entry)
    registry_path.write_text(json.dumps(registry, indent=2))

    return IncrementalResult(
        new_checkpoint=str(new_ckpt),
        promoted=bool(promote),
        new_score=new_score,
        previous_score=prev_score,
        horizon_min=horizon_min,
        data_hash=entry["data_hash"],
    )


def load_registry(runs_dir: str | Path) -> list[dict]:
    path = Path(runs_dir) / "checkpoints" / "registry.json"
    return json.loads(path.read_text()) if path.exists() else []
