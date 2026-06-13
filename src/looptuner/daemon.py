"""Live polling daemon — collect data and log predicted-vs-actual, no weight updates.

Polls Nightscout every N minutes, pulls new entries/treatments since the last-seen
cursor, appends them to the training corpus (the cached dataset), and tracks how the
twin's live forecasts compare to what actually happened. It NEVER updates model
weights — that's the nightly `train-incremental` job's responsibility. This gives the
"learning as it goes" feel without online-SGD foot-guns.

Crash-safe and idempotent: the cursor + pending predictions persist to disk every
poll, raw pulls are deduped by record id, and resuming re-reads the cursor. Stoppable
via SIGINT/SIGTERM (clean shutdown after the in-flight poll) — no data loss because
state is already on disk.

``poll_once`` is the pure, testable core (inject a fake client + model in tests);
``run_daemon`` wraps it in the sleep/signal loop.
"""

from __future__ import annotations

import json
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from looptuner.ingest.nightscout import assemble_dataset
from looptuner.ingest.schema import GRID_MINUTES
from looptuner.ingest.store import save_dataset
from looptuner.model.twin import ForwardSimulator
from looptuner.scenario import CounterfactualSpec, scenario_forecast

DEFAULT_HORIZONS_MIN = (30, 60)
DEFAULT_BACKFILL_DAYS = 7
GAP_TOLERANCE_MS = int(7.5 * 60 * 1000)


@dataclass
class PendingPrediction:
    made_at_ms: int
    target_ms: int
    horizon_min: int
    pred: float
    lo90: float
    hi90: float


@dataclass
class DaemonState:
    cursor_ms: int = 0  # latest entry timestamp seen
    pending: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> DaemonState:
        if path.exists():
            d = json.loads(path.read_text())
            return cls(cursor_ms=int(d.get("cursor_ms", 0)), pending=d.get("pending", []))
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


class DaemonPaths:
    def __init__(self, data_dir: Path, runs_dir: Path):
        self.dataset = data_dir / "dataset"
        self.daemon_dir = runs_dir / "daemon"
        self.state = self.daemon_dir / "state.json"
        self.entries = self.daemon_dir / "entries.parquet"
        self.treatments = self.daemon_dir / "treatments.parquet"
        self.predictions = runs_dir / "live_predictions.parquet"


def _record_key(rec: dict, fallback_fields: tuple[str, ...]) -> str:
    if rec.get("_id"):
        return str(rec["_id"])
    return "|".join(str(rec.get(f, "")) for f in fallback_fields)


def _append_raw(path: Path, records: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    """Append raw JSON records to a parquet store, deduped by id; return all of them."""
    new = pd.DataFrame(
        [{"key": _record_key(r, key_fields), "raw": json.dumps(r)} for r in records]
    )
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, new], ignore_index=True) if len(new) else old
    else:
        combined = new
    if len(combined):
        combined = combined.drop_duplicates(subset="key", keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path)
    return [json.loads(r) for r in combined["raw"]] if len(combined) else []


def poll_once(
    client,
    sim: ForwardSimulator,
    paths: DaemonPaths,
    horizons_min: tuple[int, ...] = DEFAULT_HORIZONS_MIN,
    conformal=None,
    now: pd.Timestamp | None = None,
    backfill_days: int = DEFAULT_BACKFILL_DAYS,
) -> dict:
    """One poll: pull new data, append to corpus, resolve + make predicted-vs-actual logs."""
    now = pd.Timestamp.now("UTC") if now is None else now
    state = DaemonState.load(paths.state)

    if state.cursor_ms == 0:
        start = now - pd.Timedelta(days=backfill_days)
    else:
        start = pd.Timestamp(state.cursor_ms, unit="ms", tz="UTC")

    new_entries = client.fetch_entries(start, now)
    new_treatments = client.fetch_treatments(start, now)
    profile_docs = client.fetch_profile()

    all_entries = _append_raw(paths.entries, new_entries, ("date",))
    all_treatments = _append_raw(
        paths.treatments, new_treatments, ("created_at", "eventType")
    )

    # Rebuild the training corpus from everything accumulated (idempotent).
    dataset = assemble_dataset(
        entries=all_entries, treatments=all_treatments, profile_docs=profile_docs,
        source="daemon",
    )
    save_dataset(dataset, paths.dataset)

    frame = dataset.frame
    bg = frame["bg"]
    latest_ms = int(frame.index[-1].timestamp() * 1000)

    # Resolve pending predictions whose target time now has an actual CGM value.
    resolved_rows = []
    still_pending = []
    # Epoch-ms per grid point, unit-safe across pandas datetime resolutions.
    bg_ms = ((frame.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy()
    for p in state.pending:
        target = int(p["target_ms"])
        if target > latest_ms:
            still_pending.append(p)
            continue
        idx = int((abs(bg_ms - target)).argmin())
        if abs(bg_ms[idx] - target) <= GAP_TOLERANCE_MS and pd.notna(bg.iloc[idx]):
            actual = float(bg.iloc[idx])
            resolved_rows.append(
                {
                    "made_at": pd.Timestamp(p["made_at_ms"], unit="ms", tz="UTC"),
                    "target": pd.Timestamp(target, unit="ms", tz="UTC"),
                    "horizon_min": p["horizon_min"],
                    "pred": p["pred"],
                    "lo90": p["lo90"],
                    "hi90": p["hi90"],
                    "actual": actual,
                    "abs_err": abs(p["pred"] - actual),
                    "in90": bool(p["lo90"] <= actual <= p["hi90"]),
                }
            )
        # If the target passed with no usable CGM (sensor gap), drop it.

    if resolved_rows:
        df = pd.DataFrame(resolved_rows)
        if paths.predictions.exists():
            df = pd.concat([pd.read_parquet(paths.predictions), df], ignore_index=True)
        paths.predictions.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(paths.predictions)

    # Make fresh forecasts from the current state (no weight update).
    made = 0
    if bg.notna().any():
        res = scenario_forecast(
            sim, dataset, CounterfactualSpec.flat(), conformal=conformal,
            horizon_min=max(horizons_min),
        )
        anchor_ms = int(pd.Timestamp(res["anchor_time"]).timestamp() * 1000)
        for h in horizons_min:
            step = h // GRID_MINUTES
            if step >= len(res["point"]):
                continue
            lo, hi = (float("nan"), float("nan"))
            if res["bands"] and 0.9 in res["bands"]:
                lo = float(res["bands"][0.9]["lo"][step])
                hi = float(res["bands"][0.9]["hi"][step])
            still_pending.append(
                asdict(
                    PendingPrediction(
                        made_at_ms=anchor_ms,
                        target_ms=anchor_ms + h * 60_000,
                        horizon_min=h,
                        pred=float(res["point"][step]),
                        lo90=lo,
                        hi90=hi,
                    )
                )
            )
            made += 1

    state.cursor_ms = max(state.cursor_ms, latest_ms)
    state.pending = still_pending
    state.save(paths.state)

    return {
        "now": str(now),
        "new_entries": len(new_entries),
        "new_treatments": len(new_treatments),
        "corpus_days": round(dataset.coverage_days(), 2),
        "resolved": len(resolved_rows),
        "pending": len(still_pending),
        "made": made,
        "cursor": str(pd.Timestamp(state.cursor_ms, unit="ms", tz="UTC")),
    }


class _Stopper:
    def __init__(self):
        self.stop = False

    def request(self, *_):
        self.stop = True


def run_daemon(
    client,
    sim: ForwardSimulator,
    paths: DaemonPaths,
    interval_min: float = 5.0,
    horizons_min: tuple[int, ...] = DEFAULT_HORIZONS_MIN,
    conformal=None,
    backfill_days: int = DEFAULT_BACKFILL_DAYS,
    max_polls: int | None = None,
    on_poll=None,
    sleep_fn=time.sleep,
) -> None:
    """Run the poll loop until stopped (SIGINT/SIGTERM) or ``max_polls`` reached."""
    stopper = _Stopper()
    try:
        signal.signal(signal.SIGINT, stopper.request)
        signal.signal(signal.SIGTERM, stopper.request)
    except ValueError:
        pass  # not in main thread (e.g. tests) — rely on max_polls

    polls = 0
    while not stopper.stop:
        try:
            summary = poll_once(
                client, sim, paths, horizons_min, conformal=conformal,
                backfill_days=backfill_days,
            )
        except Exception as e:  # one bad poll shouldn't kill the daemon
            summary = {"error": str(e)}
        if on_poll:
            on_poll(summary)
        polls += 1
        if max_polls is not None and polls >= max_polls:
            break
        if stopper.stop:
            break
        sleep_fn(interval_min * 60.0)
