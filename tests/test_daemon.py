import numpy as np
import pandas as pd

from looptuner.daemon import DaemonPaths, DaemonState, poll_once, run_daemon
from looptuner.ingest.synthetic import generate_synthetic_dataset
from looptuner.model.twin import ForwardSimulator


class FakeNightscout:
    """In-memory Nightscout stand-in: a 5-min CGM series + a flat profile, no network."""

    def __init__(self, base: pd.Timestamp, span_hours: float = 72.0):
        n = int(span_hours * 60 / 5)
        times = [base - pd.Timedelta(minutes=5 * (n - 1 - i)) for i in range(n)]
        # Gentle wave so there's signal; values in a physiological band.
        self.entries = [
            {
                "date": int(t.timestamp() * 1000),
                "sgv": float(120 + 30 * np.sin(i / 12)),
                "type": "sgv",
                "_id": f"e{i}",
            }
            for i, t in enumerate(times)
        ]

    def fetch_entries(self, start, end):
        s, e = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
        return [x for x in self.entries if s <= x["date"] <= e]

    def fetch_treatments(self, start, end):
        return []

    def fetch_profile(self):
        return [{
            "defaultProfile": "D",
            "store": {"D": {
                "timezone": "UTC",
                "sens": [{"timeAsSeconds": 0, "value": 50}],
                "carbratio": [{"timeAsSeconds": 0, "value": 10}],
                "basal": [{"timeAsSeconds": 0, "value": 0.8}],
                "target_low": [{"timeAsSeconds": 0, "value": 100}],
                "target_high": [{"timeAsSeconds": 0, "value": 110}],
            }},
        }]

    def close(self):
        pass


def _sim():
    ds, _ = generate_synthetic_dataset(n_days=4, seed=0)
    sim = ForwardSimulator.from_dataset(ds, seed=0)
    sim.fit(ds, val_days=1, epochs=15)
    return sim


def test_poll_makes_predictions_and_resolves(tmp_path):
    base = pd.Timestamp("2026-03-01 12:00", tz="UTC")
    client = FakeNightscout(base + pd.Timedelta(minutes=60))  # data available up to +60
    sim = _sim()
    paths = DaemonPaths(tmp_path / "data", tmp_path / "runs")

    # Poll 1 at `base`: builds corpus, makes 30/60-min predictions (targets in future).
    s1 = poll_once(client, sim, paths, horizons_min=(30, 60), now=base)
    assert s1["new_entries"] > 0
    assert s1["made"] == 2
    assert s1["resolved"] == 0
    assert paths.dataset.exists() and paths.state.exists()
    assert DaemonState.load(paths.state).cursor_ms > 0  # crash-safe cursor persisted

    # Poll 2 at base+35min: the 30-min target now has an actual CGM -> resolves.
    s2 = poll_once(client, sim, paths, horizons_min=(30, 60), now=base + pd.Timedelta(minutes=35))
    assert s2["resolved"] >= 1
    assert paths.predictions.exists()
    log = pd.read_parquet(paths.predictions)
    assert {"pred", "actual", "abs_err", "horizon_min"} <= set(log.columns)


def test_poll_is_idempotent(tmp_path):
    base = pd.Timestamp("2026-03-01 12:00", tz="UTC")
    client = FakeNightscout(base)
    sim = _sim()
    paths = DaemonPaths(tmp_path / "data", tmp_path / "runs")

    poll_once(client, sim, paths, now=base)
    entries_after_1 = len(pd.read_parquet(paths.entries))
    # Re-poll the same window: dedup by id means the corpus does not grow.
    poll_once(client, sim, paths, now=base)
    entries_after_2 = len(pd.read_parquet(paths.entries))
    assert entries_after_1 == entries_after_2


def test_run_daemon_max_polls(tmp_path):
    base = pd.Timestamp("2026-03-01 12:00", tz="UTC")
    client = FakeNightscout(base)
    sim = _sim()
    paths = DaemonPaths(tmp_path / "data", tmp_path / "runs")
    seen = []
    run_daemon(
        client, sim, paths, interval_min=0.0, max_polls=3,
        on_poll=seen.append, sleep_fn=lambda _s: None,
    )
    assert len(seen) == 3
    assert all("new_entries" in s or "error" in s for s in seen)
