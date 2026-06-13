"""Persist and reload a TidyDataset (parquet frame + JSON sidecar).

Keeps pulls reproducible and inspectable. The frame is parquet; profile, events,
and basal segments live in a JSON sidecar so the exact inputs can be reconstructed
without re-hitting Nightscout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from looptuner.ingest.schema import (
    BasalSegment,
    BolusEvent,
    CarbEvent,
    Profile,
    ScheduleEntry,
    TidyDataset,
)


def _profile_to_dict(p: Profile) -> dict:
    def sched(s):
        return [[e.seconds, e.value] for e in s]

    return {
        "timezone": p.timezone,
        "dia_hours": p.dia_hours,
        "isf": sched(p.isf),
        "cr": sched(p.cr),
        "basal": sched(p.basal),
        "target": sched(p.target),
    }


def _profile_from_dict(d: dict) -> Profile:
    def sched(items):
        return tuple(ScheduleEntry(int(s), float(v)) for s, v in items)

    return Profile(
        timezone=d["timezone"],
        dia_hours=d["dia_hours"],
        isf=sched(d["isf"]),
        cr=sched(d["cr"]),
        basal=sched(d["basal"]),
        target=sched(d["target"]),
    )


def save_dataset(ds: TidyDataset, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ds.frame.to_parquet(directory / "frame.parquet")
    meta = {
        "source": ds.source,
        "timezone": ds.timezone,
        "profile": _profile_to_dict(ds.profile),
        "boluses": [[b.time.isoformat(), b.units] for b in ds.boluses],
        "carbs": [[c.time.isoformat(), c.grams, c.absorption_minutes] for c in ds.carbs],
        "basal_segments": [
            [s.start.isoformat(), s.end.isoformat(), s.rate, s.is_temp]
            for s in ds.basal_segments
        ],
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))
    return directory


def load_dataset(directory: str | Path) -> TidyDataset:
    directory = Path(directory)
    frame = pd.read_parquet(directory / "frame.parquet")
    meta = json.loads((directory / "meta.json").read_text())
    boluses = [BolusEvent(pd.Timestamp(t), u) for t, u in meta["boluses"]]
    carbs = [CarbEvent(pd.Timestamp(t), g, a) for t, g, a in meta["carbs"]]
    basal = [
        BasalSegment(pd.Timestamp(s), pd.Timestamp(e), r, bool(temp))
        for s, e, r, temp in meta["basal_segments"]
    ]
    return TidyDataset(
        frame=frame,
        profile=_profile_from_dict(meta["profile"]),
        boluses=boluses,
        carbs=carbs,
        basal_segments=basal,
        timezone=meta["timezone"],
        source=meta["source"],
    )
