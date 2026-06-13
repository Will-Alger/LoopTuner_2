"""Nightscout REST API v1 client and parsers.

Pulls CGM (``entries``), treatments (boluses, carbs, temp basals), and the profile,
and maps them into the canonical ``TidyDataset``. The JSON->events parsing is split
out as pure functions so it can be unit-tested with sample payloads without a live
server (you set up ``.env`` with your own URL/token).

Auth: prefer a read token passed as ``?token=...``. Falls back to the legacy
``api-secret`` header (sha1 of the API secret) if no token is given.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd

from looptuner.ingest.schema import (
    BasalSegment,
    BolusEvent,
    CarbEvent,
    Profile,
    ScheduleEntry,
    TidyDataset,
    build_tidy_dataset,
)

MGDL_PER_MMOL = 18.018


# --------------------------------------------------------------------------- #
# Pure parsers (unit-testable)                                                 #
# --------------------------------------------------------------------------- #
def _epoch_ms_to_ts(ms: float) -> pd.Timestamp:
    return pd.Timestamp(datetime.fromtimestamp(ms / 1000.0, tz=UTC))


def parse_entries(entries: list[dict[str, Any]]) -> list[tuple[pd.Timestamp, float]]:
    """Parse ``entries.json`` SGV records into (timestamp, mg/dL) samples.

    ``sgv`` is always mg/dL in Nightscout. Non-SGV records (e.g. calibrations) are
    skipped.
    """
    out: list[tuple[pd.Timestamp, float]] = []
    for e in entries:
        if e.get("type", "sgv") != "sgv":
            continue
        sgv = e.get("sgv")
        date = e.get("date")
        if sgv is None or date is None:
            continue
        out.append((_epoch_ms_to_ts(float(date)), float(sgv)))
    return out


def _treatment_ts(t: dict[str, Any]) -> pd.Timestamp | None:
    if "created_at" in t:
        return pd.Timestamp(t["created_at"]).tz_convert("UTC") if pd.Timestamp(
            t["created_at"]
        ).tzinfo else pd.Timestamp(t["created_at"]).tz_localize("UTC")
    if "date" in t:
        return _epoch_ms_to_ts(float(t["date"]))
    return None


def parse_treatments(
    treatments: list[dict[str, Any]],
) -> tuple[list[BolusEvent], list[CarbEvent], list[BasalSegment]]:
    """Split treatments into boluses, carb entries, and temp-basal segments."""
    boluses: list[BolusEvent] = []
    carbs: list[CarbEvent] = []
    temp_basals: list[BasalSegment] = []

    for t in treatments:
        ts = _treatment_ts(t)
        if ts is None:
            continue
        insulin = t.get("insulin")
        if insulin:
            boluses.append(BolusEvent(time=ts, units=float(insulin)))
        grams = t.get("carbs")
        if grams:
            absorb = float(t.get("absorptionTime", 180.0) or 180.0)
            carbs.append(CarbEvent(time=ts, grams=float(grams), absorption_minutes=absorb))
        if t.get("eventType") == "Temp Basal" and t.get("rate") is not None:
            dur = float(t.get("duration", 30.0) or 30.0)
            temp_basals.append(
                BasalSegment(
                    start=ts,
                    end=ts + pd.Timedelta(minutes=dur),
                    rate=float(t["rate"]),
                    is_temp=True,
                )
            )
    return boluses, carbs, temp_basals


def parse_profile(profile_docs: list[dict[str, Any]]) -> Profile:
    """Parse ``profile.json`` into a ``Profile`` (ISF/CR/basal/target schedules)."""
    if not profile_docs:
        return Profile()
    doc = profile_docs[0]
    store = doc.get("store", {})
    default_name = doc.get("defaultProfile")
    prof = store.get(default_name) if default_name in store else next(iter(store.values()), {})
    if not prof:
        return Profile()

    tz = prof.get("timezone", "UTC")
    units = (prof.get("units") or "mg/dl").lower()
    to_mgdl = MGDL_PER_MMOL if units.startswith("mmol") else 1.0

    def sched(key: str, scale: float = 1.0) -> tuple[ScheduleEntry, ...]:
        items = prof.get(key, []) or []
        entries = []
        for it in items:
            secs = int(it.get("timeAsSeconds", 0))
            entries.append(ScheduleEntry(secs, float(it["value"]) * scale))
        entries.sort(key=lambda e: e.seconds)
        return tuple(entries) if entries else (ScheduleEntry(0, 0.0),)

    isf = sched("sens", scale=to_mgdl)
    cr = sched("carbratio")  # g/U, unit-independent
    basal = sched("basal")
    target_low = sched("target_low", scale=to_mgdl)
    target_high = sched("target_high", scale=to_mgdl)
    # Represent target as the midpoint of the low/high schedule (by low's breakpoints).
    target = tuple(
        ScheduleEntry(lo.seconds, (lo.value + hi.value) / 2.0)
        for lo, hi in zip(target_low, target_high, strict=False)
    ) or (ScheduleEntry(0, 105.0),)

    return Profile(
        timezone=tz,
        dia_hours=float(prof.get("dia", 6.0)),
        isf=isf,
        cr=cr,
        basal=basal,
        target=target,
    )


def expand_scheduled_basal(
    profile: Profile, start: pd.Timestamp, end: pd.Timestamp
) -> list[BasalSegment]:
    """Tile the profile's basal schedule into concrete segments over [start, end]."""
    from looptuner.ingest.schema import safe_zoneinfo

    tz = safe_zoneinfo(profile.timezone)
    segments: list[BasalSegment] = []
    day = start.tz_convert(tz).normalize()
    last = end.tz_convert(tz)
    while day <= last:
        for i, entry in enumerate(profile.basal):
            seg_start = day + pd.Timedelta(seconds=entry.seconds)
            if i + 1 < len(profile.basal):
                seg_end = day + pd.Timedelta(seconds=profile.basal[i + 1].seconds)
            else:
                seg_end = day + pd.Timedelta(days=1)
            segments.append(
                BasalSegment(
                    start=seg_start.tz_convert("UTC"),
                    end=seg_end.tz_convert("UTC"),
                    rate=entry.value,
                    is_temp=False,
                )
            )
        day = day + pd.Timedelta(days=1)
    return segments


def assemble_dataset(
    *,
    entries: list[dict[str, Any]],
    treatments: list[dict[str, Any]],
    profile_docs: list[dict[str, Any]],
    source: str = "nightscout",
) -> TidyDataset:
    """Parse raw Nightscout JSON payloads into a ``TidyDataset``."""
    bg_samples = parse_entries(entries)
    if not bg_samples:
        raise ValueError("No SGV entries returned from Nightscout.")
    boluses, carbs, temp_basals = parse_treatments(treatments)
    profile = parse_profile(profile_docs)

    times = [t for t, _ in bg_samples]
    scheduled = expand_scheduled_basal(profile, min(times), max(times))
    basal_segments = scheduled + temp_basals

    return build_tidy_dataset(
        bg_samples=bg_samples,
        boluses=boluses,
        carbs=carbs,
        basal_segments=basal_segments,
        profile=profile,
        timezone_name=profile.timezone,
        source=source,
    )


# --------------------------------------------------------------------------- #
# HTTP client                                                                  #
# --------------------------------------------------------------------------- #
class NightscoutClient:
    """Thin REST v1 client. Network use requires a real URL + token in ``.env``."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        api_secret: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token or None
        self._headers = {}
        if not self.token and api_secret:
            self._headers["api-secret"] = hashlib.sha1(api_secret.encode()).hexdigest()
        self._client = httpx.Client(timeout=timeout, headers=self._headers)

    def _params(self, extra: dict[str, Any]) -> dict[str, Any]:
        p = dict(extra)
        if self.token:
            p["token"] = self.token
        return p

    def _get(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        url = f"{self.base_url}/api/v1/{path}"
        resp = self._client.get(url, params=self._params(params))
        resp.raise_for_status()
        return resp.json()

    def fetch_entries(
        self, start: pd.Timestamp, end: pd.Timestamp, page: int = 50_000
    ) -> list[dict[str, Any]]:
        s_ms = int(start.timestamp() * 1000)
        e_ms = int(end.timestamp() * 1000)
        return self._get(
            "entries.json",
            {
                "find[date][$gte]": s_ms,
                "find[date][$lte]": e_ms,
                "count": page,
            },
        )

    def fetch_treatments(
        self, start: pd.Timestamp, end: pd.Timestamp, page: int = 50_000
    ) -> list[dict[str, Any]]:
        return self._get(
            "treatments.json",
            {
                "find[created_at][$gte]": start.isoformat(),
                "find[created_at][$lte]": end.isoformat(),
                "count": page,
            },
        )

    def fetch_profile(self) -> list[dict[str, Any]]:
        return self._get("profile.json", {})

    def fetch_dataset(self, start: pd.Timestamp, end: pd.Timestamp) -> TidyDataset:
        entries = self.fetch_entries(start, end)
        treatments = self.fetch_treatments(start, end)
        profile_docs = self.fetch_profile()
        return assemble_dataset(
            entries=entries,
            treatments=treatments,
            profile_docs=profile_docs,
            source=f"nightscout({self.base_url})",
        )

    def close(self) -> None:
        self._client.close()
