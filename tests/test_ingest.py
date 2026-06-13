import numpy as np
import pandas as pd

from looptuner.ingest.nightscout import (
    assemble_dataset,
    parse_entries,
    parse_profile,
    parse_treatments,
)
from looptuner.ingest.schema import GRID_MINUTES
from looptuner.ingest.synthetic import generate_synthetic_dataset


def test_parse_entries_filters_non_sgv():
    entries = [
        {"type": "sgv", "sgv": 120, "date": 1_700_000_000_000},
        {"type": "cal", "date": 1_700_000_300_000},
        {"sgv": 95, "date": 1_700_000_600_000},  # type defaults to sgv
    ]
    out = parse_entries(entries)
    assert len(out) == 2
    assert out[0][1] == 120.0
    assert isinstance(out[0][0], pd.Timestamp)


def test_parse_treatments_splits_types():
    treatments = [
        {"eventType": "Bolus", "insulin": 2.5, "created_at": "2026-01-01T12:00:00Z"},
        {"eventType": "Meal Bolus", "carbs": 40, "insulin": 4.0,
         "absorptionTime": 240, "created_at": "2026-01-01T12:30:00Z"},
        {"eventType": "Temp Basal", "rate": 0.0, "duration": 30,
         "created_at": "2026-01-01T13:00:00Z"},
    ]
    boluses, carbs, temps = parse_treatments(treatments)
    assert len(boluses) == 2  # the meal bolus also carries insulin
    assert len(carbs) == 1
    assert carbs[0].grams == 40.0
    assert carbs[0].absorption_minutes == 240.0
    assert len(temps) == 1
    assert temps[0].rate == 0.0
    assert (temps[0].end - temps[0].start) == pd.Timedelta(minutes=30)


def test_parse_profile_mgdl():
    docs = [{
        "defaultProfile": "Default",
        "store": {"Default": {
            "timezone": "America/New_York",
            "units": "mg/dl",
            "dia": 6,
            "sens": [{"timeAsSeconds": 0, "value": 50}],
            "carbratio": [{"timeAsSeconds": 0, "value": 10}],
            "basal": [{"timeAsSeconds": 0, "value": 0.8}],
            "target_low": [{"timeAsSeconds": 0, "value": 100}],
            "target_high": [{"timeAsSeconds": 0, "value": 110}],
        }},
    }]
    prof = parse_profile(docs)
    assert prof.timezone == "America/New_York"
    assert prof.isf[0].value == 50.0
    assert prof.cr[0].value == 10.0
    assert prof.target[0].value == 105.0


def test_parse_profile_mmol_converts_isf():
    docs = [{
        "defaultProfile": "D",
        "store": {"D": {
            "units": "mmol",
            "sens": [{"timeAsSeconds": 0, "value": 3.0}],
            "carbratio": [{"timeAsSeconds": 0, "value": 10}],
            "basal": [{"timeAsSeconds": 0, "value": 0.8}],
            "target_low": [{"timeAsSeconds": 0, "value": 5.0}],
            "target_high": [{"timeAsSeconds": 0, "value": 6.0}],
        }},
    }]
    prof = parse_profile(docs)
    assert abs(prof.isf[0].value - 3.0 * 18.018) < 1e-6


def test_assemble_dataset_end_to_end():
    entries = [
        {"type": "sgv", "sgv": 120, "date": 1_700_000_000_000},
        {"type": "sgv", "sgv": 130, "date": 1_700_000_300_000},
        {"type": "sgv", "sgv": 140, "date": 1_700_000_600_000},
    ]
    treatments = [
        {"eventType": "Meal Bolus", "carbs": 30, "insulin": 3.0,
         "created_at": "2023-11-14T22:13:20Z"},
    ]
    profile_docs = [{
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
    ds = assemble_dataset(entries=entries, treatments=treatments, profile_docs=profile_docs)
    assert ds.n_steps >= 3
    assert ds.frame["bg"].notna().sum() == 3
    assert ds.frame["carbs"].sum() == 30.0
    assert ds.frame["basal_rate"].max() > 0


def test_synthetic_dataset_shapes_and_coverage():
    ds, truth = generate_synthetic_dataset(n_days=7, seed=1)
    assert abs(ds.coverage_days() - 7.0) < 0.1
    assert ds.frame.index.freq == pd.tseries.frequencies.to_offset(f"{GRID_MINUTES}min")
    # Most of the grid should have BG (a few engineered gaps aside).
    assert ds.frame["bg"].notna().mean() > 0.9
    # Ground truth ISF shows a dawn dip (lower around 5am than midday).
    assert truth.isf_at_hour(5) < truth.isf_at_hour(13)
    assert len(ds.boluses) > 10
    assert len(ds.carbs) > 10
    # Latent BG stays in a physiological band.
    assert truth.noiseless_bg.between(40, 400).all()


def test_synthetic_is_deterministic():
    a, _ = generate_synthetic_dataset(n_days=3, seed=42)
    b, _ = generate_synthetic_dataset(n_days=3, seed=42)
    assert np.allclose(a.frame["bg"].fillna(-1), b.frame["bg"].fillna(-1))
