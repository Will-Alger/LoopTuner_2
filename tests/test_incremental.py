import json

from looptuner.incremental import load_registry, train_incremental
from looptuner.ingest.synthetic import generate_synthetic_dataset


def test_incremental_trains_tracks_and_promotes(tmp_path):
    ds, _ = generate_synthetic_dataset(n_days=6, seed=2)
    runs = tmp_path / "runs"

    res1 = train_incremental(ds, runs, epochs=60, horizon_min=60, seed=0)
    # First run: no previous model, so it must be promoted and become current.
    assert res1.promoted
    assert (runs / "twin.pt").exists()
    assert res1.new_checkpoint.endswith(".pt")
    reg = load_registry(runs)
    assert len(reg) == 1
    assert reg[0]["promoted"] is True
    assert reg[0]["data_hash"]

    # Second run: registry grows; a previous score is recorded for comparison.
    res2 = train_incremental(ds, runs, epochs=60, horizon_min=60, seed=1)
    reg = load_registry(runs)
    assert len(reg) == 2
    assert reg[1]["previous_score_mape"] is not None
    # Promotion is gated on beating the previous score on the latest day.
    assert res2.promoted == (res2.new_score <= res2.previous_score)


def test_registry_is_valid_json(tmp_path):
    ds, _ = generate_synthetic_dataset(n_days=5, seed=1)
    runs = tmp_path / "runs"
    train_incremental(ds, runs, epochs=40, seed=0)
    payload = json.loads((runs / "checkpoints" / "registry.json").read_text())
    # n_days counts distinct LOCAL dates, which can exceed the 5 days of coverage when
    # the window straddles a midnight in the local timezone.
    assert isinstance(payload, list)
    assert payload[0]["coverage_days"] == 5.0
    assert payload[0]["n_days"] >= 5
