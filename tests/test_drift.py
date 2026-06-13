import numpy as np

from looptuner.drift import compute_drift, render_drift_markdown
from looptuner.ingest.synthetic import generate_synthetic_dataset
from looptuner.model.twin import ForwardSimulator


def test_drift_report_shapes_and_render():
    ds, _ = generate_synthetic_dataset(n_days=7, seed=3)
    sim = ForwardSimulator.from_dataset(ds, seed=0)
    sim.fit(ds, val_days=1, epochs=60)
    res = compute_drift(ds, sim, horizon_min=60, days=6, anchor_stride=3)
    assert res.per_hour_mape.shape == (24,)
    assert res.n_predictions > 0
    assert isinstance(res.flags, list)
    md = render_drift_markdown(res)
    assert "Drift report" in md
    assert "Per-hour error" in md


def test_drift_flag_fires_on_injected_degradation():
    ds, _ = generate_synthetic_dataset(n_days=8, seed=5)
    sim = ForwardSimulator.from_dataset(ds, seed=0)
    sim.fit(ds, val_days=1, epochs=60)

    # Corrupt the most recent day's CGM so the model's error there spikes.
    last_day = ds.day_index().unique()[-1]
    mask = (ds.day_index() == last_day).to_numpy()
    ds.frame.loc[mask, "bg"] = np.clip(ds.frame.loc[mask, "bg"] + 120.0, 40, 400)

    res = compute_drift(ds, sim, horizon_min=60, days=8, anchor_stride=2)
    assert len(res.flags) > 0  # at least one hour should flag the degradation
