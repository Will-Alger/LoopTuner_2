import numpy as np

from looptuner.ingest.store import load_dataset, save_dataset
from looptuner.ingest.synthetic import generate_synthetic_dataset


def test_dataset_roundtrip(tmp_path):
    ds, _ = generate_synthetic_dataset(n_days=4, seed=2)
    save_dataset(ds, tmp_path / "ds")
    loaded = load_dataset(tmp_path / "ds")

    assert loaded.n_steps == ds.n_steps
    assert loaded.timezone == ds.timezone
    assert len(loaded.boluses) == len(ds.boluses)
    assert len(loaded.carbs) == len(ds.carbs)
    np.testing.assert_allclose(
        loaded.frame["bg"].fillna(-1).to_numpy(), ds.frame["bg"].fillna(-1).to_numpy()
    )
    np.testing.assert_allclose(
        loaded.frame["basal_rate"].to_numpy(), ds.frame["basal_rate"].to_numpy()
    )
    # Profile schedules preserved.
    assert loaded.profile.isf[5].value == ds.profile.isf[5].value
    assert loaded.profile.cr[12].value == ds.profile.cr[12].value
