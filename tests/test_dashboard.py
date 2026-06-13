"""Smoke test: the Streamlit dashboard runs without raising, on a tiny dataset+model."""

import importlib.util
import os

import pytest

from looptuner.ingest.store import save_dataset
from looptuner.ingest.synthetic import generate_synthetic_dataset
from looptuner.model.twin import ForwardSimulator

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs"
    (data_dir).mkdir()
    (runs_dir).mkdir()
    # Settings.load() reads these; set before it runs (load_dotenv won't override).
    monkeypatch.setenv("LOOPTUNER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("LOOPTUNER_RUNS_DIR", str(runs_dir))

    ds, _ = generate_synthetic_dataset(n_days=4, seed=0)
    save_dataset(ds, data_dir / "dataset")
    sim = ForwardSimulator.from_dataset(ds, seed=0)
    sim.fit(ds, val_days=1, epochs=20)
    sim.save(str(runs_dir / "twin.pt"))
    return tmp_path


def test_dashboard_runs_without_exception(app_env):
    script = importlib.util.find_spec("looptuner.dashboard").origin
    assert os.path.exists(script)
    at = AppTest.from_file(script, default_timeout=120)
    at.run()
    assert not at.exception
    # The title renders and at least the scenario metrics are present.
    assert any("LoopTuner" in str(t.value) for t in at.title)
    assert len(at.tabs) == 5
