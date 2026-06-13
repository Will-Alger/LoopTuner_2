import numpy as np

from looptuner.eval.baselines import persistence
from looptuner.eval.metrics import rmse
from looptuner.ingest.synthetic import generate_synthetic_dataset
from looptuner.model.kernels import (
    carb_activity,
    exponential_insulin_activity,
)
from looptuner.model.twin import ForwardSimulator


def test_insulin_activity_integrates_to_one():
    # Fine grid integral of the activity curve should be ~1 (a unit fully acts).
    t = np.arange(0, 360, 0.5)
    area = np.trapezoid(exponential_insulin_activity(t, 360, 75), t)
    assert abs(area - 1.0) < 0.02
    # No activity before delivery or after DIA.
    assert exponential_insulin_activity(np.array([-5.0]), 360, 75)[0] == 0.0
    assert exponential_insulin_activity(np.array([400.0]), 360, 75)[0] == 0.0


def test_carb_activity_integrates_to_about_one():
    t = np.arange(0, 1200, 0.5)
    area = np.trapezoid(carb_activity(t, 180), t)
    assert abs(area - 1.0) < 0.05


def test_forward_simulator_trains_and_beats_persistence():
    ds, _ = generate_synthetic_dataset(n_days=10, seed=7)
    sim = ForwardSimulator.from_dataset(ds, seed=0)
    forcing, res = sim.fit(
        ds, val_days=2, epochs=80, train_horizon_min=120, batch_windows=48
    )
    assert res.best_epoch >= 0
    val_codes = set(range(len(forcing.days) - 2, len(forcing.days)))
    metrics = sim.evaluate_horizons(forcing, val_codes, horizons_min=(60, 120))

    # The twin should beat naive persistence at a 2h horizon on well-specified data.
    starts = sim._eligible_starts(forcing, val_codes, 24)
    bg = forcing.bg.cpu().numpy()
    h = 24  # 120 min / 5
    preds_p, actual = [], []
    for s in starts:
        hist = bg[max(0, s - 6) : s + 1]
        hist = hist[np.isfinite(hist)]
        a = bg[min(forcing.n - 1, s + h)]
        if hist.size >= 2 and np.isfinite(a):
            preds_p.append(persistence(hist, h)[-1])
            actual.append(a)
    persist_rmse = rmse(np.array(preds_p), np.array(actual))
    assert metrics[120]["rmse"] < persist_rmse
    assert metrics[120]["mape"] < 15.0


def test_predict_shape_and_save_load(tmp_path):
    ds, _ = generate_synthetic_dataset(n_days=5, seed=1)
    sim = ForwardSimulator.from_dataset(ds, seed=0)
    forcing, _ = sim.fit(ds, val_days=1, epochs=20, batch_windows=32)
    traj = sim.predict(forcing, start=50, horizon_steps=24)
    assert traj.shape == (25,)
    assert np.isfinite(traj).all()

    p = tmp_path / "twin.pt"
    sim.save(str(p))
    sim2 = ForwardSimulator.load(str(p))
    traj2 = sim2.predict(forcing, start=50, horizon_steps=24)
    assert np.allclose(traj, traj2, atol=1e-4)
