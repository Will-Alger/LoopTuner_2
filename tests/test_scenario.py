import numpy as np
import pytest

from looptuner.ingest.synthetic import generate_synthetic_dataset
from looptuner.model.twin import ForwardSimulator
from looptuner.scenario import (
    CounterfactualSpec,
    calibrate_conformal,
    counterfactual_replay,
    scenario_forecast,
    summarize_forecast,
)


@pytest.fixture(scope="module")
def trained():
    ds, _ = generate_synthetic_dataset(n_days=8, seed=2)
    sim = ForwardSimulator.from_dataset(ds, seed=0)
    sim.fit(ds, val_days=1, epochs=100)
    return sim, ds


def test_forecast_length_and_future_padding(trained):
    sim, ds = trained
    res = scenario_forecast(sim, ds, CounterfactualSpec.flat(), horizon_min=360)
    assert len(res["point"]) == 360 // 5 + 1
    assert np.isfinite(res["point"]).all()


def test_added_bolus_lowers_and_carbs_raise(trained):
    sim, ds = trained
    base = scenario_forecast(sim, ds, CounterfactualSpec.flat(), horizon_min=240)
    bolus = scenario_forecast(
        sim, ds, CounterfactualSpec(added_boluses=[(0.0, 4.0)]), horizon_min=240
    )
    meal = scenario_forecast(
        sim, ds, CounterfactualSpec(added_carbs=[(0.0, 50.0, 180.0)]), horizon_min=240
    )
    assert bolus["point"][-1] < base["point"][-1] - 5
    assert meal["point"][-1] > base["point"][-1] + 5


def test_conformal_bands_present_and_ordered(trained):
    sim, ds = trained
    conf = calibrate_conformal(sim, ds)
    res = scenario_forecast(sim, ds, CounterfactualSpec.flat(), conformal=conf, horizon_min=180)
    assert 0.9 in res["bands"]
    lo, hi = res["bands"][0.9]["lo"], res["bands"][0.9]["hi"]
    assert np.all(hi >= res["point"]) and np.all(lo <= res["point"])
    # 90% band should be wider than 50% band.
    w90 = (res["bands"][0.9]["hi"] - res["bands"][0.9]["lo"]).mean()
    w50 = (res["bands"][0.5]["hi"] - res["bands"][0.5]["lo"]).mean()
    assert w90 > w50


def test_counterfactual_isf_direction(trained):
    sim, ds = trained
    base = counterfactual_replay(sim, ds, CounterfactualSpec.flat())
    more = counterfactual_replay(sim, ds, CounterfactualSpec().with_isf_scale(0, 24, 1.3))
    assert len(base["actual"]) == len(base["counterfactual"])
    # Higher ISF (more sensitive) lowers BG given the same insulin.
    assert np.nanmean(more["counterfactual"]) < np.nanmean(base["counterfactual"])


def test_summarize_forecast_keys(trained):
    sim, ds = trained
    res = scenario_forecast(sim, ds, CounterfactualSpec.flat(), horizon_min=120)
    s = summarize_forecast(res)
    for key in ("anchor_bg", "end_bg", "min_bg", "max_bg", "frac_below_70"):
        assert key in s
