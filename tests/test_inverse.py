import json

import numpy as np

from looptuner.ingest.synthetic import generate_synthetic_dataset
from looptuner.model.inverse import recommendations, run_inverse
from looptuner.recommend_report import render_inverse_markdown, write_inverse_json


def test_inverse_anchors_level_and_recovers_shape():
    ds, truth = generate_synthetic_dataset(n_days=9, seed=2)
    result = run_inverse(ds, n_models=4, epochs=70, base_seed=0)
    isf = result.isf_summary(0.9)

    # Level is pinned to the clinical profile's 24h average (identifiability fix).
    assert abs(isf["median"].mean() - result.current_isf.mean()) < 1.0
    # Shape recovers the dawn dip: more resistant (lower ISF) around 3-6am than midday.
    assert isf["median"][3:7].mean() < isf["median"][11:15].mean()
    # Positive correlation with the true ISF shape.
    corr = np.corrcoef(isf["median"], truth.isf_by_hour)[0, 1]
    assert corr > 0.3


def test_recommendations_structure_and_cr_gating():
    ds, _ = generate_synthetic_dataset(n_days=6, seed=1)
    result = run_inverse(ds, n_models=3, epochs=50, base_seed=0)
    recs = recommendations(result, coverage=0.9)
    assert len(recs) == 48  # 24 hours x {ISF, CR}
    # Hours with no announced carbs must be flagged insufficient-data for CR.
    for r in recs:
        if r.param == "CR" and result.carb_support_by_hour[r.hour] < 30.0:
            assert r.action == "insufficient-data"
    # All recommendation numbers are plain Python floats (clean JSON/printing).
    assert all(isinstance(r.proposed, float) for r in recs)


def test_inverse_report_and_json(tmp_path):
    ds, _ = generate_synthetic_dataset(n_days=6, seed=3)
    result = run_inverse(ds, n_models=3, epochs=40, base_seed=0)
    recs = recommendations(result)
    md = render_inverse_markdown(result, recs)
    assert "ISF(t)" in md and "CR(t)" in md
    assert "clinically-tuned profile" in md  # the identifiability caveat is present

    p = write_inverse_json(result, recs, tmp_path / "inv.json")
    payload = json.loads(p.read_text())
    assert len(payload["isf"]) == 24
    assert len(payload["recommendations"]) == 48
