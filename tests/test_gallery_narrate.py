import matplotlib

matplotlib.use("Agg")

from looptuner import narrate  # noqa: E402
from looptuner.backtest.engine import run_backtest  # noqa: E402
from looptuner.backtest.gallery import (  # noqa: E402
    narrator_payload,
    render_worst_miss_charts,
    worst_miss_contexts,
)
from looptuner.ingest.synthetic import generate_synthetic_dataset  # noqa: E402
from looptuner.model.twin import ForwardSimulator  # noqa: E402


def test_worst_miss_gallery(tmp_path):
    ds, _ = generate_synthetic_dataset(n_days=6, seed=2)
    sim = ForwardSimulator.from_dataset(ds, seed=0)
    sim.fit(ds, val_days=1, epochs=40)
    df, _ = run_backtest(ds, horizons_min=(60, 120), test_days=1, epochs=40, anchor_stride=4)

    contexts = worst_miss_contexts(ds, sim, df, horizon_min=120, n=5)
    assert len(contexts) == 5
    for c in contexts:
        for key in ("rank", "timestamp", "pred", "actual", "model_isf_at_hour", "pred_bg"):
            assert key in c
    # Misses are ordered by absolute error (largest first).
    errs = [abs(c["signed_err"]) for c in contexts]
    assert errs == sorted(errs, reverse=True)

    # narrator payload drops the heavy chart arrays.
    payload = narrator_payload(contexts)
    assert "pred_bg" not in payload[0] and "actual_times" not in payload[0]

    paths = render_worst_miss_charts(contexts, tmp_path / "g", narratives=["because reasons"] * 5)
    assert len(paths) == 5
    assert all(p.exists() for p in paths)


def test_narrator_soft_fails_without_key(monkeypatch):
    # Simulate the SDK being unavailable -> empty list, never raises.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("no anthropic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert narrate.narrate_misses([{"rank": 1}]) == []


def test_narrator_success_with_mock(monkeypatch):
    import anthropic

    class _Block:
        type = "text"
        text = '{"explanations": ["unannounced meal", "exercise dip"]}'

    class _Resp:
        stop_reason = "end_turn"
        content = [_Block()]

    class _Messages:
        def create(self, **kwargs):
            # Token-light contract: narrator must not enable extended thinking.
            assert "thinking" not in kwargs
            assert kwargs["model"]
            return _Resp()

    class _FakeClient:
        def __init__(self, *a, **k):
            self.messages = _Messages()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    out = narrate.narrate_misses([{"rank": 1}, {"rank": 2}], api_key="test")
    assert out == ["unannounced meal", "exercise dip"]


def test_narrator_pads_mismatched_length(monkeypatch):
    import anthropic

    class _Block:
        type = "text"
        text = '{"explanations": ["only one"]}'

    class _Resp:
        stop_reason = "end_turn"
        content = [_Block()]

    class _FakeClient:
        def __init__(self, *a, **k):
            self.messages = type("M", (), {"create": lambda self, **k: _Resp()})()

    monkeypatch.setattr(anthropic, "Anthropic", _FakeClient)
    out = narrate.narrate_misses([{"rank": 1}, {"rank": 2}, {"rank": 3}], api_key="x")
    assert out == ["only one", "", ""]
