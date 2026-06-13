# LoopTuner

A patient-specific **glucose digital twin** for Type 1 Diabetes — a learned,
differentiable forward simulator of *your own* blood-glucose dynamics, trained on
your Nightscout history.

> **Decision support only.** LoopTuner *suggests* settings with uncertainty bands.
> Every number is reviewed and entered manually in your pump app. Nothing here is
> ever wired to autonomous dosing. No dosing code, ever.

## What it does

Given your insulin and carb inputs, the twin predicts BG forward in time. Once
trained you can:

- Run **counterfactuals** ("what would my BG have looked like with ISF=42 instead
  of 50 from 4–8am?").
- Extract per-hour **ISF(t)** and **CR(t)** as functions inside the model, with
  **credible intervals** — never bare point estimates.
- **Backtest** the twin in shadow mode against trivial baselines before trusting it.

## Architecture

A **hybrid mechanistic Neural ODE** (chosen in Phase 1): a compartmental backbone
that reuses Loop's exponential insulin model, with a small neural network producing
time-of-day insulin sensitivity `S_I(t)` (≈ ISF) and carb gain (≈ CR), plus a small
regularized residual for what the physics misses. This is the most data-efficient
and interpretable option, and its physics prior prevents the causal confusion a
black-box model falls into (learning that "insulin raises BG" because boluses
precede meal spikes).

- **Runtime:** PyTorch + `torchdiffeq`.
- **Uncertainty:** conformal prediction for trajectory bands; deep ensemble /
  Laplace for ISF/CR parameter intervals.

See `project.txt` for the full spec and `docs/phase1_critique.md` for the reasoning
behind the architecture and library choices.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # installs CPU torch by default — runs anywhere
cp .env.example .env         # then fill in your Nightscout URL + token
uv run pytest                # run the test suite (uses synthetic data, no secrets)
```

### GPU setup (RTX 5070 Ti / Blackwell, sm_120, WSL2)

The GPU is an accelerator, not a requirement — every code path runs on CPU. To use
the Blackwell GPU, override the default CPU torch with a cu128 build:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch
python -c "import torch; print(torch.cuda.get_device_capability())"  # expect (12, 0)
```

If JAX/diffrax were ever considered: Blackwell sm_120 support is currently fragile
under JAX/XLA (PTX-JIT issues), which is why this project uses PyTorch/torchdiffeq.

## Status

Phase 2, building incrementally:

- [x] Nightscout ingestion → tidy time-series (+ synthetic generator for testing)
- [x] Forward simulator (hybrid Neural ODE) — beats persistence & linear baselines
      at every horizon on both synthetic and real data
- [x] Train / held-out validation + metrics (MAPE, RMSE, clinical)
- [x] CLI (`ingest` / `train` / `evaluate`) + reproducible dataset cache & provenance
- [x] Conformal uncertainty for calibrated trajectory bands
- [x] Backtest / shadow-mode harness (no future-leakage, walk-forward, vs baselines,
      calibration, error decomposition, worst-miss gallery, per-day score, persistent
      benchmark log) — `looptuner backtest` / `shadow` / `benchmark-trend`
- [x] Counterfactual replay + interactive scenario simulator (`scenario` /
      `counterfactual`) — same forward model, ISF/CR/basal overrides + added
      bolus/carb, conformal bands, <200ms point path / <2s full forecast
- [ ] Inverse fit: per-hour ISF(t)/CR(t) with uncertainty
- [ ] Backtest polish: calibration-plot/worst-miss PNGs + optional LLM narrator
- [ ] Drift monitor, live polling daemon, nightly retrain
