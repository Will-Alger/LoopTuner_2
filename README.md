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

### Interactive dashboard

```bash
uv sync --extra ui           # installs Streamlit (optional)
uv run looptuner ingest --days 30 && uv run looptuner train   # need a dataset + model
uv run looptuner ui          # opens a local dashboard at http://localhost:8501
```

The dashboard is local and read-only (no auth, nothing exposed): a scenario simulator
with live sliders + conformal bands, counterfactual day replay, backtest diagnostics,
ISF/CR credible-interval ribbons, and the drift monitor. `looptuner charts` renders the
backtest charts to PNGs without the app.

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
- [x] Inverse fit: per-hour ISF(t)/CR(t) with credible intervals via a
      leave-one-day-out ensemble; level anchored to the clinical profile (only the
      identifiable diurnal *shape* is data-driven), CR gated on carb support.
      `looptuner inverse` → Markdown + JSON recommendation report.
- [x] Drift monitor (`drift-report`) — per-hour predicted-vs-actual error over recent
      days, flagging sudden drops (retrain signal or physiology change)
- [x] Nightly retrain (`train-incremental`) — retrain on full history, promote only
      if it beats the current model on the latest day; versioned checkpoint registry
      with scores + data hash (`checkpoints`)
- [x] Interactive Streamlit dashboard (`looptuner ui`) — scenario simulator with
      live sliders + bands, counterfactual replay, backtest diagnostics, ISF/CR
      ribbons, drift; reusable matplotlib charts (`looptuner charts` → PNGs)
- [x] Backtest polish: worst-miss trajectory PNGs (`backtest --gallery`) + optional
      LLM narrator (`--narrate`, behind a flag — the LLM is a narrator at the end of
      the pipeline, never the predictor)
- [x] Live polling daemon (`daemon` / `daemon-status`) — polls Nightscout, appends to
      the corpus, logs predicted-vs-actual; crash-safe cursor, idempotent, stoppable.
      Never updates weights (that's `train-incremental`'s job).

**All Phase 2 features from the spec are now implemented.**
