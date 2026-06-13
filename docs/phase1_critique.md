# Phase 1 — Survey critique and architecture decision

Critique of the two deep-research surveys (`chatgpt_feedback.txt`,
`gemini_feedback.txt`) and the resulting build decisions.

## Decisions (locked in)

- **Architecture:** Hybrid mechanistic Neural ODE — compartmental backbone reusing
  Loop's exponential insulin model, with a small neural net producing time-of-day
  `S_I(t)` (≈ ISF) and carb gain (≈ CR) over a low-order circadian Fourier basis,
  plus an optional regularized residual. Most data-efficient and interpretable; its
  physics prior prevents the causal confusion a black-box model falls into.
- **Runtime:** PyTorch + `torchdiffeq`. Overrules Gemini's JAX/diffrax pick because
  the target GPU (RTX 5070 Ti, Blackwell sm_120, WSL2) currently has fragile JAX/XLA
  support (PTX-JIT issues) while PyTorch ships AOT cu128 kernels. diffrax's nicer
  impulse/event handling is recovered manually (we never integrate through a delta).
- **Uncertainty:** conformal prediction for trajectory bands (calibration-critical
  backtest); deep ensemble / Laplace for ISF/CR *parameter* intervals. These are two
  different questions; neither survey separated them.

## Where the surveys agree (high confidence)

- Hybrid > pure NODE, latent ODE, and discrete sequence models for *this* use case
  (counterfactuals + interpretable ISF/CR).
- Pretraining/priors matter with little data — but see the nuance below.
- Don't reinvent the heuristic tier (autotune, AutoISF) or skip trivial baselines.

## Where they disagree — and our call

- **Library:** Gemini hard-commits to JAX/diffrax; ChatGPT hedges. We pick PyTorch
  for hardware-risk reasons (above).
- **Uncertainty:** ChatGPT leans Bayesian; Gemini says conformal, skip BNNs. Both
  half-right — conformal for bands, ensemble/Laplace for parameters.
- **"Pretraining is mandatory" (Gemini):** overstated for a *hybrid* — the
  mechanistic backbone already defeats the causal-confounding failure mode, so
  pretraining is better framed as physiological priors + a synthetic warm-start.
- **Accuracy numbers:** Gemini's ~8/14/20 mg/dL RMSE table is implausibly rosy and
  internally inconsistent; ChatGPT's figures are closer to the literature. (Our real
  twin lands at ~32 mg/dL RMSE @30min on a 7-day cold start, as expected.)

## Errors / overstatements to ignore

- OhioT1DM is **Medtronic Enlite** CGM under an Ohio-University DUA — *not*
  FreeStyle Libre / Kaggle CC-BY (ChatGPT) and *not* PhysioNet (Gemini).
- `simglucose` ships the 2008 model's **30 fixed** virtual patients — you can't
  trivially "generate 100" (Gemini).
- replayBG now has a Python port (`py_replay_bg`) — not MATLAB-only (Gemini).
- Several cited 2026 papers (`Zou et al.`, `GlucoNet-MM`) are unverified — treat the
  ideas, not the citations, as load-bearing.

## What both missed (and we handle)

1. **Blackwell GPU support** — the actual hardware constraint (drove the library
   choice).
2. **Loop hands you the backbone for free** — IOB/insulin curve, COB, temp basals.
   We reuse Loop's insulin model rather than relearning insulin PK.
3. **Cold-start data regime** — with only a few days you cannot fit 48 free hourly
   ISF/CR params; we start with a smooth circadian basis and refine as data grows.
4. **Identifiability** — ISF, CR, and basal are only partially separable from
   observational data; informative priors + clean fasting windows are required.
5. **Impulse handling** — boluses/carbs are deltas; we inject them as forcings, never
   integrating through a discontinuity.

## Empirical finding from the real data (important)

The patient's Nightscout history is a **fully-closed-loop, unannounced-meals**
pattern: insulin is delivered almost entirely via Loop temp basals (hundreds of
them), with very few correction boluses and **almost no announced carbs** (~2 carb
entries per week). Consequences:

- **ISF(t) and basal:** well-identifiable (rich insulin variation + CGM response).
- **CR(t):** poorly identifiable from announced carbs alone. Extracting per-hour CR
  (a core project goal) will need either (a) more announced meals, or (b) treating
  meals as latent disturbances the residual infers — which conflates carbs with
  sensitivity. This is flagged for the user before the inverse-fit milestone.
