"""The hybrid mechanistic Neural ODE — the digital twin's vector field.

Glucose evolves under a mechanistic backbone whose *time-varying* parameters are
produced by a small neural network over circadian features:

    dG/dt = EGP(t) - S_I(t) * I_act(t) + (S_I(t) / CR(t)) * C_app(t)
            - k * (G - G_b) + r_theta(features)

* ``I_act(t)`` and ``C_app(t)`` are precomputed insulin-activity and carb-appearance
  forcings from Loop's exponential model (we don't relearn insulin PK).
* ``S_I(t)`` (≈ ISF) and ``CR(t)`` are output by ``SensitivityNet`` from a low-order
  circadian (Fourier) basis — starting smooth/flat, refining toward per-hour as data
  grows (Phase 1: 2-3 harmonics is right for only a few days of data).
* ``r_theta`` is a small, heavily-regularized residual for what the physics misses
  (off by default; the mechanistic prior is what prevents black-box causal confusion).

The state is 1-D glucose: insulin/carb compartments live in the precomputed forcings,
so this is a forced continuous-time ODE integrated with torchdiffeq. The exact same
module is used for training, backtest, and the scenario simulator — one code path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

MINUTES_PER_DAY = 1440.0


@dataclass
class TwinConfig:
    """Hyperparameters and physiological priors for the twin."""

    n_harmonics: int = 3  # circadian Fourier harmonics for ISF/CR
    hidden: int = 16  # sensitivity-net hidden width
    isf_init: float = 50.0  # mg/dL per U (profile mean)
    cr_init: float = 10.0  # g per U (profile mean)
    basal_init: float = 0.8  # U/hr (for EGP balance init)
    glucose_baseline: float = 120.0  # mg/dL homeostatic set-point
    clearance_per_min: float = 0.01  # initial homeostatic pull
    use_residual: bool = False
    residual_hidden: int = 16
    # Multiplicative bounds keep ISF/CR physiologically sane (vs. profile mean).
    sensitivity_log_clamp: float = 0.8  # exp(+-0.8) ≈ x0.45 .. x2.2


def _circadian_features(minute_of_day: Tensor, n_harmonics: int) -> Tensor:
    """Map minute-of-day (0..1440) to [sin,cos] Fourier features, shape (..., 2K)."""
    feats = []
    for k in range(1, n_harmonics + 1):
        ang = 2.0 * math.pi * k * minute_of_day / MINUTES_PER_DAY
        feats.append(torch.sin(ang))
        feats.append(torch.cos(ang))
    return torch.stack(feats, dim=-1)


class SensitivityNet(nn.Module):
    """Smooth circadian ISF(t) and CR(t) as positive functions of time-of-day.

    Outputs are ``mean * exp(clamp(delta))`` where ``delta`` starts at zero, so the
    model begins at the profile means (flat schedules) and learns diurnal variation.
    """

    def __init__(self, cfg: TwinConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = 2 * cfg.n_harmonics
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden),
            nn.Tanh(),
            nn.Linear(cfg.hidden, 2),
        )
        # Start at the profile mean: last layer ~0 so delta≈0 initially.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("log_isf_mean", torch.tensor(math.log(cfg.isf_init)))
        self.register_buffer("log_cr_mean", torch.tensor(math.log(cfg.cr_init)))

    def forward(self, minute_of_day: Tensor) -> tuple[Tensor, Tensor]:
        feats = _circadian_features(minute_of_day, self.cfg.n_harmonics)
        delta = self.net(feats)
        c = self.cfg.sensitivity_log_clamp
        d_isf = torch.clamp(delta[..., 0], -c, c)
        d_cr = torch.clamp(delta[..., 1], -c, c)
        isf = torch.exp(self.log_isf_mean + d_isf)
        cr = torch.exp(self.log_cr_mean + d_cr)
        return isf, cr

    def schedule(self, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
        """Evaluate ISF(t), CR(t) on a 24-hour hourly grid (for reporting)."""
        hours = torch.arange(24, dtype=torch.float32, device=device)
        mod = hours * 60.0
        with torch.no_grad():
            return self.forward(mod)


def _interp1d(grid: Tensor, t: Tensor) -> Tensor:
    """Linear interpolation of a 1-D ``grid`` (sampled every step) at fractional index ``t``.

    ``t`` is minutes since window start scaled to step index by the caller. Clamped at
    the ends. Differentiable w.r.t. nothing here (forcings are constants) but cheap.
    """
    n = grid.shape[0]
    ti = torch.clamp(t, 0.0, float(n - 1))
    i0 = torch.floor(ti).long()
    i1 = torch.clamp(i0 + 1, max=n - 1)
    frac = ti - i0.to(ti.dtype)
    return grid[i0] * (1.0 - frac) + grid[i1] * frac


class HybridGlucoseODE(nn.Module):
    """torchdiffeq-compatible vector field. Forcings are bound per-window via ``bind``."""

    def __init__(self, cfg: TwinConfig):
        super().__init__()
        self.cfg = cfg
        self.sens = SensitivityNet(cfg)
        # Learnable mechanistic scalars (positive via softplus where needed).
        self.raw_k = nn.Parameter(torch.tensor(_inv_softplus(cfg.clearance_per_min)))
        self.gb = nn.Parameter(torch.tensor(float(cfg.glucose_baseline)))
        egp0 = cfg.isf_init * cfg.basal_init / 60.0
        self.raw_egp = nn.Parameter(torch.tensor(_inv_softplus(max(egp0, 1e-3))))
        self.residual = (
            nn.Sequential(
                nn.Linear(2 * cfg.n_harmonics + 1, cfg.residual_hidden),
                nn.Tanh(),
                nn.Linear(cfg.residual_hidden, 1),
            )
            if cfg.use_residual
            else None
        )
        if self.residual is not None:
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

        # Per-window forcing buffers (set by bind()). Forcings are (steps, B);
        # start_minute is (B,) so a whole minibatch of windows integrates at once.
        self._i_act: Tensor | None = None
        self._c_app: Tensor | None = None
        self._start_minute: Tensor | None = None
        self._dt: float = 5.0
        # Optional per-step counterfactual multipliers on ISF(t)/CR(t) (steps, B).
        self._isf_scale: Tensor | None = None
        self._cr_scale: Tensor | None = None

    # --- bound forcing ----------------------------------------------------- #
    def bind(
        self,
        i_act: Tensor,
        c_app: Tensor,
        start_minute: Tensor,
        dt_min: float,
        isf_scale: Tensor | None = None,
        cr_scale: Tensor | None = None,
    ) -> None:
        """Attach the forcing series (steps, B) for the window batch being integrated.

        ``isf_scale`` / ``cr_scale`` are optional per-step multipliers applied to the
        learned ISF(t)/CR(t) — this is how counterfactual "what if ISF were 0.8x from
        4-8am" queries are expressed, without any separate prediction code path.
        """
        self._i_act = i_act
        self._c_app = c_app
        self._start_minute = start_minute
        self._dt = float(dt_min)
        self._isf_scale = isf_scale
        self._cr_scale = cr_scale

    @property
    def k(self) -> Tensor:
        return nn.functional.softplus(self.raw_k)

    @property
    def egp(self) -> Tensor:
        return nn.functional.softplus(self.raw_egp)

    def forward(self, t: Tensor, g: Tensor) -> Tensor:
        assert self._i_act is not None and self._start_minute is not None
        idx = t / self._dt  # minutes -> fractional step index
        i_act = _interp1d(self._i_act, idx)  # (B,)
        c_app = _interp1d(self._c_app, idx)  # (B,)
        mod = torch.remainder(self._start_minute + t, MINUTES_PER_DAY)  # (B,)
        isf, cr = self.sens(mod)  # (B,), (B,)
        if self._isf_scale is not None:
            isf = isf * _interp1d(self._isf_scale, idx)
        if self._cr_scale is not None:
            cr = cr * _interp1d(self._cr_scale, idx)
        dg = self.egp - isf * i_act + (isf / cr) * c_app - self.k * (g - self.gb)
        if self.residual is not None:
            feats = _circadian_features(mod, self.cfg.n_harmonics)  # (B, 2K)
            rin = torch.cat([feats, ((g - self.gb) / 50.0).unsqueeze(-1)], dim=-1)
            dg = dg + self.residual(rin).squeeze(-1)
        return dg


def _inv_softplus(y: float) -> float:
    """Inverse of softplus, to initialize a raw parameter so softplus(raw)=y."""
    return math.log(math.expm1(y)) if y > 0 else -5.0
