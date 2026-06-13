"""Forward simulator: train the hybrid Neural ODE and predict BG trajectories.

This is the FIRST deliverable of Phase 2 — a forward model only (no inverse problem
yet). It precomputes Loop-style insulin/carb forcings from a ``TidyDataset``, then
trains the hybrid ODE on short windows with strict held-out-day validation.

The same ``predict`` path is what the backtest and scenario simulator call, so a bug
shows up everywhere at once rather than in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor
from torchdiffeq import odeint

from looptuner.eval.metrics import clinical_metrics
from looptuner.ingest.schema import GRID_MINUTES, TidyDataset
from looptuner.model.hybrid_node import HybridGlucoseODE, TwinConfig
from looptuner.model.kernels import carb_activity_grid, insulin_activity_grid

DEFAULT_HORIZONS_MIN = (30, 60, 120, 240)


@dataclass
class ForcingData:
    """Precomputed per-grid forcing tensors for one dataset."""

    i_act: Tensor  # insulin activity per step (n,)
    c_app: Tensor  # carb appearance per step (n,)
    bg: Tensor  # observed BG per step (n,), NaN where missing
    minute_of_day: Tensor  # local minute-of-day per step (n,)
    day_codes: np.ndarray  # integer day label per step (n,)
    days: list  # ordered unique day labels
    dt: float = float(GRID_MINUTES)
    n: int = 0


def build_forcing(dataset: TidyDataset, dia_minutes: float, device: str = "cpu") -> ForcingData:
    frame = dataset.frame
    n = len(frame)
    insulin_delivery = (frame["bolus"].to_numpy() + frame["basal_insulin"].to_numpy())
    i_act = insulin_activity_grid(insulin_delivery, GRID_MINUTES, dia_minutes=dia_minutes)

    carb_series = frame["carbs"].to_numpy().astype(float)
    absorb_series = np.full(n, 180.0)
    for c in dataset.carbs:
        k = frame.index.get_indexer([c.time.floor(f"{GRID_MINUTES}min")])
        if k[0] >= 0:
            absorb_series[k[0]] = c.absorption_minutes
    c_app = carb_activity_grid(carb_series, absorb_series, GRID_MINUTES)

    day_series = dataset.day_index()
    days = sorted(day_series.unique().tolist())
    day_to_code = {d: i for i, d in enumerate(days)}
    day_codes = np.array([day_to_code[d] for d in day_series.to_numpy()], dtype=int)

    def t(x):
        return torch.tensor(np.asarray(x, dtype=np.float32), device=device)

    return ForcingData(
        i_act=t(i_act),
        c_app=t(c_app),
        bg=t(frame["bg"].to_numpy()),
        minute_of_day=t(frame["minute_of_day"].to_numpy()),
        day_codes=day_codes,
        days=days,
        dt=float(GRID_MINUTES),
        n=n,
    )


@dataclass
class TrainResult:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_val: float = float("inf")
    best_epoch: int = -1


class ForwardSimulator:
    """Trainable hybrid-ODE forward simulator over a single patient's history."""

    def __init__(self, config: TwinConfig | None = None, device: str = "cpu", seed: int = 0):
        self.cfg = config or TwinConfig()
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        self.model = HybridGlucoseODE(self.cfg).to(device)
        self.solver = "rk4"

    # --- profile-aware construction --------------------------------------- #
    @classmethod
    def from_dataset(cls, dataset: TidyDataset, device: str = "cpu", seed: int = 0, **overrides):
        """Initialize physiological priors (ISF/CR/basal/baseline) from the profile."""
        prof = dataset.profile
        mid = dataset.frame.index[len(dataset.frame) // 2]
        bg = dataset.frame["bg"].to_numpy()
        baseline = float(np.nanmedian(bg)) if np.isfinite(bg).any() else 120.0
        cfg = TwinConfig(
            isf_init=prof.isf_at(mid),
            cr_init=prof.cr_at(mid),
            basal_init=prof.basal_at(mid),
            glucose_baseline=baseline,
            **overrides,
        )
        return cls(cfg, device=device, seed=seed)

    # --- integration ------------------------------------------------------ #
    def _integrate(
        self, forcing: ForcingData, starts: np.ndarray, horizon: int, g0: Tensor
    ) -> Tensor:
        """Batched integration of windows starting at ``starts`` for ``horizon`` steps.

        Returns predicted BG of shape (horizon+1, B).
        """
        n = forcing.n
        steps = np.arange(horizon + 1)[:, None] + starts[None, :]
        steps = np.clip(steps, 0, n - 1)
        idx = torch.as_tensor(steps, device=self.device, dtype=torch.long)
        i_win = forcing.i_act[idx]  # (H+1, B)
        c_win = forcing.c_app[idx]
        start_min = forcing.minute_of_day[torch.as_tensor(starts, device=self.device)]
        self.model.bind(i_win, c_win, start_min, forcing.dt)
        t_eval = torch.arange(horizon + 1, dtype=torch.float32, device=self.device) * forcing.dt
        sol = odeint(self.model, g0, t_eval, method=self.solver)
        return sol  # (H+1, B)

    @torch.no_grad()
    def roll(
        self,
        i_act_win: np.ndarray,
        c_app_win: np.ndarray,
        start_minute: float,
        g0: float,
        dt: float = float(GRID_MINUTES),
        isf_scale: np.ndarray | None = None,
        cr_scale: np.ndarray | None = None,
    ) -> np.ndarray:
        """Integrate a single custom forcing window (used by backtest and scenarios).

        ``i_act_win`` / ``c_app_win`` have length H+1 (the forcing over the window,
        already constructed with no future-leakage). ``isf_scale`` / ``cr_scale`` are
        optional per-step multipliers for counterfactual ISF/CR overrides. Returns BG
        of length H+1.
        """
        self.model.eval()
        horizon = len(i_act_win) - 1
        i_t = torch.as_tensor(np.asarray(i_act_win, np.float32)[:, None], device=self.device)
        c_t = torch.as_tensor(np.asarray(c_app_win, np.float32)[:, None], device=self.device)
        sm = torch.as_tensor([float(start_minute)], dtype=torch.float32, device=self.device)
        isf_t = (
            torch.as_tensor(np.asarray(isf_scale, np.float32)[:, None], device=self.device)
            if isf_scale is not None
            else None
        )
        cr_t = (
            torch.as_tensor(np.asarray(cr_scale, np.float32)[:, None], device=self.device)
            if cr_scale is not None
            else None
        )
        self.model.bind(i_t, c_t, sm, dt, isf_scale=isf_t, cr_scale=cr_t)
        t_eval = torch.arange(horizon + 1, dtype=torch.float32, device=self.device) * dt
        g0t = torch.as_tensor([float(g0)], dtype=torch.float32, device=self.device)
        sol = odeint(self.model, g0t, t_eval, method=self.solver)
        return sol.squeeze(-1).cpu().numpy()

    # --- window sampling -------------------------------------------------- #
    def _eligible_starts(
        self, forcing: ForcingData, day_codes: set[int], horizon: int
    ) -> np.ndarray:
        bg = forcing.bg.cpu().numpy()
        valid_day = np.isin(forcing.day_codes, list(day_codes))
        observed_start = np.isfinite(bg)
        fits = np.arange(forcing.n) + horizon < forcing.n
        starts = np.where(valid_day & observed_start & fits)[0]
        return starts

    # --- training --------------------------------------------------------- #
    def fit(
        self,
        dataset: TidyDataset,
        val_days: int = 1,
        epochs: int = 200,
        train_horizon_min: int = 120,
        batch_windows: int = 64,
        lr: float = 0.02,
        weight_decay: float = 1e-3,
        patience: int = 40,
        verbose: bool = False,
    ) -> tuple[ForcingData, TrainResult]:
        forcing = build_forcing(dataset, dataset.profile.dia_hours * 60.0, self.device)
        n_days = len(forcing.days)
        val_days = min(val_days, max(1, n_days - 1))
        val_codes = set(range(n_days - val_days, n_days))
        train_codes = set(range(0, n_days - val_days))
        result = self._fit_with_codes(
            forcing, train_codes, val_codes, epochs, train_horizon_min,
            batch_windows, lr, weight_decay, patience, verbose,
        )
        return forcing, result

    def fit_until_day(
        self,
        dataset: TidyDataset,
        train_upto_day: int,
        epochs: int = 150,
        train_horizon_min: int = 120,
        batch_windows: int = 64,
        lr: float = 0.02,
        weight_decay: float = 1e-3,
        patience: int = 40,
    ) -> tuple[ForcingData, TrainResult]:
        """Train using ONLY days strictly before ``train_upto_day`` (walk-forward).

        The last available training day is used as validation; no test-day data is
        ever seen. This is what the backtest calls so the cursor moves forward.
        """
        forcing = build_forcing(dataset, dataset.profile.dia_hours * 60.0, self.device)
        train_upto_day = max(2, min(train_upto_day, len(forcing.days)))
        val_codes = {train_upto_day - 1}
        train_codes = set(range(0, train_upto_day - 1)) or {0}
        result = self._fit_with_codes(
            forcing, train_codes, val_codes, epochs, train_horizon_min,
            batch_windows, lr, weight_decay, patience, False,
        )
        return forcing, result

    def _fit_with_codes(
        self,
        forcing: ForcingData,
        train_codes: set[int],
        val_codes: set[int],
        epochs: int,
        train_horizon_min: int,
        batch_windows: int,
        lr: float,
        weight_decay: float,
        patience: int,
        verbose: bool,
    ) -> TrainResult:
        horizon = train_horizon_min // GRID_MINUTES
        train_starts = self._eligible_starts(forcing, train_codes, horizon)
        val_starts = self._eligible_starts(forcing, val_codes, horizon)
        if train_starts.size == 0:
            raise ValueError("No eligible training windows — need more contiguous CGM.")

        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        rng = np.random.default_rng(self.seed)
        result = TrainResult()
        best_state = None
        bad = 0

        for epoch in range(epochs):
            self.model.train()
            n_sel = min(batch_windows, train_starts.size)
            sel = rng.choice(train_starts, size=n_sel, replace=False)
            loss = self._window_loss(forcing, sel, horizon)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            opt.step()
            result.train_loss.append(float(loss.item()))

            self.model.eval()
            with torch.no_grad():
                vsel = val_starts if val_starts.size else sel
                vloss = float(self._window_loss(forcing, vsel, horizon).item())
            result.val_loss.append(vloss)

            if vloss < result.best_val - 1e-4:
                result.best_val = vloss
                result.best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if verbose and epoch % 25 == 0:
                print(f"epoch {epoch:4d}  train {loss.item():8.2f}  val {vloss:8.2f}")
            if bad >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return result

    def _window_loss(self, forcing: ForcingData, starts: np.ndarray, horizon: int) -> Tensor:
        g0 = forcing.bg[torch.as_tensor(starts, device=self.device)]
        sol = self._integrate(forcing, starts, horizon, g0)  # (H+1, B)
        steps = np.clip(np.arange(horizon + 1)[:, None] + starts[None, :], 0, forcing.n - 1)
        target = forcing.bg[torch.as_tensor(steps, device=self.device)]  # (H+1, B)
        mask = torch.isfinite(target)
        # Skip the trivially-correct first step (g0 == target[0]).
        mask[0] = False
        if mask.sum() == 0:
            return sol.sum() * 0.0
        return torch.nn.functional.huber_loss(sol[mask], target[mask], delta=15.0)

    # --- prediction ------------------------------------------------------- #
    @torch.no_grad()
    def predict(
        self, forcing: ForcingData, start: int, horizon_steps: int, g0: float | None = None
    ) -> np.ndarray:
        self.model.eval()
        init = forcing.bg[start] if g0 is None else torch.tensor(float(g0), device=self.device)
        g0t = init.reshape(1)
        sol = self._integrate(forcing, np.array([start]), horizon_steps, g0t)
        return sol.squeeze(-1).cpu().numpy()

    # --- evaluation ------------------------------------------------------- #
    @torch.no_grad()
    def evaluate_horizons(
        self,
        forcing: ForcingData,
        day_codes: set[int] | None = None,
        horizons_min: tuple[int, ...] = DEFAULT_HORIZONS_MIN,
    ) -> dict[int, dict[str, float]]:
        """Held-out accuracy at each horizon over all eligible windows in ``day_codes``."""
        self.model.eval()
        max_h = max(horizons_min) // GRID_MINUTES
        if day_codes is None:
            day_codes = set(range(len(forcing.days)))
        starts = self._eligible_starts(forcing, day_codes, max_h)
        if starts.size == 0:
            return {}
        g0 = forcing.bg[torch.as_tensor(starts, device=self.device)]
        sol = self._integrate(forcing, starts, max_h, g0).cpu().numpy()  # (max_h+1, B)
        bg_np = forcing.bg.cpu().numpy()
        out: dict[int, dict[str, float]] = {}
        for h_min in horizons_min:
            h = h_min // GRID_MINUTES
            pred = sol[h]
            actual_idx = np.clip(starts + h, 0, forcing.n - 1)
            actual = bg_np[actual_idx]
            out[h_min] = clinical_metrics(pred, actual)
        return out

    # --- persistence ------------------------------------------------------ #
    def save(self, path: str) -> None:
        torch.save({"cfg": self.cfg, "state_dict": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> ForwardSimulator:
        blob = torch.load(path, map_location=device, weights_only=False)
        sim = cls(blob["cfg"], device=device)
        sim.model.load_state_dict(blob["state_dict"])
        return sim
