"""Reusable matplotlib figures for the dashboard and for PNG report charts.

Every function returns a ``matplotlib.figure.Figure`` so it can be shown live in the
Streamlit app (``st.pyplot(fig)``) or saved to disk for the Markdown reports — one
plotting path, two consumers. Uses the Agg backend so it works headless.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from looptuner.backtest.report import twin_quality_by_day  # noqa: E402
from looptuner.drift import DriftResult  # noqa: E402
from looptuner.model.inverse import InverseResult  # noqa: E402

_TIR_LOW, _TIR_HIGH = 70.0, 180.0


def _style_bg_axis(ax) -> None:
    ax.axhspan(_TIR_LOW, _TIR_HIGH, color="#e8f5e9", zorder=0)
    ax.axhline(_TIR_LOW, color="#c62828", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("BG (mg/dL)")
    ax.grid(True, alpha=0.25)


def fig_forecast(result: dict, title: str = "Scenario forecast") -> plt.Figure:
    """Forecast trajectory with conformal bands and the no-change baseline."""
    times = result["times"]
    x = [pd.Timestamp(t) for t in times]
    point = np.asarray(result["point"])
    fig, ax = plt.subplots(figsize=(8, 4))
    _style_bg_axis(ax)
    bands = result.get("bands") or {}
    colors = {0.9: "#90caf9", 0.5: "#42a5f5"}
    for cov in sorted(bands, reverse=True):
        b = bands[cov]
        ax.fill_between(
            x, b["lo"], b["hi"], color=colors.get(cov, "#bbdefb"),
            alpha=0.4, label=f"{int(cov * 100)}% interval",
        )
    if "baseline" in result:
        ax.plot(x, result["baseline"], color="#757575", ls="--", lw=1.3, label="no change")
    ax.plot(x, point, color="#1565c0", lw=2.0, label="forecast")
    ax.scatter([x[0]], [point[0]], color="#1565c0", zorder=5)
    ax.set_title(title)
    ax.set_xlabel("time")
    fig.autofmt_xdate()
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def fig_counterfactual(result: dict) -> plt.Figure:
    """Actual vs counterfactual BG over a replayed day."""
    x = [pd.Timestamp(t) for t in result["times"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    _style_bg_axis(ax)
    ax.plot(x, result["actual"], color="#37474f", lw=1.8, label="actual")
    ax.plot(x, result["counterfactual"], color="#6a1b9a", lw=1.8, ls="-", label="counterfactual")
    ax.set_title(f"Counterfactual replay — {result.get('day', '')}")
    ax.set_xlabel("time")
    fig.autofmt_xdate()
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def fig_isf_cr(result: InverseResult, coverage: float = 0.9) -> plt.Figure:
    """ISF(t) and CR(t) median with credible-interval ribbons vs current profile."""
    isf = result.isf_summary(coverage)
    cr = result.cr_summary(coverage)
    hours = result.hours
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

    a1.fill_between(hours, isf["lo"], isf["hi"], color="#ffcc80", alpha=0.5,
                    label=f"{int(coverage * 100)}% CI")
    a1.plot(hours, isf["median"], color="#e65100", lw=2, label="learned (median)")
    a1.plot(hours, result.current_isf, color="#455a64", ls="--", lw=1.5, label="current profile")
    a1.set_title("ISF(t) — mg/dL per U")
    a1.set_xlabel("hour of day")
    a1.set_xticks(range(0, 24, 3))
    a1.grid(True, alpha=0.25)
    a1.legend(fontsize=8)

    a2.fill_between(hours, cr["lo"], cr["hi"], color="#a5d6a7", alpha=0.5,
                    label=f"{int(coverage * 100)}% CI")
    a2.plot(hours, cr["median"], color="#2e7d32", lw=2, label="learned (median)")
    a2.plot(hours, result.current_cr, color="#455a64", ls="--", lw=1.5, label="current profile")
    # Shade hours with little carb support (CR not identifiable there).
    unsupported = result.carb_support_by_hour < 30.0
    for h in hours[unsupported]:
        a2.axvspan(h - 0.5, h + 0.5, color="#bdbdbd", alpha=0.25, zorder=0)
    a2.set_title("CR(t) — g per U (grey = low carb support)")
    a2.set_xlabel("hour of day")
    a2.set_xticks(range(0, 24, 3))
    a2.grid(True, alpha=0.25)
    a2.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_calibration(df: pd.DataFrame, coverage_levels=(0.5, 0.9)) -> plt.Figure:
    """Nominal vs empirical coverage at each horizon — the calibration check."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color="#9e9e9e", ls="--", lw=1, label="perfect")
    horizons = sorted(df["horizon_min"].unique())
    for cov in coverage_levels:
        col = f"in{int(cov * 100)}"
        if col not in df:
            continue
        emp = [df[df["horizon_min"] == h][col].mean() for h in horizons]
        ax.plot([cov] * len(horizons), emp, "o", label=f"{int(cov * 100)}% nominal")
        for h, e in zip(horizons, emp, strict=False):
            ax.annotate(f"{h}m", (cov, e), fontsize=7, xytext=(4, 0),
                        textcoords="offset points")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Conformal calibration")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_accuracy_by_horizon(df: pd.DataFrame) -> plt.Figure:
    """Twin RMSE vs baselines at each horizon."""
    horizons = sorted(df["horizon_min"].unique())

    def rmse(g, col=None):
        if col is None:
            return float(np.sqrt(np.mean(g["signed_err"] ** 2)))
        d = g.dropna(subset=[col])
        return float(np.sqrt(np.mean((d[col] - d["actual"]) ** 2))) if len(d) else np.nan

    twin = [rmse(df[df["horizon_min"] == h]) for h in horizons]
    persist = [rmse(df[df["horizon_min"] == h], "persist_pred") for h in horizons]
    linear = [rmse(df[df["horizon_min"] == h], "linear_pred") for h in horizons]
    x = np.arange(len(horizons))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, twin, w, label="twin", color="#1565c0")
    ax.bar(x, persist, w, label="persistence", color="#90a4ae")
    ax.bar(x + w, linear, w, label="linear", color="#cfd8dc")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}m" for h in horizons])
    ax.set_ylabel("RMSE (mg/dL)")
    ax.set_title("Accuracy vs trivial baselines")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def fig_twin_quality(df: pd.DataFrame, ref_horizon_min: int = 120) -> plt.Figure:
    """Per-day twin-quality score over time."""
    q = twin_quality_by_day(df, ref_horizon_min)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if not q.empty:
        ax.plot(q["day"], q["twin_quality"], "o-", color="#00897b")
    ax.set_ylabel("twin quality")
    ax.set_title(f"Per-day twin quality ({ref_horizon_min}min)")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def fig_drift(res: DriftResult) -> plt.Figure:
    """Per-hour recent vs baseline error, with flagged hours highlighted."""
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(hours - 0.2, np.nan_to_num(res.baseline_hour_mape), 0.4,
           label="baseline", color="#b0bec5")
    ax.bar(hours + 0.2, np.nan_to_num(res.recent_hour_mape), 0.4,
           label="recent day", color="#ef6c00")
    flagged = {f["hour"] for f in res.flags}
    for h in flagged:
        ax.axvspan(h - 0.5, h + 0.5, color="#ffebee", zorder=0)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("MAPE (%)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title(f"Drift — {res.horizon_min}min (shaded = flagged)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig
