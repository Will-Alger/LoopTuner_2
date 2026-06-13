"""Backtest report: Markdown summary + persistent benchmark log.

Produces the multi-section report the spec asks for — headline accuracy by horizon
and hour-of-day, calibration coverage, systematic error decomposition, a worst-miss
gallery, and a per-day twin-quality score — plus an append-only parquet log so twin
quality can be charted over weeks.

Charts (calibration plot image, worst-miss trajectory PNGs) and the optional LLM
narrator are layered on top of these records in a follow-up; the numeric tables here
are the substance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _rmse(g: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean(g["signed_err"] ** 2))) if len(g) else float("nan")


def _baseline_rmse(g: pd.DataFrame, col: str) -> float:
    d = g.dropna(subset=[col])
    return float(np.sqrt(np.mean((d[col] - d["actual"]) ** 2))) if len(d) else float("nan")


def twin_quality_by_day(df: pd.DataFrame, ref_horizon_min: int = 120) -> pd.DataFrame:
    """A single 'twin quality' number per day: 100 - MAPE at a reference horizon."""
    sub = df[df["horizon_min"] == ref_horizon_min]
    if sub.empty:
        return pd.DataFrame(columns=["day", "twin_quality", "rmse", "mape", "n"])
    rows = []
    for day, g in sub.groupby("day"):
        rows.append(
            {
                "day": day,
                "twin_quality": round(100.0 - float(g["pct_err"].mean()), 1),
                "rmse": round(_rmse(g), 1),
                "mape": round(float(g["pct_err"].mean()), 1),
                "n": len(g),
            }
        )
    return pd.DataFrame(rows).sort_values("day")


def _headline_table(df: pd.DataFrame) -> str:
    lines = [
        "| Horizon | Twin RMSE | Persist | Linear | Twin MAPE% | Beats baselines |",
        "|---|---|---|---|---|---|",
    ]
    for h, g in df.groupby("horizon_min"):
        twin = _rmse(g)
        persist = _baseline_rmse(g, "persist_pred")
        linear = _baseline_rmse(g, "linear_pred")
        beats = "yes" if twin < min(persist, linear) else "**no**"
        mape = g["pct_err"].mean()
        lines.append(
            f"| {h}m | {twin:.1f} | {persist:.1f} | {linear:.1f} | {mape:.1f} | {beats} |"
        )
    return "\n".join(lines)


def _by_hour_table(df: pd.DataFrame, horizon_min: int) -> str:
    sub = df[df["horizon_min"] == horizon_min]
    lines = [
        f"Hour-of-day RMSE at {horizon_min}min:",
        "",
        "| Hour | RMSE | Bias | n |",
        "|---|---|---|---|",
    ]
    for hour, g in sub.groupby("hour"):
        lines.append(f"| {hour:02d} | {_rmse(g):.1f} | {g['signed_err'].mean():+.1f} | {len(g)} |")
    return "\n".join(lines)


def _calibration_table(df: pd.DataFrame, coverage_levels: list[float]) -> str:
    header = " | ".join(f"{int(c*100)}% cov (emp.)" for c in coverage_levels)
    lines = ["| Horizon | " + header + " |"]
    lines.append("|---|" + "|".join("---" for _ in coverage_levels) + "|")
    for h, g in df.groupby("horizon_min"):
        cells = []
        for c in coverage_levels:
            col = f"in{int(c*100)}"
            emp = float(g[col].mean()) * 100 if col in g else float("nan")
            cells.append(f"{emp:.0f}%")
        lines.append(f"| {h}m | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _decomposition_table(df: pd.DataFrame, horizon_min: int) -> str:
    sub = df[df["horizon_min"] == horizon_min]
    groups = {
        "overnight (0-6h)": sub[sub["overnight"]],
        "daytime": sub[~sub["overnight"]],
        "near a carb entry": sub[sub["near_carb"]],
        "near a bolus": sub[sub["near_bolus"]],
        "quiet (no recent treatment)": sub[~sub["near_carb"] & ~sub["near_bolus"]],
    }
    lines = [
        f"Systematic error at {horizon_min}min:",
        "",
        "| Segment | RMSE | Bias | n |",
        "|---|---|---|---|",
    ]
    for name, g in groups.items():
        if len(g):
            lines.append(f"| {name} | {_rmse(g):.1f} | {g['signed_err'].mean():+.1f} | {len(g)} |")
    return "\n".join(lines)


def _worst_misses(df: pd.DataFrame, horizon_min: int, n: int = 10) -> str:
    sub = df[df["horizon_min"] == horizon_min].nlargest(n, "abs_err")
    lines = [
        f"Worst {n} misses at {horizon_min}min (the gold for what the twin doesn't know):",
        "",
        "| Time | Pred | Actual | Err | Context |",
        "|---|---|---|---|---|",
    ]
    for _, r in sub.iterrows():
        ctx = []
        if r["overnight"]:
            ctx.append("overnight")
        if r["near_carb"]:
            ctx.append("near-carb")
        if r["near_bolus"]:
            ctx.append("near-bolus")
        lines.append(
            f"| {r['timestamp']} | {r['pred']:.0f} | {r['actual']:.0f} | "
            f"{r['signed_err']:+.0f} | {', '.join(ctx) or 'quiet'} |"
        )
    return "\n".join(lines)


def render_markdown_report(df: pd.DataFrame, meta: dict) -> str:
    if df.empty:
        return "# Backtest report\n\nNo predictions were generated (insufficient data)."
    cov_levels = meta.get("coverage_levels", [0.5, 0.9])
    ref = 120 if 120 in df["horizon_min"].unique() else int(df["horizon_min"].max())
    quality = twin_quality_by_day(df, ref)

    parts = [
        f"# Backtest report — {meta.get('mode', 'walk_forward')}",
        "",
        f"- Source: `{meta.get('source')}`  |  Coverage: {meta.get('coverage_days')} days  "
        f"|  Predictions: {len(df)}",
        f"- Horizons: {meta.get('horizons_min')} min  |  Conformal levels: "
        f"{[int(c*100) for c in cov_levels]}%",
        "",
        "## a) Headline accuracy vs trivial baselines",
        "",
        _headline_table(df),
        "",
        "## b) Calibration (does X% confidence contain the actual X% of the time?)",
        "",
        _calibration_table(df, cov_levels),
        "",
        "Conformal guarantees the *marginal* rate; large gaps here flag where the "
        "exchangeability assumption (e.g. a changed infusion site) breaks down.",
        "",
        f"## c) Systematic error decomposition (at {ref}min)",
        "",
        _decomposition_table(df, ref),
        "",
        f"## Accuracy by hour of day (at {ref}min)",
        "",
        _by_hour_table(df, ref),
        "",
        f"## d) Worst-prediction gallery (at {ref}min)",
        "",
        _worst_misses(df, ref),
        "",
        "## e) Per-day twin quality (higher is better)",
        "",
        "| Day | Twin quality | RMSE | MAPE% | n |",
        "|---|---|---|---|---|",
    ]
    for _, r in quality.iterrows():
        parts.append(
            f"| {r['day']} | {r['twin_quality']} | {r['rmse']} | {r['mape']} | {int(r['n'])} |"
        )
    return "\n".join(parts)


def append_benchmark_log(path: str | Path, df: pd.DataFrame, meta: dict) -> Path:
    """Append one row per horizon to a persistent parquet log for trend charting."""
    path = Path(path)
    rows = []
    run_ts = pd.Timestamp.now("UTC")
    for h, g in df.groupby("horizon_min"):
        rows.append(
            {
                "run_ts": run_ts,
                "mode": meta.get("mode"),
                "source": meta.get("source"),
                "coverage_days": meta.get("coverage_days"),
                "horizon_min": int(h),
                "n": len(g),
                "rmse": _rmse(g),
                "mape": float(g["pct_err"].mean()),
                "bias": float(g["signed_err"].mean()),
                "cov50": float(g["in50"].mean()) if "in50" in g else np.nan,
                "cov90": float(g["in90"].mean()) if "in90" in g else np.nan,
            }
        )
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path)
    return path
