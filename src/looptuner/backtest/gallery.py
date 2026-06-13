"""Worst-prediction gallery: reconstruct and chart the backtest's biggest misses.

For the top-N largest-error predictions, re-roll the (no-leakage) anchored trajectory
with the current model and chart it against the actual CGM, marking the insulin/carb
inputs in the window. Each miss also carries a compact context dict — that's what the
optional LLM narrator (looptuner.narrate) explains in one sentence.

This is "the gold for understanding what the twin doesn't know" from the spec.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from looptuner.backtest.engine import BacktestArrays
from looptuner.ingest.schema import GRID_MINUTES, TidyDataset
from looptuner.model.twin import ForwardSimulator


def worst_miss_contexts(
    dataset: TidyDataset,
    sim: ForwardSimulator,
    records: pd.DataFrame,
    horizon_min: int = 120,
    n: int = 10,
    context_lookback_min: int = 60,
) -> list[dict]:
    """Build chart-ready + narrator-ready context dicts for the N worst misses."""
    arr = BacktestArrays.from_dataset(dataset)
    h = horizon_min // GRID_MINUTES
    sub = records[records["horizon_min"] == horizon_min].nlargest(n, "abs_err")
    isf_sched, cr_sched = (t.cpu().numpy() for t in sim.model.sens.schedule())
    lookback = context_lookback_min // GRID_MINUTES

    out: list[dict] = []
    for rank, (_, row) in enumerate(sub.iterrows(), start=1):
        a = int(row["anchor"])
        lo = max(0, a - lookback)
        hi = min(arr.n - 1, a + h)
        actual_times = arr.timestamps[lo : hi + 1]
        actual_bg = arr.bg[lo : hi + 1]

        i_win, c_win = arr.anchored_window(a, h)
        pred = sim.roll(i_win, c_win, arr.minute_of_day[a], arr.bg[a])
        pred_times = [
            arr.timestamps[a] + pd.Timedelta(minutes=j * GRID_MINUTES) for j in range(h + 1)
        ]

        # IOB / COB proxies over the recent window before the anchor.
        win = slice(max(0, a - 48), a + 1)  # last 4h
        iob_recent = float(arr.bolus[win].sum())
        cob_recent = float(arr.carbs[win].sum())
        hour = int(arr.minute_of_day[a] // 60)

        bolus_marks = [
            (arr.timestamps[k], float(arr.bolus[k]))
            for k in range(lo, hi + 1)
            if arr.bolus[k] > 0
        ]
        carb_marks = [
            (arr.timestamps[k], float(arr.carbs[k]))
            for k in range(lo, hi + 1)
            if arr.carbs[k] > 0
        ]

        out.append(
            {
                "rank": rank,
                "anchor": a,
                "timestamp": str(arr.timestamps[a]),
                "hour": hour,
                "start_bg": round(float(arr.bg[a]), 0),
                "pred": round(float(row["pred"]), 0),
                "actual": round(float(row["actual"]), 0),
                "signed_err": round(float(row["signed_err"]), 0),
                "horizon_min": horizon_min,
                "overnight": bool(row["overnight"]),
                "near_carb": bool(row["near_carb"]),
                "near_bolus": bool(row["near_bolus"]),
                "iob_recent_u": round(iob_recent, 1),
                "cob_recent_g": round(cob_recent, 0),
                "model_isf_at_hour": round(float(isf_sched[hour]), 1),
                "model_cr_at_hour": round(float(cr_sched[hour]), 1),
                # Chart arrays:
                "actual_times": list(actual_times),
                "actual_bg": actual_bg,
                "pred_times": pred_times,
                "pred_bg": pred,
                "bolus_marks": bolus_marks,
                "carb_marks": carb_marks,
                "title": (
                    f"#{rank}: {arr.timestamps[a]} — predicted {row['pred']:.0f}, "
                    f"actual {row['actual']:.0f} ({row['signed_err']:+.0f}) at {horizon_min}min"
                ),
            }
        )
    return out


def render_worst_miss_charts(
    contexts: list[dict], out_dir: str | Path, narratives: list[str] | None = None
) -> list[Path]:
    """Render each worst-miss context to a PNG; attach narratives if provided."""
    from looptuner import plots

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, ctx in enumerate(contexts):
        if narratives and i < len(narratives):
            ctx = {**ctx, "narrative": narratives[i]}
        fig = plots.fig_worst_miss(ctx)
        p = out_dir / f"worst_miss_{ctx['rank']:02d}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        paths.append(p)
    return paths


def render_gallery_markdown(
    contexts: list[dict], chart_paths: list[Path], narratives: list[str] | None = None
) -> str:
    """A Markdown gallery embedding each worst-miss chart + context (+ narrative)."""
    parts = ["# Worst-prediction gallery", "",
             "The largest-error predictions — what the twin doesn't yet know.", ""]
    for i, ctx in enumerate(contexts):
        parts.append(f"## #{ctx['rank']} — {ctx['timestamp']}")
        parts.append("")
        parts.append(f"![worst miss {ctx['rank']}]({chart_paths[i].name})")
        parts.append("")
        parts.append(
            f"- Predicted **{ctx['pred']:.0f}**, actual **{ctx['actual']:.0f}** "
            f"({ctx['signed_err']:+.0f}) at {ctx['horizon_min']}min; start BG "
            f"{ctx['start_bg']:.0f}, hour {ctx['hour']:02d}."
        )
        parts.append(
            f"- Recent IOB ~{ctx['iob_recent_u']}U, COB ~{ctx['cob_recent_g']:.0f}g; "
            f"model ISF {ctx['model_isf_at_hour']}, CR {ctx['model_cr_at_hour']} at this hour."
        )
        if narratives and i < len(narratives) and narratives[i]:
            parts.append(f"- **Likely cause:** {narratives[i]}")
        parts.append("")
    return "\n".join(parts)


def narrator_payload(contexts: list[dict]) -> list[dict]:
    """Strip the heavy chart arrays, leaving only the compact fields the LLM needs."""
    keep = (
        "rank", "timestamp", "hour", "start_bg", "pred", "actual", "signed_err",
        "horizon_min", "overnight", "near_carb", "near_bolus", "iob_recent_u",
        "cob_recent_g", "model_isf_at_hour", "model_cr_at_hour",
    )
    return [{k: c[k] for k in keep} for c in contexts]
