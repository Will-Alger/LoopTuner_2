"""Human-readable ISF/CR recommendation report (Markdown + JSON).

Renders the inverse-fit ensemble into a settings report with credible intervals.
Suggestions only — every number is for the user to review and enter manually.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from looptuner.model.inverse import InverseResult, Recommendation


def _fmt_hour(h: int) -> str:
    return f"{h:02d}:00"


def render_inverse_markdown(
    result: InverseResult, recs: list[Recommendation], coverage: float = 0.9
) -> str:
    pct = int(coverage * 100)
    isf_recs = {r.hour: r for r in recs if r.param == "ISF"}
    cr_recs = {r.hour: r for r in recs if r.param == "CR"}
    n_cr_estimable = sum(1 for r in cr_recs.values() if r.action != "insufficient-data")

    parts = [
        "# ISF / CR recommendation report",
        "",
        "> Decision support only. These are *suggestions* with uncertainty — review "
        "every number and enter changes manually in Loop. Nothing here doses.",
        "",
        "> The absolute ISF/CR *level* is confounded with endogenous glucose production "
        "and is not identifiable from CGM alone, so each schedule's 24h average is "
        "pinned to your clinically-tuned profile. What the data drives — and what these "
        "suggestions are about — is the time-of-day *shape* (when you're relatively "
        "more or less sensitive).",
        "",
        f"- Ensemble of {result.n_models} leave-one-day-out twins; "
        f"{pct}% credible intervals from the ensemble spread.",
        f"- CR is estimable for {n_cr_estimable}/24 hours given your announced-carb "
        "history; the rest are flagged insufficient-data (log meals to unlock them).",
        "",
        f"## ISF(t) — mg/dL per unit ({pct}% CI)",
        "",
        "| Hour | Current | Proposed | CI | Suggestion | Confidence |",
        "|---|---|---|---|---|---|",
    ]
    for h in range(24):
        r = isf_recs[h]
        parts.append(
            f"| {_fmt_hour(h)} | {r.current} | {r.proposed} | "
            f"[{r.lo}, {r.hi}] | {r.action} | {r.confidence} |"
        )

    parts += [
        "",
        f"## CR(t) — g per unit ({pct}% CI)",
        "",
        "| Hour | Current | Proposed | CI | Carb support (g) | Suggestion | Confidence |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in range(24):
        r = cr_recs[h]
        support = result.carb_support_by_hour[h]
        parts.append(
            f"| {_fmt_hour(h)} | {r.current} | {r.proposed} | [{r.lo}, {r.hi}] | "
            f"{support:.0f} | {r.action} | {r.confidence} |"
        )

    actionable = [
        r for r in recs if r.action in {"raise", "lower"} and r.confidence in {"high", "medium"}
    ]
    parts += ["", "## Actionable suggestions (medium+ confidence)", ""]
    if actionable:
        parts.append("| Hour | Param | Current | Proposed | CI | Action |")
        parts.append("|---|---|---|---|---|---|")
        for r in sorted(actionable, key=lambda x: (x.param, x.hour)):
            parts.append(
                f"| {_fmt_hour(r.hour)} | {r.param} | {r.current} | {r.proposed} | "
                f"[{r.lo}, {r.hi}] | {r.action} |"
            )
    else:
        parts.append(
            "None yet — the data doesn't confidently support changing any setting. "
            "This is expected with limited history; revisit as more days accumulate."
        )
    return "\n".join(parts)


def write_inverse_json(
    result: InverseResult, recs: list[Recommendation], path: str | Path, coverage: float = 0.9
) -> Path:
    isf = result.isf_summary(coverage)
    cr = result.cr_summary(coverage)
    payload = {
        "coverage": coverage,
        "n_models": result.n_models,
        "isf": [
            {
                "hour": h,
                "current": float(result.current_isf[h]),
                "median": float(isf["median"][h]),
                "lo": float(isf["lo"][h]),
                "hi": float(isf["hi"][h]),
            }
            for h in range(24)
        ],
        "cr": [
            {
                "hour": h,
                "current": float(result.current_cr[h]),
                "median": float(cr["median"][h]),
                "lo": float(cr["lo"][h]),
                "hi": float(cr["hi"][h]),
                "carb_support_g": float(result.carb_support_by_hour[h]),
            }
            for h in range(24)
        ],
        "recommendations": [asdict(r) for r in recs],
    }
    def _np(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(type(o))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_np))
    return path
