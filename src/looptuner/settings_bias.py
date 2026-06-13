"""Settings-bias report — observational, model-free detection of aggressive settings.

The forward twin anchors ISF/CR *level* to your clinical profile (the level isn't
identifiable from CGM alone), so it deliberately won't tell you your whole carb ratio
or basal is too strong. This report answers that question from a different direction:
purely from outcomes in your CGM + treatment history, does glucose systematically end
up *low* after meals, after corrections, or overnight?

It is decision support, not a dosing tool: it surfaces directional patterns and points
you to a conversation with your care team. It never recommends a dose or a setting
number, and it does not use the neural model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from looptuner.backtest.engine import BacktestArrays
from looptuner.ingest.schema import GRID_MINUTES, TidyDataset

HYPO = 70.0
SEVERE = 54.0
TIR_LOW, TIR_HIGH = 70.0, 180.0


@dataclass
class PostBolusStats:
    n: int
    frac_to_hypo: float  # fraction of boluses followed by BG < 70 within the lookahead
    frac_below_target: float
    median_nadir: float
    median_drop: float  # start BG minus the lowest BG reached

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "frac_to_hypo": round(self.frac_to_hypo, 3),
            "frac_below_target": round(self.frac_below_target, 3),
            "median_nadir": round(self.median_nadir, 0),
            "median_drop": round(self.median_drop, 0),
        }


@dataclass
class SettingsBiasResult:
    n_days: float
    target_mean: float
    tir: float
    tbr_70: float
    tbr_54: float
    tar_180: float
    meal_bolus: PostBolusStats
    correction_bolus: PostBolusStats
    overnight_drift_mgdl_per_h: float
    n_fasting_nights: int
    tbr_by_hour: np.ndarray
    hypo_attribution: dict = field(default_factory=dict)
    n_hypo_episodes: int = 0


def _post_bolus_stats(
    arr: BacktestArrays, target: np.ndarray, idxs: np.ndarray, lookahead: int
) -> PostBolusStats:
    nadirs, drops, hypo_hits, below_target = [], [], 0, 0
    for k in idxs:
        end = min(arr.n, k + lookahead + 1)
        window = arr.bg[k:end]
        window = window[np.isfinite(window)]
        if window.size == 0:
            continue
        start_seg = arr.bg[max(0, k - 2) : k + 1]
        start_seg = start_seg[np.isfinite(start_seg)]
        start = float(start_seg[-1]) if start_seg.size else float(window[0])
        nadir = float(window.min())
        nadirs.append(nadir)
        drops.append(start - nadir)
        hypo_hits += int(nadir < HYPO)
        below_target += int(nadir < target[k])
    n = len(nadirs)
    if n == 0:
        return PostBolusStats(0, float("nan"), float("nan"), float("nan"), float("nan"))
    return PostBolusStats(
        n=n,
        frac_to_hypo=hypo_hits / n,
        frac_below_target=below_target / n,
        median_nadir=float(np.median(nadirs)),
        median_drop=float(np.median(drops)),
    )


def _overnight_drift(arr: BacktestArrays) -> tuple[float, int]:
    """Mean fasting-overnight BG slope (mg/dL per hour) across nights with no inputs."""
    hour = (arr.minute_of_day // 60).astype(int)
    slopes = []
    for code in np.unique(arr.day_codes):
        night = np.where((arr.day_codes == code) & (hour >= 0) & (hour < 6))[0]
        if night.size < 12:
            continue
        if arr.bolus[night].sum() > 0 or arr.carbs[night].sum() > 0:
            continue  # not fasting
        bg = arr.bg[night]
        ok = np.isfinite(bg)
        if ok.sum() < 6:
            continue
        x_hours = (np.arange(night.size) * GRID_MINUTES / 60.0)[ok]
        slope = float(np.polyfit(x_hours, bg[ok], 1)[0])
        slopes.append(slope)
    if not slopes:
        return float("nan"), 0
    return float(np.mean(slopes)), len(slopes)


def _hypo_attribution(arr: BacktestArrays, lookback: int) -> tuple[dict, int]:
    """Attribute hypo time (BG<70) to the most recent likely cause."""
    bg = arr.bg
    below = np.isfinite(bg) & (bg < HYPO)
    buckets = {"post_meal": 0, "post_correction": 0, "overnight": 0, "other": 0}
    episodes = 0
    i = 0
    while i < arr.n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j < arr.n and below[j]:
            j += 1
        episodes += 1
        length = j - i
        look = slice(max(0, i - lookback), i + 1)
        meal_bolus = arr.near_carb[look] & (arr.bolus[look] > 0)
        meal = bool((arr.carbs[look] > 0).any() or meal_bolus.any())
        corr = bool((arr.bolus[look] > 0).any())
        hour = int(arr.minute_of_day[i] // 60)
        if meal:
            buckets["post_meal"] += length
        elif corr:
            buckets["post_correction"] += length
        elif hour < 6:
            buckets["overnight"] += length
        else:
            buckets["other"] += length
        i = j
    total = sum(buckets.values())
    fracs = {k: (v / total if total else 0.0) for k, v in buckets.items()}
    return fracs, episodes


def compute_settings_bias(
    dataset: TidyDataset, lookahead_min: int = 180, lookback_min: int = 180
) -> SettingsBiasResult:
    arr = BacktestArrays.from_dataset(dataset)
    prof = dataset.profile
    target = np.array([prof.target_at(ts) for ts in arr.timestamps], dtype=float)
    lookahead = lookahead_min // GRID_MINUTES
    lookback = lookback_min // GRID_MINUTES

    bg = arr.bg[np.isfinite(arr.bg)]
    tir = float(np.mean((bg >= TIR_LOW) & (bg <= TIR_HIGH))) if bg.size else float("nan")
    tbr_70 = float(np.mean(bg < HYPO)) if bg.size else float("nan")
    tbr_54 = float(np.mean(bg < SEVERE)) if bg.size else float("nan")
    tar = float(np.mean(bg > TIR_HIGH)) if bg.size else float("nan")

    bolus_idx = np.where(arr.bolus > 0)[0]
    meal_idx = bolus_idx[arr.near_carb[bolus_idx]]
    corr_idx = bolus_idx[~arr.near_carb[bolus_idx]]

    hour = (arr.minute_of_day // 60).astype(int)
    tbr_by_hour = np.full(24, np.nan)
    for hr in range(24):
        sel = arr.bg[(hour == hr) & np.isfinite(arr.bg)]
        if sel.size:
            tbr_by_hour[hr] = float(np.mean(sel < HYPO))

    drift, n_nights = _overnight_drift(arr)
    attribution, n_episodes = _hypo_attribution(arr, lookback)

    return SettingsBiasResult(
        n_days=round(dataset.coverage_days(), 2),
        target_mean=float(np.nanmean(target)),
        tir=tir,
        tbr_70=tbr_70,
        tbr_54=tbr_54,
        tar_180=tar,
        meal_bolus=_post_bolus_stats(arr, target, meal_idx, lookahead),
        correction_bolus=_post_bolus_stats(arr, target, corr_idx, lookahead),
        overnight_drift_mgdl_per_h=drift,
        n_fasting_nights=n_nights,
        tbr_by_hour=tbr_by_hour,
        hypo_attribution=attribution,
        n_hypo_episodes=n_episodes,
    )


def _bolus_finding(stats: PostBolusStats, kind: str, setting: str) -> str:
    if stats.n < 5:
        return f"- {kind}: only {stats.n} events — too few to read a pattern yet."
    pct = stats.frac_to_hypo * 100
    line = (
        f"- {kind}: {stats.n} events; **{pct:.0f}%** were followed by a low (<70) within "
        f"3h (median nadir {stats.median_nadir:.0f}, typical drop {stats.median_drop:.0f})."
    )
    if pct >= 25:
        line += (
            f" This is a notable rate — consistent with {kind.lower()} insulin running "
            f"strong ({setting} possibly too aggressive). Worth reviewing with your care team."
        )
    elif pct <= 5:
        line += f" Low hypo rate — no sign {setting} is too aggressive here."
    return line


def render_settings_bias_markdown(res: SettingsBiasResult) -> str:
    parts = [
        "# Settings-bias report (observational)",
        "",
        "> Decision support only — patterns from your own CGM + treatments, **not** the "
        "neural model and **not** a dose recommendation. Use it to spot tendencies to "
        "review with your care team; never change a setting from this alone.",
        "",
        f"- Window: {res.n_days} days  |  target ~{res.target_mean:.0f} mg/dL  |  "
        f"{res.n_hypo_episodes} hypo episodes (<70)",
        f"- Time in range {res.tir * 100:.0f}%  |  below 70: **{res.tbr_70 * 100:.1f}%**  "
        f"|  below 54: {res.tbr_54 * 100:.1f}%  |  above 180: {res.tar_180 * 100:.0f}%",
        "",
        "## Do you go low after dosing?",
        "",
        _bolus_finding(res.meal_bolus, "Meal boluses", "carb ratio"),
        _bolus_finding(res.correction_bolus, "Correction boluses", "ISF"),
        "",
        "## Overnight (fasting) drift",
        "",
    ]
    if res.n_fasting_nights == 0:
        parts.append("- No clean fasting nights found in this window.")
    else:
        d = res.overnight_drift_mgdl_per_h
        line = (
            f"- Across {res.n_fasting_nights} fasting nights, glucose drifts "
            f"**{d:+.1f} mg/dL per hour** (0–6am)."
        )
        if d <= -5:
            line += (
                " A steady downward drift while fasting is consistent with net overnight "
                "over-delivery (basal may be high). Review with your care team."
            )
        elif d >= 5:
            line += " An upward drift overnight may reflect dawn phenomenon or low basal."
        else:
            line += " Roughly flat — no strong overnight bias."
        parts.append(line)

    parts += ["", "## Where do the lows come from?", ""]
    if res.n_hypo_episodes == 0:
        parts.append("- No hypos in this window.")
    else:
        a = res.hypo_attribution
        parts.append("| Source | Share of low time |")
        parts.append("|---|---|")
        labels = {
            "post_meal": "After a meal bolus",
            "post_correction": "After a correction",
            "overnight": "Overnight / fasting",
            "other": "Other",
        }
        for key, label in labels.items():
            parts.append(f"| {label} | {a.get(key, 0.0) * 100:.0f}% |")
        parts.append("")
        parts.append(
            "If most of your low time follows meal boluses, your example (overriding 5→3 "
            "units and still going low) is a *pattern*, not a one-off — the conversation "
            "to have is about meal dosing, not the twin."
        )
    return "\n".join(parts)
