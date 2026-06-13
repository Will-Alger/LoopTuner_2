"""LoopTuner Streamlit dashboard.

Run with ``looptuner ui`` (or ``streamlit run src/looptuner/dashboard.py``). Local,
single-user, read-only over your cached dataset + saved model — no auth, no server to
expose. Decision support only: it shows forecasts and suggestions; it never doses.

The heavy lifting is the same library the CLI uses (scenario_forecast, run_backtest,
run_inverse, compute_drift) — the UI is just another front-end over one code path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from looptuner import plots
from looptuner.config import Settings
from looptuner.ingest.store import load_dataset
from looptuner.model.twin import ForwardSimulator
from looptuner.scenario import (
    CounterfactualSpec,
    calibrate_conformal,
    counterfactual_replay,
    scenario_forecast,
    summarize_forecast,
)

st.set_page_config(page_title="LoopTuner", page_icon="🩸", layout="wide")

DATASET_SUBDIR = "dataset"
MODEL_FILE = "twin.pt"


@st.cache_resource(show_spinner=False)
def _load(settings_data_dir: str, settings_runs_dir: str):
    ds = load_dataset(Path(settings_data_dir) / DATASET_SUBDIR)
    model_path = Path(settings_runs_dir) / MODEL_FILE
    sim = ForwardSimulator.load(str(model_path)) if model_path.exists() else None
    return ds, sim


@st.cache_resource(show_spinner=False)
def _conformal(settings_runs_dir: str, _ds, _sim):
    return calibrate_conformal(_sim, _ds)


def _hour_range(label: str, key: str) -> tuple[int, int]:
    lo, hi = st.slider(label, 0, 24, (0, 24), key=key)
    return lo, hi


def main() -> None:
    st.title("🩸 LoopTuner — glucose digital twin")
    st.caption(
        "Decision support only. Suggestions with uncertainty — review every number "
        "and enter changes manually in Loop. Nothing here doses."
    )
    settings = Settings.load()

    try:
        ds, sim = _load(str(settings.data_dir), str(settings.runs_dir))
    except FileNotFoundError:
        st.error(
            "No cached dataset found. Run `looptuner ingest` (or `looptuner ingest "
            "--synthetic`) first, then `looptuner train`."
        )
        return
    if sim is None:
        st.error("No trained model found at runs/twin.pt. Run `looptuner train` first.")
        return

    s = ds.summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coverage", f"{s['coverage_days']} d")
    c2.metric("CGM present", f"{s['bg_present_frac'] * 100:.0f}%")
    c3.metric("Boluses", s["n_boluses"])
    c4.metric("Carb entries", s["n_carb_entries"])

    tab_scn, tab_cf, tab_diag, tab_rec, tab_drift, tab_bias = st.tabs(
        ["Scenario", "Counterfactual day", "Backtest", "ISF/CR", "Drift", "Settings bias"]
    )

    # --- Scenario simulator ------------------------------------------------ #
    with tab_scn:
        st.subheader("What-if forecast from now")
        left, right = st.columns([1, 2])
        with left:
            bolus = st.number_input("Bolus now (U)", 0.0, 20.0, 0.0, 0.5)
            carbs = st.number_input("Carbs now (g)", 0.0, 200.0, 0.0, 5.0)
            absorb = st.number_input("Carb absorption (min)", 60.0, 480.0, 180.0, 15.0)
            horizon = st.slider("Horizon (min)", 60, 360, 240, 30)
            isf_scale = st.slider("ISF ×", 0.5, 2.0, 1.0, 0.05)
            cr_scale = st.slider("CR ×", 0.5, 2.0, 1.0, 0.05)
            isf_from, isf_to = _hour_range("ISF/CR override hours", "scn_hours")
            show_bands = st.checkbox("Conformal bands", value=True)

        spec = CounterfactualSpec()
        if isf_scale != 1.0:
            spec.with_isf_scale(isf_from, isf_to, isf_scale)
        if cr_scale != 1.0:
            spec.with_cr_scale(isf_from, isf_to, cr_scale)
        if bolus > 0:
            spec.added_boluses.append((0.0, bolus))
        if carbs > 0:
            spec.added_carbs.append((0.0, carbs, absorb))

        conf = _conformal(str(settings.runs_dir), ds, sim) if show_bands else None
        res = scenario_forecast(sim, ds, spec, conformal=conf, horizon_min=horizon)
        summ = summarize_forecast(res)
        with right:
            st.pyplot(plots.fig_forecast(res), use_container_width=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "End BG", f"{summ['end_bg']:.0f}",
            f"{summ['delta_vs_baseline_end']:+.0f} vs no-change",
        )
        m2.metric("Min", f"{summ['min_bg']:.0f}")
        m3.metric("Max", f"{summ['max_bg']:.0f}")
        m4.metric("Time <70", f"{summ['frac_below_70'] * 100:.0f}%")

    # --- Counterfactual replay --------------------------------------------- #
    with tab_cf:
        st.subheader("Replay a real day under different settings")
        days = [str(d) for d in ds.day_index().unique()]
        day = st.selectbox("Day", days, index=len(days) - 1)
        cc1, cc2, cc3 = st.columns(3)
        isf_s = cc1.slider("ISF ×", 0.5, 2.0, 1.0, 0.05, key="cf_isf")
        cr_s = cc2.slider("CR ×", 0.5, 2.0, 1.0, 0.05, key="cf_cr")
        basal = cc3.slider("Basal override U/hr (0=keep)", 0.0, 3.0, 0.0, 0.1, key="cf_basal")
        spec = CounterfactualSpec()
        if isf_s != 1.0:
            spec.with_isf_scale(0, 24, isf_s)
        if cr_s != 1.0:
            spec.with_cr_scale(0, 24, cr_s)
        if basal > 0:
            spec.basal_rate_by_hour = np.full(24, basal)
        rep = counterfactual_replay(sim, ds, spec, day=day)
        st.pyplot(plots.fig_counterfactual(rep), use_container_width=True)

    # --- Backtest diagnostics ---------------------------------------------- #
    with tab_diag:
        st.subheader("Backtest diagnostics")
        records = sorted((settings.runs_dir / "reports").glob("*_records.parquet"))
        if not records:
            st.info("No backtest records yet. Run `looptuner backtest` (or `shadow`).")
        else:
            df = pd.read_parquet(records[-1])
            st.caption(f"Latest: {records[-1].name} ({len(df)} predictions)")
            d1, d2 = st.columns(2)
            d1.pyplot(plots.fig_accuracy_by_horizon(df), use_container_width=True)
            d2.pyplot(plots.fig_calibration(df), use_container_width=True)
            st.pyplot(plots.fig_twin_quality(df), use_container_width=True)

    # --- ISF/CR recommendations -------------------------------------------- #
    with tab_rec:
        st.subheader("Per-hour ISF(t) / CR(t) with credible intervals")
        st.caption(
            "Level is pinned to your clinical profile (not identifiable from CGM); "
            "the data drives the time-of-day shape. CR needs announced carbs."
        )
        models = st.slider("Ensemble size", 3, 12, 6)
        epochs = st.slider("Epochs/member", 50, 300, 150, 25)
        if st.button("Run inverse fit (slow)"):
            from looptuner.model.inverse import run_inverse

            with st.spinner(f"Training {models} ensemble members..."):
                result = run_inverse(ds, n_models=models, epochs=epochs)
            st.session_state["inverse"] = result
        if "inverse" in st.session_state:
            st.pyplot(plots.fig_isf_cr(st.session_state["inverse"]), use_container_width=True)

    # --- Drift ------------------------------------------------------------- #
    with tab_drift:
        st.subheader("Drift monitor")
        horizon = st.slider("Horizon (min)", 30, 240, 60, 30, key="drift_h")
        days_n = st.slider("Days", 3, 14, 7, key="drift_days")
        if st.button("Run drift check"):
            from looptuner.drift import compute_drift

            with st.spinner("Scoring recent days..."):
                dr = compute_drift(ds, sim, horizon_min=horizon, days=days_n)
            st.session_state["drift"] = dr
        if "drift" in st.session_state:
            dr = st.session_state["drift"]
            st.pyplot(plots.fig_drift(dr), use_container_width=True)
            if dr.flags:
                st.warning(
                    "Flagged hours: "
                    + ", ".join(f"{f['hour']:02d}:00" for f in dr.flags)
                    + " — consider retraining (`train-incremental`) or check for a "
                    "new infusion site / illness."
                )
            else:
                st.success("No sudden per-hour accuracy drops — the twin is tracking.")

    # --- Settings bias (observational, model-free) ------------------------- #
    with tab_bias:
        st.subheader("Settings-bias check (observational, not the model)")
        st.caption(
            "Do you systematically go low after meals/corrections or overnight? Patterns "
            "to review with your care team — never a dose recommendation."
        )
        from looptuner.settings_bias import compute_settings_bias, render_settings_bias_markdown

        res = compute_settings_bias(ds)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Time in range", f"{res.tir * 100:.0f}%")
        b2.metric("Below 70", f"{res.tbr_70 * 100:.1f}%")
        b3.metric("Above 180", f"{res.tar_180 * 100:.0f}%")
        b4.metric("Overnight drift", f"{res.overnight_drift_mgdl_per_h:+.1f}/h")
        st.pyplot(plots.fig_settings_bias(res), use_container_width=True)
        st.markdown(render_settings_bias_markdown(res))


main()
