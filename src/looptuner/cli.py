"""LoopTuner command-line interface.

Subcommands wrap the same importable functions you can call from a notebook — the
CLI is a thin shell over the library, never a separate code path.

    looptuner ingest   --days 30          # pull Nightscout -> cached tidy dataset
    looptuner train    --epochs 300       # fit the twin, save checkpoint + metadata
    looptuner evaluate                    # held-out accuracy vs trivial baselines

Decision support only. No dosing commands exist, by design.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from looptuner.config import Settings, dataframe_hash, write_run_metadata
from looptuner.eval.baselines import linear_extrapolation, persistence
from looptuner.eval.metrics import rmse
from looptuner.ingest.schema import GRID_MINUTES
from looptuner.ingest.store import load_dataset, save_dataset
from looptuner.model.twin import DEFAULT_HORIZONS_MIN, ForwardSimulator, build_forcing

app = typer.Typer(
    add_completion=False, help="Personal T1D glucose digital twin (suggestions only)."
)
console = Console()

DATASET_SUBDIR = "dataset"
MODEL_FILE = "twin.pt"


def _dataset_path(settings: Settings) -> Path:
    return settings.data_dir / DATASET_SUBDIR


@app.command()
def ingest(
    days: int = typer.Option(30, help="How many days of history to pull."),
    use_synthetic: bool = typer.Option(
        False, "--synthetic", help="Generate synthetic data instead."
    ),
    synthetic_days: int = typer.Option(14, help="Days of synthetic data if --synthetic."),
    seed: int = typer.Option(0, help="Synthetic seed."),
):
    """Pull data from Nightscout (or generate synthetic) into a cached tidy dataset."""
    settings = Settings.load()
    if use_synthetic:
        from looptuner.ingest.synthetic import generate_synthetic_dataset

        ds, _ = generate_synthetic_dataset(
            n_days=synthetic_days, seed=seed, timezone_name=settings.local_timezone
        )
    else:
        if not settings.nightscout_url:
            console.print(
                "[red]NIGHTSCOUT_URL not set. Copy .env.example to .env and fill it in.[/]"
            )
            raise typer.Exit(1)
        from looptuner.ingest.nightscout import NightscoutClient

        client = NightscoutClient(
            settings.nightscout_url,
            token=settings.nightscout_token,
            api_secret=settings.nightscout_api_secret,
        )
        end = pd.Timestamp.now("UTC")
        start = end - pd.Timedelta(days=days)
        with console.status(f"Pulling {days} days from Nightscout..."):
            ds = client.fetch_dataset(start, end)
        client.close()

    out = _dataset_path(settings)
    save_dataset(ds, out)
    summary = ds.summary()
    console.print_json(json.dumps(summary))
    console.print(f"[green]Saved dataset[/] -> {out}  (hash {dataframe_hash(ds.frame)})")


@app.command()
def train(
    epochs: int = typer.Option(300, help="Training epochs."),
    val_days: int = typer.Option(1, help="Held-out validation days (most recent)."),
    harmonics: int = typer.Option(3, help="Circadian Fourier harmonics for ISF/CR."),
    train_horizon_min: int = typer.Option(120, help="Prediction window length (min)."),
    residual: bool = typer.Option(False, "--residual", help="Enable the neural residual term."),
    device: str = typer.Option("cpu", help="cpu or cuda."),
    seed: int = typer.Option(0),
):
    """Fit the twin on the cached dataset and save a checkpoint with provenance."""
    settings = Settings.load()
    ds = load_dataset(_dataset_path(settings))
    sim = ForwardSimulator.from_dataset(
        ds, device=device, seed=seed, n_harmonics=harmonics, use_residual=residual
    )
    with console.status(f"Training {epochs} epochs..."):
        forcing, result = sim.fit(
            ds, val_days=val_days, epochs=epochs, train_horizon_min=train_horizon_min
        )
    model_path = settings.runs_dir / MODEL_FILE
    sim.save(str(model_path))
    write_run_metadata(
        settings.runs_dir / "train_meta.json",
        data_hash=dataframe_hash(ds.frame),
        source=ds.source,
        coverage_days=ds.coverage_days(),
        epochs=epochs,
        best_val=result.best_val,
        best_epoch=result.best_epoch,
        harmonics=harmonics,
        residual=residual,
    )
    console.print(
        f"[green]Trained[/] best_val={result.best_val:.2f} @epoch {result.best_epoch}; "
        f"saved -> {model_path}"
    )


@app.command()
def evaluate(
    horizons: str = typer.Option(
        ",".join(str(h) for h in DEFAULT_HORIZONS_MIN), help="Comma-separated horizons (min)."
    ),
    val_days: int = typer.Option(1, help="Held-out days to evaluate (most recent)."),
):
    """Score the saved twin on held-out days vs persistence and linear baselines."""
    settings = Settings.load()
    ds = load_dataset(_dataset_path(settings))
    sim = ForwardSimulator.load(str(settings.runs_dir / MODEL_FILE))
    forcing = build_forcing(ds, ds.profile.dia_hours * 60.0)
    horizons_min = tuple(int(x) for x in horizons.split(","))
    n_days = len(forcing.days)
    val_codes = set(range(max(0, n_days - val_days), n_days))
    metrics = sim.evaluate_horizons(forcing, val_codes, horizons_min=horizons_min)

    table = Table(title="Held-out accuracy: twin vs trivial baselines")
    cols = ("Horizon", "Twin RMSE", "Persist", "Linear", "Verdict", "MAPE%", "Hypo recall", "n")
    for col in cols:
        table.add_column(col)
    bg = forcing.bg.cpu().numpy()
    import numpy as np

    for h_min in horizons_min:
        h = h_min // GRID_MINUTES
        starts = sim._eligible_starts(forcing, val_codes, h)
        pp, pl, act = [], [], []
        for s in starts:
            hist = bg[max(0, s - 6) : s + 1]
            hist = hist[np.isfinite(hist)]
            a = bg[min(forcing.n - 1, s + h)]
            if hist.size >= 2 and np.isfinite(a):
                pp.append(persistence(hist, h)[-1])
                pl.append(linear_extrapolation(hist, h)[-1])
                act.append(a)
        bp, bl = rmse(np.array(pp), np.array(act)), rmse(np.array(pl), np.array(act))
        mm = metrics.get(h_min, {})
        twin = mm.get("rmse", float("nan"))
        verdict = "[green]beats[/]" if twin < min(bp, bl) else "[yellow]loses[/]"
        recall = mm.get("hypo_recall", float("nan"))
        table.add_row(
            f"{h_min}m",
            f"{twin:.1f}",
            f"{bp:.1f}",
            f"{bl:.1f}",
            verdict,
            f"{mm.get('mape', float('nan')):.1f}",
            f"{recall:.2f}" if recall == recall else "—",
            str(int(mm.get("n", 0))),
        )
    console.print(table)


@app.command()
def backtest(
    horizons: str = typer.Option(",".join(str(h) for h in DEFAULT_HORIZONS_MIN)),
    test_days: int = typer.Option(3, help="Most-recent days to walk forward over."),
    epochs: int = typer.Option(150, help="Epochs per expanding-window retrain."),
    stride: int = typer.Option(1, help="Anchor stride (1 = every 5-min step)."),
    seed: int = typer.Option(0),
):
    """Walk-forward backtest: predict the past with no future leakage, vs baselines."""
    from looptuner.backtest import render_markdown_report, run_backtest
    from looptuner.backtest.report import append_benchmark_log

    settings = Settings.load()
    ds = load_dataset(_dataset_path(settings))
    horizons_min = tuple(int(x) for x in horizons.split(","))

    def progress(d, codes):
        console.print(f"  walk-forward: day {codes.index(d) + 1}/{len(codes)} (retraining)...")

    df, meta = run_backtest(
        ds, horizons_min=horizons_min, test_days=test_days, epochs=epochs,
        anchor_stride=stride, seed=seed, progress=progress,
    )
    _write_backtest_outputs(settings, df, meta, tag="backtest")
    append_benchmark_log(settings.runs_dir / "benchmark_log.parquet", df, meta)
    console.print(render_markdown_report(df, meta))


@app.command()
def shadow(
    hours: float = typer.Option(24.0, help="Trailing window to health-check."),
    horizons: str = typer.Option("30,60,120"),
    epochs: int = typer.Option(200),
):
    """Fast daily health-check: backtest only the last N hours against prior history."""
    from looptuner.backtest import run_backtest
    from looptuner.backtest.report import append_benchmark_log

    settings = Settings.load()
    ds = load_dataset(_dataset_path(settings))
    horizons_min = tuple(int(x) for x in horizons.split(","))
    df, meta = run_backtest(
        ds, horizons_min=horizons_min, epochs=epochs, last_hours=hours
    )
    _write_backtest_outputs(settings, df, meta, tag="shadow")
    append_benchmark_log(settings.runs_dir / "benchmark_log.parquet", df, meta)
    if df.empty:
        console.print("[yellow]No predictions in the shadow window.[/]")
        return
    ref = 120 if 120 in df["horizon_min"].unique() else int(df["horizon_min"].max())
    sub = df[df["horizon_min"] == ref]
    worst = sub.loc[sub["abs_err"].idxmax()]
    console.print(
        f"[bold]Shadow ({hours:.0f}h)[/]: {ref}min MAPE "
        f"{sub['pct_err'].mean():.1f}%  RMSE {(sub['signed_err'] ** 2).mean() ** 0.5:.1f}.  "
        f"Worst miss {worst['timestamp']}: predicted {worst['pred']:.0f}, actual "
        f"{worst['actual']:.0f} ({worst['signed_err']:+.0f})."
    )


@app.command()
def scenario(
    bolus: float = typer.Option(0.0, help="Hypothetical bolus now (U)."),
    carbs: float = typer.Option(0.0, help="Hypothetical carbs now (g)."),
    carb_absorb: float = typer.Option(180.0, help="Carb absorption (min)."),
    at_offset_min: float = typer.Option(0.0, help="Minutes from now for the input."),
    isf_scale: float = typer.Option(1.0, help="Multiply ISF over [from_hour,to_hour)."),
    cr_scale: float = typer.Option(1.0, help="Multiply CR over [from_hour,to_hour)."),
    from_hour: int = typer.Option(0),
    to_hour: int = typer.Option(24),
    basal: float = typer.Option(-1.0, help="Forward basal override U/hr (<0 = keep)."),
    horizon_min: int = typer.Option(360, help="Forecast horizon (min)."),
    no_bands: bool = typer.Option(False, "--no-bands", help="Skip conformal bands (faster)."),
):
    """Forecast BG for the next 1-6h from the current state under a what-if input."""
    from looptuner.scenario import (
        CounterfactualSpec,
        calibrate_conformal,
        scenario_forecast,
        summarize_forecast,
    )

    settings = Settings.load()
    ds = load_dataset(_dataset_path(settings))
    sim = ForwardSimulator.load(str(settings.runs_dir / MODEL_FILE))
    spec = CounterfactualSpec()
    if isf_scale != 1.0:
        spec.with_isf_scale(from_hour, to_hour, isf_scale)
    if cr_scale != 1.0:
        spec.with_cr_scale(from_hour, to_hour, cr_scale)
    if bolus > 0:
        spec.added_boluses.append((at_offset_min, bolus))
    if carbs > 0:
        spec.added_carbs.append((at_offset_min, carbs, carb_absorb))
    if basal >= 0:
        import numpy as np

        spec.basal_rate_by_hour = np.full(24, basal)

    conf = None if no_bands else calibrate_conformal(sim, ds)
    res = scenario_forecast(sim, ds, spec, conformal=conf, horizon_min=horizon_min)
    s = summarize_forecast(res)
    console.print(
        f"[bold]Forecast from {res['anchor_time']} (BG {s['anchor_bg']:.0f})[/]  "
        f"end {s['end_bg']:.0f}, min {s['min_bg']:.0f}, max {s['max_bg']:.0f}, "
        f"time<70 {s['frac_below_70']*100:.0f}%, Δ vs no-change {s['delta_vs_baseline_end']:+.0f}"
    )
    _print_trajectory(res)


@app.command()
def counterfactual(
    day: str = typer.Option(None, help="Day to replay (YYYY-MM-DD; default last)."),
    isf_scale: float = typer.Option(1.0),
    cr_scale: float = typer.Option(1.0),
    from_hour: int = typer.Option(0),
    to_hour: int = typer.Option(24),
    basal: float = typer.Option(-1.0, help="Basal override U/hr (<0 = keep)."),
):
    """Replay a real day under proposed settings vs what actually happened."""
    import numpy as np

    from looptuner.scenario import CounterfactualSpec, counterfactual_replay

    settings = Settings.load()
    ds = load_dataset(_dataset_path(settings))
    sim = ForwardSimulator.load(str(settings.runs_dir / MODEL_FILE))
    spec = CounterfactualSpec()
    if isf_scale != 1.0:
        spec.with_isf_scale(from_hour, to_hour, isf_scale)
    if cr_scale != 1.0:
        spec.with_cr_scale(from_hour, to_hour, cr_scale)
    if basal >= 0:
        spec.basal_rate_by_hour = np.full(24, basal)

    res = counterfactual_replay(sim, ds, spec, day=day)
    cf, act = res["counterfactual"], res["actual"]
    console.print(
        f"[bold]Counterfactual replay {res['day']}[/]  "
        f"actual mean {np.nanmean(act):.0f} (TIR {_tir(act)*100:.0f}%) -> "
        f"counterfactual mean {np.nanmean(cf):.0f} (TIR {_tir(cf)*100:.0f}%)"
    )
    table = Table(title="Hourly: actual vs counterfactual BG")
    for c in ("Time", "Actual", "Counterfactual", "Δ"):
        table.add_column(c)
    for i in range(0, len(res["times"]), 12):  # hourly
        a, c = act[i], cf[i]
        astr = f"{a:.0f}" if np.isfinite(a) else "—"
        table.add_row(str(res["times"][i])[11:16], astr, f"{c:.0f}",
                      f"{c-a:+.0f}" if np.isfinite(a) else "—")
    console.print(table)


def _tir(bg, low=70, high=180):
    import numpy as np

    b = np.asarray(bg, float)
    b = b[np.isfinite(b)]
    return float(np.mean((b >= low) & (b <= high))) if b.size else float("nan")


def _print_trajectory(res: dict) -> None:

    table = Table(title="Predicted trajectory")
    cols = ["Time", "BG"]
    has_bands = bool(res["bands"])
    if has_bands:
        cols += ["90% lo", "90% hi"]
    for c in cols:
        table.add_column(c)
    pt = res["point"]
    for i in range(0, len(res["times"]), 6):  # every 30 min
        row = [str(res["times"][i])[11:16], f"{pt[i]:.0f}"]
        if has_bands and 0.9 in res["bands"]:
            b = res["bands"][0.9]
            row += [f"{b['lo'][i]:.0f}", f"{b['hi'][i]:.0f}"]
        table.add_row(*row)
    console.print(table)


@app.command()
def inverse(
    models: int = typer.Option(8, help="Ensemble size (leave-one-day-out)."),
    epochs: int = typer.Option(200, help="Epochs per ensemble member."),
    coverage: float = typer.Option(0.9, help="Credible-interval coverage."),
    seed: int = typer.Option(0),
):
    """Extract per-hour ISF(t)/CR(t) with credible intervals; write a settings report."""
    from looptuner.model.inverse import recommendations, run_inverse
    from looptuner.recommend_report import render_inverse_markdown, write_inverse_json

    settings = Settings.load()
    ds = load_dataset(_dataset_path(settings))

    def progress(i, n):
        console.print(f"  ensemble member {i + 1}/{n} (leave-one-day-out)...")

    result = run_inverse(
        ds, n_models=models, epochs=epochs, base_seed=seed,
        coverage_levels=(0.5, coverage), progress=progress,
    )
    recs = recommendations(result, coverage=coverage)
    md = render_inverse_markdown(result, recs, coverage=coverage)
    reports = settings.runs_dir / "reports"
    ts = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S")
    (reports / f"inverse_{ts}.md").write_text(md)
    write_inverse_json(result, recs, reports / f"inverse_{ts}.json", coverage=coverage)
    console.print(f"[green]Wrote[/] {reports}/inverse_{ts}.md (+ .json)")
    console.print(md)


@app.command(name="train-incremental")
def train_incremental_cmd(
    epochs: int = typer.Option(300, help="Training epochs."),
    horizon_min: int = typer.Option(60, help="Validation horizon (min)."),
    seed: int = typer.Option(0),
):
    """Retrain on full history; promote only if it beats the current model (latest day)."""
    from looptuner.incremental import train_incremental

    settings = Settings.load()
    ds = load_dataset(_dataset_path(settings))
    with console.status(f"Retraining on {ds.coverage_days():.1f} days..."):
        res = train_incremental(
            ds, settings.runs_dir, epochs=epochs, horizon_min=horizon_min, seed=seed
        )
    prev = "—" if res.previous_score != res.previous_score else f"{res.previous_score:.1f}"
    verdict = (
        "[green]promoted to current[/]" if res.promoted else "[yellow]kept previous current[/]"
    )
    console.print(
        f"New model {horizon_min}min MAPE {res.new_score:.1f}% (previous {prev}%) -> {verdict}\n"
        f"checkpoint: {res.new_checkpoint}"
    )


@app.command()
def checkpoints():
    """List tracked model checkpoints and their validation scores."""
    from looptuner.incremental import load_registry

    settings = Settings.load()
    registry = load_registry(settings.runs_dir)
    if not registry:
        console.print("[yellow]No checkpoints yet — run `train-incremental`.[/]")
        raise typer.Exit(0)
    table = Table(title="Checkpoint registry")
    for c in ("Timestamp", "Val MAPE%", "Prev", "Promoted", "Days", "Data hash"):
        table.add_column(c)
    for e in registry:
        table.add_row(
            e["timestamp"], str(e["val_score_mape"]),
            str(e.get("previous_score_mape", "—")), "yes" if e["promoted"] else "no",
            str(e["n_days"]), e["data_hash"],
        )
    console.print(table)


@app.command(name="drift-report")
def drift_report(
    horizon_min: int = typer.Option(60, help="Horizon to monitor (min)."),
    days: int = typer.Option(7, help="Trailing window to check."),
):
    """Per-hour predicted-vs-actual error over recent days; flags sudden drops."""
    from looptuner.drift import compute_drift, render_drift_markdown

    settings = Settings.load()
    ds = load_dataset(_dataset_path(settings))
    sim = ForwardSimulator.load(str(settings.runs_dir / MODEL_FILE))
    with console.status(f"Scoring last {days} days at {horizon_min}min..."):
        res = compute_drift(ds, sim, horizon_min=horizon_min, days=days)
    reports = settings.runs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S")
    (reports / f"drift_{ts}.md").write_text(render_drift_markdown(res))
    console.print(render_drift_markdown(res))
    if res.flags:
        hrs = ", ".join(f"{f['hour']:02d}:00" for f in res.flags)
        console.print(f"[yellow]Flagged hours: {hrs}[/]")


@app.command(name="benchmark-trend")
def benchmark_trend():
    """Show twin quality over time from the persistent benchmark log."""
    settings = Settings.load()
    log_path = settings.runs_dir / "benchmark_log.parquet"
    if not log_path.exists():
        console.print("[yellow]No benchmark log yet — run `backtest` or `shadow` first.[/]")
        raise typer.Exit(0)
    log = pd.read_parquet(log_path)
    table = Table(title="Benchmark trend (per run, per horizon)")
    for c in ("Run (UTC)", "Mode", "Horizon", "RMSE", "MAPE%", "Cov90%", "n"):
        table.add_column(c)
    for _, r in log.sort_values("run_ts").iterrows():
        table.add_row(
            str(r["run_ts"])[:19], str(r["mode"]), f"{int(r['horizon_min'])}m",
            f"{r['rmse']:.1f}", f"{r['mape']:.1f}",
            f"{r['cov90'] * 100:.0f}" if pd.notna(r["cov90"]) else "—", str(int(r["n"])),
        )
    console.print(table)


def _write_backtest_outputs(settings: Settings, df: pd.DataFrame, meta: dict, tag: str) -> None:
    from looptuner.backtest import render_markdown_report

    reports = settings.runs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%S")
    (reports / f"{tag}_{ts}.md").write_text(render_markdown_report(df, meta))
    if not df.empty:
        df.to_parquet(reports / f"{tag}_{ts}_records.parquet")
    console.print(f"[green]Wrote[/] {reports}/{tag}_{ts}.md")


if __name__ == "__main__":
    app()
