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
