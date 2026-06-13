"""Walk-forward backtest / shadow-mode harness — the project's core deliverable.

Predicts BG forward through past data in strict chronological order, using only
information available before each anchor (no future-leakage), with conformal
intervals, baseline comparison, calibration, and error decomposition.
"""

from looptuner.backtest.engine import BacktestArrays, run_backtest
from looptuner.backtest.report import render_markdown_report, twin_quality_by_day

__all__ = [
    "BacktestArrays",
    "run_backtest",
    "render_markdown_report",
    "twin_quality_by_day",
]
