"""Charts for backtest results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

from .metrics import drawdown_series

PALETTE = ["#1f3b73", "#e76f51", "#2a9d8f", "#8d99ae", "#b5838d"]


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def equity_and_drawdown(series: dict[str, pd.Series], path: str | Path, title: str) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for (name, r), color in zip(series.items(), PALETTE):
        wealth = (1 + r.dropna()).cumprod()
        ax1.plot(wealth.index, wealth, label=name, color=color, lw=1.4)
        dd = drawdown_series(r)
        ax2.plot(dd.index, 100 * dd, color=color, lw=1)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title(title)
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)
    _save(fig, path)


def bar_by_bucket(table: pd.DataFrame, path: str | Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(table.columns)
    x = range(len(table.index))
    for j, (col, color) in enumerate(zip(table.columns, PALETTE)):
        ax.bar([i + j * width for i in x], table[col], width=width, label=col, color=color)
    ax.set_xticks([i + width * (len(table.columns) - 1) / 2 for i in x])
    ax.set_xticklabels(table.index)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, path)
