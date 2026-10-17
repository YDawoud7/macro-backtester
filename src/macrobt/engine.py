"""Vectorised portfolio backtest.

Conventions
-----------
* ``returns`` holds per-period *excess* returns (over the funding rate) for
  each asset, indexed by period end. A long/short futures-style book earns
  ``sum_i w_i * r_i`` on top of cash, so excess returns are what matter.
* ``weights`` holds target weights (fraction of NAV, can be negative or
  leveraged) decided at the end of period t using information up to t.
* Targets are applied with a lag: weights set at t earn the returns of
  period t + lag. ``lag=1`` is the honest default; ``lag=0`` would let a
  signal trade on the same return it was computed from.
* Between rebalances positions drift with their returns. Turnover at a
  rebalance is measured against the drifted book, and transaction costs are
  charged as ``cost_bps * |trade|`` on that turnover.

Everything is computed with array operations over the whole sample, so a
20-asset monthly backtest over 25 years runs in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics


@dataclass
class BacktestResult:
    returns: pd.Series  # net portfolio excess return per period
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series  # sum of |trades| as a fraction of NAV
    weights: pd.DataFrame  # weights actually held during each period
    contributions: pd.DataFrame  # per-asset gross return contribution
    periods_per_year: int = 12
    name: str = ""
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        s = metrics.summary(self.returns, self.periods_per_year, self.turnover)
        s["gross_sharpe"] = metrics.sharpe_ratio(self.gross_returns, self.periods_per_year)
        s["ann_cost"] = float(self.costs.mean() * self.periods_per_year)
        return s

    def wealth(self) -> pd.Series:
        return (1 + self.returns.fillna(0)).cumprod()


def run_backtest(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float | pd.Series | dict = 0.0,
    lag: int = 1,
    periods_per_year: int = 12,
    name: str = "",
) -> BacktestResult:
    """Simulate holding ``weights`` (lagged) against ``returns``.

    ``cost_bps`` can be a scalar or a per-asset mapping of one-way costs in
    basis points of traded notional.
    """
    if lag < 0:
        raise ValueError("lag must be non-negative")
    cols = returns.columns
    w_target = weights.reindex(index=returns.index, columns=cols)
    r = returns.astype(float)

    # Weights held during period t were decided at t - lag.
    held = w_target.shift(lag)
    held = held.where(r.notna(), 0.0).fillna(0.0)  # no position in an asset without a return
    r_filled = r.fillna(0.0)

    contrib = held * r_filled
    gross = contrib.sum(axis=1)

    # Book at the end of period t-1 after drifting through period t-1's return,
    # expressed as a fraction of end-of-period NAV.
    prev_held = held.shift(1).fillna(0.0)
    prev_r = r_filled.shift(1).fillna(0.0)
    prev_gross = gross.shift(1).fillna(0.0)
    drifted = prev_held * (1 + prev_r)
    drifted = drifted.div(1 + prev_gross, axis=0)

    trades = (held - drifted).abs()
    if isinstance(cost_bps, dict):
        cost_bps = pd.Series(cost_bps)
    if isinstance(cost_bps, pd.Series):
        cost_vec = cost_bps.reindex(cols).fillna(0.0).to_numpy() * 1e-4
    else:
        cost_vec = np.full(len(cols), float(cost_bps) * 1e-4)
    costs = trades.mul(cost_vec, axis=1).sum(axis=1)
    net = gross - costs

    # Trim the warm-up before the first position.
    active = held.abs().sum(axis=1) > 0
    if active.any():
        start = active.idxmax()
        sl = slice(start, None)
    else:
        sl = slice(None)

    return BacktestResult(
        returns=net.loc[sl],
        gross_returns=gross.loc[sl],
        costs=costs.loc[sl],
        turnover=trades.sum(axis=1).loc[sl],
        weights=held.loc[sl],
        contributions=contrib.loc[sl],
        periods_per_year=periods_per_year,
        name=name,
    )
