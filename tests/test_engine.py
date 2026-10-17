import numpy as np
import pandas as pd
import pytest

from macrobt.engine import run_backtest

IDX = pd.date_range("2020-01-31", periods=6, freq="ME")


def frame(data):
    return pd.DataFrame(data, index=IDX)


def test_weights_are_lagged_one_period():
    r = frame({"A": [0.0, 0.10, -0.05, 0.02, 0.0, 0.0]})
    w = frame({"A": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]})  # decided at t0
    res = run_backtest(r, w)
    # Position held only in period 1, so only the +10% is earned.
    assert res.gross_returns.sum() == pytest.approx(0.10)


def test_perfect_foresight_needs_lag_zero():
    rng = np.random.default_rng(0)
    r = frame({"A": rng.normal(0, 0.05, 6)})
    w = np.sign(r)  # uses the same-period return
    honest = run_backtest(r, w, lag=1).gross_returns
    cheating = run_backtest(r, w, lag=0).gross_returns
    assert cheating.sum() == pytest.approx(r["A"].abs().sum())
    assert honest.sum() < cheating.sum()


def test_costs_charged_on_turnover():
    r = frame({"A": [0.0] * 6, "B": [0.0] * 6})
    w = frame({"A": [1, 1, -1, -1, 0, 0], "B": [0, 0, 0, 0, 0, 0]})
    res = run_backtest(r, w, cost_bps=10)
    # trades: enter +1 (1), flip to -1 (2), exit (1) => 4 units * 10bp
    assert res.turnover.sum() == pytest.approx(4.0)
    assert res.costs.sum() == pytest.approx(4 * 0.001)


def test_drift_reduces_rebalancing_trade():
    # Holding 50/50; A rallies 20% so the drifted book is overweight A.
    r = frame({"A": [0.0, 0.20, 0.0, 0.0, 0.0, 0.0], "B": [0.0] * 6})
    w = frame({"A": [0.5] * 6, "B": [0.5] * 6})
    res = run_backtest(r, w)
    drifted_a = 0.5 * 1.2 / 1.1
    drifted_b = 0.5 / 1.1
    expected_trade = abs(0.5 - drifted_a) + abs(0.5 - drifted_b)
    assert res.turnover.loc[IDX[2]] == pytest.approx(expected_trade)


def test_per_asset_costs_and_missing_returns():
    r = frame({"A": [0.0] * 6, "B": [np.nan, np.nan, 0.0, 0.0, 0.0, 0.0]})
    w = frame({"A": [1.0] * 6, "B": [1.0] * 6})
    res = run_backtest(r, w, cost_bps={"A": 0, "B": 50})
    # B can only be held once it has returns
    assert res.weights.loc[IDX[1], "B"] == 0
    assert res.weights.loc[IDX[2], "B"] == 1
    assert res.costs.sum() == pytest.approx(0.005)
