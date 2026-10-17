import numpy as np
import pandas as pd
import pytest

from macrobt import metrics as m


def series(values, freq="ME"):
    return pd.Series(values, index=pd.date_range("2020-01-31", periods=len(values), freq=freq))


def test_annualised_return_of_constant_returns():
    r = series([0.01] * 24)
    assert m.annualised_return(r, 12) == pytest.approx(1.01**12 - 1)


def test_sharpe_matches_definition():
    rng = np.random.default_rng(1)
    r = series(rng.normal(0.005, 0.03, 240))
    expected = r.mean() / r.std(ddof=1) * np.sqrt(12)
    assert m.sharpe_ratio(r) == pytest.approx(expected)


def test_max_drawdown_simple_path():
    # wealth: 1.1, 0.99, 1.188 -> peak 1.1 then trough 0.99 => -10%
    r = series([0.10, -0.10, 0.20])
    assert m.max_drawdown(r) == pytest.approx(-0.10)


def test_drawdown_counts_loss_from_start():
    r = series([-0.05, 0.02])
    assert m.drawdown_series(r).iloc[0] == pytest.approx(-0.05)


def test_hit_rate_ignores_flat_periods():
    assert m.hit_rate(series([0.01, -0.01, 0.0, 0.02])) == pytest.approx(2 / 3)


def test_summary_keys_and_nan_handling():
    r = series([np.nan, 0.01, -0.02, 0.03])
    s = m.summary(r, turnover=series([0, 0.5, 0.5, 0.5]))
    assert s["periods"] == 3
    assert s["ann_turnover"] == pytest.approx(0.5 * 12)
    assert {"sharpe", "max_drawdown", "calmar", "skew"} <= s.keys()
