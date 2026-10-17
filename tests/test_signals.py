import numpy as np
import pandas as pd
import pytest

from macrobt.portfolio import combine, ex_ante_vol, vol_target
from macrobt.signals import carry_weights, ewma_vol, trailing_return, tsmom_weights

IDX = pd.date_range("2000-01-31", periods=60, freq="ME")


def test_trailing_return_compounds():
    r = pd.DataFrame({"A": [0.1, 0.1, 0.1]}, index=IDX[:3])
    out = trailing_return(r, 2)
    assert np.isnan(out["A"].iloc[0])
    assert out["A"].iloc[1] == pytest.approx(0.21)


def test_tsmom_follows_trend_and_scales_by_vol():
    rng = np.random.default_rng(3)
    up = 0.02 + rng.normal(0, 0.01, 60)  # steady riser, low vol
    down = -0.02 + rng.normal(0, 0.04, 60)  # faller, high vol
    r = pd.DataFrame({"UP": up, "DOWN": down}, index=IDX)
    w = tsmom_weights(r, lookback=12).dropna()
    assert (w["UP"] > 0).all() and (w["DOWN"] < 0).all()
    assert w["UP"].abs().mean() > w["DOWN"].abs().mean()


def test_tsmom_uses_no_future_data():
    rng = np.random.default_rng(4)
    r = pd.DataFrame({"A": rng.normal(0, 0.03, 60), "B": rng.normal(0, 0.03, 60)}, index=IDX)
    w_full = tsmom_weights(r)
    w_cut = tsmom_weights(r.iloc[:40])
    pd.testing.assert_frame_equal(w_full.iloc[:40], w_cut)


def test_ewma_vol_of_constant_vol_series():
    rng = np.random.default_rng(5)
    r = pd.DataFrame({"A": rng.normal(0, 0.05, 2000)}, index=pd.date_range("1900-01-31", periods=2000, freq="ME"))
    assert ewma_vol(r, com=60).iloc[-500:].mean().iloc[0] == pytest.approx(0.05 * np.sqrt(12), rel=0.1)


def test_carry_longs_high_yielders():
    diff = pd.DataFrame([[3.0, 2.0, 1.0, 0.0, -1.0, -2.0]], columns=list("ABCDEF"), index=IDX[:1])
    w = carry_weights(diff, k=2)
    assert list(w.iloc[0]) == pytest.approx([0.5, 0.5, 0, 0, -0.5, -0.5])
    assert w.sum(axis=1).iloc[0] == pytest.approx(0.0)


def test_carry_requires_enough_assets():
    diff = pd.DataFrame([[1.0, 2.0, np.nan, np.nan, np.nan, np.nan]], columns=list("ABCDEF"), index=IDX[:1])
    assert carry_weights(diff, k=1).isna().all().all()


def test_vol_target_hits_target_on_iid_returns():
    rng = np.random.default_rng(6)
    idx = pd.date_range("1950-01-31", periods=600, freq="ME")
    r = pd.DataFrame(rng.normal(0, 0.04, (600, 3)), index=idx, columns=list("ABC"))
    w = pd.DataFrame(1.0, index=idx, columns=list("ABC"))
    scaled = vol_target(w, r, target=0.10, max_leverage=100)
    realised = (scaled.shift(1) * r).sum(axis=1).iloc[50:].std() * np.sqrt(12)
    assert realised == pytest.approx(0.10, rel=0.1)


def test_ex_ante_vol_single_asset():
    rng = np.random.default_rng(7)
    idx = pd.date_range("1990-01-31", periods=100, freq="ME")
    r = pd.DataFrame({"A": rng.normal(0, 0.03, 100)}, index=idx)
    w = pd.DataFrame({"A": 2.0}, index=idx)
    v = ex_ante_vol(w, r, window=36)
    assert v.iloc[-1] == pytest.approx(2 * r["A"].iloc[-36:].std() * np.sqrt(12))


def test_combine_allocations():
    a = pd.DataFrame({"X": [1.0, 1.0]}, index=IDX[:2])
    b = pd.DataFrame({"Y": [2.0, np.nan]}, index=IDX[:2])
    c = combine({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    assert c.loc[IDX[0]].tolist() == [0.5, 1.0]
    assert np.isnan(c.loc[IDX[1], "Y"])
