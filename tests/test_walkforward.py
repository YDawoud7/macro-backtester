import numpy as np
import pandas as pd

from macrobt.walkforward import select_parameters, stitch_weights

IDX = pd.date_range("2000-01-31", periods=120, freq="ME")


def test_picks_the_better_candidate():
    rng = np.random.default_rng(0)
    good = rng.normal(0.02, 0.02, 120)
    bad = rng.normal(-0.01, 0.02, 120)
    choice = select_parameters(pd.DataFrame({"good": good, "bad": bad}, index=IDX), train_periods=36, min_periods=24)
    assert (choice.dropna() == "good").all()
    assert choice.iloc[:23].isna().all()


def test_selection_does_not_peek():
    rng = np.random.default_rng(1)
    r = pd.DataFrame({"a": rng.normal(0, 0.02, 120), "b": rng.normal(0, 0.02, 120)}, index=IDX)
    c1 = select_parameters(r, train_periods=36, min_periods=24)
    r2 = r.copy()
    r2.iloc[80:] = r2.iloc[80:] * -10  # change the future only
    c2 = select_parameters(r2, train_periods=36, min_periods=24)
    pd.testing.assert_series_equal(c1.iloc[:80], c2.iloc[:80])


def test_stitch_weights():
    choice = pd.Series(["x", "x", "y"], index=IDX[:3])
    wx = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=IDX[:3])
    wy = pd.DataFrame({"A": [-1.0, -1.0, -1.0]}, index=IDX[:3])
    out = stitch_weights(choice, {"x": wx, "y": wy})
    assert out["A"].tolist() == [1.0, 1.0, -1.0]
