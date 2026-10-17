import numpy as np
import pandas as pd
import pytest

from macrobt.data.fred import parse_fred_csv
from macrobt.data.panel import FX, US_RATE, etf_excess_returns, fx_excess_returns
from macrobt.data.yahoo import parse_chart_json


def test_parse_fred_handles_dots_and_blanks():
    text = "observation_date,DGS2,DEXUSEU\n2024-01-02,4.33,.\n2024-01-03,,1.09\n"
    df = parse_fred_csv(text)
    assert df.index[0] == pd.Timestamp("2024-01-02")
    assert np.isnan(df.loc["2024-01-02", "DEXUSEU"])
    assert df.loc["2024-01-03", "DEXUSEU"] == pytest.approx(1.09)


def test_parse_fred_legacy_date_column():
    df = parse_fred_csv("DATE,TB3MS\n2020-01-01,1.52\n")
    assert df.index.name == "date"


def test_parse_yahoo_chart():
    payload = {"chart": {"result": [{
        "meta": {"gmtoffset": -14400},
        "timestamp": [1704205800, 1704292200],
        "indicators": {"adjclose": [{"adjclose": [100.0, None]}]},
    }]}}
    s = parse_chart_json(payload)
    assert len(s) == 1 and s.iloc[0] == 100.0
    assert s.index[0] == pd.Timestamp("2024-01-02")


def test_etf_excess_return_subtracts_lagged_tbill():
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-28"])
    prices = pd.DataFrame({"SPY": [100.0, 110.0, 99.0]}, index=idx)
    tbill = pd.Series([6.0, 12.0, 0.0], index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]))
    r = etf_excess_returns(prices, tbill)
    assert r.loc["2024-02-29", "SPY"] == pytest.approx(0.10 - 0.005)
    assert r.loc["2024-03-31", "SPY"] == pytest.approx(-0.10 - 0.01)


def test_fx_excess_return_includes_carry_and_inverts_quotes():
    days = pd.to_datetime(["2024-01-31", "2024-02-29"])
    spot = pd.DataFrame(index=days)
    for ccy, (sid, direct, _) in FX.items():
        spot[sid] = 1.0
    spot.loc["2024-02-29", "DEXJPUS"] = 1.25  # yen weakens 20% vs USD
    months = pd.to_datetime(["2024-01-01", "2024-02-01"])
    rates = pd.DataFrame({rid: 1.0 for _, (_, _, rid) in FX.items()}, index=months)
    rates[US_RATE] = 5.0
    rates["IR3TIB01AUM156N"] = 7.0
    r, diff = fx_excess_returns(spot, rates)
    feb = pd.Timestamp("2024-02-29")
    assert r.loc[feb, "JPY"] == pytest.approx(1 / 1.25 - 1 + (1.0 - 5.0) / 1200)
    assert r.loc[feb, "AUD"] == pytest.approx((7.0 - 5.0) / 1200)
    assert diff.loc[feb, "AUD"] == pytest.approx(2.0)
