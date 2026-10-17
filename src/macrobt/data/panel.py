"""Build the monthly return panel used by the strategies.

Assets
------
* 10 ETFs across equities, government bonds and commodities. Monthly total
  returns from adjusted closes, minus the 3-month T-bill rate.
* 9 G10 currencies against the USD. The excess return of holding currency
  j for a month is the spot move plus the interest differential earned:

      r_j(t) = S_j(t) / S_j(t-1) - 1 + (i_j(t-1) - i_US(t-1)) / 12

  with S in USD per unit of foreign currency and rates from the OECD 3-month
  interbank series on FRED. This is the return of a rolled 1-month forward,
  up to the small cross term, because covered interest parity ties the
  forward discount to the rate differential.

Rates for month t-1 are used for month t, so nothing in a return depends on
information from inside the month.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import fred

ETFS = {
    "SPY": "equity", "EFA": "equity", "EEM": "equity", "EWJ": "equity",
    "IEF": "bond", "TLT": "bond", "TIP": "bond",
    "GLD": "commodity", "SLV": "commodity", "DBC": "commodity",
}

# currency: (FRED spot id, quoted as USD per FCY?, FRED 3m rate id)
FX = {
    "EUR": ("DEXUSEU", True, "IR3TIB01EZM156N"),
    "GBP": ("DEXUSUK", True, "IR3TIB01GBM156N"),
    "AUD": ("DEXUSAL", True, "IR3TIB01AUM156N"),
    "NZD": ("DEXUSNZ", True, "IR3TIB01NZM156N"),
    "JPY": ("DEXJPUS", False, "IR3TIB01JPM156N"),
    "CAD": ("DEXCAUS", False, "IR3TIB01CAM156N"),
    "CHF": ("DEXSZUS", False, "IR3TIB01CHM156N"),
    "NOK": ("DEXNOUS", False, "IR3TIB01NOM156N"),
    "SEK": ("DEXSDUS", False, "IR3TIB01SEM156N"),
}
US_RATE = "IR3TIB01USM156N"
TBILL = "TB3MS"

FRED_SERIES = [v[0] for v in FX.values()] + [v[2] for v in FX.values()] + [US_RATE, TBILL]

# One-way transaction costs in bps of notional traded, by asset class.
COSTS_BPS = {"equity": 5.0, "bond": 5.0, "commodity": 10.0, "fx": 3.0}

# OECD rates are published with a lag; carry a stale print forward at most this many months.
MAX_RATE_FILL = 12


@dataclass
class Panel:
    returns: pd.DataFrame  # monthly excess returns, all assets
    rate_diff: pd.DataFrame  # foreign minus US rate (% p.a.), FX only, known at month end
    asset_class: dict[str, str]

    @property
    def costs_bps(self) -> dict[str, float]:
        return {a: COSTS_BPS[c] for a, c in self.asset_class.items()}

    def subset(self, classes: set[str]) -> pd.DataFrame:
        cols = [a for a, c in self.asset_class.items() if c in classes]
        return self.returns[cols]


def month_end(df: pd.DataFrame) -> pd.DataFrame:
    """Last observation in each calendar month, indexed by month end."""
    return df.resample("ME").last()


def drop_incomplete_last_month(df: pd.DataFrame, last_obs: pd.Timestamp) -> pd.DataFrame:
    """Drop the final month if the data stops well before its end."""
    if len(df) and (df.index[-1] - last_obs).days > 5:
        return df.iloc[:-1]
    return df


def etf_excess_returns(prices: pd.DataFrame, tbill_pct: pd.Series) -> pd.DataFrame:
    monthly = month_end(prices)
    monthly = drop_incomplete_last_month(monthly, prices.dropna(how="all").index[-1])
    r = monthly.pct_change(fill_method=None)
    rf = (tbill_pct.resample("ME").last() / 1200).shift(1).reindex(r.index)
    return r.sub(rf, axis=0)


def fx_excess_returns(spot_daily: pd.DataFrame, rates_monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usd_per_fcy = {}
    for ccy, (sid, direct, _) in FX.items():
        s = spot_daily[sid]
        usd_per_fcy[ccy] = s if direct else 1.0 / s
    spot = month_end(pd.DataFrame(usd_per_fcy))
    spot = drop_incomplete_last_month(spot, spot_daily.dropna(how="all").index[-1])

    rates = rates_monthly.resample("ME").last().ffill(limit=MAX_RATE_FILL)
    us = rates[US_RATE]
    diff = pd.DataFrame({ccy: rates[rid] - us for ccy, (_, _, rid) in FX.items()}).reindex(spot.index)
    carry = diff.shift(1) / 1200
    r = spot.pct_change(fill_method=None) + carry
    return r, diff


def load_panel(data_dir: str | Path = "data") -> Panel:
    """Assemble the panel from files written by ``scripts/download_data.py``."""
    data_dir = Path(data_dir)
    raw = fred.load(FRED_SERIES, data_dir / "fred")
    spot_ids = [v[0] for v in FX.values()]
    rate_ids = [v[2] for v in FX.values()] + [US_RATE]
    spot = raw[spot_ids].dropna(how="all")
    rates = raw[rate_ids].dropna(how="all")

    prices = pd.read_csv(data_dir / "etf_prices.csv", index_col=0, parse_dates=True).sort_index()
    etf = etf_excess_returns(prices[list(ETFS)], raw[TBILL].dropna())
    fx, diff = fx_excess_returns(spot, rates)

    returns = pd.concat([etf, fx], axis=1).sort_index()
    returns = returns.loc[returns.notna().any(axis=1)]
    # Drop the first row: pct_change leaves it empty and rates need one month of history.
    returns = returns.iloc[1:]
    classes = {**ETFS, **{c: "fx" for c in FX}}
    return Panel(returns=returns.replace([np.inf, -np.inf], np.nan), rate_diff=diff.reindex(returns.index), asset_class=classes)
