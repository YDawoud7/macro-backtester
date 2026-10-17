"""Trading signals. Every function only uses data up to and including t.

Time-series momentum (Moskowitz, Ooi and Pedersen, 2012)
    Go long an asset whose excess return over the past ``lookback`` periods
    is positive and short if negative, sized inversely to its ex-ante
    volatility so each position carries similar risk.

FX carry
    Each period rank currencies by their short-rate differential versus the
    USD. Go long the ``k`` highest yielders and short the ``k`` lowest,
    equally weighted, so the book is dollar neutral.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol(returns: pd.DataFrame, com: float = 12, min_periods: int = 12, periods_per_year: int = 12) -> pd.DataFrame:
    """Annualised exponentially weighted volatility (demeaned with the EW mean)."""
    var = returns.ewm(com=com, min_periods=min_periods).var(bias=False)
    return np.sqrt(var * periods_per_year)


def trailing_return(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Compounded return over the last ``lookback`` periods; NaN until a full window exists."""
    log_r = np.log1p(returns)
    return np.expm1(log_r.rolling(lookback, min_periods=lookback).sum())


def tsmom_weights(
    returns: pd.DataFrame,
    lookback: int = 12,
    vol_com: float = 12,
    asset_vol_target: float = 0.40,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Volatility-scaled time-series momentum weights.

    Each asset gets ``sign(trailing return) * asset_vol_target / sigma``,
    then the book is divided by the number of assets with a signal that
    period, so adding assets diversifies rather than levers up. 40% per asset
    follows Moskowitz, Ooi and Pedersen; the portfolio is usually re-scaled
    to a portfolio-level target afterwards.
    """
    sig = np.sign(trailing_return(returns, lookback))
    vol = ewma_vol(returns, com=vol_com, min_periods=max(12, lookback // 2), periods_per_year=periods_per_year)
    raw = sig * asset_vol_target / vol
    raw = raw.replace([np.inf, -np.inf], np.nan)
    n_active = raw.notna().sum(axis=1).replace(0, np.nan)
    return raw.div(n_active, axis=0)


def carry_weights(rate_diff: pd.DataFrame, k: int = 3, min_assets: int = 6) -> pd.DataFrame:
    """Long the top ``k`` and short the bottom ``k`` by rate differential.

    Weights are +1/k and -1/k (gross 2, net 0). Periods with fewer than
    ``min_assets`` valid differentials get no position.
    """
    ranks = rate_diff.rank(axis=1, method="first")
    n = rate_diff.notna().sum(axis=1)
    longs = ranks.gt(n - k, axis=0) & rate_diff.notna()
    shorts = ranks.le(k, axis=0) & rate_diff.notna()
    w = longs.astype(float) / k - shorts.astype(float) / k
    w[n < min_assets] = np.nan
    return w.where(rate_diff.notna())
