"""Performance statistics for a series of periodic returns.

All functions take simple (not log) returns and a ``periods_per_year``
argument (12 for monthly, 252 for daily) and ignore leading NaNs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(r: pd.Series) -> pd.Series:
    return r.dropna()


def annualised_return(r: pd.Series, periods_per_year: int = 12) -> float:
    """Geometric annualised return."""
    r = _clean(r)
    if len(r) == 0:
        return float("nan")
    growth = float(np.prod(1 + r.to_numpy()))
    return growth ** (periods_per_year / len(r)) - 1


def annualised_vol(r: pd.Series, periods_per_year: int = 12) -> float:
    return float(_clean(r).std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(r: pd.Series, periods_per_year: int = 12) -> float:
    """Annualised Sharpe ratio of excess returns (arithmetic mean / std)."""
    r = _clean(r)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def sortino_ratio(r: pd.Series, periods_per_year: int = 12) -> float:
    r = _clean(r)
    downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    return float(r.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else float("nan")


def drawdown_series(r: pd.Series) -> pd.Series:
    """Drawdown from the running peak of cumulative wealth (<= 0)."""
    wealth = (1 + _clean(r)).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    return pd.Series(wealth.to_numpy() / peak - 1, index=wealth.index)


def max_drawdown(r: pd.Series) -> float:
    dd = drawdown_series(r)
    return float(dd.min()) if len(dd) else float("nan")


def calmar_ratio(r: pd.Series, periods_per_year: int = 12) -> float:
    mdd = max_drawdown(r)
    return annualised_return(r, periods_per_year) / abs(mdd) if mdd < 0 else float("nan")


def hit_rate(r: pd.Series) -> float:
    r = _clean(r)
    r = r[r != 0]
    return float((r > 0).mean()) if len(r) else float("nan")


def summary(r: pd.Series, periods_per_year: int = 12, turnover: pd.Series | None = None) -> dict:
    """Headline statistics as a flat dict (ratios as decimals)."""
    r = _clean(r)
    out = {
        "start": str(r.index[0].date()) if hasattr(r.index[0], "date") else str(r.index[0]),
        "end": str(r.index[-1].date()) if hasattr(r.index[-1], "date") else str(r.index[-1]),
        "periods": int(len(r)),
        "ann_return": annualised_return(r, periods_per_year),
        "ann_vol": annualised_vol(r, periods_per_year),
        "sharpe": sharpe_ratio(r, periods_per_year),
        "sortino": sortino_ratio(r, periods_per_year),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar_ratio(r, periods_per_year),
        "hit_rate": hit_rate(r),
        "skew": float(r.skew()),
    }
    if turnover is not None:
        out["ann_turnover"] = float(turnover.reindex(r.index).mean() * periods_per_year)
    return out


def sharpe_t_stat(r: pd.Series) -> float:
    """t-statistic of the mean return (equivalently Sharpe * sqrt(T) per period)."""
    r = _clean(r)
    return float(r.mean() / r.std(ddof=1) * np.sqrt(len(r)))
