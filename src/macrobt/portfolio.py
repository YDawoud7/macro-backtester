"""Portfolio construction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ex_ante_vol(weights: pd.DataFrame, returns: pd.DataFrame, window: int = 36, min_periods: int = 24,
                periods_per_year: int = 12) -> pd.Series:
    """Annualised volatility of today's weights under a trailing sample covariance.

    The covariance at t uses returns up to and including t, which is known
    when the weights for t+1 are set.
    """
    cols = weights.columns
    r = returns.reindex(columns=cols)
    out = pd.Series(np.nan, index=weights.index)
    r_np = r.to_numpy()
    w_np = weights.reindex(r.index).to_numpy()
    for i in range(len(r.index)):
        lo = max(0, i - window + 1)
        win = r_np[lo : i + 1]
        w = w_np[i]
        mask = ~np.isnan(w) & (w != 0)
        if mask.sum() == 0 or (i + 1 - lo) < min_periods:
            continue
        sub = win[:, mask]
        sub = sub[~np.isnan(sub).any(axis=1)]
        if len(sub) < min_periods:
            continue
        cov = np.cov(sub, rowvar=False, ddof=1)
        cov = np.atleast_2d(cov)
        var = float(w[mask] @ cov @ w[mask])
        out.iloc[i] = np.sqrt(max(var, 0.0) * periods_per_year)
    return out


def vol_target(weights: pd.DataFrame, returns: pd.DataFrame, target: float = 0.10, window: int = 36,
               min_periods: int = 24, max_leverage: float = 4.0, periods_per_year: int = 12) -> pd.DataFrame:
    """Scale the whole book so its ex-ante volatility equals ``target``.

    Gross leverage (sum of |w|) is capped at ``max_leverage``.
    """
    vol = ex_ante_vol(weights, returns, window, min_periods, periods_per_year)
    scale = target / vol
    scaled = weights.mul(scale, axis=0)
    gross = scaled.abs().sum(axis=1)
    cap = (max_leverage / gross).clip(upper=1.0)
    return scaled.mul(cap, axis=0)


def combine(weight_sets: dict[str, pd.DataFrame], allocations: dict[str, float]) -> pd.DataFrame:
    """Sum several weight frames with the given allocations (union of assets)."""
    cols = sorted(set().union(*[w.columns for w in weight_sets.values()]))
    idx = sorted(set().union(*[w.index for w in weight_sets.values()]))
    total = pd.DataFrame(0.0, index=pd.DatetimeIndex(idx), columns=cols)
    any_valid = pd.DataFrame(False, index=total.index, columns=cols)
    for name, w in weight_sets.items():
        aligned = w.reindex(index=total.index, columns=cols)
        total += aligned.fillna(0.0) * allocations.get(name, 0.0)
        any_valid |= aligned.notna()
    return total.where(any_valid)
