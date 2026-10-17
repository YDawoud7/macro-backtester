"""Portfolio construction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ex_ante_vol(weights: pd.DataFrame, returns: pd.DataFrame, window: int = 36, min_periods: int = 24,
                periods_per_year: int = 12) -> pd.Series:
    """Annualised volatility of today's weights under a trailing sample covariance.

    The covariance at t uses returns up to and including t, which is known
    when the weights for t+1 are set.

    Assets have different start dates, so the covariance is estimated
    pairwise: each entry uses the months where both assets have returns
    (at least ``pair_min`` of them). An asset whose own variance can't be
    estimated yet is left out of the calculation; a pair without enough
    overlap contributes no covariance term.
    """
    cols = weights.columns
    r = returns.reindex(columns=cols)
    out = pd.Series(np.nan, index=weights.index)
    w_frame = weights.reindex(r.index)
    pair_min = 12
    for i in range(len(r.index)):
        if i + 1 < min_periods:
            continue
        w = w_frame.iloc[i]
        active = w.index[w.notna() & (w != 0)]
        if len(active) == 0:
            continue
        win = r[active].iloc[max(0, i - window + 1) : i + 1]
        cov = win.cov(min_periods=pair_min)
        ok = cov.index[np.isfinite(np.diag(cov.to_numpy()))]
        if len(ok) == 0:
            continue
        c = cov.loc[ok, ok].fillna(0.0).to_numpy()
        wv = w[ok].to_numpy()
        out.iloc[i] = np.sqrt(max(float(wv @ c @ wv), 0.0) * periods_per_year)
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
