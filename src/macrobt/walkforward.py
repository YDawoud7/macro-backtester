"""Walk-forward parameter selection.

At each refit date the candidate with the best trailing Sharpe ratio over the
previous ``train_periods`` is chosen and its weights are used until the next
refit. Candidate returns at time t only depend on data up to t, and the
selection at a refit date only looks at returns up to that date, so the
stitched strategy is genuinely out of sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_parameters(
    candidate_returns: pd.DataFrame,
    train_periods: int = 60,
    refit_every: int = 12,
    min_periods: int = 36,
    periods_per_year: int = 12,
) -> pd.Series:
    """Chosen candidate (column label) for each period, forward filled between refits.

    The label at refit date t is chosen from returns up to and including t
    and applies to weights set from t onward (which earn returns from t+1).
    """
    idx = candidate_returns.index
    choice = pd.Series(index=idx, dtype=object)
    first_valid = candidate_returns.dropna(how="all").index
    if len(first_valid) == 0:
        return choice
    start = idx.get_loc(first_valid[0])
    for i in range(start + min_periods - 1, len(idx), refit_every):
        window = candidate_returns.iloc[max(0, i - train_periods + 1) : i + 1]
        mu = window.mean()
        sd = window.std(ddof=1)
        sharpe = (mu / sd * np.sqrt(periods_per_year)).where(window.count() >= min_periods)
        if sharpe.notna().any():
            choice.iloc[i] = sharpe.idxmax()
    return choice.ffill()


def stitch_weights(choice: pd.Series, candidate_weights: dict) -> pd.DataFrame:
    """Weights of the chosen candidate at each date."""
    cols = sorted(set().union(*[w.columns for w in candidate_weights.values()]))
    out = pd.DataFrame(np.nan, index=choice.index, columns=cols)
    for label, w in candidate_weights.items():
        mask = choice == label
        if mask.any():
            out.loc[mask] = w.reindex(index=choice.index, columns=cols).loc[mask].to_numpy()
    return out
