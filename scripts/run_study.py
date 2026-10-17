"""Run the full study and write results/ and figures/.

    python scripts/download_data.py   # once
    python scripts/run_study.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from macrobt import metrics
from macrobt.data.panel import load_panel
from macrobt.engine import run_backtest
from macrobt.plotting import bar_by_bucket, equity_and_drawdown
from macrobt.portfolio import combine, vol_target
from macrobt.signals import carry_weights, tsmom_weights
from macrobt.walkforward import select_parameters, stitch_weights

TARGET_VOL = 0.10
START = "2001-01-31"  # first month with equities, bonds and FX all available
LOOKBACKS = [1, 3, 6, 9, 12, 18, 24]
OUT, FIG = Path("results"), Path("figures")


def tsmom_book(returns: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    return vol_target(tsmom_weights(returns, lookback=lookback), returns, TARGET_VOL)


def carry_book(returns: pd.DataFrame, rate_diff: pd.DataFrame) -> pd.DataFrame:
    return vol_target(carry_weights(rate_diff, k=3), returns, TARGET_VOL)


def fmt_table(rows: dict[str, dict], keys: list[tuple[str, str, str]]) -> str:
    header = "| Strategy | " + " | ".join(k[1] for k in keys) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in keys) + " |"
    lines = [header, sep]
    for name, s in rows.items():
        cells = []
        for key, _, fmt in keys:
            v = s.get(key, float("nan"))
            cells.append(format(v, fmt) if isinstance(v, (int, float)) and not np.isnan(v) else "n/a")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


KEYS = [
    ("ann_return", "Return", ".1%"),
    ("ann_vol", "Vol", ".1%"),
    ("sharpe", "Sharpe", ".2f"),
    ("gross_sharpe", "Gross Sharpe", ".2f"),
    ("max_drawdown", "Max DD", ".1%"),
    ("ann_turnover", "Turnover (x/yr)", ".1f"),
    ("ann_cost", "Cost (/yr)", ".2%"),
    ("skew", "Skew", ".2f"),
]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    panel = load_panel("data")
    R = panel.returns
    fx_cols = [a for a, c in panel.asset_class.items() if c == "fx"]
    costs = panel.costs_bps
    report: dict = {"sample_end": str(R.index[-1].date())}
    md: list[str] = []

    # ---- 1. Core strategies on the common sample ---------------------------
    w_tsmom = tsmom_book(R, 12)
    w_carry = carry_book(R[fx_cols], panel.rate_diff)
    w_combo = vol_target(combine({"tsmom": w_tsmom, "carry": w_carry}, {"tsmom": 0.5, "carry": 0.5}), R, TARGET_VOL)

    runs = {}
    for name, w in [("TSMOM (12m, 19 assets)", w_tsmom), ("FX carry (3x3)", w_carry), ("50/50 blend", w_combo)]:
        res = run_backtest(R, w.loc[START:], cost_bps=costs, name=name)
        runs[name] = res
    rows = {k: v.summary() for k, v in runs.items()}
    corr = pd.DataFrame({k: v.returns for k, v in runs.items()}).corr()
    report["core"] = rows
    report["core_correlation"] = corr.round(3).to_dict()
    md += ["## Core results (net of costs, " + rows["TSMOM (12m, 19 assets)"]["start"][:7] + " to "
           + rows["TSMOM (12m, 19 assets)"]["end"][:7] + ")", "", fmt_table(rows, KEYS), "",
           "Correlation of monthly returns:", "", corr.round(2).to_markdown(), ""]
    equity_and_drawdown({k: v.returns for k, v in runs.items()}, FIG / "core_equity.png",
                        "Net excess returns, 10% vol target")

    # ---- 2. Lookback sensitivity ------------------------------------------
    lb_rows, cand_returns, cand_weights = {}, {}, {}
    for lb in LOOKBACKS:
        w = tsmom_book(R, lb)
        res = run_backtest(R, w, cost_bps=costs)
        cand_returns[lb] = res.returns
        cand_weights[lb] = w
        sub = run_backtest(R, w.loc[START:], cost_bps=costs)
        lb_rows[f"{lb}m"] = sub.summary()
    report["lookback"] = lb_rows
    md += ["## TSMOM lookback sensitivity", "", fmt_table(lb_rows, KEYS), ""]

    # ---- 3. Walk-forward lookback selection --------------------------------
    cand = pd.DataFrame(cand_returns)
    choice = select_parameters(cand, train_periods=60, refit_every=12, min_periods=36)
    w_wf = stitch_weights(choice, cand_weights)
    wf = run_backtest(R, w_wf.loc[START:], cost_bps=costs, name="Walk-forward")
    fixed = runs["TSMOM (12m, 19 assets)"]
    wf_rows = {"Fixed 12m": fixed.summary(), "Walk-forward (5y trailing Sharpe, annual refit)": wf.summary()}
    picks = choice.loc[START:].groupby(choice.loc[START:].index.year).first().astype(int)
    report["walk_forward"] = {"rows": wf_rows, "picks_by_year": {int(k): int(v) for k, v in picks.items()}}
    md += ["## Walk-forward lookback selection", "", fmt_table(wf_rows, KEYS), "",
           "Lookback chosen each year: " + ", ".join(f"{y}: {v}m" for y, v in picks.items()), ""]

    # ---- 4. Costs and execution delay --------------------------------------
    rob = {}
    for mult in [0, 1, 2, 5]:
        res = run_backtest(R, w_tsmom.loc[START:], cost_bps={k: v * mult for k, v in costs.items()})
        rob[f"Costs x{mult}"] = res.summary()
    rob["Extra 1-month delay"] = run_backtest(R, w_tsmom.loc[START:], cost_bps=costs, lag=2).summary()
    report["robustness_tsmom"] = rob
    md += ["## TSMOM robustness", "", fmt_table(rob, KEYS), ""]

    # ---- 5. Behaviour in equity sell-offs ----------------------------------
    spy = R["SPY"].loc[fixed.returns.index]
    buckets = pd.qcut(spy, 5, labels=["Worst 20%", "Q2", "Q3", "Q4", "Best 20%"])
    by_bucket = pd.DataFrame({
        "SPY": spy.groupby(buckets, observed=True).mean(),
        "TSMOM": fixed.returns.groupby(buckets, observed=True).mean(),
        "FX carry": runs["FX carry (3x3)"].returns.reindex(spy.index).groupby(buckets, observed=True).mean(),
    }) * 100
    report["spy_quintiles_pct_per_month"] = by_bucket.round(3).to_dict()
    md += ["## Average monthly return by SPY quintile (%)", "", by_bucket.round(2).to_markdown(), ""]
    bar_by_bucket(by_bucket, FIG / "spy_quintiles.png", "Average monthly return by SPY excess return quintile",
                  "% per month")

    # ---- 6. Long history: FX only ------------------------------------------
    w_fx_carry = carry_book(R[fx_cols], panel.rate_diff)
    w_fx_tsmom = tsmom_book(R[fx_cols], 12)
    long_runs = {
        "FX carry": run_backtest(R[fx_cols], w_fx_carry, cost_bps=costs),
        "FX TSMOM (12m)": run_backtest(R[fx_cols], w_fx_tsmom, cost_bps=costs),
    }
    long_rows = {k: v.summary() for k, v in long_runs.items()}
    # Sub-periods for carry, which is known for crash risk
    carry_r = long_runs["FX carry"].returns
    for a, b in [("1983", "1999"), ("2000", "2012"), ("2013", "2026")]:
        seg = carry_r.loc[a:b]
        if len(seg) > 24:
            long_rows[f"FX carry {a}-{b}"] = metrics.summary(seg)
    report["fx_long_history"] = long_rows
    md += ["## FX strategies over the full FRED history", "", fmt_table(long_rows, KEYS), ""]
    equity_and_drawdown({k: v.returns for k, v in long_runs.items()}, FIG / "fx_long_history.png",
                        "G10 FX strategies, net, 10% vol target")

    # ---- 7. Per-asset contribution of TSMOM --------------------------------
    contrib = fixed.contributions.sum().sort_values(ascending=False) * 100
    by_class = fixed.contributions.T.groupby(pd.Series(panel.asset_class)).sum().T.sum() * 100
    report["tsmom_contribution_pct"] = {"by_asset": contrib.round(2).to_dict(), "by_class": by_class.round(2).to_dict()}
    md += ["## TSMOM gross contribution by asset class (sum of monthly % returns)", "",
           by_class.round(1).to_frame("contribution").to_markdown(), ""]

    (OUT / "summary.json").write_text(json.dumps(report, indent=2, default=float))
    (OUT / "results.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
