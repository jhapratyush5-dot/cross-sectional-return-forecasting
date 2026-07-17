"""
Evaluation.

R-squared is close to meaningless for cross-sectional return forecasting — the
noise floor is so high that a genuinely useful model will still have an R^2
near zero. What matters is whether the model *ranks* stocks correctly, and
whether that ranking survives being turned into a portfolio.

Hence the two metrics here:
  * Information Coefficient (IC): rank correlation between prediction and
    realized return, computed per date. An average rank IC of 0.02-0.05 is a
    real, tradeable signal in equities.
  * Quantile spread backtest: long the top quintile, short the bottom quintile,
    equal-weighted, rebalanced each period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def information_coefficient(preds: pd.DataFrame) -> pd.Series:
    """Per-date Spearman rank correlation between y_pred and y_true."""

    def _ic(block: pd.DataFrame) -> float:
        if len(block) < 5 or block["y_pred"].nunique() < 2:
            return np.nan
        rho, _ = spearmanr(block["y_pred"], block["y_true"])
        return float(rho)

    return preds.groupby("date").apply(_ic).rename("ic")


def quantile_backtest(preds: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """
    Sort each cross-section into quantiles by prediction and compute the
    equal-weighted realized return of each bucket, plus the long/short spread.
    """
    df = preds.copy()

    def _assign(block: pd.DataFrame) -> pd.Series:
        # `rank(method="first")` avoids ties collapsing into a single bucket.
        ranks = block["y_pred"].rank(method="first")
        return pd.qcut(ranks, n_quantiles, labels=False, duplicates="drop")

    df["quantile"] = df.groupby("date", sort=False, group_keys=False).apply(_assign)
    df = df.dropna(subset=["quantile"])
    df["quantile"] = df["quantile"].astype(int)

    by_q = (
        df.groupby(["date", "quantile"])["y_true"].mean().unstack("quantile")
    )
    by_q.columns = [f"q{int(c) + 1}" for c in by_q.columns]

    top, bottom = f"q{n_quantiles}", "q1"
    by_q["long_short"] = by_q[top] - by_q[bottom]
    return by_q


def summarize(preds: pd.DataFrame, periods_per_year: int = 12) -> dict:
    """Headline statistics for the strategy."""
    ic = information_coefficient(preds).dropna()
    bt = quantile_backtest(preds)
    ls = bt["long_short"].dropna()

    ic_mean = float(ic.mean())
    ic_std = float(ic.std(ddof=0))

    # Information ratio of the IC series ("t-stat of the signal").
    ic_ir = ic_mean / ic_std * np.sqrt(len(ic)) if ic_std > 0 else np.nan

    ann_ret = float(ls.mean() * periods_per_year)
    ann_vol = float(ls.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    cum = (1.0 + ls).cumprod()
    max_dd = float((cum / cum.cummax() - 1.0).min())

    return {
        "n_periods": int(len(ic)),
        "mean_ic": ic_mean,
        "ic_t_stat": float(ic_ir),
        "ic_hit_rate": float((ic > 0).mean()),
        "ls_ann_return": ann_ret,
        "ls_ann_vol": ann_vol,
        "ls_sharpe": float(sharpe),
        "ls_max_drawdown": max_dd,
    }


def format_summary(stats: dict) -> str:
    """Render the summary dict as a readable block."""
    lines = [
        "",
        "=" * 46,
        "  OUT-OF-SAMPLE RESULTS (walk-forward)",
        "=" * 46,
        f"  Test periods           : {stats['n_periods']}",
        f"  Mean IC                : {stats['mean_ic']:+.4f}",
        f"  IC t-stat              : {stats['ic_t_stat']:+.2f}",
        f"  IC hit rate            : {stats['ic_hit_rate']:.1%}",
        "-" * 46,
        f"  L/S annualized return  : {stats['ls_ann_return']:+.2%}",
        f"  L/S annualized vol     : {stats['ls_ann_vol']:.2%}",
        f"  L/S Sharpe             : {stats['ls_sharpe']:+.2f}",
        f"  L/S max drawdown       : {stats['ls_max_drawdown']:.2%}",
        "=" * 46,
        "",
    ]
    return "\n".join(lines)
