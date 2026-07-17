"""
Feature engineering.

The single most important rule in this module: every feature at date `t` is
computed only from information available at or before `t`, and the target is the
*forward* return from `t` to `t+1`. Any leakage of future information produces
backtests that look excellent and fail in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "mom_12_1",
    "reversal_1m",
    "vol_6m",
    "dist_from_high_12m",
    "turnover_z",
]


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Construct cross-sectional features and the forward-return target.

    Parameters
    ----------
    panel : long-format DataFrame with [date, ticker, close] (+ optional volume)

    Returns
    -------
    DataFrame with [date, ticker] + FEATURE_COLUMNS + ["fwd_ret"], where
    `fwd_ret` is the return from t to t+1.
    """
    df = panel.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker", sort=False)["close"]

    # --- Trailing return blocks -------------------------------------------
    df["ret_1m"] = g.pct_change(1)

    # 12-1 momentum: cumulative return over the last 12 periods, skipping the
    # most recent one. Skipping the last period is standard practice: it removes
    # the short-term reversal effect that otherwise contaminates momentum.
    price_lag_1 = g.shift(1)
    price_lag_12 = g.shift(12)
    df["mom_12_1"] = (price_lag_1 / price_lag_12) - 1.0

    # Short-term reversal: last period's return, negated so that higher = more
    # attractive (losers tend to bounce).
    df["reversal_1m"] = -df["ret_1m"]

    # --- Risk / dispersion -------------------------------------------------
    df["vol_6m"] = (
        df.groupby("ticker", sort=False)["ret_1m"]
        .rolling(6, min_periods=6)
        .std()
        .reset_index(level=0, drop=True)
    )

    # --- Price position ----------------------------------------------------
    rolling_high = (
        g.rolling(12, min_periods=12).max().reset_index(level=0, drop=True)
    )
    df["dist_from_high_12m"] = df["close"] / rolling_high - 1.0

    # --- Liquidity ---------------------------------------------------------
    if "volume" in df.columns:
        dollar_vol = df["close"] * df["volume"]
        df["turnover_z"] = np.log1p(dollar_vol)
    else:
        df["turnover_z"] = 0.0

    # --- Target: forward return (t -> t+1) ---------------------------------
    df["fwd_ret"] = g.shift(-1) / df["close"] - 1.0

    out = df[["date", "ticker", *FEATURE_COLUMNS, "fwd_ret"]]
    return out.dropna().sort_values(["date", "ticker"]).reset_index(drop=True)


def cross_sectional_zscore(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Standardize features *within each date*.

    Cross-sectional (rather than pooled) standardization is deliberate: we are
    ranking stocks against each other at a point in time, not comparing across
    time. This also neutralizes market-wide level shifts, so the model learns
    relative rather than directional signal.

    Outliers are winsorized at +/- 3 sd to stop a single name from dominating.
    """
    columns = columns or FEATURE_COLUMNS
    out = df.copy()

    def _z(block: pd.DataFrame) -> pd.DataFrame:
        mu = block.mean()
        sd = block.std(ddof=0).replace(0.0, np.nan)
        z = (block - mu) / sd
        return z.clip(-3.0, 3.0).fillna(0.0)

    out[columns] = (
        out.groupby("date", sort=False)[columns]
        .transform(lambda s: _z(s.to_frame()).iloc[:, 0])
    )
    return out
