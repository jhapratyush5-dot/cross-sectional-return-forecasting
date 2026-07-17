"""
Data loading for the cross-sectional return forecasting pipeline.

The pipeline expects a long-format panel with one row per (date, ticker):

    date        ticker   close    volume
    2015-01-30  AAA      101.2    1_200_000
    2015-01-30  BBB       57.8      840_000
    ...

`load_panel` reads such a CSV. `make_synthetic_panel` generates a panel with a
known, weak factor structure so the pipeline is runnable and testable without a
paid market-data subscription.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "ticker", "close"]


def load_panel(path: str) -> pd.DataFrame:
    """Load a long-format price panel from CSV and validate its schema."""
    df = pd.read_csv(path, parse_dates=["date"])

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Panel is missing required column(s): {missing}")

    if df.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("Panel contains duplicate (date, ticker) rows.")

    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def make_synthetic_panel(
    n_tickers: int = 120,
    n_periods: int = 180,
    seed: int = 7,
    signal_strength: float = 0.004,
) -> pd.DataFrame:
    """
    Generate a synthetic monthly price panel with a deliberately weak factor
    structure.

    Construction
    ------------
    Each period, a stock's return is driven by:
      * a market component shared by all names,
      * a persistent per-stock "quality" factor,
      * a mild mean-reversion term on last period's return,
      * idiosyncratic noise, which dominates.

    `signal_strength` is intentionally small. Real cross-sectional equity signals
    have information coefficients in the 0.02-0.05 range; generating data with a
    strong, easily-learned signal would produce flattering, unrealistic results.

    Returns
    -------
    DataFrame with columns [date, ticker, close, volume].
    """
    rng = np.random.default_rng(seed)

    tickers = [f"SYN{i:03d}" for i in range(n_tickers)]
    dates = pd.date_range("2010-01-31", periods=n_periods, freq="ME")

    # Slowly time-varying per-stock factor exposure (the "learnable" part).
    # Making it drift rather than stay fixed prevents the model from simply
    # memorizing a static ranking, which is not how real factor exposures behave.
    quality = rng.normal(0.0, 1.0, size=n_tickers)

    prices = np.zeros((n_periods, n_tickers))
    prices[0] = rng.uniform(20, 200, size=n_tickers)

    prev_ret = np.zeros(n_tickers)

    for t in range(1, n_periods):
        # Let exposures drift so the signal decays and must be re-learned.
        quality = 0.97 * quality + rng.normal(0.0, 0.24, size=n_tickers)

        market = rng.normal(0.006, 0.040)
        idio = rng.normal(0.0, 0.085, size=n_tickers)

        ret = (
            market
            + signal_strength * quality
            - 0.05 * prev_ret          # mild short-term reversal
            + idio
        )
        ret = np.clip(ret, -0.5, 0.5)

        prices[t] = prices[t - 1] * (1.0 + ret)
        prev_ret = ret

    panel = pd.DataFrame(prices, index=dates, columns=tickers)
    panel = (
        panel.stack()
        .rename("close")
        .rename_axis(["date", "ticker"])
        .reset_index()
    )

    panel["volume"] = rng.lognormal(13.5, 0.6, size=len(panel)).round()
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)
