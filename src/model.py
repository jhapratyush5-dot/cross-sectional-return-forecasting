"""
Models and walk-forward validation.

Standard k-fold cross-validation is invalid for time-series/panel data: it
trains on the future to predict the past. This module uses an expanding-window
walk-forward scheme, which mirrors how the model would actually be deployed —
at each rebalance date, fit only on history, then predict the next period.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

try:
    from lightgbm import LGBMRegressor

    _HAS_LGBM = True
except ImportError:  # pragma: no cover
    _HAS_LGBM = False

from .features import FEATURE_COLUMNS


def make_model(kind: str = "ridge", seed: int = 7):
    """Factory for the supported model types."""
    if kind == "ridge":
        # Heavy regularization is appropriate: the signal-to-noise ratio in
        # cross-sectional equity returns is very low, and an unregularized fit
        # will happily memorize noise.
        return Ridge(alpha=10.0)

    if kind == "lgbm":
        if not _HAS_LGBM:
            raise ImportError("lightgbm is not installed; use kind='ridge'.")
        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            random_state=seed,
            verbose=-1,
        )

    raise ValueError(f"Unknown model kind: {kind!r}")


@dataclass
class WalkForwardConfig:
    min_train_periods: int = 36   # periods of history before first prediction
    step: int = 1                 # rebalance frequency, in periods


def walk_forward_predict(
    df: pd.DataFrame,
    kind: str = "ridge",
    config: WalkForwardConfig | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Run an expanding-window walk-forward backtest.

    At each test date t:
        train on all rows with date < t
        predict the cross-section at date t

    Returns
    -------
    DataFrame with [date, ticker, y_true, y_pred] for every out-of-sample date.
    """
    config = config or WalkForwardConfig()
    dates = np.sort(df["date"].unique())

    if len(dates) <= config.min_train_periods:
        raise ValueError(
            f"Need more than {config.min_train_periods} periods; got {len(dates)}."
        )

    test_dates = dates[config.min_train_periods :: config.step]
    frames: list[pd.DataFrame] = []

    for t in test_dates:
        train = df[df["date"] < t]
        test = df[df["date"] == t]

        if train.empty or test.empty:
            continue

        model = make_model(kind, seed=seed)
        model.fit(train[FEATURE_COLUMNS].to_numpy(), train["fwd_ret"].to_numpy())

        preds = model.predict(test[FEATURE_COLUMNS].to_numpy())

        frames.append(
            pd.DataFrame(
                {
                    "date": test["date"].to_numpy(),
                    "ticker": test["ticker"].to_numpy(),
                    "y_true": test["fwd_ret"].to_numpy(),
                    "y_pred": preds,
                }
            )
        )

    if not frames:
        raise RuntimeError("Walk-forward produced no predictions.")

    return pd.concat(frames, ignore_index=True)
