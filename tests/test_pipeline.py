"""
Tests for the forecasting pipeline.

The most important test here is `test_no_lookahead_in_features`: silent lookahead
bias is the single most common way a backtest ends up looking profitable and
being worthless.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import make_synthetic_panel
from src.evaluate import information_coefficient, quantile_backtest, summarize
from src.features import FEATURE_COLUMNS, build_features, cross_sectional_zscore
from src.model import WalkForwardConfig, walk_forward_predict


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(n_tickers=40, n_periods=90, seed=1)


@pytest.fixture(scope="module")
def feats(panel):
    return cross_sectional_zscore(build_features(panel))


def test_panel_schema(panel):
    assert {"date", "ticker", "close", "volume"} <= set(panel.columns)
    assert not panel.duplicated(subset=["date", "ticker"]).any()


def test_features_have_no_nans(feats):
    assert not feats[FEATURE_COLUMNS + ["fwd_ret"]].isna().any().any()


def test_no_lookahead_in_features(panel):
    """
    Truncating the panel must not change features already computed on earlier
    dates. If it does, a feature is peeking at the future.
    """
    full = build_features(panel)

    cutoff = np.sort(panel["date"].unique())[-10]
    truncated = build_features(panel[panel["date"] <= cutoff])

    # Compare on dates strictly before the cutoff (the final date legitimately
    # loses its forward return when the panel is truncated).
    common = truncated["date"] < cutoff
    trunc_cmp = truncated[common].set_index(["date", "ticker"])[FEATURE_COLUMNS]
    full_cmp = (
        full.set_index(["date", "ticker"])
        .loc[trunc_cmp.index, FEATURE_COLUMNS]
    )

    pd.testing.assert_frame_equal(trunc_cmp, full_cmp, atol=1e-10)


def test_target_is_forward_looking(panel):
    """`fwd_ret` at date t must equal the realized t -> t+1 return."""
    feats = build_features(panel)
    row = feats.iloc[0]

    prices = (
        panel[panel["ticker"] == row["ticker"]]
        .set_index("date")["close"]
        .sort_index()
    )
    dates = prices.index
    i = dates.get_loc(row["date"])

    expected = prices.iloc[i + 1] / prices.iloc[i] - 1.0
    assert row["fwd_ret"] == pytest.approx(expected, rel=1e-9)


def test_zscore_is_cross_sectional(feats):
    """Within each date, standardized features should be ~zero-mean."""
    means = feats.groupby("date")[FEATURE_COLUMNS].mean().abs().max().max()
    assert means < 0.5  # winsorizing prevents exactly zero


def test_walk_forward_is_out_of_sample(feats):
    """Predictions must only exist for dates after the training window."""
    cfg = WalkForwardConfig(min_train_periods=24)
    preds = walk_forward_predict(feats, kind="ridge", config=cfg)

    dates = np.sort(feats["date"].unique())
    first_test = dates[cfg.min_train_periods]

    assert preds["date"].min() >= first_test
    assert set(preds.columns) == {"date", "ticker", "y_true", "y_pred"}


def test_quantile_backtest_shape(feats):
    preds = walk_forward_predict(
        feats, kind="ridge", config=WalkForwardConfig(min_train_periods=24)
    )
    bt = quantile_backtest(preds, n_quantiles=5)
    assert "long_short" in bt.columns
    assert len(bt) == preds["date"].nunique()


def test_summary_keys(feats):
    preds = walk_forward_predict(
        feats, kind="ridge", config=WalkForwardConfig(min_train_periods=24)
    )
    stats = summarize(preds)
    for key in ("mean_ic", "ls_sharpe", "ls_max_drawdown", "ic_t_stat"):
        assert key in stats


def test_shuffled_target_has_no_signal(feats):
    """
    Sanity check: destroying the relationship between features and target should
    collapse the IC toward zero. If it doesn't, something is leaking.
    """
    rng = np.random.default_rng(0)
    noise = feats.copy()
    noise["fwd_ret"] = rng.permutation(noise["fwd_ret"].to_numpy())

    preds = walk_forward_predict(
        noise, kind="ridge", config=WalkForwardConfig(min_train_periods=24)
    )
    ic = information_coefficient(preds).dropna()
    assert abs(ic.mean()) < 0.05
