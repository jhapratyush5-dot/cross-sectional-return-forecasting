# Cross-Sectional Equity Return Forecasting

A semi-systematic pipeline that ranks stocks by expected forward return using
multi-factor signals, validated with walk-forward backtesting and evaluated the
way quantitative equity strategies are actually judged — by information
coefficient and long/short quantile spread, not by R².

---

## Why this is built the way it is

Most "stock prediction" projects fail for the same three reasons. This one is
designed around avoiding them:

**1. Lookahead bias.** Every feature at date `t` uses only data available at or
before `t`; the target is the *forward* return from `t` to `t+1`. This is
enforced by a test (`test_no_lookahead_in_features`) that truncates the panel and
asserts previously-computed features are unchanged.

**2. Invalid validation.** K-fold cross-validation trains on the future to
predict the past. This pipeline uses an **expanding-window walk-forward** scheme:
at each rebalance date, fit only on history, then predict the next cross-section
— exactly how the model would be deployed.

**3. Meaningless metrics.** Cross-sectional equity returns have a very low
signal-to-noise ratio; a genuinely useful model still has an R² near zero. What
matters is *ranking*. This project reports:

- **Information Coefficient (IC)** — per-date Spearman rank correlation between
  prediction and realized return. A mean IC of 0.02–0.05 is a real, tradeable
  signal in equities.
- **Quantile spread backtest** — long the top quintile, short the bottom,
  equal-weighted, rebalanced each period, reported with Sharpe and max drawdown.

---

## Results (synthetic data, out-of-sample)

The repo ships with a synthetic panel generator so the pipeline is fully
reproducible without a market-data subscription. The generator embeds a
deliberately **weak, decaying factor structure** — signal strength is tuned so
that realized IC lands in a realistic range. (An earlier version produced a
Sharpe of 13, which was a bug in the data generator, not a discovery.)

| Model | Mean IC | IC t-stat | Hit rate | L/S Sharpe | Max DD |
|-------|--------:|----------:|---------:|-----------:|-------:|
| Ridge | +0.042 | +5.67 | 69.5% | +1.53 | −7.0% |
| LightGBM | +0.020 | +2.47 | 58.0% | +1.05 | −12.3% |

*131 out-of-sample monthly periods, 120 names, expanding window with 36 periods
of minimum training history.*

**The interesting result is that Ridge beats LightGBM.** This is not a bug — it's
the expected outcome in a low signal-to-noise regime. Gradient boosting has
enough capacity to fit the noise, while a heavily-regularized linear model is
forced to capture only the persistent linear structure. It's a useful reminder
that model complexity is not free.

> ⚠️ These numbers describe synthetic data with a known embedded signal. They
> demonstrate that the pipeline **recovers a signal it should recover** and
> **rejects one it shouldn't** (see `test_shuffled_target_has_no_signal`). They
> are not a claim about live-market performance.

---

## Features

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| `mom_12_1` | Return over t−12 → t−1 | Classic momentum; skipping the most recent month removes short-term reversal contamination |
| `reversal_1m` | Negated last-period return | Short-horizon losers tend to bounce |
| `vol_6m` | 6-period std. dev. of returns | Low-volatility anomaly / risk control |
| `dist_from_high_12m` | Price ÷ 12-period high − 1 | Position within the recent range |
| `turnover_z` | log(1 + dollar volume) | Liquidity proxy |

All features are **cross-sectionally z-scored within each date** and winsorized
at ±3σ. Standardizing per-date (rather than pooled) is deliberate: the model
should rank names against each other at a point in time, not compare across
regimes. It also neutralizes market-wide level shifts, so the model learns
relative rather than directional signal.

---

## Usage

```bash
pip install -r requirements.txt

# Run on generated synthetic data
python -m src.run --synthetic --model ridge
python -m src.run --synthetic --model lgbm

# Run on your own price panel
python -m src.run --data data/prices.csv --model ridge --out preds.csv

# Tests
pytest tests/ -q
```

### Input format

A long-format CSV with one row per (date, ticker):

```csv
date,ticker,close,volume
2015-01-30,AAA,101.20,1200000
2015-01-30,BBB,57.80,840000
```

---

## Structure

```
src/
├── data.py       # panel loading + synthetic generator
├── features.py   # feature engineering (no-lookahead discipline)
├── model.py      # Ridge / LightGBM + walk-forward validation
├── evaluate.py   # IC, quantile backtest, summary stats
└── run.py        # CLI entry point
tests/
└── test_pipeline.py   # incl. lookahead + shuffled-target checks
```

---

## Limitations & next steps

Honest about what this doesn't do yet:

- **No transaction costs or slippage.** A monthly-rebalanced quintile spread
  incurs real turnover; net of costs the Sharpe would fall meaningfully.
- **No survivorship-bias-free universe.** Real backtests need point-in-time
  constituent lists and delisting returns.
- **Price-based features only.** Fundamental data (valuation, quality,
  revisions) and alternative data would add orthogonal signal.
- **Equal-weighted buckets, no risk model.** A production version would
  neutralize sector and factor exposures and size positions by risk.

Planned: transaction-cost modelling, sector neutralization, and an ensemble that
blends the linear and non-linear models rather than choosing between them.

---

## License

MIT
