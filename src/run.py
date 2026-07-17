"""
Pipeline entry point.

Examples
--------
    # Run on generated synthetic data (no data subscription needed)
    python -m src.run --synthetic --model lgbm

    # Run on your own price panel
    python -m src.run --data data/prices.csv --model ridge
"""

from __future__ import annotations

import argparse

from .data import load_panel, make_synthetic_panel
from .evaluate import format_summary, summarize
from .features import build_features, cross_sectional_zscore
from .model import WalkForwardConfig, walk_forward_predict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-sectional equity return forecasting pipeline."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="Path to a long-format price panel CSV.")
    src.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a synthetic panel instead of loading data.",
    )
    parser.add_argument("--model", choices=["ridge", "lgbm"], default="ridge")
    parser.add_argument("--min-train", type=int, default=36)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", help="Optional path to write predictions CSV.")
    args = parser.parse_args()

    if args.synthetic:
        print("Generating synthetic panel...")
        panel = make_synthetic_panel(seed=args.seed)
    else:
        print(f"Loading panel from {args.data}...")
        panel = load_panel(args.data)

    print(
        f"  {panel['ticker'].nunique()} tickers x "
        f"{panel['date'].nunique()} periods"
    )

    print("Building features...")
    feats = build_features(panel)
    feats = cross_sectional_zscore(feats)
    print(f"  {len(feats):,} usable (date, ticker) rows after dropna")

    print(f"Running walk-forward backtest [{args.model}]...")
    preds = walk_forward_predict(
        feats,
        kind=args.model,
        config=WalkForwardConfig(min_train_periods=args.min_train),
        seed=args.seed,
    )

    stats = summarize(preds)
    print(format_summary(stats))

    if args.out:
        preds.to_csv(args.out, index=False)
        print(f"Predictions written to {args.out}")


if __name__ == "__main__":
    main()
