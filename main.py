from __future__ import annotations

import argparse
from pathlib import Path

from analysis import run_analysis


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Lab 3: causal and adaptive Kalman filtering")
    parser.add_argument("--data", type=Path, default=root / "data/cyberpunk_2077_monthly_reviews.csv")
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument("--simulations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.simulations < 30:
        parser.error("--simulations must be at least 30 for ensemble confidence intervals")
    run_analysis(args.data, args.output, args.simulations, args.seed)


if __name__ == "__main__":
    main()
