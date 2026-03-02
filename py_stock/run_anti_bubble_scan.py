"""
run_anti_bubble_scan.py
=======================
CLI entry point for the Anti-Bubble Small-Cap Tech Scanner.
Place this file in the project root (same level as main.py).

Usage:
    python run_anti_bubble_scan.py
    python run_anti_bubble_scan.py --budget 50000 --top 10 --workers 20
    python run_anti_bubble_scan.py --ai-threshold 35 --bubble-threshold 55

Arguments:
    --budget        Total investment budget in USD        (default: 100000)
    --top           Target number of portfolio holdings   (default: 15)
    --workers       Parallel threads                      (default: 16)
    --stage1-max    Max stocks to keep after Stage 1      (default: 300)
    --stage2-max    Max stocks to keep after Stage 2      (default: 100)
    --ai-threshold  Max AI dependency score (0-100)       (default: 45)
    --bubble-threshold  Max bubble risk score (0-100)     (default: 65)
    --safety-threshold  Min safety score (0-100)          (default: 30)
    --max-beta      Max portfolio beta vs QQQ             (default: 0.40)
    --min-upside    Min upside % to consider a stock      (default: 2.0)
    --quiet         Suppress stage-by-stage progress      (default: False)

The scanner runs a 4-stage funnel:
  Stage 0  Fetch full US equity universe (~8,000 tickers)
  Stage 1  Cheap metadata filter: cap, sector, volume   -> ~200-300
  Stage 2  News AI screen: AI bubble dependency scoring -> ~80-120
  Stage 3  Full analysis: valuation + bubble + safety   -> top 15-25

Then builds a mean-variance optimized portfolio that:
  - Maximizes Sharpe ratio
  - Keeps portfolio beta vs QQQ (AI bubble proxy) <= max-beta
  - Limits each position to 2%-20% of budget
"""

import sys
import argparse

# Ensure models/ and data/ are on the path
sys.path.insert(0, '.')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Anti-bubble small-cap tech scanner & portfolio optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--budget',           type=float, default=100_000,
                        help='Total investment budget in USD (default: 100000)')
    parser.add_argument('--top',              type=int,   default=15,
                        help='Target number of portfolio holdings (default: 15)')
    parser.add_argument('--workers',          type=int,   default=16,
                        help='Parallel threads (default: 16)')
    parser.add_argument('--stage1-max',       type=int,   default=300,
                        help='Max stocks after Stage 1 metadata filter (default: 300)')
    parser.add_argument('--stage2-max',       type=int,   default=100,
                        help='Max stocks after Stage 2 news screen (default: 100)')
    parser.add_argument('--ai-threshold',     type=float, default=45.0,
                        help='Max AI bubble dependency score 0-100 (default: 45)')
    parser.add_argument('--bubble-threshold', type=float, default=65.0,
                        help='Max bubble risk score 0-100 (default: 65)')
    parser.add_argument('--safety-threshold', type=float, default=30.0,
                        help='Min safety score 0-100 (default: 30)')
    parser.add_argument('--max-beta',         type=float, default=0.40,
                        help='Max portfolio beta vs QQQ (default: 0.40)')
    parser.add_argument('--min-upside',       type=float, default=2.0,
                        help='Min valuation upside %% to include a stock (default: 2.0)')
    parser.add_argument('--quiet',            action='store_true',
                        help='Suppress stage-by-stage progress output')
    args = parser.parse_args()

    print(f"""
Anti-Bubble Small-Cap Tech Scanner
===================================
Budget          : ${args.budget:,.0f}
Target holdings : {args.top}
Workers         : {args.workers}
AI dep limit    : {args.ai_threshold}
Bubble risk lim : {args.bubble_threshold}
Safety min      : {args.safety_threshold}
Max QQQ beta    : {args.max_beta}
Min upside      : {args.min_upside}%
""")

    from models.anti_bubble_scanner import AntiBubbleScanner

    scanner = AntiBubbleScanner(
        max_workers=args.workers,
        max_stage1_results=args.stage1_max,
        max_stage2_results=args.stage2_max,
        ai_dep_threshold=args.ai_threshold,
        bubble_threshold=args.bubble_threshold,
        safety_threshold=args.safety_threshold,
        min_upside_pct=args.min_upside,
        max_portfolio_beta=args.max_beta,
    )

    try:
        portfolio = scanner.run(
            budget=args.budget,
            top_n=args.top,
            verbose=not args.quiet,
        )
        portfolio.display()

    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("\nTips to widen the search:")
        print("  --ai-threshold 60       (allow more AI-adjacent stocks)")
        print("  --bubble-threshold 75   (allow higher bubble risk)")
        print("  --safety-threshold 20   (lower quality bar)")
        print("  --min-upside 0          (include fairly-valued stocks)")
        sys.exit(1)


if __name__ == '__main__':
    main()