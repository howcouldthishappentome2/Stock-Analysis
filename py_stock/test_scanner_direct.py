#!/usr/bin/env python
"""Direct test of the scanner"""
import sys
sys.path.insert(0, '.')

from models.optimized_market_scanner import OptimizedStockMarketScanner

print("Testing dividend stock scanner...")
scanner = OptimizedStockMarketScanner(max_workers=4, stock_type='dividend')
opportunities = scanner.scan_market(use_tech_stocks=False)

print(f"\nFound {len(opportunities)} dividend stock opportunities")
for opp in opportunities[:5]:
    print(f"  {opp.ticker}: ${opp.current_price:.2f} -> ${opp.fair_value:.2f} (Profit: {opp.probability_of_profit*100:.0f}%)")
