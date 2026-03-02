#!/usr/bin/env python
"""Test scanner with single stock"""
import sys
sys.path.insert(0, '.')

from models.optimized_market_scanner import OptimizedStockMarketScanner

# Test with just JNJ
class TestScanner(OptimizedStockMarketScanner):
    DIVIDEND_STOCKS = ['JNJ']

scanner = TestScanner(max_workers=1, stock_type='dividend')
print("Scanning 1 dividend stock...")
opportunities = scanner.scan_market(use_tech_stocks=False)

print(f"Found {len(opportunities)} opportunities")
if opportunities:
    opp = opportunities[0]
    print(f"  {opp.ticker}: ${opp.current_price:.2f} -> ${opp.fair_value:.2f}")
else:
    print("No opportunities found!")
