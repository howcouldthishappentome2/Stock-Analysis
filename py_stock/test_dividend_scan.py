#!/usr/bin/env python
"""Test scanner with actual dividend stocks from app.py setup"""
import sys
sys.path.insert(0, '.')

from models.optimized_market_scanner import OptimizedStockMarketScanner

# This mimics what app.py does 
scanner = OptimizedStockMarketScanner(
    tickers=OptimizedStockMarketScanner.DIVIDEND_STOCKS, 
    max_workers=4, 
    stock_type='dividend'
)

print(f"Scanning {len(scanner.tickers)} dividend stocks with scanner setup...")
opportunities = scanner.scan_market(use_tech_stocks=False)

print(f"\nFound {len(opportunities)} opportunities")
for i, opp in enumerate(opportunities[:3], 1):
    print(f"{i}. {opp.ticker}: ${opp.current_price:.2f} -> ${opp.fair_value:.2f} ({opp.probability_of_profit*100:.0f}% prob)")
