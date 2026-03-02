#!/usr/bin/env python
"""Test with 1 worker and 1 stock"""
import sys
sys.path.insert(0, '.')

from models.optimized_market_scanner import OptimizedStockMarketScanner

# Test with just 1 stock and 1 worker to see what happens
class SingleStockScanner(OptimizedStockMarketScanner):
    DIVIDEND_STOCKS = ['JNJ']

scanner = SingleStockScanner(max_workers=1, stock_type='dividend')
print("Starting scan...")

import time
start = time.time()

try:
    opportunities = scanner.scan_market(use_tech_stocks=False)
    elapsed = time.time() - start
    print(f"Found {len(opportunities)} opportunities in {elapsed:.1f}s")
except KeyboardInterrupt:
    print("Interrupted!")
