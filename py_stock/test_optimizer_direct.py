#!/usr/bin/env python
import sys
sys.path.insert(0, '/c/Users/Patrick Ma/OneDrive - The University of Melbourne/py_stock')

from models.portfolio_optimizer import PortfolioOptimizer

test_stocks = [
    {'ticker': 'NVDA', 'upside_downside_pct': 0.452},
    {'ticker': 'MSFT', 'upside_downside_pct': 0.268},
]

optimizer = PortfolioOptimizer()
allocation = optimizer.suggest_allocation(test_stocks, budget=100000, risk_tolerance='MODERATE')
result = optimizer.format_allocation_for_json(allocation)

print('First allocation:')
alloc = result['allocations'][0]
print(f"  ticker: {alloc['ticker']}")
print(f"  weight_pct: {alloc['weight_pct']} (type: {type(alloc['weight_pct']).__name__})")

import json
print('\nFull result (first 500 chars):')
print(json.dumps(result, indent=2)[:500])
