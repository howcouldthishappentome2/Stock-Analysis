#!/usr/bin/env python
import requests
import json

# Test portfolio allocation endpoint
test_data = {
    'stocks': [
        {'ticker': 'NVDA', 'upside_downside_pct': 0.452},
        {'ticker': 'MSFT', 'upside_downside_pct': 0.268},
        {'ticker': 'GOOGL', 'upside_downside_pct': 0.158},
        {'ticker': 'AAPL', 'upside_downside_pct': 0.122},
    ],
    'budget': 100000,
    'risk_tolerance': 'MODERATE'
}

try:
    response = requests.post('http://localhost:3000/api/allocate-portfolio', json=test_data)
    data = response.json()
    
    print('Response status:', response.status_code)
    print('\nAllocations:')
    for alloc in data.get('allocations', []):
        print(f"  {alloc['ticker']}: weight_pct={alloc['weight_pct']} (type: {type(alloc['weight_pct']).__name__})")
        # Test that weight_pct is callable with toFixed
        if isinstance(alloc['weight_pct'], (int, float)):
            print(f"    -> {alloc['weight_pct']:.1f}% ✓")
        else:
            print(f"    -> ERROR: not a number! ✗")
    
    print('\nPortfolio metrics:')
    for k, v in data.get('portfolio_metrics', {}).items():
        print(f"  {k}: {v}")
        
except Exception as e:
    print(f'Error: {e}')
