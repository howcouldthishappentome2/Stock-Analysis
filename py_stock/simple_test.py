#!/usr/bin/env python
import requests

test_data = {
    'stocks': [{'ticker': 'NVDA', 'upside_downside_pct': 0.45}],
    'budget': 100000,
    'risk_tolerance': 'MODERATE'
}

r = requests.post('http://localhost:3000/api/allocate-portfolio', json=test_data)
data = r.json()

if 'allocations' in data and data['allocations']:
    alloc = data['allocations'][0]
    weight_pct = alloc.get('weight_pct')
    print(f'weight_pct: {weight_pct}')
    print(f'type: {type(weight_pct).__name__}')
    
    if isinstance(weight_pct, (int, float)):
        print('Result: SUCCESS - weight_pct is a number')
    else:
        print('Result: FAIL - weight_pct is not a number')
else:
    print('Error:', data)
