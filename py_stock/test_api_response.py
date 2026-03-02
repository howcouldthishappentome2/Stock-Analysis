#!/usr/bin/env python
import requests
import json

response = requests.get('http://localhost:3000/api/scan-anti-bubble?demo=true')
data = response.json()

print('Response keys:', sorted(data.keys()))
print('candidates_found:', data.get('candidates_found'))
print('Has portfolio_stats:', 'portfolio_stats' in data)
print('Has all_candidates:', 'all_candidates' in data)
print('Has budget:', 'budget' in data)
print('Has portfolio:', 'portfolio' in data)

if 'portfolio_stats' in data:
    print('\nportfolio_stats:')
    for k, v in data['portfolio_stats'].items():
        print(f'  {k}: {v}')

if 'all_candidates' in data:
    print(f'\nall_candidates count: {len(data["all_candidates"])}')
    print('Tickers:', [c['ticker'] for c in data['all_candidates']])

print('\nFull response structure:')
print(json.dumps(data, indent=2)[:500])
