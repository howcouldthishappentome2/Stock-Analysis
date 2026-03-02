import requests, json
r = requests.get('http://127.0.0.1:3000/api/scan-market?limit=1')
print('Status', r.status_code)
data = r.json()
import pprint
pprint.pprint(data)
print('\nDividend opp keys:', list(data['dividend_stocks']['opportunities'][0].keys()))
print('Tech opp keys:', list(data['tech_stocks']['opportunities'][0].keys()))
