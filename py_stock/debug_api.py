import requests
import json
import time

time.sleep(2)

print("Testing /api/scan-market endpoint...")
try:
    r = requests.get('http://localhost:3000/api/scan-market?limit=3', timeout=180)
    print(f"Status: {r.status_code}")
    data = r.json()
    
    print(f"\nResponse keys: {list(data.keys())}")
    
    if 'error' in data:
        print(f"Error: {data['error']}")
    else:
        div_stocks = data.get('dividend_stocks', {})
        print(f"\nDividend stocks response:")
        print(f"  Count: {div_stocks.get('count')}")
        print(f"  Opportunities: {len(div_stocks.get('opportunities', []))}")
        if div_stocks.get('opportunities'):
            for s in div_stocks['opportunities'][:2]:
                print(f"    - {s['ticker']}: ${s.get('current_price')} -> ${s.get('fair_value')}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
