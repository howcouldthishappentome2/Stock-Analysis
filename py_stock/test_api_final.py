import requests
import time
import json

time.sleep(3)
print("Testing dividend stock scan via API...")
try:
    r = requests.get('http://localhost:3000/api/scan-market?limit=3', timeout=180)
    print(f"Status: {r.status_code}")
    d = r.json()
    
    if 'error' in d:
        print(f"API Error: {d['error']}")
    else:
        div_stocks = d.get('dividend_stocks', {}).get('opportunities', [])
        tech_stocks = d.get('tech_stocks', {}).get('opportunities', [])
        print(f"Dividend stocks found: {len(div_stocks)}")
        print(f"Tech stocks found: {len(tech_stocks)}")
        
        if div_stocks:
            print("\nFirst 2 dividend stocks:")
            for stock in div_stocks[:2]:
                print(f"  {stock['ticker']}: ${stock['current_price']} -> ${stock['fair_value']}")
                
except Exception as e:
    print(f"Error: {e}")
