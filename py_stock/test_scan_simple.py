"""Simple test to verify scan-market API is working"""
import urllib.request
import json
import sys

print("Testing scan-market API...")
print("This may take 1-3 minutes as it fetches real stock data...", flush=True)

try:
    url = 'http://localhost:3000/api/scan-market?limit=2'
    print(f"Fetching: {url}", flush=True)
    
    with urllib.request.urlopen(url, timeout=300) as response:
        data = json.loads(response.read().decode())
        
        dividend_count = data.get('dividend_stocks', {}).get('count', 0)
        tech_count = data.get('tech_stocks', {}).get('count', 0)
        
        print(f"\nAPI Response Successful!")
        print(f"Dividend stocks found: {dividend_count}")
        print(f"Tech stocks found: {tech_count}")
        
        if dividend_count > 0:
            div_stock = data['dividend_stocks']['opportunities'][0]
            print(f"\nExample dividend stock: {div_stock['ticker']}")
            print(f"  Safety Score: {div_stock.get('safety_score', 'N/A')}")
            
        if tech_count > 0:
            tech_stock = data['tech_stocks']['opportunities'][0]
            print(f"\nExample tech stock: {tech_stock['ticker']}")
            print(f"  Safety Score: {tech_stock.get('safety_score', 'N/A')}")
            print(f"  Has Bubble Alerts: {'bubble_alerts' in tech_stock}")
            
        print("\nScan Market button should now work!")
        sys.exit(0)

except Exception as e:
    print(f"ERROR: {str(e)}", flush=True)
    sys.exit(1)
