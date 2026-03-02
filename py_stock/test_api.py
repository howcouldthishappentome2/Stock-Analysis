import requests
import time
time.sleep(3)
try:
    print("Making request to /api/scan-market...")
    r = requests.get('http://localhost:3000/api/scan-market?limit=5', timeout=120)
    print(f"Status: {r.status_code}")
    d = r.json()
    div_count = len(d.get('dividend_stocks', {}).get('opportunities', []))
    tech_count = len(d.get('tech_stocks', {}).get('opportunities', []))
    print(f"Dividend stocks: {div_count}")
    print(f"Tech stocks: {tech_count}")
    if div_count > 0:
        print(f"First dividend: {d['dividend_stocks']['opportunities'][0]['ticker']}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
