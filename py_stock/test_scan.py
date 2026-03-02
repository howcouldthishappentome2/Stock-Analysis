import subprocess
import time
import requests
import json
import sys

# Start Flask in background
proc = subprocess.Popen(['python', 'app.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
time.sleep(5)  # Wait for Flask to start

try:
    # Test the endpoint  
    print('Testing /api/scan-market endpoint...')
    response = requests.get('http://localhost:3000/api/scan-market?limit=3', timeout=30)
    print('Status:', response.status_code)
    data = response.json()
    if 'error' in data:
        print('Error:', data['error'])
    else:
        div_count = len(data.get('dividend_stocks', {}).get('opportunities', []))
        tech_count = len(data.get('tech_stocks', {}).get('opportunities', []))
        print(f'[OK] Dividend stocks: {div_count}')
        print(f'[OK] Tech stocks: {tech_count}')
        if div_count > 0:
            ticker = data['dividend_stocks']['opportunities'][0]['ticker']
            print(f'  Sample dividend stock: {ticker}')
        if tech_count > 0:
            ticker = data['tech_stocks']['opportunities'][0]['ticker']
            print(f'  Sample tech stock: {ticker}')
except Exception as e:
    import traceback
    print('Error:', type(e).__name__, str(e)[:100])
    traceback.print_exc()
finally:
    proc.terminate()
    proc.wait()
