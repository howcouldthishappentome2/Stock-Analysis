"""Test the scan-market API endpoint"""
import requests
import json

print("Testing /api/scan-market endpoint...")
try:
    response = requests.get('http://localhost:3000/api/scan-market?limit=3', timeout=120)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        data = response.json()
        print("\nSuccess! Response structure:")
        print(f"- Keys: {list(data.keys())}")
        if 'dividend_stocks' in data:
            print(f"- Dividend stocks count: {data['dividend_stocks'].get('count', 0)}")
        if 'tech_stocks' in data:
            print(f"- Tech stocks count: {data['tech_stocks'].get('count', 0)}")
    else:
        print(f"\nError Response:\n{response.text}")
except Exception as e:
    print(f"Exception: {str(e)}")
    import traceback
    traceback.print_exc()
