import requests
try:
    r = requests.get('http://localhost:3000/api/scan-market?limit=2', timeout=30)
    print(f'Status: {r.status_code}')
    d = r.json()
    div_count = len(d.get('dividend_stocks', {}).get('opportunities', []))
    tech_count = len(d.get('tech_stocks', {}).get('opportunities', []))
    print(f'Dividend stocks: {div_count}')
    print(f'Tech stocks: {tech_count}')
    if div_count > 0:
        print(f'  First dividend: {d["dividend_stocks"]["opportunities"][0]["ticker"]}')
    if tech_count > 0:
        print(f'  First tech: {d["tech_stocks"]["opportunities"][0]["ticker"]}')
except Exception as e:
    print(f'Error: {e}')
