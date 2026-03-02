import requests, json
print('calling scan...')
r = requests.get('http://127.0.0.1:3000/api/scan-market', timeout=120)
print('status', r.status_code)
data = r.json()
print(json.dumps(data, indent=2)[:2000])
