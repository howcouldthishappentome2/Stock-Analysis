import requests
r = requests.post('http://127.0.0.1:3000/api/analyze', json={'ticker':'AAPL'})
print('STATUS', r.status_code)
print(r.text[:1000])
