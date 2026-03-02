#!/usr/bin/env python
import requests
import json

# Test portfolio allocation with the new gold-heavy allocation
test_data = {
    'stocks': [
        {'ticker': 'NVDA', 'upside_downside_pct': 0.452},
        {'ticker': 'MSFT', 'upside_downside_pct': 0.268},
        {'ticker': 'GOOGL', 'upside_downside_pct': 0.158},
        {'ticker': 'AAPL', 'upside_downside_pct': 0.122},
    ],
    'budget': 100000,
    'risk_tolerance': 'MODERATE'
}

r = requests.post('http://localhost:3000/api/allocate-portfolio', json=test_data)
data = r.json()

print("=== PORTFOLIO ALLOCATION WITH INCREASED GOLD ===\n")

print("Asset Breakdown:")
print("-" * 60)
for alloc in data.get('allocations', []):
    weight_pct = alloc['weight_pct']
    ticker = alloc['ticker']
    asset_class = alloc.get('asset_class', 'N/A')
    
    # Highlight gold
    marker = " 🟡 GOLD" if ticker == 'GLD' else ""
    print(f"{ticker:6} {asset_class:15} {weight_pct:6.2f}% | ${(weight_pct/100 * 100000):>8,.0f}{marker}")

print("\n" + "-" * 60)
print("\nAsset Class Breakdown:")
for asset_class, pct in sorted(data['portfolio_metrics']['asset_class_breakdown'].items()):
    bar = "█" * int(pct * 2)
    print(f"  {asset_class:15} {pct:6.1f}%  {bar}")

print("\nPortfolio Metrics:")
print(f"  Total Risk Score: {data['portfolio_metrics']['total_risk_score']}")
print(f"  Expected Volatility: {data['portfolio_metrics']['expected_volatility_pct']:.2f}%")
print(f"  Diversification Score: {data['portfolio_metrics']['diversification_score']:.1f}")

# Calculate gold exposure
gold_weight = next((a['weight_pct'] for a in data['allocations'] if a['ticker'] == 'GLD'), 0)
print(f"\n🟡 GOLD ALLOCATION: {gold_weight:.2f}% of portfolio (${gold_weight/100 * 100000:,.0f})")
