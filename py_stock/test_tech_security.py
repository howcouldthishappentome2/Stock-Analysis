"""Test tech stock security analyzer integration"""
import sys
sys.path.insert(0, '.')

from models.optimized_market_scanner import OptimizedStockMarketScanner

# Test with just one tech stock
scanner = OptimizedStockMarketScanner(
    tickers=['AAPL'],  # Test with Apple
    max_workers=1
)

print("Testing tech stock security analyzer with AAPL...")
opportunities = scanner.scan_market(use_tech_stocks=True)

if opportunities:
    for opp in opportunities:
        print(f"\nTicker: {opp.ticker}")
        print(f"Current Price: ${opp.current_price:.2f}")
        print(f"Fair Value: ${opp.fair_value:.2f}")
        print(f"Recommendation: {opp.recommendation}")
        print(f"Safety Score: {opp.safety_score}")
        if opp.security_alerts:
            print(f"Security Alerts ({len(opp.security_alerts)}):")
            for alert in opp.security_alerts:
                print(f"  - {alert['risk_type']}: {alert['description']} (severity: {alert['severity']})")
else:
    print("No opportunities found (stock likely failed security checks)")
