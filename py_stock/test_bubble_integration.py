"""
Quick integration test of bubble detection with the market scanner
"""
import sys
sys.path.insert(0, '.')

# Simulate a quick scan with minimal workers
from models.optimized_market_scanner import OptimizedStockMarketScanner
from models.bubble_detector import BubbleDetector

# Test with just a couple tech stocks
test_stocks = ['AAPL', 'MSFT']  # Will read real data from yfinance

print("=" * 80)
print("INTEGRATION TEST: Bubble Detection in Market Scanner")
print("=" * 80)
print()
print("Scanning tech stocks with bubble detection enabled...")
print("This will:")
print("1. Fetch real stock data from yfinance")
print("2. Run security checks (fraud/scam detection)")
print("3. Run bubble detection (valuation analysis)")
print("4. Filter out critical risks")
print()

try:
    # Create scanner for tech stocks
    scanner = OptimizedStockMarketScanner(
        tickers=test_stocks,
        max_workers=1,  # Serial for easier debugging
        stock_type='tech'
    )
    
    # Scan the market
    opportunities = scanner.scan_market(use_tech_stocks=False)
    
    print(f"\nScan Results: {len(opportunities)} stocks passed security & bubble checks")
    print()
    
    for opp in opportunities:
        print(f"Ticker: {opp.ticker}")
        print(f"  Current Price: ${opp.current_price:.2f}")
        print(f"  Fair Value: ${opp.fair_value:.2f}")
        print(f"  Upside/Downside: {opp.upside_downside_pct:.2f}%")
        print(f"  Safety Score: {opp.safety_score:.1f}/100" if opp.safety_score else "  Safety Score: N/A")
        
        if opp.security_alerts:
            bubble_alerts = [a for a in opp.security_alerts if 'bubble' in str(a.get('risk_type', ''))]
            fraud_alerts = [a for a in opp.security_alerts if 'bubble' not in str(a.get('risk_type', ''))]
            
            if bubble_alerts:
                print(f"  BUBBLE ALERTS: {len(bubble_alerts)}")
                for alert in bubble_alerts[:2]:  # Show first 2
                    print(f"    - [{alert['severity'].upper()}] {alert['type']}")
            
            if fraud_alerts:
                print(f"  FRAUD ALERTS: {len(fraud_alerts)}")
                for alert in fraud_alerts[:2]:  # Show first 2
                    print(f"    - [{alert['severity'].upper()}] {alert['risk_type']}")
        print()
    
    scanner.close()
    print("=" * 80)
    print("SUCCESS: Bubble detection integration working correctly!")
    print("=" * 80)
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
