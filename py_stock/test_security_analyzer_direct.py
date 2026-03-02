"""Direct test of TechStockSecurityAnalyzer with mock stock data"""
import sys
sys.path.insert(0, '.')

from models.stock_params import StockParams
from models.tech_stock_analyzer import TechStockSecurityAnalyzer
from datetime import datetime, timedelta

# Create mock stock data for a tech stock
mock_stock = StockParams(
    ticker='TEST',
    current_price=150.0,
    dividend_yield=0.0,
    dividend_per_share=0.0,
    growth_rate=0.25,
    volatility=0.35,  # 35% volatility
    pe_ratio=35.0,
    payout_ratio=0.0,
    earnings_per_share=2.5,
    book_value_per_share=15.0,
    debt_to_equity=0.8
)

# Add extended attributes that tech_stock_analyzer might use
mock_stock.week_52_high = 160.0  # Close to current - should trigger peak warning
mock_stock.week_52_low = 90.0
mock_stock.ytd_return = 0.35  # Up 35% year-to-date
mock_stock.float_shares = 50000000  # 50M shares float - NOT low float

print("Testing TechStockSecurityAnalyzer with mock stock data...")
print(f"Stock: {mock_stock.ticker}")
print(f"Current Price: ${mock_stock.current_price}")
print(f"52-week High: ${mock_stock.week_52_high}")
print(f"P/E Ratio: {mock_stock.pe_ratio}")
print(f"Growth Rate: {mock_stock.growth_rate*100}%")
print(f"YTD Return: {mock_stock.ytd_return*100}%")
print()

analyzer = TechStockSecurityAnalyzer(mock_stock)
is_safe, alerts = analyzer.analyze()
safety_score = analyzer.get_safety_score()

print(f"Is Safe: {is_safe}")
print(f"Safety Score: {safety_score}/100")
print(f"Number of Alerts: {len(alerts)}")
print()

if alerts:
    print("Security Alerts:")
    for alert in alerts:
        print(f"  [{alert.severity.upper()}] {alert.risk_type}:")
        print(f"    Description: {alert.description}")
        print(f"    Confidence: {alert.confidence*100:.0f}%")
        print()
else:
    print("No alerts - stock appears safe")
