"""Test bubble detection with tech stock simulation"""
import sys
sys.path.insert(0, '.')

from models.stock_params import StockParams
from models.bubble_detector import BubbleDetector
from models.tech_stock_analyzer import TechStockSecurityAnalyzer

# Test Case 1: Normal tech stock (reasonable valuation)
print("=" * 80)
print("TEST 1: Normal Tech Stock (Apple-like profile)")
print("=" * 80)
normal_stock = StockParams(
    ticker='NORMAL',
    current_price=150.0,
    dividend_yield=0.005,
    dividend_per_share=0.96,
    growth_rate=0.10,  # 10% growth
    volatility=0.22,
    pe_ratio=25.0,     # Fair PE for tech
    payout_ratio=0.25,
    earnings_per_share=6.0,
    book_value_per_share=15.0,
    debt_to_equity=0.5
)
normal_stock.week_52_high = 160.0
normal_stock.ytd_return = 0.15

detector = BubbleDetector(normal_stock, sector='technology')
metrics = detector.detect_bubble()
print(f"Bubble Risk Score: {metrics.bubble_risk_score:.1f}/100")
print(f"Valuation Percentile: {metrics.valuation_percentile:.1f}th")
print(f"Risk Level: {detector.get_bubble_risk_level(metrics.bubble_risk_score)}")
print(f"Alerts: {len(metrics.alerts)}")
for alert in metrics.alerts:
    print(f"  - [{alert.severity.upper()}] {alert.type}: {alert.description[:60]}...")
print()

# Test Case 2: AI Bubble Stock (Nvidia-like) - High PE, circular capex
print("=" * 80)
print("TEST 2: AI Bubble Stock (Nvidia pattern - high PE, negative FCF)")
print("=" * 80)
bubble_stock = StockParams(
    ticker='NVDA_BUBBLE',
    current_price=300.0,
    dividend_yield=0.0,
    dividend_per_share=0.0,
    growth_rate=0.05,  # Only 5% actual growth
    volatility=0.45,   # High volatility
    pe_ratio=60.0,     # Extremely high PE
    payout_ratio=0.0,
    earnings_per_share=5.0,
    book_value_per_share=10.0,
    debt_to_equity=2.5
)
bubble_stock.week_52_high = 280.0  # Near peak
bubble_stock.ytd_return = 0.80     # Up 80% - parabolic
bubble_stock.free_cash_flow = -1000000  # Negative! Massive capex spending
bubble_stock.profit_margin = 0.03  # Only 3% margin

detector = BubbleDetector(bubble_stock, sector='ai')
metrics = detector.detect_bubble()
print(f"Bubble Risk Score: {metrics.bubble_risk_score:.1f}/100")
print(f"Valuation Percentile: {metrics.valuation_percentile:.1f}th")
print(f"Risk Level: {detector.get_bubble_risk_level(metrics.bubble_risk_score)}")
print(f"Alerts: {len(metrics.alerts)}")
for alert in metrics.alerts:
    print(f"  - [{alert.severity.upper()}] {alert.type}:")
    print(f"    {alert.description[:80]}...")
print()

# Test Case 3: Dot-com crash equivalent
print("=" * 80)
print("TEST 3: Extreme Bubble (2000 Dot-com equivalent)")
print("=" * 80)
dotcom_stock = StockParams(
    ticker='BUBBLE2000',
    current_price=500.0,
    dividend_yield=0.0,
    dividend_per_share=0.0,
    growth_rate=0.0,   # No actual earnings/growth
    volatility=0.60,
    pe_ratio=200.0,    # Insane PE
    payout_ratio=0.0,
    earnings_per_share=-1.0,  # Negative earnings!
    book_value_per_share=5.0,
    debt_to_equity=5.0  # Extreme debt
)
dotcom_stock.week_52_high = 520.0
dotcom_stock.ytd_return = 1.50  # Up 150%
dotcom_stock.free_cash_flow = -5000000

detector = BubbleDetector(dotcom_stock, sector='technology')
metrics = detector.detect_bubble()
print(f"Bubble Risk Score: {metrics.bubble_risk_score:.1f}/100")
print(f"Valuation Percentile: {metrics.valuation_percentile:.1f}th")
print(f"Risk Level: {detector.get_bubble_risk_level(metrics.bubble_risk_score)}")
print(f"Alerts: {len(metrics.alerts)}")
for alert in metrics.alerts:
    print(f"  - [{alert.severity.upper()}] {alert.type}:")
    print(f"    {alert.description}")
print()

# Test Case 4: Tech Stock Analyzer Integration
print("=" * 80)
print("TEST 4: TechStockSecurityAnalyzer with Bubble Detection")
print("=" * 80)
analyzer = TechStockSecurityAnalyzer(bubble_stock)
is_safe, alerts = analyzer.analyze()
safety_score = analyzer.get_safety_score()
print(f"Is Safe: {is_safe}")
print(f"Safety Score: {safety_score:.1f}/100")
print(f"Total Alerts: {len(alerts)}")
bubble_alerts = [a for a in alerts if 'bubble' in a.risk_type]
print(f"Bubble-related Alerts: {len(bubble_alerts)}")
for alert in bubble_alerts:
    print(f"  - [{alert.severity.upper()}] {alert.risk_type}:")
    print(f"    {alert.description[:70]}...")
