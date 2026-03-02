#!/usr/bin/env python
"""Direct test of the scanner - just test one stock"""
import sys
sys.path.insert(0, '.')

from data.stock_data import StockDataCollector, InterestRateDataCollector
from models.recommendation_engine import RecommendationEngine
from models.stock_params import StockParams

# Test just one stock
ticker = 'JNJ'
print(f"Testing {ticker}...")

try:
    # Collect stock data
    collector = StockDataCollector(ticker)
    print(f"  Fetching {ticker} data...")
    stock_params = collector.fetch_stock_data()
    print(f"  Price: ${stock_params.current_price}")
    print(f"  Dividend Yield: {stock_params.dividend_yield:.4f}")
    
    # Get interest rates
    ir_collector = InterestRateDataCollector()
    ir_params = ir_collector.calibrate_ir_model()
    rates = ir_collector.get_current_rates()
    print(f"  Risk-free rate: {rates['risk_free_rate']:.4f}")
    
    # Generate recommendation
    print(f"  Generating recommendation...")
    engine = RecommendationEngine(
        stock_params,
        ir_params,
        risk_free_rate=rates['risk_free_rate']
    )
    recommendation = engine.generate_recommendation()
    print(f"  Recommendation: {recommendation.action.value}")
    print(f"  Fair Value: ${recommendation.fair_value:.2f}")
    print(f"  Upside: {recommendation.upside_downside_pct:.2f}%")
    print(f"Success!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
