import os
import sys
import json
# Ensure project root is on path
root = os.path.dirname(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

from models.optimized_market_scanner import OptimizedStockMarketScanner

scanner = OptimizedStockMarketScanner(tickers=['JNJ','PG','KO','PEP','MO'], max_workers=4)
results = scanner.get_top_opportunities(n=5)
print('Found', len(results))
for r in results:
    print(json.dumps({
        'ticker': r.ticker,
        'current_price': r.current_price,
        'fair_value': r.fair_value,
        'upside_downside_pct': r.upside_downside_pct,
        'dividend_yield': r.dividend_yield,
        'recommendation': r.recommendation,
        'confidence': r.confidence,
        'probability_of_profit': r.probability_of_profit,
        'expected_holding_months': r.expected_holding_months,
        'expected_return': r.expected_return,
        'exit_signal': r.exit_signal,
        'key_risk': r.key_risk
    }))
