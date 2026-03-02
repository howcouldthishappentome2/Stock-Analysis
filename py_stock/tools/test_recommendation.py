from models.stock_params import StockParams, InterestRateParams
from models.recommendation_engine import RecommendationEngine

s=StockParams(
    ticker='TST',
    current_price=100.0,
    dividend_yield=0.04,
    dividend_per_share=4.0,
    growth_rate=0.02,
    volatility=0.2,
    pe_ratio=15.0,
    payout_ratio=0.5,
    earnings_per_share=8.0,
    book_value_per_share=50.0,
    debt_to_equity=0.3
)

ir=InterestRateParams(r0=0.03,kappa=0.2,theta=0.04,sigma=0.01,model_type='Vasicek')

engine=RecommendationEngine(s, ir, risk_free_rate=0.03)
rec=engine.generate_recommendation()
print('FAIR_VALUE:', rec.fair_value)
print('ACTION:', rec.action)
print('CONFIDENCE:', rec.confidence)
print('DIV_PROJ_FAIR(upside %):', rec.target_price_upside)
