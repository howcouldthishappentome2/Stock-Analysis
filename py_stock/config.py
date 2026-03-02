"""Configuration file for stock analysis system"""

# Reddit API credentials (optional)
REDDIT_CLIENT_ID = None
REDDIT_CLIENT_SECRET = None
REDDIT_USER_AGENT = "stock_analyzer/1.0"

# Analysis Parameters
DEFAULT_MONTE_CARLO_PATHS = 10000
DEFAULT_HOLDING_PERIOD_DAYS = 365
DEFAULT_PORTFOLIO_VALUE = 100000

# Output Settings
OUTPUT_DIR = "output"

# Risk Thresholds
FRAUD_THRESHOLDS = {
    'volume_spike_extreme': 10.0,
    'volume_spike_high': 5.0,
    'price_spike_extreme': 5.0,
    'price_spike_high': 3.0,
}

# Stock Recommendation System Parameters
DIVIDEND_MODEL_PARAMS = {
    'high_growth_years': 5,
    'stable_growth_rate': 0.03,
    'equity_risk_premium': 0.05,
}

# Interest Rate Model Parameters
INTEREST_RATE_MODEL = {
    'model_type': 'CIR',  # 'CIR' or 'Vasicek'
    'kappa': 0.15,  # Mean reversion speed
    'sigma': 0.01,  # Volatility
}

# Position Sizing Parameters
POSITION_SIZING = {
    'max_single_position': 0.10,  # Max 10% in single stock
    'strong_buy_allocation': 0.05,  # 5% for STRONG_BUY
    'buy_allocation': 0.03,  # 3% for BUY
    'hold_allocation': 0.01,  # 1% for HOLD
}

# Risk Tolerance Adjustments
RISK_ADJUSTMENTS = {
    'CONSERVATIVE': 0.7,
    'MODERATE': 1.0,
    'AGGRESSIVE': 1.3,
}

# Dividend Stock Screening Criteria
DIVIDEND_SCREENING = {
    'min_dividend_yield': 0.02,  # 2%
    'max_pe_ratio': 25,
    'max_debt_to_equity': 2.0,
    'min_payout_coverage': 0.60,  # At least 60% dividend coverage
}

# Web Server Settings
FLASK_DEBUG = False  # Disable debug mode for stable performance
FLASK_USE_RELOADER = False  # Disable auto-reload to prevent slowdowns
FLASK_PORT = 3000
FLASK_HOST = '0.0.0.0'

# Data Collection
YFINANCE_PERIOD = '5y'  # Historical data period
YFINANCE_INTERVAL = '1d'  # Daily data
