from dataclasses import dataclass
from typing import Optional

@dataclass
class StockParams:
    """Parameters for individual stock simulation"""
    ticker: str
    current_price: float
    dividend_yield: float
    dividend_per_share: float
    growth_rate: float
    volatility: float
    pe_ratio: float
    payout_ratio: float
    earnings_per_share: float
    book_value_per_share: float
    debt_to_equity: float
    jump_intensity: Optional[float] = None
    jump_mean: Optional[float] = None
    jump_std: Optional[float] = None
    beta: Optional[float] = None
    beta_r2: Optional[float] = None

@dataclass
class InterestRateParams:
    """Parameters for CIR/Vasicek interest rate model"""
    r0: float  # Initial rate
    kappa: float  # Mean reversion speed
    theta: float  # Long-term mean
    sigma: float  # Volatility
    model_type: str = "CIR"  # "CIR" or "Vasicek"

@dataclass
class DividendGrowthParams:
    """Parameters for dividend growth model"""
    dividend_per_share: float
    growth_rate: float  # Historical/expected growth rate
    required_return: float  # Discount rate
    years_to_forecast: int = 5