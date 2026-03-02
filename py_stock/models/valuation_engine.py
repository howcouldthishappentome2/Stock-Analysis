"""
Stock Valuation Engine - DCF, valuation ratios, and buy/sell signals
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict
from .stock_params import StockParams


@dataclass
class ValuationMetrics:
    """Stock valuation results"""
    ticker: str
    current_price: float
    fair_value: float
    upside_downside: float  # %
    pe_ratio: float
    historical_pe: float  # For comparison
    pb_ratio: float
    payout_ratio: float
    dividend_coverage: str
    overall_rating: str  # BUY, HOLD, SELL


class StockValuationEngine:
    """Complete stock valuation using multiple methods"""
    
    def __init__(self, stock: StockParams, risk_free_rate: float = 0.04):
        """
        Initialize valuation engine
        
        Args:
            stock: StockParams object
            risk_free_rate: Risk-free rate for discount calculations
        """
        self.stock = stock
        self.risk_free_rate = risk_free_rate
        self.equity_risk_premium = 0.05  # Typical equity risk premium
        self.required_return = risk_free_rate + self.equity_risk_premium
        
    def calculate_fair_value(self) -> Tuple[float, str]:
        """
        Calculate fair value using multiple methods and average
        Methods: Dividend Discount, P/E based, P/B based, DCF
        """
        valuations = []
        weights = []
        
        # Method 1: Dividend Discount Model (50% weight for dividend stocks)
        if self.stock.dividend_per_share > 0:
            ddm_value = self._dividend_discount_model()
            valuations.append(ddm_value)
            weights.append(0.40)
        
        # Method 2: P/E Multiple (30% weight)
        if self.stock.pe_ratio > 0:
            pe_value = self._pe_multiple_valuation()
            valuations.append(pe_value)
            weights.append(0.30)
        
        # Method 3: P/B Multiple (20% weight)
        if self.stock.book_value_per_share > 0:
            pb_value = self._pb_multiple_valuation()
            valuations.append(pb_value)
            weights.append(0.20)
        
        # Method 4: Free Cash Flow based (if available - fallback to earnings)
        fcf_value = self._fcf_based_valuation()
        valuations.append(fcf_value)
        if self.stock.dividend_per_share > 0:
            weights.append(0.10)
        else:
            weights.append(0.30)
        
        # Weighted average fair value
        total_weight = sum(weights)
        fair_value = sum(v * w for v, w in zip(valuations, weights)) / total_weight
        
        current_price = self.stock.current_price
        upside_downside = (fair_value - current_price) / current_price
        
        if upside_downside > 0.15:
            assessment = "Undervalued"
        elif upside_downside < -0.15:
            assessment = "Overvalued"
        else:
            assessment = "Fair Value"
        
        return fair_value, assessment
    
    def _dividend_discount_model(self) -> float:
        """Gordon Growth Model valuation"""
        if self.required_return <= self.stock.growth_rate:
            # Use alternative method
            return self.stock.earnings_per_share * 10 / self.stock.pe_ratio if self.stock.pe_ratio > 0 else self.stock.current_price
        
        next_dividend = self.stock.dividend_per_share * (1 + self.stock.growth_rate)
        return next_dividend / (self.required_return - self.stock.growth_rate)
    
    def _pe_multiple_valuation(self) -> float:
        """P/E ratio based valuation"""
        # Industry avg P/E for dividend stocks typically 15-20
        industry_avg_pe = 18
        return self.stock.earnings_per_share * industry_avg_pe
    
    def _pb_multiple_valuation(self) -> float:
        """P/B ratio based valuation"""
        # Industry avg P/B for dividend stocks typically 1.5-2.5
        industry_avg_pb = 2.0
        return self.stock.book_value_per_share * industry_avg_pb
    
    def _fcf_based_valuation(self) -> float:
        """Free Cash Flow based valuation"""
        # Approximate FCF = EPS * (1 - growth_rate) if growth sustainable
        # Conservative estimate
        sustainable_payout = self.stock.earnings_per_share * (1 - min(self.stock.growth_rate, 0.5))
        fcf_yield = self.required_return
        return sustainable_payout / fcf_yield if fcf_yield > 0 else self.stock.current_price
    
    def analyze_valuation_ratios(self) -> ValuationMetrics:
        """
        Comprehensive valuation ratio analysis
        """
        fair_value, assessment = self.calculate_fair_value()
        upside_downside = (fair_value - self.stock.current_price) / self.stock.current_price
        
        # Dividend coverage
        if self.stock.earnings_per_share > 0:
            payout_ratio = self.stock.dividend_per_share / self.stock.earnings_per_share
            if payout_ratio < 0.60:
                dividend_coverage = "Safe (Payout < 60%)"
            elif payout_ratio < 0.80:
                dividend_coverage = "Moderate (60-80%)"
            else:
                dividend_coverage = "High Risk (>80%)"
        else:
            payout_ratio = 0
            dividend_coverage = "Insufficient data"
        
        # Determine rating
        if upside_downside > 0.20:
            rating = "STRONG BUY"
        elif upside_downside > 0.10:
            rating = "BUY"
        elif upside_downside > -0.10:
            rating = "HOLD"
        elif upside_downside > -0.20:
            rating = "SELL"
        else:
            rating = "STRONG SELL"
        
        return ValuationMetrics(
            ticker=self.stock.ticker,
            current_price=self.stock.current_price,
            fair_value=fair_value,
            upside_downside=upside_downside * 100,
            pe_ratio=self.stock.pe_ratio,
            historical_pe=18,  # Industry average
            pb_ratio=self.stock.current_price / self.stock.book_value_per_share if self.stock.book_value_per_share > 0 else 0,
            payout_ratio=payout_ratio,
            dividend_coverage=dividend_coverage,
            overall_rating=rating
        )
    
    def sensitivity_analysis(self, 
                            rate_change_range: List[float] = None,
                            growth_change_range: List[float] = None) -> Dict:
        """
        Sensitivity analysis on key assumptions
        
        Args:
            rate_change_range: Range of discount rate changes [-2%, +2%]
            growth_change_range: Range of growth rate changes [-2%, +2%]
        """
        if rate_change_range is None:
            rate_change_range = np.linspace(-0.02, 0.02, 5)
        if growth_change_range is None:
            growth_change_range = np.linspace(-0.02, 0.02, 5)
        
        sensitivity_matrix = np.zeros((len(rate_change_range), len(growth_change_range)))
        
        for i, rate_delta in enumerate(rate_change_range):
            for j, growth_delta in enumerate(growth_change_range):
                temp_required_return = self.required_return + rate_delta
                temp_growth = self.stock.growth_rate + growth_delta
                
                if temp_required_return <= temp_growth:
                    sensitivity_matrix[i, j] = self.stock.current_price
                else:
                    next_div = self.stock.dividend_per_share * (1 + temp_growth)
                    fair_val = next_div / (temp_required_return - temp_growth)
                    sensitivity_matrix[i, j] = fair_val
        
        return {
            'sensitivity_matrix': sensitivity_matrix,
            'rate_changes': [f"{x:.1%}" for x in rate_change_range],
            'growth_changes': [f"{x:.1%}" for x in growth_change_range],
            'base_case': self.calculate_fair_value()[0]
        }
    
    def quality_score(self) -> Tuple[float, str]:
        """
        Calculate stock quality score (0-100)
        Factors: Dividend safety, growth sustainability, valuation, leverage
        """
        scores = []
        
        # Dividend Safety (max 30 points)
        if self.stock.earnings_per_share > 0:
            payout = self.stock.dividend_per_share / self.stock.earnings_per_share
            if payout < 0.40:
                scores.append(30)
            elif payout < 0.60:
                scores.append(25)
            elif payout < 0.80:
                scores.append(15)
            else:
                scores.append(5)
        else:
            scores.append(5)
        
        # Growth Quality (max 25 points)
        if self.stock.growth_rate > 0.10:
            scores.append(25)
        elif self.stock.growth_rate > 0.05:
            scores.append(20)
        elif self.stock.growth_rate > 0.03:
            scores.append(15)
        else:
            scores.append(10)
        
        # Valuation (max 25 points)
        _, assessment = self.calculate_fair_value()
        if assessment == "Undervalued":
            scores.append(25)
        elif assessment == "Fair Value":
            scores.append(15)
        else:
            scores.append(5)
        
        # Leverage/Debt (max 20 points)
        if self.stock.debt_to_equity < 0.5:
            scores.append(20)
        elif self.stock.debt_to_equity < 1.0:
            scores.append(15)
        elif self.stock.debt_to_equity < 1.5:
            scores.append(10)
        else:
            scores.append(5)
        
        total_score = sum(scores)
        
        if total_score >= 80:
            quality = "Excellent"
        elif total_score >= 70:
            quality = "Good"
        elif total_score >= 60:
            quality = "Fair"
        else:
            quality = "Poor"
        
        return total_score, quality
