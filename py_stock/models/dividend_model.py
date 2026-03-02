"""
Dividend Growth Model - Gordon Growth Model and dividend analysis
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from .stock_params import DividendGrowthParams, StockParams
from typing import Optional


# Local import to avoid potential circular imports at module load
def _get_expected_rate(analyzer, t: float) -> float:
    try:
        return analyzer.model.closed_form_expected_rate(t)
    except Exception:
        # Fallback to current rate if analyzer unavailable
        return getattr(analyzer, 'params', None).r0 if getattr(analyzer, 'params', None) else 0.03


@dataclass
class DividendProjection:
    """Dividend projection results"""
    current_dividend: float
    projected_dividends: List[float]
    years: List[int]
    fair_value: float
    upside_downside: float  # % difference from current price
    valuation_quality: str  # "Undervalued", "Fair", "Overvalued"


class DividendGrowthModel:
    """Gordon Growth Model for dividend-paying stocks"""
    
    def __init__(self, stock_params: StockParams, interest_rate: float):
        """
        Initialize dividend growth model
        
        Args:
            stock_params: Stock parameters
            interest_rate: Current risk-free rate for discount rate calculation
        """
        self.stock_params = stock_params
        self.interest_rate = interest_rate
        self.risk_premium = 0.05  # Equity risk premium
        self.required_return = interest_rate + self.risk_premium
        
    def gordon_growth_model(self) -> Tuple[float, str]:
        """
        Calculate fair value using Gordon Growth Model
        Formula: P = D1 / (r - g)
        where D1 = next year's dividend, r = required return, g = growth rate
        
        Returns:
            Tuple of (fair_value, assessment)
        """
        if self.required_return <= self.stock_params.growth_rate:
            # Model invalid when growth >= return
            return 0, "Invalid (growth >= required return)"
        
        next_dividend = self.stock_params.dividend_per_share * (1 + self.stock_params.growth_rate)
        fair_value = next_dividend / (self.required_return - self.stock_params.growth_rate)
        
        current_price = self.stock_params.current_price
        upside_downside = (fair_value - current_price) / current_price
        
        if upside_downside > 0.15:
            assessment = "Undervalued"
        elif upside_downside < -0.15:
            assessment = "Overvalued"
        else:
            assessment = "Fair"
            
        return fair_value, assessment
    
    def multi_stage_dividend_growth(self, 
                                   high_growth_years: int = 5,
                                   high_growth_rate: float = None) -> DividendProjection:
        """
        Multi-stage dividend growth model (2-stage)
        High growth period followed by stable growth
        
        Args:
            high_growth_years: Years of high growth period
            high_growth_rate: High growth rate (if None, use stock's growth rate)
            
        Returns:
            DividendProjection object with projections
        """
        if high_growth_rate is None:
            high_growth_rate = self.stock_params.growth_rate
        
        stable_growth_rate = 0.03  # Conservative long-term growth (GDP-like)
        current_dividend = self.stock_params.dividend_per_share
        
        # High growth phase projections
        projected_dividends = []
        years = []
        pv_high_growth = 0
        
        for year in range(1, high_growth_years + 1):
            projected_div = current_dividend * ((1 + high_growth_rate) ** year)
            projected_dividends.append(projected_div)
            years.append(year)
            discount_factor = (1 + self.required_return) ** year
            pv_high_growth += projected_div / discount_factor
        
        # Terminal value (stable growth phase)
        terminal_dividend = projected_dividends[-1] * (1 + stable_growth_rate)
        
        if self.required_return <= stable_growth_rate:
            terminal_value = 0
        else:
            terminal_value = terminal_dividend / (self.required_return - stable_growth_rate)
        
        pv_terminal = terminal_value / ((1 + self.required_return) ** high_growth_years)
        
        # Fair value
        fair_value = pv_high_growth + pv_terminal
        
        current_price = self.stock_params.current_price
        upside_downside = (fair_value - current_price) / current_price
        
        if upside_downside > 0.15:
            valuation_quality = "Undervalued"
        elif upside_downside < -0.15:
            valuation_quality = "Overvalued"
        else:
            valuation_quality = "Fair"
        
        return DividendProjection(
            current_dividend=current_dividend,
            projected_dividends=projected_dividends,
            years=years,
            fair_value=fair_value,
            upside_downside=upside_downside,
            valuation_quality=valuation_quality
        )
    
    def dividend_safety_check(self) -> Tuple[float, str]:
        """
        Check if dividend is sustainable
        Returns dividend cover ratio and sustainability assessment
        
        Formula: Payout Ratio = Dividends / Earnings
        Safe if < 60%, Caution if 60-80%, Risky if > 80%
        """
        if self.stock_params.earnings_per_share <= 0:
            return 0, "Insufficient data"
        
        payout_ratio = self.stock_params.dividend_per_share / self.stock_params.earnings_per_share
        
        if payout_ratio < 0.60:
            assessment = "Safe - Room to grow dividend"
        elif payout_ratio < 0.80:
            assessment = "Caution - Limited growth room"
        else:
            assessment = "Risky - Dividend may be cut"
        
        return payout_ratio, assessment
    
    def yield_analysis(self) -> dict:
        """
        Analyze dividend yield and compare to benchmarks
        """
        yield_pct = self.stock_params.dividend_yield * 100
        
        # Typical benchmarks
        risk_free_yield = self.interest_rate * 100
        high_yield_threshold = 5.0  # %
        low_yield_threshold = 2.0  # %
        
        if yield_pct > high_yield_threshold:
            yield_assessment = "High yield - Verify sustainability"
        elif yield_pct > low_yield_threshold:
            yield_assessment = "Attractive yield"
        else:
            yield_assessment = "Low yield - Consider capital appreciation"
        
        excess_yield = yield_pct - risk_free_yield
        
        return {
            'current_yield': yield_pct,
            'risk_free_yield': risk_free_yield,
            'excess_yield': excess_yield,
            'assessment': yield_assessment
        }

    def discounted_ddm_with_rate_analyzer(self,
                                          rate_analyzer: Optional[object] = None,
                                          high_growth_years: int = 5,
                                          stable_growth_rate: float = 0.03) -> DividendProjection:
        """
        Discount dividend cash flows using an interest-rate forecast provided by an
        InterestRateAnalyzer (uses closed-form expected rates per year). This produces
        a DDM-style fair value where each year's dividend is discounted by the
        expected short rates (annual compounding approximated).

        Args:
            rate_analyzer: InterestRateAnalyzer instance (optional)
            high_growth_years: years to project high growth before terminal
            stable_growth_rate: terminal long-term dividend growth

        Returns:
            DividendProjection with discounted fair value and metadata
        """
        current_dividend = self.stock_params.dividend_per_share

        projected_dividends = []
        years = []
        pv_sum = 0.0

        # Use expected rates provider
        for year in range(1, high_growth_years + 1):
            projected_div = current_dividend * ((1 + self.stock_params.growth_rate) ** year)
            projected_dividends.append(projected_div)
            years.append(year)

            # Get expected short rate for this year (annual approximation)
            expected_r = _get_expected_rate(rate_analyzer, float(year)) if rate_analyzer else self.interest_rate

            # Discount factor using continuous compounding approximation
            discount_factor = np.exp(-expected_r * year)
            pv_sum += projected_div * discount_factor

        # Terminal value at end of high growth period using Gordon formula
        terminal_dividend = projected_dividends[-1] * (1 + stable_growth_rate)
        # Use expected long-term rate at terminal horizon
        expected_terminal_r = _get_expected_rate(rate_analyzer, float(high_growth_years)) if rate_analyzer else self.interest_rate

        if expected_terminal_r <= stable_growth_rate:
            terminal_value = 0.0
        else:
            terminal_value = terminal_dividend / (expected_terminal_r - stable_growth_rate)

        # Discount terminal back to present
        discount_terminal = np.exp(-expected_terminal_r * high_growth_years)
        pv_terminal = terminal_value * discount_terminal

        fair_value = pv_sum + pv_terminal

        current_price = self.stock_params.current_price
        upside_downside = (fair_value - current_price) / current_price if current_price > 0 else 0.0

        if upside_downside > 0.15:
            valuation_quality = "Undervalued"
        elif upside_downside < -0.15:
            valuation_quality = "Overvalued"
        else:
            valuation_quality = "Fair"

        return DividendProjection(
            current_dividend=current_dividend,
            projected_dividends=projected_dividends,
            years=years,
            fair_value=fair_value,
            upside_downside=upside_downside,
            valuation_quality=valuation_quality
        )

    def monte_carlo_discounted_ddm(self,
                                   rate_analyzer: Optional[object] = None,
                                   high_growth_years: int = 5,
                                   stable_growth_rate: float = 0.03,
                                   num_paths: int = 1000,
                                   num_steps_per_year: int = 1) -> DividendProjection:
        """
        Monte-Carlo path-averaged DDM: simulate interest-rate paths and discount
        each path's dividend cash flows by that path's short rates, then average
        PVs across all simulated paths to produce a stochastic fair value.

        Args:
            rate_analyzer: InterestRateAnalyzer instance providing .model.simulate_paths()
            high_growth_years: projection horizon before terminal
            stable_growth_rate: terminal dividend growth rate
            num_paths: Monte Carlo paths
            num_steps_per_year: subdivisions per year (1 = annual)

        Returns:
            DividendProjection
        """
        current_dividend = self.stock_params.dividend_per_share
        projected_dividends = []
        years = list(range(1, high_growth_years + 1))

        # Build dividends per year (deterministic growth assumption)
        for year in years:
            projected_dividends.append(current_dividend * ((1 + self.stock_params.growth_rate) ** year))

        # If rate_analyzer provided, simulate paths; otherwise fallback to deterministic rate
        if rate_analyzer is None:
            # Fallback: same as discounted_ddm_with_rate_analyzer
            return self.discounted_ddm_with_rate_analyzer(rate_analyzer=None,
                                                          high_growth_years=high_growth_years,
                                                          stable_growth_rate=stable_growth_rate)

        # Simulate annual-ish paths: use num_steps = high_growth_years * num_steps_per_year
        sim = rate_analyzer.model.simulate_paths(T=high_growth_years,
                                                num_steps=high_growth_years * num_steps_per_year,
                                                num_paths=num_paths)

        paths = sim.paths  # shape: (num_paths, num_steps+1)
        # Coarsen to yearly by taking values at integer years (assuming equally spaced)
        step_idx = [int(i * num_steps_per_year) for i in range(1, high_growth_years + 1)]
        # rates_by_year shape: (num_paths, high_growth_years)
        rates_by_year = paths[:, step_idx]

        pvs = []
        for i in range(rates_by_year.shape[0]):
            rates = rates_by_year[i, :]
            # cumulative continuous discount exponents per year
            cum_rates = np.cumsum(rates / float(num_steps_per_year))  # if subannual, adjust

            pv = 0.0
            for t, div in enumerate(projected_dividends):
                discount = np.exp(-cum_rates[t])
                pv += div * discount

            # terminal value using path's terminal short rate
            r_T = rates[-1]
            terminal_div = projected_dividends[-1] * (1 + stable_growth_rate)
            if r_T <= stable_growth_rate:
                terminal_value = 0.0
            else:
                terminal_value = terminal_div / (r_T - stable_growth_rate)

            # discount terminal
            discount_terminal = np.exp(-cum_rates[-1])
            pv += terminal_value * discount_terminal

            pvs.append(pv)

        # Average across paths
        fair_value = float(np.mean(pvs)) if len(pvs) > 0 else 0.0

        current_price = self.stock_params.current_price
        upside_downside = (fair_value - current_price) / current_price if current_price > 0 else 0.0

        if upside_downside > 0.15:
            valuation_quality = "Undervalued"
        elif upside_downside < -0.15:
            valuation_quality = "Overvalued"
        else:
            valuation_quality = "Fair"

        return DividendProjection(
            current_dividend=current_dividend,
            projected_dividends=projected_dividends,
            years=years,
            fair_value=fair_value,
            upside_downside=upside_downside,
            valuation_quality=valuation_quality
        )
