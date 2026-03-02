"""
Stock Market Scanner - Find profitable dividend stocks with profit probability analysis
"""
import asyncio
from typing import List, Tuple, Dict
from dataclasses import dataclass
from .recommendation_engine import RecommendationEngine, ActionType
from .stock_params import InterestRateParams
from data.stock_data import StockDataCollector, InterestRateDataCollector


@dataclass
class StockOpportunity:
    """Represents a stock opportunity ranked by profit probability"""
    ticker: str
    current_price: float
    fair_value: float
    upside_downside_pct: float
    dividend_yield: float
    recommendation: str
    confidence: float
    probability_of_profit: float  # 0-1
    expected_holding_months: int
    expected_return: float  # Annual %
    exit_signal: str  # When to sell
    key_risk: str


class StockMarketScanner:
    """Scan market for profitable dividend stocks"""
    
    # Popular dividend-paying stocks to scan
    POPULAR_DIVIDEND_STOCKS = [
        # Consumer Staples
        'JNJ', 'PG', 'KO', 'PEP', 'MO', 'PM',
        # Utilities
        'NEE', 'DUK', 'SO', 'D', 'EXC',
        # Financials
        'JPM', 'BAC', 'GS', 'USB', 'WFC',
        # Industrials
        'MMM', 'CAT', 'BA', 'LMT', 'RTX',
        # Healthcare
        'PFE', 'ABBV', 'MRK', 'AMGN', 'CVS',
        # Real Estate
        'VTR', 'STAG', 'O', 'VICI', 'COLD',
        # Energy
        'MPC', 'PSX', 'VLO', 'CVX', 'XOM',
        # Telecom
        'T', 'VZ', 'CMCSA',
        # Consumer Discretionary
        'MCD', 'NKE', 'SBUX', 'TJX',
        # Technology (selective)
        'MSFT', 'IBM', 'CSCO', 'INTC'
    ]
    
    def __init__(self, tickers: List[str] = None):
        """
        Initialize scanner
        
        Args:
            tickers: List of tickers to scan (default: popular dividend stocks)
        """
        self.tickers = tickers or self.POPULAR_DIVIDEND_STOCKS
        self.ir_collector = InterestRateDataCollector()
        self.ir_params = self.ir_collector.calibrate_ir_model()
        self.rates = self.ir_collector.get_current_rates()
    
    def scan_market(self) -> List[StockOpportunity]:
        """
        Scan market and return ranked opportunities by profit probability
        
        Returns:
            List of StockOpportunity sorted by probability of profit
        """
        opportunities = []
        successful_scans = 0
        failed_scans = 0
        
        for ticker in self.tickers:
            try:
                opportunity = self._analyze_stock(ticker)
                if opportunity:
                    opportunities.append(opportunity)
                    successful_scans += 1
            except Exception as e:
                failed_scans += 1
                print(f"  [FAIL] {ticker}: {str(e)[:50]}")
                continue
        
        print(f"\nScanning complete: {successful_scans} stocks analyzed, {failed_scans} failed")
        
        # Sort by probability of profit (descending)
        opportunities.sort(key=lambda x: x.probability_of_profit, reverse=True)
        return opportunities
    
    def _analyze_stock(self, ticker: str) -> StockOpportunity:
        """Analyze single stock and create opportunity"""
        
        # Collect data
        collector = StockDataCollector(ticker)
        stock_params = collector.fetch_stock_data()
        
        # Skip non-dividend stocks
        if stock_params.dividend_per_share <= 0 or stock_params.dividend_yield < 0.01:
            return None
        
        # Generate recommendation
        engine = RecommendationEngine(
            stock_params,
            self.ir_params,
            risk_free_rate=self.rates['risk_free_rate']
        )
        recommendation = engine.generate_recommendation(
            investor_budget=100000,
            investor_risk_tolerance="MODERATE"
        )
        
        # Calculate probability of profit
        prob_profit = self._calculate_profit_probability(
            recommendation.upside_downside_pct,
            recommendation.recommended_holding_period_months,
            stock_params.volatility,
            recommendation.confidence
        )
        
        # Expected annual return
        annual_return = (stock_params.dividend_yield * 100) + (recommendation.upside_downside_pct / (recommendation.recommended_holding_period_months / 12))
        
        # Determine exit signal
        exit_signal = self._determine_exit_signal(
            recommendation.action,
            recommendation.position_sizing.take_profit_price,
            recommendation.position_sizing.stop_loss_price,
            recommendation.recommended_holding_period_months
        )
        
        # Key risk
        key_risk = recommendation.risks[0] if recommendation.risks else "Market volatility"
        
        return StockOpportunity(
            ticker=ticker,
            current_price=stock_params.current_price,
            fair_value=recommendation.fair_value,
            upside_downside_pct=recommendation.upside_downside_pct,
            dividend_yield=stock_params.dividend_yield * 100,
            recommendation=recommendation.action.value,
            confidence=recommendation.confidence * 100,
            probability_of_profit=prob_profit,
            expected_holding_months=recommendation.recommended_holding_period_months,
            expected_return=annual_return,
            exit_signal=exit_signal,
            key_risk=key_risk
        )
    
    def _calculate_profit_probability(self,
                                     upside_downside_pct: float,
                                     months: int,
                                     volatility: float,
                                     confidence: float) -> float:
        """
        Calculate probability of profit using simplified Monte Carlo approach
        
        Uses normal distribution to estimate probability of positive return
        """
        # Base probability from upside potential
        if upside_downside_pct > 0:
            base_prob = min(0.95, 0.5 + (upside_downside_pct / 100) * 0.2)
        else:
            base_prob = max(0.05, 0.5 + (upside_downside_pct / 100) * 0.2)
        
        # Adjust for time - longer holding periods reduce volatility risk
        time_adjustment = min(1.0, 1 - (volatility * 0.1) + (months / 12) * 0.05)
        
        # Adjust for confidence in analysis
        confidence_adjustment = confidence / 100
        
        # Combined probability
        prob = base_prob * time_adjustment * confidence_adjustment
        
        return max(0.01, min(0.99, prob))
    
    def _determine_exit_signal(self,
                             action: ActionType,
                             take_profit: float,
                             stop_loss: float,
                             holding_months: int) -> str:
        """Determine when to sell or if to hold forever"""
        
        if action in [ActionType.STRONG_BUY, ActionType.BUY]:
            if holding_months >= 36:
                return f"Sell at ${take_profit:.2f} OR hold for dividend growth"
            else:
                return f"Sell at ${take_profit:.2f} (target: {holding_months} months)"
        elif action == ActionType.HOLD:
            return f"Hold for dividend income, Sell if drops below ${stop_loss:.2f}"
        else:  # SELL or STRONG_SELL
            return f"Exit at market, Stop loss at ${stop_loss:.2f}"
    
    def get_top_opportunities(self, n: int = 10) -> List[StockOpportunity]:
        """Get top N opportunities"""
        opportunities = self.scan_market()
        return opportunities[:n]
    
    def format_results(self, opportunities: List[StockOpportunity]) -> str:
        """Format results as readable string"""
        if not opportunities:
            return "No opportunities found"
        
        output = "\n" + "="*120 + "\n"
        output += "TOP STOCK OPPORTUNITIES - RANKED BY PROBABILITY OF PROFIT\n"
        output += "="*120 + "\n\n"
        
        for i, opp in enumerate(opportunities, 1):
            output += f"{i}. {opp.ticker}\n"
            output += f"   Current Price: ${opp.current_price:.2f} | Fair Value: ${opp.fair_value:.2f}\n"
            output += f"   Upside/Downside: {opp.upside_downside_pct:.2f}% | Dividend Yield: {opp.dividend_yield:.2f}%\n"
            output += f"   Probability of Profit: {opp.probability_of_profit*100:.1f}%\n"
            output += f"   Expected Annual Return: {opp.expected_return:.2f}%\n"
            output += f"   Recommendation: {opp.recommendation} (Confidence: {opp.confidence:.0f}%)\n"
            output += f"   Holding Period: {opp.expected_holding_months} months\n"
            output += f"   Exit Signal: {opp.exit_signal}\n"
            output += f"   Key Risk: {opp.key_risk}\n"
            output += "\n"
        
        output += "="*120 + "\n"
        return output
