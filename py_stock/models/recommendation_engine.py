"""
Stock Recommendation Engine - Generate buy/sell signals, position sizing, and holding periods
"""
from dataclasses import dataclass
from typing import Tuple, List, Dict
from enum import Enum
import numpy as np
from models.stock_params import StockParams, InterestRateParams
from models.valuation_engine import StockValuationEngine
from models.dividend_model import DividendGrowthModel
from models.interest_rate_model import InterestRateAnalyzer


class ActionType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class PositionSizing:
    """Position sizing recommendation"""
    initial_investment: float
    position_size_pct: float  # % of portfolio
    number_of_shares: float
    recommended_buy_price: float
    stop_loss_price: float
    take_profit_price: float


@dataclass
class TradingRecommendation:
    """Complete trading recommendation"""
    ticker: str
    action: ActionType
    confidence: float  # 0-1
    fair_value: float
    target_price_upside: float
    current_price: float
    upside_downside_pct: float
    recommended_holding_period_months: int
    position_sizing: PositionSizing
    key_catalysts: List[str]
    risks: List[str]
    rationale: str
    alpha: float = 0.0  # CAPM Jensen's alpha (annualized)
    news_sentiment: str = "neutral"  # bullish/neutral/bearish from AI analysis
    news_sentiment_score: float = 0.0  # -1.0 to 1.0
    news_keywords: List[str] = None  # Key topics from recent news
    news_validation_status: str = "not_analyzed"  # not_analyzed, validated, moderate, low


class RecommendationEngine:
    """Generate investment recommendations"""
    
    def __init__(self, 
                 stock: StockParams,
                 interest_params: InterestRateParams,
                 risk_free_rate: float = 0.04,
                 stock_type: str = 'dividend'):
        """
        Initialize recommendation engine
        
        Args:
            stock: Stock parameters
            interest_params: Interest rate model parameters
            risk_free_rate: Current risk-free rate
            stock_type: Type of stock - 'dividend' or 'tech'. Tech stocks skip dividend & Fama-French models
        """
        self.stock = stock
        self.interest_params = interest_params
        self.risk_free_rate = risk_free_rate
        self.equity_risk_premium = 0.05
        self.stock_type = stock_type
        
        # Initialize sub-models
        self.valuation = StockValuationEngine(stock, risk_free_rate)
        self.dividend_model = DividendGrowthModel(stock, risk_free_rate)
        self.rate_analyzer = InterestRateAnalyzer(interest_params)
        
    def generate_recommendation(self, investor_budget: float = 100000,
                              investor_risk_tolerance: str = "MODERATE",
                              use_holistic_engine: bool = True) -> TradingRecommendation:
        """
        Generate complete trading recommendation.

        When use_holistic_engine=True (default), delegates to HolisticSignalEngine
        which addresses three critical issues with the pure DDM approach:
          1. Real-time price  — uses yfinance fast_info.last_price instead of
             stale 1-2 day-old closing prices
          2. Ex-dividend awareness — detects price drops caused by dividend payments
             and prevents false SELL signals around ex-dividend dates
          3. Multi-factor signal aggregation — combines valuation (30%), news AI
             sentiment (25%), price momentum (20%), dividend health (15%), and
             quality/balance sheet (10%) for a holistic recommendation that is
             resistant to single-factor noise

        Args:
            investor_budget: Total portfolio size
            investor_risk_tolerance: "CONSERVATIVE", "MODERATE", "AGGRESSIVE"
            use_holistic_engine: If True, use holistic engine (default: True).
                                 Set False to use base DDM-only engine.

        Returns:
            TradingRecommendation object
        """
        if use_holistic_engine:
            try:
                return self._generate_holistic_recommendation(
                    investor_budget, investor_risk_tolerance
                )
            except Exception as e:
                print(f"  [HolisticEngine] Failed ({e}), falling back to base engine")
                # fall through to base engine below

        # ── BASE ENGINE (fallback) ────────────────────────────────────────────
        # Get valuations
        try:
            fair_value, val_assessment = self.valuation.calculate_fair_value()
            fair_value, val_assessment = self.valuation.calculate_fair_value()
        except:
            # Fallback for stocks with missing data
            fair_value = self.stock.current_price * 1.1
            val_assessment = "Estimated"
            
        try:
            valuation_metrics = self.valuation.analyze_valuation_ratios()
        except:
            # Create a minimal valuation metrics object for problematic stocks
            from dataclasses import dataclass
            @dataclass
            class MinimalMetrics:
                ticker: str
                current_price: float
                fair_value: float
                upside_downside: float
                pe_ratio: float
                historical_pe: float
                pb_ratio: float
                payout_ratio: float
                dividend_coverage: str
                overall_rating: str
            
            valuation_metrics = MinimalMetrics(
                ticker=self.stock.ticker,
                current_price=self.stock.current_price,
                fair_value=fair_value,
                upside_downside=(fair_value - self.stock.current_price) / self.stock.current_price * 100,
                pe_ratio=self.stock.pe_ratio if self.stock.pe_ratio > 0 else 0,
                historical_pe=18,
                pb_ratio=0,
                payout_ratio=0,
                dividend_coverage="N/A",
                overall_rating="HOLD"
            )
        
        try:
            quality_score, quality = self.valuation.quality_score()
        except:
            quality_score = 50
            quality = "Estimated"
        
        # Get dividend analysis using Monte Carlo path-averaged discounting
        # Skip for tech stocks (dividend models don't work for growth stocks)
        if self.stock_type == 'dividend':
            dividend_projection = self.dividend_model.monte_carlo_discounted_ddm(
                rate_analyzer=self.rate_analyzer,
                high_growth_years=5,
                stable_growth_rate=0.03,
                num_paths=500,
                num_steps_per_year=1
            )
            payout_ratio, div_safety = self.dividend_model.dividend_safety_check()
            
            # Get interest rate outlook
            rate_scenarios = self.rate_analyzer.rate_scenario_analysis()
            rate_impact = self.rate_analyzer.impact_on_dividend_yield(self.stock.dividend_yield)
        else:
            # For tech stocks, use simple growth projection
            # dividend_projection will use fair_value from valuation engine instead of dividend DDM
            from dataclasses import dataclass
            @dataclass
            class SimplePriceProjection:
                upside_downside: float
            
            # Use valuation metrics for price projection (no dividend dependency)
            upside_value = (valuation_metrics.upside_downside / 100) if valuation_metrics else 0.1
            dividend_projection = SimplePriceProjection(
                upside_downside=upside_value
            )
            payout_ratio = 0.0
            div_safety = "N/A"
            rate_scenarios = {'current_rate': 0.04, 'base_case': 0.04}
            rate_impact = {'rate_change': 0.0}  # For tech stocks, no dividend impact
        
        # Calculate action signal
        action, confidence = self._calculate_action_signal(
            valuation_metrics, 
            quality_score,
            rate_scenarios,
            investor_risk_tolerance
        )
        
        # Position sizing
        position = self._calculate_position_sizing(
            fair_value, 
            action, 
            investor_budget,
            investor_risk_tolerance
        )
        
        # Holding period
        holding_period = self._calculate_holding_period(
            valuation_metrics.upside_downside,
            self.stock.growth_rate,
            quality_score
        )
        
        # Key catalysts and risks
        catalysts = self._identify_catalysts()
        risks = self._identify_risks(rate_impact, payout_ratio)
        
        # Generate rationale
        rationale = self._generate_rationale(
            action,
            valuation_metrics,
            dividend_projection,
            rate_impact,
            quality
        )

        # Estimate CAPM expected return and alpha
        market_vol_proxy = 0.15
        beta = self.stock.beta if getattr(self.stock, 'beta', None) is not None else (
            (self.stock.volatility / market_vol_proxy) if market_vol_proxy > 0 else 1.0
        )

        capm_expected_return = self.risk_free_rate + beta * self.equity_risk_premium

        # Annualize valuation-based expected return: dividend yield + annualized capital gain to fair value
        years = max(1, holding_period / 12)
        capital_gain = 0.0
        if self.stock.current_price > 0:
            capital_gain = (fair_value - self.stock.current_price) / self.stock.current_price
        annualized_capital_gain = capital_gain / years
        valuation_annual_return = self.stock.dividend_yield + annualized_capital_gain

        alpha = valuation_annual_return - capm_expected_return
        
        return TradingRecommendation(
            ticker=self.stock.ticker,
            action=action,
            confidence=confidence,
            fair_value=fair_value,
            target_price_upside=dividend_projection.upside_downside * 100,
            current_price=self.stock.current_price,
            upside_downside_pct=valuation_metrics.upside_downside,
            recommended_holding_period_months=holding_period,
            position_sizing=position,
            key_catalysts=catalysts,
            risks=risks,
            rationale=rationale
            ,
            alpha=alpha
        )
    
    def _calculate_action_signal(self, 
                                 valuation_metrics,
                                 quality_score: float,
                                 rate_scenarios: dict,
                                 risk_tolerance: str) -> Tuple[ActionType, float]:
        """Calculate buy/sell action and confidence"""
        
        # Base signal from valuation
        upside = valuation_metrics.upside_downside
        
        if upside > 25:
            base_action = ActionType.STRONG_BUY
            base_confidence = min(0.95, 0.7 + upside / 100)
        elif upside > 10:
            base_action = ActionType.BUY
            base_confidence = 0.7 + upside / 50
        elif upside > -10:
            base_action = ActionType.HOLD
            base_confidence = 0.6
        elif upside > -25:
            base_action = ActionType.SELL
            base_confidence = 0.7
        else:
            base_action = ActionType.STRONG_SELL
            base_confidence = min(0.95, 0.7 + abs(upside) / 100)
        
        # Adjust for quality
        quality_adjustment = quality_score / 100
        
        # Adjust for interest rate outlook
        rate_change = rate_scenarios.get('base_case', rate_scenarios['current_rate']) - rate_scenarios['current_rate']
        if rate_change > 0.01:  # Rates rising
            rate_adjustment = 0.9  # Reduce confidence for dividend stocks
        elif rate_change < -0.01:  # Rates falling
            rate_adjustment = 1.1  # Increase confidence
        else:
            rate_adjustment = 1.0
        
        # Adjust for risk tolerance
        if risk_tolerance == "CONSERVATIVE":
            if quality_score < 70:
                base_action = ActionType.HOLD  # More conservative
                base_confidence *= 0.9
        elif risk_tolerance == "AGGRESSIVE":
            if quality_score > 60:
                if base_action == ActionType.BUY:
                    base_action = ActionType.STRONG_BUY
                    base_confidence = min(0.95, base_confidence * 1.1)
        
        # Final confidence
        final_confidence = min(0.99, base_confidence * quality_adjustment * rate_adjustment)
        
        return base_action, max(0.5, final_confidence)
    
    def _calculate_position_sizing(self,
                                  fair_value: float,
                                  action: ActionType,
                                  investor_budget: float,
                                  risk_tolerance: str) -> PositionSizing:
        """Calculate position sizing based on action and risk tolerance"""
        
        current_price = self.stock.current_price
        
        # Position size as % of portfolio
        if action == ActionType.STRONG_BUY:
            base_pct = 0.05  # 5%
        elif action == ActionType.BUY:
            base_pct = 0.03  # 3%
        elif action == ActionType.HOLD:
            base_pct = 0.01  # 1%
        else:  # SELL or STRONG_SELL
            base_pct = 0.00
        
        # Adjust for risk tolerance
        if risk_tolerance == "CONSERVATIVE":
            base_pct *= 0.7
        elif risk_tolerance == "AGGRESSIVE":
            base_pct *= 1.3
        
        # Cap position size
        position_size_pct = min(0.10, base_pct)  # Max 10% in single stock
        initial_investment = investor_budget * position_size_pct
        number_of_shares = initial_investment / current_price
        
        # Price targets
        recommended_buy_price = fair_value * 0.95  # 5% margin of safety
        take_profit_price = fair_value * 1.15  # 15% target upside
        stop_loss_price = fair_value * 0.85  # 15% stop loss
        
        return PositionSizing(
            initial_investment=initial_investment,
            position_size_pct=position_size_pct * 100,
            number_of_shares=number_of_shares,
            recommended_buy_price=recommended_buy_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
    
    def _calculate_holding_period(self, 
                                 upside_downside: float,
                                 growth_rate: float,
                                 quality_score: float) -> int:
        """Calculate recommended holding period in months"""
        
        # Base holding period
        if upside_downside > 25:
            base_holding = 24  # 2 years
        elif upside_downside > 10:
            base_holding = 18  # 1.5 years
        elif upside_downside > -10:
            base_holding = 12  # 1 year
        else:
            base_holding = 6  # Exit sooner
        
        # Adjust for growth rate
        if growth_rate > 0.10:
            base_holding = min(36, base_holding + 12)
        elif growth_rate > 0.05:
            base_holding = base_holding
        else:
            base_holding = max(6, base_holding - 6)
        
        # Adjust for quality
        if quality_score >= 80:
            base_holding = min(36, base_holding + 6)
        elif quality_score < 60:
            base_holding = max(6, base_holding - 6)
        
        return base_holding
    
    def _identify_catalysts(self) -> List[str]:
        """Identify key catalysts for stock movement"""
        catalysts = []
        
        if self.stock.dividend_yield > 0.04:
            catalysts.append("High dividend yield attractive in current rate environment")
        
        if self.stock.growth_rate > 0.08:
            catalysts.append("Strong dividend growth trajectory")
        
        if self.stock.payout_ratio < 0.50:
            catalysts.append("Significant room for dividend increases")
        
        if self.stock.pe_ratio < 15:
            catalysts.append("Trading at discount to historical average")
        
        if self.stock.debt_to_equity < 0.5:
            catalysts.append("Strong balance sheet with low leverage")
        
        return catalysts if catalysts else ["Stable dividend income", "Defensive characteristics"]
    
    def _identify_risks(self, rate_impact: dict, payout_ratio: float) -> List[str]:
        """Identify key risks"""
        risks = []
        
        if rate_impact['rate_change'] > 0.01:
            risks.append("Rising interest rates could reduce attractiveness")
        
        if payout_ratio > 0.75:
            risks.append("High payout ratio limits dividend growth")
        
        if self.stock.debt_to_equity > 1.5:
            risks.append("High leverage increases financial risk")
        
        if self.stock.growth_rate < 0.03:
            risks.append("Limited growth prospects")
        
        if self.stock.pe_ratio > 25:
            risks.append("Premium valuation leaves little room for error")
        
        return risks if risks else ["General market volatility"]
    
    def _generate_rationale(self,
                           action: ActionType,
                           valuation_metrics,
                           dividend_projection,
                           rate_impact: dict,
                           quality: str) -> str:
        """Generate investment rationale"""
        
        upside = dividend_projection.upside_downside
        
        if action == ActionType.STRONG_BUY:
            rationale = f"{self.stock.ticker} is significantly undervalued with {upside:.1%} potential upside. "
            rationale += f"The stock has a quality rating of {quality} and offers an attractive dividend yield "
            rationale += f"with sustainable growth prospects. "
        elif action == ActionType.BUY:
            rationale = f"{self.stock.ticker} offers attractive value with {upside:.1%} upside potential. "
            rationale += f"The dividend is well-covered and positioned to grow. "
        elif action == ActionType.HOLD:
            rationale = f"{self.stock.ticker} is fairly valued at current levels. "
            rationale += f"The stock can be held for its dividend income. "
        elif action == ActionType.SELL:
            rationale = f"{self.stock.ticker} appears overvalued with limited upside. "
            rationale += f"Consider taking profits or holding for income only. "
        else:  # STRONG_SELL
            rationale = f"{self.stock.ticker} is significantly overvalued. "
            rationale += f"Risk/reward is unfavorable at current levels. "
        
        if abs(rate_impact['rate_change']) > 0.01:
            direction = "rising" if rate_impact['rate_change'] > 0 else "falling"
            rationale += f"Interest rates are expected to trend {direction}, which may impact valuation. "
        
        return rationale

    def _generate_holistic_recommendation(self,
                                           investor_budget: float,
                                           investor_risk_tolerance: str) -> 'TradingRecommendation':
        """
        Delegate to HolisticSignalEngine and convert its HolisticResult back into
        a TradingRecommendation for full API compatibility.
        """
        from models.holistic_signal_engine import HolisticSignalEngine

        engine = HolisticSignalEngine(
            ticker=self.stock.ticker,
            stock_params=self.stock,
            ir_params=self.interest_params,
            risk_free_rate=self.risk_free_rate,
            stock_type=self.stock_type,
            use_news_ai=True,
        )

        result = engine.analyze(
            investor_budget=investor_budget,
            investor_risk_tolerance=investor_risk_tolerance,
        )

        # Update stock's current_price with real-time price if available
        if result.using_realtime and result.realtime_price > 0:
            self.stock.current_price = result.realtime_price

        # Map action string to ActionType enum
        action = ActionType[result.action]

        # Position sizing using existing method
        position = self._calculate_position_sizing(
            result.fair_value,
            action,
            investor_budget,
            investor_risk_tolerance,
        )

        # Holding period
        holding_period = self._calculate_holding_period(
            result.upside_downside_pct,
            self.stock.growth_rate,
            50,  # default quality score; holistic engine already weighted it
        )

        # CAPM alpha
        beta = getattr(self.stock, 'beta', None) or 1.0
        capm_expected_return = self.risk_free_rate + beta * self.equity_risk_premium
        years = max(1, holding_period / 12)
        capital_gain = (result.fair_value - self.stock.current_price) / max(1, self.stock.current_price)
        annualized_cg = capital_gain / years
        valuation_return = self.stock.dividend_yield + annualized_cg
        alpha = valuation_return - capm_expected_return

        # Build rationale suffix with ex-div + data freshness notes
        rationale = result.rationale
        if result.using_realtime:
            rationale += f" [Price: real-time ${result.realtime_price:.2f}]"
        if result.near_ex_dividend:
            rationale += (
                f" [Ex-dividend ${result.ex_dividend_amount:.3f} on {result.ex_dividend_date}"
                f"{' — drop explained by dividend payment' if result.price_drop_explained_by_dividend else ''}]"
            )

        return TradingRecommendation(
            ticker=self.stock.ticker,
            action=action,
            confidence=result.confidence,
            fair_value=result.fair_value,
            target_price_upside=result.upside_downside_pct,
            current_price=self.stock.current_price,
            upside_downside_pct=result.upside_downside_pct,
            recommended_holding_period_months=holding_period,
            position_sizing=position,
            key_catalysts=result.catalysts,
            risks=result.risks,
            rationale=rationale,
            alpha=alpha,
            news_sentiment=result.news_sentiment,
            news_sentiment_score=result.news_sentiment_score,
            news_keywords=result.news_keywords or [],
            news_validation_status=result.news_validation_status,
        )