"""
Portfolio Optimizer - Asset allocation and weight suggestions
Balances high-risk tech stocks with safer, negatively-correlated assets
"""
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class AssetAllocation:
    """Asset allocation suggestion"""
    ticker: str
    asset_class: str  # 'tech', 'dividend', 'commodity', 'currency', 'property'
    risk_level: str   # 'high', 'moderate', 'low'
    suggested_weight: float  # 0.0 to 1.0
    rationale: str
    volatility_estimate: float  # Annualized volatility
    correlation_to_tech: float  # Correlation to tech-heavy portfolio


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation"""
    allocations: List[AssetAllocation]
    total_risk_score: float  # 0-100
    expected_volatility: float  # Portfolio volatility
    diversification_score: float  # 0-100
    asset_class_breakdown: Dict[str, float]  # Breakdown by class
    rationale: str


class PortfolioOptimizer:
    """
    Portfolio optimization and weight suggestion engine
    Considers correlations, volatility, and risk profiles
    """
    
    # Asset characteristics (volatility, correlation to tech)
    ASSET_CHARACTERISTICS = {
        # Tech Stocks
        'MSFT': {'asset_class': 'tech', 'volatility': 0.28, 'tech_correlation': 1.0, 'risk_level': 'high'},
        'AAPL': {'asset_class': 'tech', 'volatility': 0.30, 'tech_correlation': 1.0, 'risk_level': 'high'},
        'NVDA': {'asset_class': 'tech', 'volatility': 0.42, 'tech_correlation': 1.0, 'risk_level': 'high'},
        'GOOGL': {'asset_class': 'tech', 'volatility': 0.32, 'tech_correlation': 1.0, 'risk_level': 'high'},
        'META': {'asset_class': 'tech', 'volatility': 0.38, 'tech_correlation': 1.0, 'risk_level': 'high'},
        'AMZN': {'asset_class': 'tech', 'volatility': 0.35, 'tech_correlation': 1.0, 'risk_level': 'high'},
        
        # Dividend Stocks (Lower risk)
        'JNJ': {'asset_class': 'dividend', 'volatility': 0.18, 'tech_correlation': 0.3, 'risk_level': 'low'},
        'PG': {'asset_class': 'dividend', 'volatility': 0.16, 'tech_correlation': 0.25, 'risk_level': 'low'},
        'KO': {'asset_class': 'dividend', 'volatility': 0.20, 'tech_correlation': 0.28, 'risk_level': 'low'},
        'PEP': {'asset_class': 'dividend', 'volatility': 0.19, 'tech_correlation': 0.27, 'risk_level': 'low'},
        'NEE': {'asset_class': 'dividend', 'volatility': 0.22, 'tech_correlation': 0.35, 'risk_level': 'moderate'},
        'JPM': {'asset_class': 'dividend', 'volatility': 0.26, 'tech_correlation': 0.55, 'risk_level': 'moderate'},
        'BAC': {'asset_class': 'dividend', 'volatility': 0.28, 'tech_correlation': 0.60, 'risk_level': 'moderate'},
        'PFE': {'asset_class': 'dividend', 'volatility': 0.24, 'tech_correlation': 0.20, 'risk_level': 'low'},
        'MRK': {'asset_class': 'dividend', 'volatility': 0.22, 'tech_correlation': 0.22, 'risk_level': 'low'},
        
        # Commodities
        'GLD': {'asset_class': 'commodity', 'volatility': 0.14, 'tech_correlation': -0.15, 'risk_level': 'moderate'},  # Gold ETF
        'UUU': {'asset_class': 'commodity', 'volatility': 0.38, 'tech_correlation': -0.05, 'risk_level': 'high'},      # Uranium
        'SLV': {'asset_class': 'commodity', 'volatility': 0.20, 'tech_correlation': -0.10, 'risk_level': 'moderate'},  # Silver ETF
        'DBC': {'asset_class': 'commodity', 'volatility': 0.25, 'tech_correlation': -0.12, 'risk_level': 'moderate'},  # Commodities basket
        
        # Real Estate
        'VNQ': {'asset_class': 'property', 'volatility': 0.22, 'tech_correlation': 0.45, 'risk_level': 'moderate'},    # Real estate ETF
        'SPY': {'asset_class': 'equity_blend', 'volatility': 0.18, 'tech_correlation': 0.70, 'risk_level': 'moderate'}, # S&P 500
        
        # Bonds (Safest)
        'IEF': {'asset_class': 'bonds', 'volatility': 0.08, 'tech_correlation': -0.20, 'risk_level': 'low'},            # US Treasury
        'AGG': {'asset_class': 'bonds', 'volatility': 0.06, 'tech_correlation': -0.15, 'risk_level': 'low'},            # Bond ETF
    }
    
    def __init__(self):
        """Initialize portfolio optimizer"""
        self.characteristics = self.ASSET_CHARACTERISTICS
    
    def _allocate_stocks_optimized(self, recommended_stocks: List[Dict], 
                                    total_weight: float, min_diversification: float) -> List[AssetAllocation]:
        """
        Allocate stocks using Sharpe ratio optimization to maximize risk-adjusted returns
        
        Weights stocks based on:
        1. Expected return (upside potential) - PRIMARY DRIVER
        2. Volatility/Risk (risk-adjusted return via Sharpe ratio)
        3. Minimum diversification to avoid concentration risk
        
        Args:
            recommended_stocks: List of recommended stock dictionaries
            total_weight: Total weight to allocate to stocks (0-1)
            min_diversification: Minimum weight per position (prevents over-concentration)
            
        Returns:
            List of AssetAllocation objects with optimized weights
        """
        allocations = []
        
        if not recommended_stocks:
            return allocations
        
        # Calculate Sharpe ratio for each stock (return / volatility)
        # Assuming risk-free rate of 4% and assuming upside directly correlates to expected return
        risk_free_rate = 0.04
        stock_metrics = []
        
        for stock_dict in recommended_stocks:
            ticker = stock_dict.get('ticker', stock_dict.get('name', ''))
            upside = stock_dict.get('upside_downside_pct', 0) / 100  # Convert to decimal
            volatility = self._get_volatility(ticker)
            
            # Sharpe Ratio = (Expected Return - Risk Free Rate) / Volatility
            # If upside is negative, treat as downside risk
            expected_return = upside + risk_free_rate  # Total expected return
            
            if volatility > 0:
                sharpe_ratio = expected_return / volatility
            else:
                sharpe_ratio = expected_return * 10  # Avoid division by zero
            
            # Only positive expected returns should get significant allocation
            if expected_return <= 0:
                sharpe_ratio = 0
            
            stock_metrics.append({
                'ticker': ticker,
                'upside': upside,
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': max(0, sharpe_ratio),  # No negative Sharpe ratios
                'asset_class': self._get_asset_class(ticker),
                'risk_level': self._get_risk_level(ticker),
                'tech_correlation': self._get_tech_correlation(ticker),
            })
        
        # Normalize Sharpe ratios to weights
        total_sharpe = sum(m['sharpe_ratio'] for m in stock_metrics)
        
        if total_sharpe > 0:
            # Allocate proportionally to Sharpe ratio, respecting minimum diversification
            for metric in stock_metrics:
                # Base weight from Sharpe ratio
                base_weight = (metric['sharpe_ratio'] / total_sharpe) * total_weight
                
                # Apply minimum diversification floor (don't go below min_diversification)
                # and cap to prevent single position from being too large
                min_allocation = min_diversification * total_weight
                max_allocation = 0.35 * total_weight  # Max 35% of stock allocation to single position
                
                final_weight = max(min_allocation, min(base_weight, max_allocation))
                
                rationale = self._get_allocation_rationale(
                    metric['ticker'],
                    metric['asset_class'],
                    metric['tech_correlation'],
                    metric['volatility']
                )
                
                # Add expected return and Sharpe ratio to rationale
                rationale += f" | Expected Return: {metric['expected_return']*100:.1f}% | Sharpe: {metric['sharpe_ratio']:.2f}"
                
                allocations.append(AssetAllocation(
                    ticker=metric['ticker'],
                    asset_class=metric['asset_class'],
                    risk_level=metric['risk_level'],
                    suggested_weight=final_weight,
                    rationale=rationale,
                    volatility_estimate=metric['volatility'],
                    correlation_to_tech=metric['tech_correlation']
                ))
        else:
            # If no positive Sharpe ratios, allocate equally
            equal_weight = total_weight / len(recommended_stocks)
            for metric in stock_metrics:
                allocations.append(AssetAllocation(
                    ticker=metric['ticker'],
                    asset_class=metric['asset_class'],
                    risk_level=metric['risk_level'],
                    suggested_weight=equal_weight,
                    rationale=f"Equal weight allocation (limited positive expected returns)",
                    volatility_estimate=metric['volatility'],
                    correlation_to_tech=metric['tech_correlation']
                ))
        
        return allocations
    
    def suggest_allocation(self, 
                          recommended_stocks: List[Dict],
                          budget: float = 100000,
                          risk_tolerance: str = 'MODERATE') -> PortfolioAllocation:
        """
        Suggest portfolio allocation based on recommended stocks
        Optimizes for maximum risk-adjusted returns using Sharpe ratio weighting
        
        Args:
            recommended_stocks: List of recommended stock dicts with ticker, upside, etc.
            budget: Total investment budget
            risk_tolerance: 'CONSERVATIVE', 'MODERATE', or 'AGGRESSIVE'
            
        Returns:
            PortfolioAllocation with weight suggestions maximized for profit
        """
        allocations = []
        
        # Extract tickers from recommendations
        recommended_tickers = [s.get('ticker', s.get('name', '')) for s in recommended_stocks]
        
        # Separate tech and non-tech stocks
        tech_stocks = [t for t in recommended_tickers if self._get_asset_class(t) == 'tech']
        non_tech_stocks = [t for t in recommended_tickers if self._get_asset_class(t) != 'tech']
        
        # Determine risk-based allocation (more aggressive for higher returns)
        if risk_tolerance == 'CONSERVATIVE':
            stocks_weight = 0.45
            defensive_weight = 0.35
            alternatives_weight = 0.20  # Increased gold/hedge allocation
            min_diversification = 0.05  # Min 5% per position
        elif risk_tolerance == 'AGGRESSIVE':
            stocks_weight = 0.70
            defensive_weight = 0.10
            alternatives_weight = 0.20  # Increased gold/hedge allocation
            min_diversification = 0.02  # Min 2% per position
        else:  # MODERATE (optimized for good returns with reasonable risk)
            stocks_weight = 0.60
            defensive_weight = 0.15
            alternatives_weight = 0.25  # Increased gold/hedge allocation
            min_diversification = 0.03  # Min 3% per position
        
        # Allocate stocks using Sharpe ratio optimization for maximum risk-adjusted returns
        stock_allocations = self._allocate_stocks_optimized(
            recommended_stocks, stocks_weight, min_diversification
        )
        allocations.extend(stock_allocations)
        
        # Add defensive stocks (low correlation to tech)
        if defensive_weight > 0:
            defensive_candidates = [
                ('JNJ', 'Pharmaceutical - Low correlation to tech'),
                ('PG', 'Consumer Staples - Recession resistant'),
                ('PFE', 'Healthcare - Defensive'),
                ('KO', 'Beverages - Consumer staple'),
            ]
            
            for ticker, desc in defensive_candidates:
                if ticker not in recommended_tickers:
                    weight = defensive_weight / len(defensive_candidates)
                    allocations.append(AssetAllocation(
                        ticker=ticker,
                        asset_class='dividend',
                        risk_level='low',
                        suggested_weight=weight,
                        rationale=f"{desc} - Balances tech portfolio risk",
                        volatility_estimate=self._get_volatility(ticker),
                        correlation_to_tech=self._get_tech_correlation(ticker)
                    ))
        
        # Add alternative assets (negative correlation hedge)
        if alternatives_weight > 0:
            # Gold gets higher weight as inflation hedge and USD crisis insurance
            # Allocate: GLD 50%, IEF (Treasuries) 40%, SLV (Silver) 10%
            alternative_allocations = [
                ('GLD', 'Gold ETF - Negative correlation hedge, inflation protection, USD stability hedge', 0.50),
                ('IEF', 'US Treasury Bonds - Safe haven, negative correlation', 0.40),
                ('SLV', 'Silver - Inflation hedge, low correlation', 0.10),
            ]
            
            for ticker, desc, ratio in alternative_allocations:
                weight = alternatives_weight * ratio
                allocations.append(AssetAllocation(
                    ticker=ticker,
                    asset_class=self._get_asset_class(ticker),
                    risk_level=self._get_risk_level(ticker),
                    suggested_weight=weight,
                    rationale=desc,
                    volatility_estimate=self._get_volatility(ticker),
                    correlation_to_tech=self._get_tech_correlation(ticker)
                ))
        
        # Normalize weights to sum to 1.0
        total_weight = sum(a.suggested_weight for a in allocations)
        if total_weight > 0:
            allocations = [
                AssetAllocation(
                    ticker=a.ticker,
                    asset_class=a.asset_class,
                    risk_level=a.risk_level,
                    suggested_weight=a.suggested_weight / total_weight,
                    rationale=a.rationale,
                    volatility_estimate=a.volatility_estimate,
                    correlation_to_tech=a.correlation_to_tech
                )
                for a in allocations
            ]
        
        # Calculate portfolio metrics
        portfolio_volatility = self._calculate_portfolio_volatility(allocations)
        risk_score = self._calculate_risk_score(allocations)
        diversification_score = self._calculate_diversification_score(allocations)
        
        # Create asset class breakdown
        asset_class_breakdown = {}
        for allocation in allocations:
            asset_class_breakdown[allocation.asset_class] = \
                asset_class_breakdown.get(allocation.asset_class, 0) + allocation.suggested_weight
        
        rationale = self._get_portfolio_rationale(allocations, risk_tolerance, portfolio_volatility)
        
        return PortfolioAllocation(
            allocations=allocations,
            total_risk_score=risk_score,
            expected_volatility=portfolio_volatility,
            diversification_score=diversification_score,
            asset_class_breakdown=asset_class_breakdown,
            rationale=rationale
        )
    
    def _get_asset_class(self, ticker: str) -> str:
        """Get asset class for ticker"""
        if ticker in self.characteristics:
            return self.characteristics[ticker].get('asset_class', 'unknown')
        return 'unknown'
    
    def _get_volatility(self, ticker: str) -> float:
        """Get volatility for ticker"""
        if ticker in self.characteristics:
            return self.characteristics[ticker].get('volatility', 0.25)
        # Default: moderate volatility
        return 0.25
    
    def _get_tech_correlation(self, ticker: str) -> float:
        """Get correlation to tech for ticker"""
        if ticker in self.characteristics:
            return self.characteristics[ticker].get('tech_correlation', 0.5)
        return 0.5
    
    def _get_risk_level(self, ticker: str) -> str:
        """Get risk level for ticker"""
        if ticker in self.characteristics:
            return self.characteristics[ticker].get('risk_level', 'moderate')
        return 'moderate'
    
    def _get_allocation_rationale(self, ticker: str, asset_class: str, 
                                   tech_corr: float, volatility: float) -> str:
        """Get rationale for allocation"""
        if tech_corr < 0:
            return f"{ticker} ({asset_class}) - Negative correlation provides downside protection"
        elif tech_corr < 0.4:
            return f"{ticker} ({asset_class}) - Low correlation diversifies portfolio"
        elif asset_class == 'tech':
            return f"{ticker} ({asset_class}) - Growth exposure with high upside potential"
        else:
            return f"{ticker} ({asset_class}) - Balanced risk-return profile"
    
    def _calculate_portfolio_volatility(self, allocations: List[AssetAllocation]) -> float:
        """Calculate portfolio volatility (simplified)"""
        # Simplified: weighted average of volatilities adjusted for correlations
        weighted_vol = sum(a.suggested_weight * a.volatility_estimate for a in allocations)
        
        # Reduce by diversification benefit (rough approximation)
        num_assets = len(allocations)
        diversification_benefit = 1.0 - (min(num_assets - 1, 5) * 0.05)
        
        return weighted_vol * diversification_benefit
    
    def _calculate_risk_score(self, allocations: List[AssetAllocation]) -> float:
        """Calculate overall portfolio risk score (0-100)"""
        risk_scores = {'high': 75, 'moderate': 50, 'low': 25}
        
        weighted_risk = sum(
            a.suggested_weight * risk_scores.get(a.risk_level, 50)
            for a in allocations
        )
        
        return min(100, max(0, weighted_risk))
    
    def _calculate_diversification_score(self, allocations: List[AssetAllocation]) -> float:
        """Calculate diversification score (0-100)"""
        # Based on number of asset classes and weight distribution
        asset_classes = set(a.asset_class for a in allocations)
        num_classes = len(asset_classes)
        
        # Check Herfindahl concentration
        herfindahl = sum(a.suggested_weight ** 2 for a in allocations)
        
        # Score: more classes and lower concentration = higher score
        class_score = min(100, num_classes * 20)  # Max 100 at 5+ classes
        concentration_score = (1 - min(herfindahl, 1)) * 100
        
        return (class_score + concentration_score) / 2
    
    def _get_portfolio_rationale(self, allocations: List[AssetAllocation],
                                  risk_tolerance: str, volatility: float) -> str:
        """Get overall portfolio rationale"""
        asset_classes = set(a.asset_class for a in allocations)
        
        rationale = f"Diversified portfolio ({len(asset_classes)} asset classes) "
        rationale += f"with {risk_tolerance.lower()} risk profile. "
        
        # Identify correlation benefits
        negative_corr = [a for a in allocations if a.correlation_to_tech < 0]
        if negative_corr:
            rationale += f"{len(negative_corr)} assets provide downside protection. "
        
        rationale += f"Expected portfolio volatility: {volatility*100:.1f}% annually."
        
        return rationale
    
    def format_allocation_for_json(self, allocation: PortfolioAllocation) -> Dict:
        """Format allocation for JSON response"""
        return {
            'allocations': [
                {
                    'ticker': a.ticker,
                    'asset_class': a.asset_class,
                    'weight': round(a.suggested_weight * 100, 2),
                    'weight_pct': round(a.suggested_weight * 100, 2),
                    'risk_level': a.risk_level,
                    'rationale': a.rationale,
                    'volatility': round(a.volatility_estimate * 100, 2),
                    'correlation_to_tech': round(a.correlation_to_tech, 2),
                }
                for a in allocation.allocations
            ],
            'portfolio_metrics': {
                'total_risk_score': round(allocation.total_risk_score, 1),
                'expected_volatility_pct': round(allocation.expected_volatility * 100, 2),
                'diversification_score': round(allocation.diversification_score, 1),
                'asset_class_breakdown': {
                    k: round(v * 100, 2) for k, v in allocation.asset_class_breakdown.items()
                }
            },
            'portfolio_rationale': allocation.rationale
        }
