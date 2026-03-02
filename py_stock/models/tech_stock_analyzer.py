"""
Tech Stock Analyzer - Security checks for growth stocks
Detects pump-and-dumps, scams, unsustainable growth, peak prices, and bubble indicators
"""
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from models.stock_params import StockParams
from models.recommendation_engine import RecommendationEngine, ActionType
from models.bubble_detector import BubbleDetector


@dataclass
class SecurityAlert:
    """Security concern detected"""
    risk_type: str  # 'peak_price', 'unsustainable_growth', 'long_growth_period', 'pump_dump', 'scam_signal'
    severity: str   # 'low', 'medium', 'high', 'critical'
    description: str
    confidence: float  # 0-1


class TechStockSecurityAnalyzer:
    """Analyzes tech stocks for fraud, manipulation, and unsustainable valuations"""
    
    def __init__(self, stock_params: StockParams):
        self.stock = stock_params
        self.alerts: List[SecurityAlert] = []
    
    def analyze(self) -> Tuple[bool, List[SecurityAlert]]:
        """
        Analyze tech stock for security concerns
        
        Returns:
            (is_safe, alerts) - True if stock passes security checks, list of alerts
        """
        self.alerts = []
        
        # Run all security checks
        self._check_peak_price()
        self._check_unsustainable_growth()
        self._check_long_growth_period()
        self._check_pump_and_dump_signals()
        self._check_scam_indicators()
        
        # Check for bubble indicators
        self._check_bubble_risk()
        
        # If any critical alerts, stock is unsafe
        critical_alerts = [a for a in self.alerts if a.severity == 'critical']
        is_safe = len(critical_alerts) == 0
        
        return is_safe, self.alerts
    
    def _check_peak_price(self) -> None:
        """
        Detect if stock is at or near its peak price
        Check if current price is near 52-week high
        """
        if not hasattr(self.stock, 'week_52_high') or self.stock.week_52_high is None:
            return
        
        if self.stock.week_52_high <= 0:
            return
        
        # Calculate how close to peak
        distance_to_peak = (self.stock.week_52_high - self.stock.current_price) / self.stock.week_52_high
        
        # If within 5% of 52-week high, it's potentially at peak
        if distance_to_peak < 0.05:
            self.alerts.append(SecurityAlert(
                risk_type='peak_price',
                severity='high',
                description=f'Stock at {100-distance_to_peak*100:.1f}% of 52-week high (${self.stock.current_price:.2f} vs ${self.stock.week_52_high:.2f})',
                confidence=0.9
            ))
        # If within 10% of peak - medium concern
        elif distance_to_peak < 0.10:
            self.alerts.append(SecurityAlert(
                risk_type='peak_price',
                severity='medium',
                description=f'Stock near 52-week high ({100-distance_to_peak*100:.1f}% of peak)',
                confidence=0.8
            ))
    
    def _check_unsustainable_growth(self) -> None:
        """
        Detect if growth rate is unsustainable
        High growth + high valuation = red flag
        """
        if not self.stock.growth_rate or self.stock.growth_rate <= 0:
            return
        
        # Get P/E ratio for valuation check
        pe_ratio = self.stock.pe_ratio if self.stock.pe_ratio and self.stock.pe_ratio > 0 else None
        
        if not pe_ratio:
            return
        
        # PEG ratio: P/E to growth ratio (lower is better, >2 is expensive)
        peg_ratio = pe_ratio / (self.stock.growth_rate * 100) if self.stock.growth_rate > 0 else None
        
        if peg_ratio and peg_ratio > 3.0:
            self.alerts.append(SecurityAlert(
                risk_type='unsustainable_growth',
                severity='high',
                description=f'PEG ratio {peg_ratio:.2f} indicates growth may be unsustainable (P/E: {pe_ratio:.1f}, Growth: {self.stock.growth_rate*100:.1f}%)',
                confidence=0.85
            ))
        elif peg_ratio and peg_ratio > 2.0:
            self.alerts.append(SecurityAlert(
                risk_type='unsustainable_growth',
                severity='medium',
                description=f'PEG ratio {peg_ratio:.2f} suggests growth may be overpriced',
                confidence=0.70
            ))
    
    def _check_long_growth_period(self) -> None:
        """
        Detect if stock has been in sustained growth for too long
        Potential sign of hedge fund accumulation before coordinated short
        """
        if not hasattr(self.stock, 'eps_history') or not self.stock.eps_history:
            return
        
        # Check if EPS has been consistently rising
        if len(self.stock.eps_history) < 8:  # Need at least 2 years of quarterly data
            return
        
        # Analyze growth consistency over last 8 quarters (2 years)
        recent_eps = self.stock.eps_history[-8:]
        
        if len(recent_eps) < 8:
            return
        
        # Count consecutive quarters of growth
        consecutive_growth = 0
        max_consecutive = 0
        
        for i in range(1, len(recent_eps)):
            if recent_eps[i] and recent_eps[i-1] and recent_eps[i] > recent_eps[i-1]:
                consecutive_growth += 1
                max_consecutive = max(max_consecutive, consecutive_growth)
            else:
                consecutive_growth = 0
        
        # If 6+ consecutive quarters of growth (1.5+ years), potential red flag
        if max_consecutive >= 6:
            self.alerts.append(SecurityAlert(
                risk_type='long_growth_period',
                severity='medium',
                description=f'{max_consecutive} consecutive quarters of EPS growth - potential accumulation phase before planned short',
                confidence=0.65
            ))
        # If 8 consecutive quarters (full 2 years), high concern
        elif max_consecutive >= 8:
            self.alerts.append(SecurityAlert(
                risk_type='long_growth_period',
                severity='high',
                description=f'2+ years of consistent EPS growth - potential hedge fund accumulation before coordinated short',
                confidence=0.75
            ))
    
    def _check_pump_and_dump_signals(self) -> None:
        """
        Detect pump-and-dump signals:
        - Unusual volume spikes
        - High volatility with uptrends
        - Price momentum reversal patterns
        """
        if not self.stock.volatility or self.stock.volatility <= 0:
            return
        
        # High volatility + high recent gains = potential pump
        if hasattr(self.stock, 'ytd_return') and self.stock.ytd_return:
            ytd_return = self.stock.ytd_return
            
            # High volatility (>50%) + strong gains (>30% YTD) = pump signal
            if self.stock.volatility > 0.50 and ytd_return > 0.30:
                self.alerts.append(SecurityAlert(
                    risk_type='pump_dump',
                    severity='high',
                    description=f'High volatility ({self.stock.volatility*100:.1f}%) with strong YTD gains ({ytd_return*100:.1f}%) - potential pump signal',
                    confidence=0.75
                ))
            # Moderate volatility + very strong gains
            elif self.stock.volatility > 0.40 and ytd_return > 0.50:
                self.alerts.append(SecurityAlert(
                    risk_type='pump_dump',
                    severity='medium',
                    description=f'Strong gains ({ytd_return*100:.1f}% YTD) with elevated volatility - verify fundamentals',
                    confidence=0.65
                ))
    
    def _check_scam_indicators(self) -> None:
        """
        Detect scam/fraud indicators:
        - Negative earnings with high valuation
        - Extreme debt levels
        - Delisting risk
        """
        # Check for unprofitable company with high P/E
        if hasattr(self.stock, 'earnings_per_share') and self.stock.earnings_per_share and self.stock.earnings_per_share < 0:
            if self.stock.pe_ratio and self.stock.pe_ratio > 0:
                self.alerts.append(SecurityAlert(
                    risk_type='scam_signal',
                    severity='critical',
                    description='Negative earnings but positive P/E ratio - accounting irregularities or penny stock pump',
                    confidence=0.90
                ))
        
        # Check debt levels if available
        if hasattr(self.stock, 'debt_to_equity') and self.stock.debt_to_equity:
            if self.stock.debt_to_equity > 5.0:
                self.alerts.append(SecurityAlert(
                    risk_type='scam_signal',
                    severity='high',
                    description=f'Debt-to-equity ratio {self.stock.debt_to_equity:.1f} - extreme leverage risk',
                    confidence=0.75
                ))
            elif self.stock.debt_to_equity > 3.0:
                self.alerts.append(SecurityAlert(
                    risk_type='scam_signal',
                    severity='medium',
                    description=f'High debt-to-equity ratio {self.stock.debt_to_equity:.1f}',
                    confidence=0.65
                ))
        
        # Check for extremely low float (susceptible to manipulation)
        if hasattr(self.stock, 'shares_outstanding') and self.stock.shares_outstanding:
            if self.stock.shares_outstanding < 10_000_000:  # Less than 10M shares
                self.alerts.append(SecurityAlert(
                    risk_type='scam_signal',
                    severity='high',
                    description=f'Very low share count ({self.stock.shares_outstanding/1_000_000:.1f}M) - susceptible to manipulation',
                    confidence=0.70
                ))
    
    def get_safety_score(self) -> float:
        """
        Returns safety score 0-100
        100 = completely safe
        0 = critically unsafe
        """
        if not self.alerts:
            return 100.0
        
        score = 100.0
        
        for alert in self.alerts:
            penalty = 0
            if alert.severity == 'low':
                penalty = 5 * alert.confidence
            elif alert.severity == 'medium':
                penalty = 15 * alert.confidence
            elif alert.severity == 'high':
                penalty = 30 * alert.confidence
            elif alert.severity == 'critical':
                penalty = 50 * alert.confidence
            
            score -= penalty
        
        return max(0, score)
    
    def _check_bubble_risk(self) -> None:
        """
        Check for bubble indicators using historical valuation trends
        Includes AI bubble detection (Nvidia circular accounting pattern)
        """
        # Use BubbleDetector to analyze valuation
        bubble_detector = BubbleDetector(self.stock, sector='ai' if 'nvidia' in self.stock.ticker.lower() or 'nvda' in self.stock.ticker.lower() else 'technology')
        bubble_metrics = bubble_detector.detect_bubble()
        
        # Convert bubble alerts to security alerts
        for bubble_alert in bubble_metrics.alerts:
            # Map bubble alert severity to security alert severity
            severity_map = {
                'critical': 'critical',
                'high': 'high',
                'medium': 'medium',
                'low': 'low'
            }
            
            self.alerts.append(SecurityAlert(
                risk_type='bubble_risk:' + bubble_alert.type,
                severity=severity_map.get(bubble_alert.severity, 'medium'),
                description=bubble_alert.description,
                confidence=bubble_alert.confidence
            ))
