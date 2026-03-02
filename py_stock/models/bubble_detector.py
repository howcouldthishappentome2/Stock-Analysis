"""
Bubble Detection Engine - Identifies market bubbles using historical valuation trends
Uses long-term historical data to detect when markets/stocks are overvalued relative to fundamentals
"""
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
from .stock_params import StockParams


@dataclass
class BubbleAlert:
    """Alert indicating bubble risk"""
    type: str  # "valuation", "momentum", "sector", "macro"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    confidence: float  # 0-1


@dataclass
class BubbleMetrics:
    """Calculated bubble metrics for a stock or market"""
    bubble_risk_score: float  # 0-100 (higher = more bubble risk)
    valuation_percentile: float  # 0-100 (where stock stands historically)
    is_overvalued: bool
    alerts: List[BubbleAlert]


class BubbleDetector:
    """
    Detects bubbles by comparing current valuations to historical trends
    
    Historical Context:
    - Modern data (1926+): S&P 500 averages, Fama-French data
    - Extended history (1871+): Shiller CAPE ratio data shows:
      * Normal range: PE 15-25
      * Bubble territory: PE > 30
      * Crash risk: PE > 35
    - Sector bubbles: Tech 1999-2000 (PE 100+), Nifty Fifty (1970s, PE 40+), Dot-com crash
    
    Key Historical Lessons:
    - 1929: Banking/stocks bubble (PE ~30) → Great Depression
    - 1973: Nifty Fifty bubble (PE 40+) → Lost 60%+ (1973-1982)
    - 1987: S&P 500 crashed 20% in one day (PE inversion)
    - 1999-2000: Tech bubble (Nasdaq PE ~200) → Lost 78% at trough
    - 2008: Housing/finance bubble → Lost 57% (2007-2009)
    - 2021-2022: Fed inflation created valuation bubble
    - 2024-2025: AI bubble (Nvidia circular accounting, massive capex vs revenue)
    """
    
    # Historical valuation benchmarks (derived from Shiller CAPE and modern data)
    HISTORICAL_PE_STATISTICS = {
        'mean': 18.0,          # Long-term average PE (1871-present)
        'median': 17.5,        # Median PE historically
        'std_dev': 4.5,        # Standard deviation
        'safety_range': (14.0, 25.0),  # Normal range (mean ± 1.5 std)
        'warning_threshold': 28.0,      # Yellow flag (>2 std above mean)
        'bubble_threshold': 35.0,       # Red flag (>4 std above mean)
        'crash_threshold': 45.0,        # Critical (>6 std above mean)
    }
    
    # Sector-specific bubble indicators (AI/Tech especially prone)
    SECTOR_PE_THRESHOLDS = {
        'technology': {'normal': 25, 'warning': 35, 'bubble': 50, 'critical': 70},
        'growth': {'normal': 30, 'warning': 45, 'bubble': 65, 'critical': 100},
        'dividend': {'normal': 18, 'warning': 23, 'bubble': 30, 'critical': 40},
        'ai': {'normal': 35, 'warning': 50, 'bubble': 75, 'critical': 150},  # AI bubble thresholds
        'ai_adjacent': {'normal': 28, 'warning': 40, 'bubble': 55, 'critical': 85},  # Companies with major AI subsidiaries/investments (MSFT/OpenAI, GOOGL/DeepMind, AMZN/Anthropic, META)
    }
    
    # PEG ratio thresholds (growth-adjusted)
    PEG_THRESHOLDS = {
        'undervalued': (0.0, 0.5),
        'fair': (0.5, 1.5),
        'overvalued': (1.5, 3.0),
        'bubble': (3.0, 10.0),
        'extreme_bubble': (10.0, float('inf'))
    }
    
    # Price-to-Sales historical benchmarks
    PSR_THRESHOLDS = {
        'undervalued': (0.0, 0.5),
        'fair': (0.5, 2.0),
        'overvalued': (2.0, 5.0),
        'bubble': (5.0, 10.0),
        'extreme': (10.0, float('inf'))
    }
    
    def __init__(self, stock: StockParams, sector: str = 'technology'):
        """
        Initialize bubble detector
        
        Args:
            stock: Stock parameters
            sector: Stock sector for sector-specific bubble thresholds
                   Options: 'technology', 'growth', 'dividend', 'ai'
        """
        self.stock = stock
        self.sector = sector.lower()
        
    def detect_bubble(self) -> BubbleMetrics:
        """
        Comprehensive bubble detection using multiple metrics
        
        Returns:
            BubbleMetrics with risk score and alerts
        """
        alerts = []
        risk_scores = []
        
        # Check PE-based bubble indicators
        pe_score, pe_alerts = self._check_pe_bubble()
        alerts.extend(pe_alerts)
        risk_scores.append(pe_score)
        
        # Check PEG-based bubble indicators
        peg_score, peg_alerts = self._check_peg_bubble()
        alerts.extend(peg_alerts)
        risk_scores.append(peg_score)
        
        # Check Price-to-Sales bubble indicators
        ps_score, ps_alerts = self._check_ps_bubble()
        alerts.extend(ps_alerts)
        risk_scores.append(ps_score)
        
        # Check revenue/earnings quality
        quality_score, quality_alerts = self._check_fundamentals_quality()
        alerts.extend(quality_alerts)
        risk_scores.append(quality_score)
        
        # Check for circular accounting (Nvidia/Google pattern)
        accounting_score, accounting_alerts = self._check_circular_accounting()
        alerts.extend(accounting_alerts)
        risk_scores.append(accounting_score)
        
        # Check AI subsidiary/investment exposure (contagion risk from AI bubble)
        ai_sub_score, ai_sub_alerts = self._check_ai_subsidiary_exposure()
        alerts.extend(ai_sub_alerts)
        risk_scores.append(ai_sub_score)
        
        # Check momentum/technical bubble signals
        momentum_score, momentum_alerts = self._check_momentum_bubble()
        alerts.extend(momentum_alerts)
        risk_scores.append(momentum_score)
        
        # Calculate overall bubble risk (average of component scores)
        bubble_risk_score = np.mean([s for s in risk_scores if s is not None])
        
        # Calculate valuation percentile (where stock sits historically)
        valuation_percentile = self._calculate_percentile()
        
        # Determine if overvalued
        is_overvalued = bubble_risk_score > 50
        
        return BubbleMetrics(
            bubble_risk_score=min(100, max(0, bubble_risk_score)),
            valuation_percentile=valuation_percentile,
            is_overvalued=is_overvalued,
            alerts=alerts
        )
    
    def _check_pe_bubble(self) -> Tuple[float, List[BubbleAlert]]:
        """Check PE ratio against historical benchmarks"""
        alerts = []
        risk_score = 0.0
        
        if not self.stock.pe_ratio or self.stock.pe_ratio <= 0:
            return 0.0, alerts
        
        pe = self.stock.pe_ratio
        thresholds = self.SECTOR_PE_THRESHOLDS.get(self.sector, self.SECTOR_PE_THRESHOLDS['technology'])
        historical = self.HISTORICAL_PE_STATISTICS
        
        # Calculate how many standard deviations above mean
        z_score = (pe - historical['mean']) / historical['std_dev']
        
        # Check for extreme valuations
        if pe > thresholds['critical']:
            alerts.append(BubbleAlert(
                type='valuation',
                severity='critical',
                description=f'PE ratio {pe:.1f} is CRITICAL - exceeds historical crash threshold ({thresholds["critical"]:.0f}). '
                           f'For reference: Nasdaq PE in 2000 crash was ~200, Nifty Fifty peak was ~45+',
                confidence=0.95
            ))
            risk_score = 95.0
        elif pe > thresholds['bubble']:
            alerts.append(BubbleAlert(
                type='valuation',
                severity='high',
                description=f'PE ratio {pe:.1f} is in BUBBLE TERRITORY ({thresholds["bubble"]:.0f}+). '
                           f'Historical reference: Tech bubble 1999 peaked at PE 100+',
                confidence=0.90
            ))
            risk_score = 80.0
        elif pe > thresholds['warning']:
            alerts.append(BubbleAlert(
                type='valuation',
                severity='medium',
                description=f'PE ratio {pe:.1f} is elevated, {z_score:.1f} standard deviations above historical mean ({historical["mean"]:.1f}). '
                           f'Warning level for {self.sector} sector is {thresholds["warning"]:.0f}',
                confidence=0.85
            ))
            risk_score = 60.0
        elif pe > thresholds['normal']:
            alerts.append(BubbleAlert(
                type='valuation',
                severity='low',
                description=f'PE ratio {pe:.1f} is above sector average ({thresholds["normal"]:.0f})',
                confidence=0.70
            ))
            risk_score = 40.0
        else:
            risk_score = max(0, (pe - historical['mean']) / (historical['std_dev'] * 2) * 50)
        
        return risk_score, alerts
    
    def _check_peg_bubble(self) -> Tuple[float, List[BubbleAlert]]:
        """Check PEG ratio for bubble indicators"""
        alerts = []
        risk_score = 0.0
        
        if not self.stock.pe_ratio or self.stock.pe_ratio <= 0 or not self.stock.growth_rate or self.stock.growth_rate <= 0:
            return 0.0, alerts
        
        peg_ratio = self.stock.pe_ratio / (self.stock.growth_rate * 100)
        
        if peg_ratio > 10.0:
            alerts.append(BubbleAlert(
                type='valuation',
                severity='critical',
                description=f'PEG ratio {peg_ratio:.2f} indicates growth expectations are severely detached from reality. '
                           f'PE is 10x the growth rate - company would need {self.stock.growth_rate*1000:.0f}% annual growth to justify valuation',
                confidence=0.95
            ))
            risk_score = 90.0
        elif peg_ratio > 3.0:
            alerts.append(BubbleAlert(
                type='valuation',
                severity='high',
                description=f'PEG ratio {peg_ratio:.2f} is in BUBBLE zone. Market is pricing in unrealistic sustained growth.',
                confidence=0.90
            ))
            risk_score = 75.0
        elif peg_ratio > 1.5:
            alerts.append(BubbleAlert(
                type='valuation',
                severity='medium',
                description=f'PEG ratio {peg_ratio:.2f} suggests growth is already priced in.',
                confidence=0.80
            ))
            risk_score = 55.0
        
        return risk_score, alerts
    
    def _check_ps_bubble(self) -> Tuple[float, List[BubbleAlert]]:
        """Check Price-to-Sales ratio for bubble patterns"""
        alerts = []
        risk_score = 0.0
        
        if not hasattr(self.stock, 'price_to_sales') or not self.stock.price_to_sales:
            return 0.0, alerts
        
        ps = self.stock.price_to_sales
        thresholds = self.PSR_THRESHOLDS
        
        if ps > 10.0:
            alerts.append(BubbleAlert(
                type='valuation',
                severity='critical',
                description=f'Price-to-Sales ratio {ps:.1f} is extreme. Company is valued at {ps:.0f}x revenue. '
                           f'Dot-com bubble peaks saw PSR >10 for many companies that lost 80%+',
                confidence=0.95
            ))
            risk_score = 90.0
        elif ps > 5.0:
            alerts.append(BubbleAlert(
                type='valuation',
                severity='high',
                description=f'Price-to-Sales ratio {ps:.1f} is in bubble zone. Selling {ps:.0f}x annual revenue.',
                confidence=0.90
            ))
            risk_score = 75.0
        elif ps > 2.0:
            alerts.append(BubbleAlert(
                type='valuation',
                severity='medium',
                description=f'Price-to-Sales ratio {ps:.1f} is elevated. Historical median is 1.5-2.0',
                confidence=0.80
            ))
            risk_score = 50.0
        
        return risk_score, alerts
    
    def _check_fundamentals_quality(self) -> Tuple[float, List[BubbleAlert]]:
        """Check if earnings/revenue growth justifies valuation - AI bubble indicator"""
        alerts = []
        risk_score = 0.0
        
        # Major bubble warning: High PE but low/negative earnings growth
        if self.stock.pe_ratio and self.stock.pe_ratio > 30:
            if not self.stock.growth_rate or self.stock.growth_rate < 0.1:
                alerts.append(BubbleAlert(
                    type='fundamentals',
                    severity='critical',
                    description=f'MAJOR RED FLAG: High PE ratio ({self.stock.pe_ratio:.1f}) but weak growth ({self.stock.growth_rate*100:.1f}% or below). '
                               f'This is the AI bubble pattern - paying for earnings that don\'t materialize. '
                               f'Similar to 2024 Nvidia circular accounting where CAPEX spending inflates valuations.',
                    confidence=0.95
                ))
                risk_score = 85.0
            elif self.stock.growth_rate < 0.2:
                alerts.append(BubbleAlert(
                    type='fundamentals',
                    severity='high',
                    description=f'Elevated PE ({self.stock.pe_ratio:.1f}) with below-average growth ({self.stock.growth_rate*100:.1f}%). '
                               f'Growth not justifying valuation.',
                    confidence=0.90
                ))
                risk_score = 65.0
        
        # Check profitability quality
        if hasattr(self.stock, 'profit_margin') and self.stock.profit_margin:
            if self.stock.profit_margin < 0.05:  # <5% margin
                if self.stock.pe_ratio and self.stock.pe_ratio > 20:
                    alerts.append(BubbleAlert(
                        type='fundamentals',
                        severity='high',
                        description=f'Company has low profit margins ({self.stock.profit_margin*100:.1f}%) but high PE ({self.stock.pe_ratio:.1f}). '
                                   f'Unlikely to sustain current valuation. Bubble indicator.',
                        confidence=0.85
                    ))
                    risk_score = 70.0
        
        return risk_score, alerts
    
    def _check_circular_accounting(self) -> Tuple[float, List[BubbleAlert]]:
        """
        Check for circular accounting patterns (Nvidia/Google pattern)
        When large capex spending inflates valuations through circular reinvestment
        """
        alerts = []
        risk_score = 0.0
        
        # Check if company has high capex but low cash flow from operations
        if hasattr(self.stock, 'free_cash_flow') and self.stock.free_cash_flow is not None:
            # If FCF is very low or negative, capex-heavy spending is destroying value
            if self.stock.free_cash_flow < 0:
                if hasattr(self.stock, 'earnings_per_share') and self.stock.earnings_per_share and self.stock.earnings_per_share > 0:
                    alerts.append(BubbleAlert(
                        type='accounting',
                        severity='critical',
                        description=f'ACCOUNTING ALERT: Negative free cash flow despite positive earnings. '
                                   f'Company is spending more than it generates - classic bubble pattern. '
                                   f'Similar to Nvidia 2024-2025 massive capex spending inflating stock price while burning cash.',
                        confidence=0.95
                    ))
                    risk_score = 85.0
        
        # Check for high debt from capex spending
        if self.stock.debt_to_equity and self.stock.debt_to_equity > 3.0:
            if hasattr(self.stock, 'free_cash_flow') and self.stock.free_cash_flow and self.stock.free_cash_flow < 0:
                alerts.append(BubbleAlert(
                    type='accounting',
                    severity='high',
                    description=f'CIRCULAR ACCOUNTING RED FLAG: Debt-to-equity {self.stock.debt_to_equity:.1f} with negative FCF. '
                               f'Company is financing capex through debt while destroying cash value.',
                    confidence=0.90
                ))
                risk_score = 75.0
        
        return risk_score, alerts
    
    def _check_ai_subsidiary_exposure(self) -> Tuple[float, List[BubbleAlert]]:
        """
        Check if this company has significant AI subsidiary or investment exposure
        that creates indirect bubble risk (e.g., Microsoft/OpenAI, Google/DeepMind,
        Amazon/Anthropic, Meta/FAIR).
        
        These companies benefit from AI hype but are also exposed to AI bubble contagion:
        - Valuation premiums driven by AI subsidiary value (hard to verify)
        - Massive capex commitments to AI infrastructure
        - Revenue projections increasingly tied to unproven AI monetization
        """
        alerts = []
        risk_score = 0.0

        # Map of tickers to their AI exposure details
        AI_ADJACENT_COMPANIES = {
            'MSFT': {
                'name': 'Microsoft',
                'ai_entity': 'OpenAI (~$13B+ investment, ~49% stake)',
                'risk': 'Azure AI revenue growth assumptions and OpenAI valuation embedded in stock price.',
                'capex_risk': True,
            },
            'GOOGL': {
                'name': 'Alphabet/Google',
                'ai_entity': 'Google DeepMind + Gemini AI',
                'risk': 'Core search revenue threatened by AI; massive spend on TPU/AI infrastructure.',
                'capex_risk': True,
            },
            'GOOG': {
                'name': 'Alphabet/Google',
                'ai_entity': 'Google DeepMind + Gemini AI',
                'risk': 'Core search revenue threatened by AI; massive spend on TPU/AI infrastructure.',
                'capex_risk': True,
            },
            'AMZN': {
                'name': 'Amazon',
                'ai_entity': 'Anthropic (~$4B investment) + AWS AI services',
                'risk': 'AWS AI growth priced in; Anthropic stake is speculative and illiquid.',
                'capex_risk': True,
            },
            'META': {
                'name': 'Meta',
                'ai_entity': 'FAIR (Meta AI Research) + Llama models',
                'risk': 'Metaverse pivot failed; AI capex ramping sharply with unclear monetization timeline.',
                'capex_risk': True,
            },
            'ORCL': {
                'name': 'Oracle',
                'ai_entity': 'AI cloud infrastructure partner (OpenAI, Cohere)',
                'risk': 'Stock re-rated as AI play; cloud AI revenue growth expectations may be overstated.',
                'capex_risk': False,
            },
            'CRM': {
                'name': 'Salesforce',
                'ai_entity': 'Einstein AI / Agentforce',
                'risk': 'AI feature monetization unproven at scale; premium valuation requires AI uptake.',
                'capex_risk': False,
            },
        }

        ticker = getattr(self.stock, 'ticker', '').upper()
        exposure = AI_ADJACENT_COMPANIES.get(ticker)

        if exposure is None:
            return 0.0, alerts

        # Base risk for having significant AI exposure
        base_risk = 30.0
        description_parts = [
            f'AI SUBSIDIARY EXPOSURE: {exposure["name"]} has significant AI investment/subsidiary risk via {exposure["ai_entity"]}. '
            f'{exposure["risk"]}'
        ]

        # Escalate risk if PE is already elevated (AI premium stacked on top)
        pe = self.stock.pe_ratio or 0
        if pe > 35:
            base_risk += 20.0
            description_parts.append(
                f'Current PE ({pe:.1f}) suggests AI subsidiary value is already priced in as a premium. '
                f'If AI monetization disappoints, multiple compression is likely.'
            )

        # Escalate if high capex spending is confirmed
        if exposure['capex_risk']:
            fcf = getattr(self.stock, 'free_cash_flow', None)
            if fcf is not None and fcf < 0:
                base_risk += 20.0
                description_parts.append(
                    'Negative FCF confirms heavy AI infrastructure capex spend. '
                    'Value destruction risk if AI ROI does not materialize within 2-3 years.'
                )
            else:
                base_risk += 10.0
                description_parts.append(
                    'Company is known to be ramping AI capex significantly. '
                    'Watch free cash flow for signs of value erosion.'
                )

        # Determine severity
        if base_risk >= 65:
            severity = 'high'
        elif base_risk >= 45:
            severity = 'medium'
        else:
            severity = 'low'

        alerts.append(BubbleAlert(
            type='ai_subsidiary_exposure',
            severity=severity,
            description=' | '.join(description_parts),
            confidence=0.80
        ))
        risk_score = base_risk

        return risk_score, alerts

    def _check_momentum_bubble(self) -> Tuple[float, List[BubbleAlert]]:
        """Check for parabolic price rises and momentum bubble patterns"""
        alerts = []
        risk_score = 0.0
        
        # Check YTD return (momentum indicator)
        if hasattr(self.stock, 'ytd_return') and self.stock.ytd_return:
            if self.stock.ytd_return > 0.5:  # Up 50%+ in last year
                # Check if justified by growth
                if self.stock.growth_rate and self.stock.growth_rate < 0.25:
                    alerts.append(BubbleAlert(
                        type='momentum',
                        severity='high',
                        description=f'Stock up {self.stock.ytd_return*100:.0f}% YTD but growth rate only {self.stock.growth_rate*100:.1f}%. '
                                   f'Parabolic rise likely driven by momentum/speculation rather than fundamentals.',
                        confidence=0.85
                    ))
                    risk_score = 65.0
            elif self.stock.ytd_return > 0.3:  # Up 30%+
                alerts.append(BubbleAlert(
                    type='momentum',
                    severity='medium',
                    description=f'Stock up {self.stock.ytd_return*100:.0f}% YTD - momentum-driven rally. Watch for reversal.',
                    confidence=0.75
                ))
                risk_score = 50.0
        
        return risk_score, alerts
    
    def _calculate_percentile(self) -> float:
        """
        Calculate where this stock sits relative to historical valuation distribution
        
        Returns:
            0-100 percentile (100 = most overvalued historically)
        """
        if not self.stock.pe_ratio or self.stock.pe_ratio <= 0:
            return 50.0
        
        historical = self.HISTORICAL_PE_STATISTICS
        # Use z-score to calculate percentile
        z_score = (self.stock.pe_ratio - historical['mean']) / historical['std_dev']
        
        # Convert z-score to percentile using normal distribution approximation
        # z-score of 0 = 50th percentile, z-score of 2 = 97.7th percentile
        percentile = 50 + (z_score * 15.87)  # Rough approximation
        
        return max(0, min(100, percentile))
    
    def get_bubble_risk_level(self, risk_score: float) -> str:
        """Get human-readable bubble risk level"""
        if risk_score >= 75:
            return 'CRITICAL_BUBBLE'
        elif risk_score >= 60:
            return 'HIGH_BUBBLE_RISK'
        elif risk_score >= 45:
            return 'MODERATE_BUBBLE_RISK'
        elif risk_score >= 30:
            return 'LOW_BUBBLE_RISK'
        else:
            return 'SAFE'


class MarketBubbleIndex:
    """
    Tracks overall market bubble risk across multiple stocks
    Uses historical market-wide bubble periods to predict crashes
    """
    
    # Historical bubble periods and their characteristics
    HISTORICAL_BUBBLES = {
        'great_depression_1929': {
            'period': '1926-1929',
            'peak_pe': 32.0,
            'crash': -89,  # -89%
            'duration_months': 45,
            'cause': 'Unbridled stock speculation, margin abuse'
        },
        'nifty_fifty_1970s': {
            'period': '1970-1975',
            'peak_pe': 42.0,
            'crash': -62,
            'duration_months': 60,
            'cause': 'Growth stock bubble (like today\'s AI), momentum buying'
        },
        'tech_bubble_1999_2000': {
            'period': '1995-2000',
            'peak_pe': 200.0,  # Nasdaq PE
            'crash': -78,
            'duration_months': 60,
            'cause': 'Internet/dot-com bubble, no earnings requirement'
        },
        'financial_crisis_2007_2008': {
            'period': '2003-2008',
            'peak_pe': 25.0,
            'crash': -57,
            'duration_months': 36,
            'cause': 'Housing bubble, leveraged subprimes'
        },
        'current_ai_bubble_2024_2025': {
            'period': '2020-2025',
            'peak_pe': 35.0,  # S&P 500, Nvidia even higher at ~60
            'crash': -40,  # Potential
            'duration_months': None,  # Still ongoing
            'cause': 'AI hype, circular capex spending (Nvidia), low earnings growth'
        }
    }
    
    def __init__(self, stock_list: List[StockParams]):
        """Initialize market bubble index with stock list"""
        self.stocks = stock_list
    
    def calculate_market_bubble_index(self) -> Tuple[float, str, List[str]]:
        """
        Calculate overall market bubble risk (0-100)
        
        Returns:
            (bubble_score, risk_level, key_warnings)
        """
        bubble_scores = []
        warnings = []
        
        # Analyze each stock
        for stock in self.stocks:
            detector = BubbleDetector(stock, sector=self._infer_sector(stock.ticker))
            metrics = detector.detect_bubble()
            bubble_scores.append(metrics.bubble_risk_score)
            
            # Collect critical alerts
            for alert in metrics.alerts:
                if alert.severity == 'critical':
                    warnings.append(f"{stock.ticker}: {alert.description}")
        
        # Calculate average bubble risk
        avg_bubble_risk = np.mean(bubble_scores) if bubble_scores else 50.0
        
        # Check for systemic bubble patterns
        overvalued_count = sum(1 for s in bubble_scores if s > 60)
        systemic_bubble = overvalued_count / len(self.stocks) > 0.5 if self.stocks else False
        
        if systemic_bubble and avg_bubble_risk > 65:
            risk_level = 'MARKET_WIDE_BUBBLE'
            warnings.append('WARNING: Multiple stocks showing bubble signals - market-wide risk')
        elif avg_bubble_risk >= 60:
            risk_level = 'HIGH_BUBBLE_RISK'
        elif avg_bubble_risk >= 45:
            risk_level = 'MODERATE_BUBBLE_RISK'
        else:
            risk_level = 'SAFE'
        
        return avg_bubble_risk, risk_level, warnings
    
    def _infer_sector(self, ticker: str) -> str:
        """Infer sector from ticker"""
        ai_stocks = ['NVIDIA', 'NVDA', 'TSLA', 'PLTR', 'SMCI']
        # Companies with major AI subsidiaries/investments: use stricter 'ai_adjacent' sector
        ai_adjacent_stocks = ['MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'ORCL', 'CRM']
        if ticker.upper() in ai_stocks:
            return 'ai'
        elif ticker.upper() in ai_adjacent_stocks:
            return 'ai_adjacent'
        elif ticker.upper() in ['AAPL']:
            return 'growth'
        else:
            return 'technology'