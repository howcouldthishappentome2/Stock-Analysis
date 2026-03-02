"""
Optimized Stock Market Scanner - Using parallel processing and caching
"""
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from dataclasses import dataclass
from models.recommendation_engine import RecommendationEngine, ActionType
from models.stock_params import InterestRateParams, StockParams
from models.tech_stock_analyzer import TechStockSecurityAnalyzer
from data.stock_data import StockDataCollector, InterestRateDataCollector
from data.cache import StockDataCache

# Semaphore for yfinance API calls (allow 8 concurrent requests with rate limiting)
_yfinance_semaphore = threading.Semaphore(8)
_api_call_time = time.time()
_api_call_lock = threading.Lock()


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
    probability_of_profit: float
    expected_holding_months: int
    expected_return: float
    exit_signal: str
    key_risk: str
    beta: float = None
    beta_r2: float = None
    safety_score: float = None  # 0-100 for tech stocks; None for dividend stocks
    security_alerts: List[dict] = None  # List of alert dicts for tech stocks; None for dividend stocks
    _from_cache: bool = False  # Flag to indicate if result came from fallback demo data
    bubble_risk_score: float = None  # populated when bubble detection runs
    bubble_alerts: List[dict] = None


# Fallback demo data - used when yfinance times out or is unavailable
DEMO_STOCK_DATA = {
    # Dividend stocks with realistic current yields
    'JNJ': {
        'current_price': 158.42,
        'dividend_yield': 0.0289,  # ~2.89%
        'dividend_per_share': 4.60,
        'beta': 0.67,
        'beta_r2': 0.88,
        'growth_rate': 0.04,
        'volatility': 0.16,
        'pe_ratio': 26.5,
        'eps': 5.97,
        'payout_ratio': 0.77,
        'debt_to_equity': 0.82
    },
    'PG': {
        'current_price': 166.35,
        'dividend_yield': 0.0245,  # ~2.45%
        'dividend_per_share': 4.07,
        'beta': 0.62,
        'beta_r2': 0.85,
        'growth_rate': 0.02,
        'volatility': 0.15,
        'pe_ratio': 29.2,
        'eps': 5.71,
        'payout_ratio': 0.70,
        'debt_to_equity': 0.58
    },
    'KO': {
        'current_price': 63.45,
        'dividend_yield': 0.0305,  # ~3.05%
        'dividend_per_share': 1.94,
        'beta': 0.89,
        'beta_r2': 0.82,
        'growth_rate': 0.01,
        'volatility': 0.18,
        'pe_ratio': 25.8,
        'eps': 2.46,
        'payout_ratio': 0.79,
        'debt_to_equity': 1.21
    },
    'PEP': {
        'current_price': 190.28,
        'dividend_yield': 0.0266,  # ~2.66%
        'dividend_per_share': 5.08,
        'beta': 0.84,
        'beta_r2': 0.81,
        'growth_rate': 0.05,
        'volatility': 0.17,
        'pe_ratio': 28.9,
        'eps': 6.59,
        'payout_ratio': 0.77,
        'debt_to_equity': 0.95
    },
    'NEE': {
        'current_price': 45.62,
        'dividend_yield': 0.0344,  # ~3.44%
        'dividend_per_share': 1.57,
        'beta': 1.02,
        'beta_r2': 0.79,
        'growth_rate': 0.06,
        'volatility': 0.22,
        'pe_ratio': 22.4,
        'eps': 2.04,
        'payout_ratio': 0.77,
        'debt_to_equity': 1.58
    },
    'DUK': {
        'current_price': 87.35,
        'dividend_yield': 0.0392,  # ~3.92%
        'dividend_per_share': 3.43,
        'beta': 0.94,
        'beta_r2': 0.84,
        'growth_rate': 0.02,
        'volatility': 0.20,
        'pe_ratio': 26.1,
        'eps': 3.34,
        'payout_ratio': 0.83,
        'debt_to_equity': 1.45
    },
    'JPM': {
        'current_price': 197.85,
        'dividend_yield': 0.0255,  # ~2.55%
        'dividend_per_share': 5.05,
        'beta': 1.28,
        'beta_r2': 0.86,
        'growth_rate': 0.08,
        'volatility': 0.24,
        'pe_ratio': 13.2,
        'eps': 15.02,
        'payout_ratio': 0.32,
        'debt_to_equity': 1.18
    },
    'BAC': {
        'current_price': 33.95,
        'dividend_yield': 0.0375,  # ~3.75%
        'dividend_per_share': 1.27,
        'beta': 1.35,
        'beta_r2': 0.88,
        'growth_rate': 0.06,
        'volatility': 0.26,
        'pe_ratio': 11.5,
        'eps': 2.95,
        'payout_ratio': 0.30,
        'debt_to_equity': 0.98
    },
    'PFE': {
        'current_price': 26.48,
        'dividend_yield': 0.0619,  # ~6.19%
        'dividend_per_share': 1.64,
        'beta': 0.81,
        'beta_r2': 0.75,
        'growth_rate': -0.05,
        'volatility': 0.28,
        'pe_ratio': 12.8,
        'eps': 2.07,
        'payout_ratio': 0.79,
        'debt_to_equity': 0.67
    },
    'MRK': {
        'current_price': 97.62,
        'dividend_yield': 0.0338,  # ~3.38%
        'dividend_per_share': 3.30,
        'beta': 0.75,
        'beta_r2': 0.80,
        'growth_rate': 0.03,
        'volatility': 0.19,
        'pe_ratio': 24.1,
        'eps': 4.05,
        'payout_ratio': 0.81,
        'debt_to_equity': 0.43
    },
    # Tech growth stocks with realistic yields
    'MSFT': {
        'current_price': 417.35,
        'dividend_yield': 0.0084,  # ~0.84%
        'dividend_per_share': 3.51,
        'beta': 0.93,
        'beta_r2': 0.85,
        'growth_rate': 0.12,
        'volatility': 0.26,
        'pe_ratio': 33.8,
        'eps': 12.34,
        'payout_ratio': 0.27,
        'debt_to_equity': 0.11
    },
    'AAPL': {
        'current_price': 189.95,
        'dividend_yield': 0.0046,  # ~0.46%
        'dividend_per_share': 0.88,
        'beta': 1.18,
        'beta_r2': 0.87,
        'growth_rate': 0.08,
        'volatility': 0.28,
        'pe_ratio': 28.5,
        'eps': 6.68,
        'payout_ratio': 0.13,
        'debt_to_equity': 0.12
    },
    'GOOGL': {
        'current_price': 139.25,
        'dividend_yield': 0.0,  # No dividend
        'dividend_per_share': 0.0,
        'beta': 1.02,
        'beta_r2': 0.82,
        'growth_rate': 0.15,
        'volatility': 0.30,
        'pe_ratio': 22.5,
        'eps': 6.18,
        'payout_ratio': 0.0,
        'debt_to_equity': 0.08
    },
    'NVDA': {
        'current_price': 135.71,
        'dividend_yield': 0.0,  # No dividend
        'dividend_per_share': 0.0,
        'beta': 1.95,
        'beta_r2': 0.76,
        'growth_rate': 0.45,
        'volatility': 0.48,
        'pe_ratio': 65.2,
        'eps': 2.08,
        'payout_ratio': 0.0,
        'debt_to_equity': 0.08
    },
    'AMZN': {
        'current_price': 198.35,
        'dividend_yield': 0.0,  # No dividend
        'dividend_per_share': 0.0,
        'beta': 1.18,
        'beta_r2': 0.80,
        'growth_rate': 0.18,
        'volatility': 0.32,
        'pe_ratio': 42.3,
        'eps': 4.69,
        'payout_ratio': 0.0,
        'debt_to_equity': 0.35
    },
    'META': {
        'current_price': 453.88,
        'dividend_yield': 0.0,  # No dividend
        'dividend_per_share': 0.0,
        'beta': 1.23,
        'beta_r2': 0.77,
        'growth_rate': 0.25,
        'volatility': 0.35,
        'pe_ratio': 35.1,
        'eps': 12.95,
        'payout_ratio': 0.0,
        'debt_to_equity': 0.05
    },
    'CRM': {
        'current_price': 257.42,
        'dividend_yield': 0.0,  # No dividend
        'dividend_per_share': 0.0,
        'beta': 1.28,
        'beta_r2': 0.71,
        'growth_rate': 0.11,
        'volatility': 0.33,
        'pe_ratio': 180.3,
        'eps': 1.43,
        'payout_ratio': 0.0,
        'debt_to_equity': 0.21
    },
    'ADBE': {
        'current_price': 589.24,
        'dividend_yield': 0.0,  # No dividend
        'dividend_per_share': 0.0,
        'beta': 1.12,
        'beta_r2': 0.73,
        'growth_rate': 0.13,
        'volatility': 0.30,
        'pe_ratio': 65.8,
        'eps': 8.94,
        'payout_ratio': 0.0,
        'debt_to_equity': 0.04
    },
    'INTC': {
        'current_price': 42.53,
        'dividend_yield': 0.0452,  # ~4.52%
        'dividend_per_share': 1.92,
        'beta': 0.95,
        'beta_r2': 0.74,
        'growth_rate': -0.10,
        'volatility': 0.32,
        'pe_ratio': 18.6,
        'eps': 2.28,
        'payout_ratio': 0.84,
        'debt_to_equity': 0.33
    },
    'IBM': {
        'current_price': 197.64,
        'dividend_yield': 0.0398,  # ~3.98%
        'dividend_per_share': 7.87,
        'beta': 0.88,
        'beta_r2': 0.75,
        'growth_rate': 0.04,
        'volatility': 0.24,
        'pe_ratio': 21.2,
        'eps': 9.33,
        'payout_ratio': 0.84,
        'debt_to_equity': 0.47
    },
}


class OptimizedStockMarketScanner:
    """Fast stock scanner using parallel processing and caching"""
    
    # Top 10 dividend stocks - highest quality (reduced for faster scanning)
    DIVIDEND_STOCKS = [
        'JNJ', 'PG', 'KO', 'PEP',  # Consumer Staples
        'NEE', 'DUK',              # Utilities
        'JPM', 'BAC',              # Financials
        'PFE', 'MRK'               # Healthcare
    ]
    
    # Top 10 growth stocks - largest cap tech (reduced for faster scanning)
    TECH_GROWTH_STOCKS = [
        'MSFT', 'AAPL', 'GOOGL', 'NVDA',  # Mega-cap tech
        'AMZN', 'META',                    # Large-cap growth
        'CRM', 'ADBE',                     # Enterprise software
        'INTC', 'IBM'                      # Traditional tech
    ]
    
    POPULAR_DIVIDEND_STOCKS = DIVIDEND_STOCKS  # For backwards compatibility
    
    def __init__(self, tickers: List[str] = None, max_workers: int = 8, stock_type: str = 'dividend'):
        """
        Initialize optimized scanner
        
        Args:
            tickers: List of tickers to scan
            max_workers: Number of parallel workers (default: 48 for ultra-fast scanning)
            stock_type: Type of stocks being scanned ('dividend' or 'tech')
        """
        self.tickers = tickers or self.POPULAR_DIVIDEND_STOCKS
        self.max_workers = max_workers
        self.stock_type = stock_type
        # Don't create cache here - create in each worker thread instead
        self.ir_collector = InterestRateDataCollector()
        self.ir_params = self.ir_collector.calibrate_ir_model()
        self.rates = self.ir_collector.get_current_rates()
    
    def scan_market(self, use_tech_stocks: bool = False) -> List[StockOpportunity]:
        """
        Scan market using parallel processing
        
        Args:
            use_tech_stocks: If True, scan tech stocks; if False, scan dividend stocks
        
        Returns:
            List of StockOpportunity sorted by probability of profit
        """
        # Select which list to scan
        scan_tickers = self.TECH_GROWTH_STOCKS if use_tech_stocks else self.tickers
        stock_type = "tech" if use_tech_stocks else "dividend"
        display_type = "tech growth" if use_tech_stocks else "dividend"
        
        print(f"Starting parallel scan of {len(scan_tickers)} {display_type} stocks with {self.max_workers} workers...")
        start_time = time.time()
        
        opportunities = []
        successful_scans = 0
        failed_scans = 0
        cached_hits = 0
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._analyze_stock, ticker, stock_type): ticker 
                for ticker in scan_tickers
            }
            
            # Process completed futures as they finish
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result:
                        opportunities.append(result)
                        successful_scans += 1
                        # Check if was from cache
                        if hasattr(result, '_from_cache'):
                            cached_hits += 1
                except Exception as e:
                    failed_scans += 1
                    print(f"  [FAIL] {ticker}: {str(e)[:50]}")
        
        elapsed_time = time.time() - start_time
        
        # Sort by probability of profit
        opportunities.sort(key=lambda x: x.probability_of_profit, reverse=True)
        
        print(f"\nScan complete in {elapsed_time:.2f}s")
        print(f"  Successful: {successful_scans} | Failed: {failed_scans} | Cached hits: {cached_hits}")
        
        return opportunities
    
    def _analyze_stock(self, ticker: str, stock_type: str = 'dividend', timeout_sec: int = 15) -> StockOpportunity:
        """Analyze single stock (called in parallel), with fallback to demo data"""
        import signal
        import os
        
        # Create cache connection in this thread (SQLite thread-safety)
        cache = StockDataCache()
        
        # Check if we should force demo mode
        force_demo = os.environ.get('DEMO_MODE', '').lower() == 'true'
        
        try:
            # Try to get from cache first (always fastest)
            cached_analysis = cache.get_analysis(ticker)
            if cached_analysis:
                print(f"  [OK] {ticker} (cached)")
                return self._build_opportunity_from_cache(cached_analysis)
            
            # Check if we have cached stock data
            stock_params = cache.get_stock_data(ticker)
            
            if not stock_params:
                # Try fetching from yfinance ONLY if not in demo mode
                if not force_demo:
                    try:
                        # Use semaphore to rate-limit concurrent API calls
                        with _yfinance_semaphore:
                            print(f"  -> Fetching {ticker}...")
                            collector = StockDataCollector(ticker)
                            stock_params = collector.fetch_stock_data()
                        print(f"  [OK] {ticker} fetched (live)")
                    except Exception as fetch_err:
                        # Fallback to demo data if yfinance fails
                        print(f"  ~ {ticker} falling back to demo data ({str(fetch_err)[:30]})")
                        if ticker in DEMO_STOCK_DATA:
                            stock_params = self._build_stock_params_from_demo(ticker)
                            print(f"  [OK] {ticker} using demo data")
                        else:
                            print(f"  [FAIL] {ticker} no demo data available")
                            return None
                else:
                    # Force demo mode
                    if ticker in DEMO_STOCK_DATA:
                        stock_params = self._build_stock_params_from_demo(ticker)
                        print(f"  [OK] {ticker} using demo data (forced)")
                    else:
                        print(f"  [FAIL] {ticker} no demo data available")
                        return None
                
                # Validate stock data
                if not stock_params or stock_params.current_price <= 0:
                    return None
                
                # Cache the stock data
                try:
                    cache.cache_stock_data(stock_params)
                except Exception as cache_err:
                    print(f"  [WARN] {ticker} cache write failed: {str(cache_err)[:50]}")
            else:
                # Validate cached stock data
                if stock_params.current_price <= 0:
                    return None
            
            # Generate recommendation
            engine = RecommendationEngine(
                stock_params,
                self.ir_params,
                risk_free_rate=self.rates['risk_free_rate'],
                stock_type=stock_type  # Skip dividend/Fama-French models for tech stocks
            )
            # Use the simpler base engine during scans to avoid torch/holistic crashes
            recommendation = engine.generate_recommendation(
                investor_budget=100000,
                investor_risk_tolerance="MODERATE",
                use_holistic_engine=False
            )
            
            # Security checks for tech stocks (NOT for dividend stocks)
            safety_score = None
            security_alerts = None
            
            if stock_type == 'tech':
                try:
                    # Run tech stock security analyzer
                    print(f"    [Analyzing security for {ticker}]")
                    security_analyzer = TechStockSecurityAnalyzer(stock_params)
                    is_safe, alerts = security_analyzer.analyze()
                    print(f"      Analyze result: is_safe={is_safe}, alerts_count={len(alerts)}")
                    safety_score = security_analyzer.get_safety_score()
                    print(f"      Safety score: {safety_score}")
                    
                    # Convert alerts to dict format for JSON serialization
                    security_alerts = [
                        {
                            'risk_type': alert.risk_type,
                            'severity': alert.severity,
                            'description': alert.description,
                            'confidence': alert.confidence
                        }
                        for alert in alerts
                    ]
                    print(f"      Converted {len(security_alerts)} alerts to dict format")
                    
                    # Only filter out stocks with FRAUD-related critical alerts
                    # Bubble/valuation risks are warnings, not automatic exclusions
                    fraud_related_risks = ['pump_dump', 'scam_signal', 'unsustainable_growth', 'long_growth_period']
                    fraud_critical_alerts = [
                        a for a in alerts 
                        if a.severity == 'critical' and any(risk in a.risk_type for risk in fraud_related_risks)
                    ]
                    has_fraud_critical = len(fraud_critical_alerts) > 0
                    has_very_low_safety = safety_score < 25  # Extremely low (was 40, too aggressive)
                    
                    if has_fraud_critical or has_very_low_safety:
                        # This stock is too risky; don't include in results
                        print(f"      Filtered out: fraud_critical={has_fraud_critical}, low_safety={has_very_low_safety}")
                        return None
                    print(f"      Passed security checks")
                except Exception as e:
                    # Log error but continue processing (don't let one tech stock crash the whole scan)
                    import traceback
                    print(f"    [Error analyzing {ticker}] {str(e)}")
                    print(f"      Traceback: {traceback.format_exc()[:300]}")
                    # Return None to skip this stock if security analysis fails
                    return None

                # also run bubble detection and append any alerts
                try:
                    from models.bubble_detector import BubbleDetector
                    bubble_metrics = BubbleDetector(stock_params).detect_bubble()
                    bubble_alerts = []
                    for a in bubble_metrics.alerts:
                        bubble_alerts.append({
                            'type': a.type,
                            'severity': a.severity,
                            'description': a.description,
                            'confidence': a.confidence,
                        })
                    # ensure security_alerts exists and extend with bubble alerts
                    if security_alerts is None:
                        security_alerts = []
                    security_alerts.extend(bubble_alerts)
                    # record overall bubble score on opportunity later
                    bubble_score = bubble_metrics.bubble_risk_score
                except Exception:
                    bubble_score = None
            else:
                bubble_score = None
            
            # Calculate metrics
            prob_profit = self._calculate_profit_probability(
                recommendation.upside_downside_pct,
                recommendation.recommended_holding_period_months,
                stock_params.volatility,
                recommendation.confidence
            )
            
            annual_return = (stock_params.dividend_yield * 100) + (
                recommendation.upside_downside_pct / (recommendation.recommended_holding_period_months / 12)
            )
            
            exit_signal = self._determine_exit_signal(
                recommendation.action,
                recommendation.position_sizing.take_profit_price,
                recommendation.position_sizing.stop_loss_price,
                recommendation.recommended_holding_period_months
            )
            
            key_risk = recommendation.risks[0] if recommendation.risks else "Market volatility"
            
            # Cache analysis results (including current price and dividend yield)
            analysis_data = {
                'ticker': ticker,
                'current_price': stock_params.current_price,
                'dividend_yield': stock_params.dividend_yield,
                'fair_value': recommendation.fair_value,
                'upside_downside': recommendation.upside_downside_pct,
                'recommendation': recommendation.action.value,
                'confidence': recommendation.confidence * 100,
                'probability_of_profit': prob_profit,
                'holding_months': recommendation.recommended_holding_period_months,
                'expected_return': annual_return,
                'exit_signal': exit_signal,
                'key_risk': key_risk,
                'beta': stock_params.beta,
                'beta_r2': stock_params.beta_r2,
                'safety_score': safety_score,
                'security_alerts': security_alerts,
                'bubble_risk_score': bubble_score,
                'bubble_alerts': bubble_alerts if 'bubble_alerts' in locals() else [],
            }
            cache.cache_analysis(ticker, analysis_data)
            
            opportunity = StockOpportunity(
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
                key_risk=key_risk,
                beta=stock_params.beta,
                beta_r2=stock_params.beta_r2,
                safety_score=safety_score,
                security_alerts=security_alerts,
                bubble_risk_score=bubble_score,
                bubble_alerts=bubble_alerts if 'bubble_alerts' in locals() else []
            )
            
            return opportunity
        
        finally:
            cache.close()
    
    def _is_value_stock(self, stock: StockParams) -> bool:
        """
        Filter out growth stocks - keep only value/dividend stocks
        This model doesn't work well for growth or tech stocks
        """
        # Exclude stocks with very high P/E ratios (growth indicator)
        if stock.pe_ratio and stock.pe_ratio > 25:
            return False
        
        # Exclude stocks with very high growth rates but low dividends (growth stocks)
        if stock.growth_rate and stock.dividend_yield:
            # If growth rate > 15% and dividend yield < 2%, likely a growth stock
            if stock.growth_rate > 0.15 and stock.dividend_yield < 0.02:
                return False
        
        # Exclude stocks with very low dividend payout (might be reinvesting all earnings)
        if stock.payout_ratio and stock.payout_ratio < 0.20:
            return False
        
        return True
    
    def _build_stock_params_from_demo(self, ticker: str) -> StockParams:
        """Build StockParams from demo data fallback"""
        if ticker not in DEMO_STOCK_DATA:
            return None
        
        demo = DEMO_STOCK_DATA[ticker]
        return StockParams(
            ticker=ticker,
            current_price=demo['current_price'],
            dividend_yield=demo['dividend_yield'],
            dividend_per_share=demo['dividend_per_share'],
            growth_rate=demo['growth_rate'],
            volatility=demo['volatility'],
            beta=demo.get('beta'),
            beta_r2=demo.get('beta_r2'),
            pe_ratio=demo['pe_ratio'],
            payout_ratio=demo['payout_ratio'],
            earnings_per_share=demo['eps'],
            book_value_per_share=0.0,  # Not in demo data
            debt_to_equity=demo['debt_to_equity']
        )
    
    def _build_opportunity_from_cache(self, cached: Dict) -> StockOpportunity:
        """Build opportunity from cached analysis"""
        opp = StockOpportunity(
            ticker=cached['ticker'],
            current_price=cached.get('current_price', 0),
            fair_value=cached['fair_value'],
            upside_downside_pct=cached['upside_downside'],
            dividend_yield=cached.get('dividend_yield', 0),
            recommendation=cached['recommendation'],
            confidence=cached['confidence'],
            probability_of_profit=cached['probability_of_profit'],
            expected_holding_months=cached['holding_months'],
            expected_return=cached['expected_return'],
            exit_signal=cached['exit_signal'],
            key_risk=cached['key_risk'],
            beta=cached.get('beta'),
            beta_r2=cached.get('beta_r2'),
            safety_score=cached.get('safety_score'),
            security_alerts=cached.get('security_alerts')
        )
        opp._from_cache = True
        return opp
    
    def _calculate_profit_probability(self, upside_downside_pct: float, months: int,
                                     volatility: float, confidence: float) -> float:
        """Calculate probability of profit"""
        if upside_downside_pct > 0:
            base_prob = min(0.95, 0.5 + (upside_downside_pct / 100) * 0.2)
        else:
            base_prob = max(0.05, 0.5 + (upside_downside_pct / 100) * 0.2)
        
        time_adjustment = min(1.0, 1 - (volatility * 0.1) + (months / 12) * 0.05)
        confidence_adjustment = confidence / 100
        
        prob = base_prob * time_adjustment * confidence_adjustment
        return max(0.01, min(0.99, prob))
    
    def _determine_exit_signal(self, action: ActionType, take_profit: float,
                              stop_loss: float, holding_months: int) -> str:
        """Determine when to sell"""
        if action in [ActionType.STRONG_BUY, ActionType.BUY]:
            if holding_months >= 36:
                return f"Sell at ${take_profit:.2f} OR hold for dividend growth"
            else:
                return f"Sell at ${take_profit:.2f} (target: {holding_months} months)"
        elif action == ActionType.HOLD:
            return f"Hold for dividend income, Sell if drops below ${stop_loss:.2f}"
        else:
            return f"Exit at market, Stop loss at ${stop_loss:.2f}"
    
    def get_top_opportunities(self, n: int = 15) -> List[StockOpportunity]:
        """Get top N opportunities"""
        opportunities = self.scan_market()
        return opportunities[:n]
    
    def format_results(self, opportunities: List[StockOpportunity]) -> str:
        """Format results as readable string"""
        if not opportunities:
            return "No opportunities found"
        
        output = "\n" + "="*130 + "\n"
        output += "TOP STOCK OPPORTUNITIES - RANKED BY PROBABILITY OF PROFIT (OPTIMIZED PARALLEL SCAN)\n"
        output += "="*130 + "\n\n"
        
        for i, opp in enumerate(opportunities, 1):
            output += f"{i}. {opp.ticker}\n"
            output += f"   Price: ${opp.current_price:.2f} -> Fair Value: ${opp.fair_value:.2f}\n"
            output += f"   Upside: {opp.upside_downside_pct:.2f}% | Yield: {opp.dividend_yield:.2f}%\n"
            output += f"   [TARGET] Profit Probability: {opp.probability_of_profit*100:.0f}% | Return: {opp.expected_return:.2f}%/yr\n"
            output += f"   Recommendation: {opp.recommendation} (Confidence: {opp.confidence:.0f}%)\n"
            output += f"   Hold: {opp.expected_holding_months}mo | Exit: {opp.exit_signal}\n"
            output += f"   [WARN] Risk: {opp.key_risk}\n\n"
        
        output += "="*130 + "\n"
        return output
    
    def get_cache_info(self) -> str:
        """Get cache information"""
        cache = StockDataCache()
        stats = cache.get_cache_stats()
        cache.close()
        return f"\nCache Info: {stats['cached_stocks']} stocks, {stats['cached_analyses']} analyses, {stats['db_size_mb']:.2f} MB"
    
    def close(self):
        """No-op since each worker thread closes its own connection"""
        pass