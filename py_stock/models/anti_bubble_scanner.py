"""
Anti-Bubble Small-Cap Tech Scanner & Portfolio Optimizer
=========================================================
Lives in: models/anti_bubble_scanner.py

Scans the ENTIRE US stock market using a 4-stage funnel to find small/mid-cap
tech stocks that are NOT reliant on the AI bubble and would be relatively
unaffected (or benefit) from an AI bubble pop.

PIPELINE OVERVIEW
-----------------
Stage 0 - Universe Pull          (~seconds)
  yfinance screener + exchange listing -> filter to tech/software sectors
  ~8,000 stocks -> ~1,200-1,500 tech-adjacent tickers

Stage 1 - Cheap Metadata Filter  (parallel, ~1-2 min)
  Fetch only yf.Ticker.info (no history) to check:
    * Market cap: $150M-$50B (small to mid cap)
    * Sector/Industry: Technology or related
    * Min avg volume > 40k shares/day (liquidity)
    * PE ratio < 500 (not completely nonsensical)
    * Not in known AI-bubble-dependent set
  ~1,500 -> ~200-300 candidates

Stage 2 - News AI Screen         (parallel, ~2-4 min)
  For each survivor, fetch recent headlines via NewsAggregator and score
  AI bubble dependency using the enhanced AINewsSentimentAnalyzer.
  Drop stocks where ai_bubble_dependency_score > threshold.
  ~300 -> ~80-120 candidates

Stage 3 - Full Analysis          (parallel, ~3-8 min)
  Run RecommendationEngine + BubbleDetector + TechStockSecurityAnalyzer.
  Compute beta vs QQQ (AI/tech bubble proxy).
  Apply final filters: bubble_risk < 65, safety_score > 30, upside > 2%.
  ~120 -> top 15-25 stocks

Portfolio Optimization
  Mean-variance optimize with max Sharpe, subject to:
    * Portfolio beta vs QQQ <= 0.4 (anti-correlation constraint)
    * 2%-20% position limits

Usage:
    from models.anti_bubble_scanner import AntiBubbleScanner
    scanner = AntiBubbleScanner(max_workers=16)
    portfolio = scanner.run(budget=100_000, top_n=15)
    portfolio.display()
"""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Dict, Set
import numpy as np

# ---------------------------------------------------------------------------
# Known AI-bubble-dependent tickers - always excluded
# ---------------------------------------------------------------------------
AI_BUBBLE_EXCLUSIONS: Set[str] = {
    'NVDA', 'AMD', 'SMCI', 'TSLA', 'PLTR', 'MSFT', 'GOOGL', 'GOOG',
    'AMZN', 'META', 'ORCL', 'CRM', 'AAPL', 'ARM', 'MRVL', 'AVGO',
    'ANET', 'DELL', 'HPE', 'SNOW', 'DDOG', 'MDB', 'NET', 'CRWD',
    'ZS', 'OKTA', 'PANW', 'AI', 'BBAI', 'SOUN', 'IONQ', 'RGTI',
    'QBTS', 'QUBT', 'ARQQ', 'ASTS', 'RKLB',
}

# Tech-related sectors and industries to keep in Stage 1
TECH_SECTORS = {
    'Technology', 'Communication Services', 'Industrials',
    'Health Care', 'Financial Services',
}
TECH_INDUSTRIES = {
    'Software--Application', 'Software--Infrastructure',
    'Software-Application', 'Software-Infrastructure',
    'Semiconductors', 'Semiconductor Equipment & Materials',
    'Electronic Components', 'Information Technology Services',
    'Computer Hardware', 'Communication Equipment',
    'Scientific & Technical Instruments', 'Health Information Services',
    'Internet Content & Information', 'Electronic Gaming & Multimedia',
    'Electronics & Computer Distribution',
}

# Rate limiter for yfinance API calls
_yf_semaphore = threading.Semaphore(10)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateStock:
    """A stock that survived all filters and full analysis."""
    ticker: str
    current_price: float
    fair_value: float
    upside_pct: float
    expected_annual_return: float
    volatility: float
    beta_vs_qqq: float
    bubble_risk_score: float
    safety_score: float
    ai_bubble_dependency_score: float
    news_sentiment: str
    probability_of_profit: float
    market_cap_b: float
    pe_ratio: float
    sector: str = ''
    industry: str = ''


@dataclass
class PortfolioAllocation:
    """Optimized portfolio output."""
    stocks: List[CandidateStock]
    weights: np.ndarray
    expected_portfolio_return: float
    expected_portfolio_volatility: float
    sharpe_ratio: float
    portfolio_beta_vs_qqq: float
    avg_bubble_risk: float
    avg_ai_dependency: float
    budget: float

    def display(self) -> None:
        """Pretty-print the portfolio to stdout."""
        print()
        print("=" * 90)
        print("  ANTI-BUBBLE SMALL-CAP TECH PORTFOLIO")
        print("=" * 90)
        print(f"  Budget                 : ${self.budget:>12,.0f}")
        print(f"  Expected Annual Return : {self.expected_portfolio_return * 100:>8.1f}%")
        print(f"  Expected Volatility    : {self.expected_portfolio_volatility * 100:>8.1f}%")
        print(f"  Sharpe Ratio           : {self.sharpe_ratio:>8.2f}")
        bmark = ('Bubble-resilient' if self.portfolio_beta_vs_qqq < 0.35
                 else 'Some AI correlation')
        print(f"  Portfolio beta vs QQQ  : {self.portfolio_beta_vs_qqq:>8.3f}  ({bmark})")
        print(f"  Avg Bubble Risk        : {self.avg_bubble_risk:>8.1f} / 100")
        print(f"  Avg AI Dependency      : {self.avg_ai_dependency:>8.1f} / 100")
        print()

        hdr = (f"{'#':<3} {'Ticker':<8} {'Weight':>7} {'$Amount':>11} {'Upside':>8} "
               f"{'Ret%':>7} {'Vol%':>7} {'B(QQQ)':>8} {'Bubble':>7} "
               f"{'AI-Dep':>7} {'Sentiment':>10}")
        print(hdr)
        print("-" * 90)

        for i, (s, w) in enumerate(zip(self.stocks, self.weights), 1):
            amt = w * self.budget
            print(
                f"{i:<3} {s.ticker:<8} {w * 100:>6.1f}% ${amt:>10,.0f} "
                f"{s.upside_pct:>+7.1f}% "
                f"{s.expected_annual_return * 100:>6.1f}% "
                f"{s.volatility * 100:>6.1f}% "
                f"{s.beta_vs_qqq:>8.3f} "
                f"{s.bubble_risk_score:>6.0f} "
                f"{s.ai_bubble_dependency_score:>6.0f} "
                f"{s.news_sentiment:>10}"
            )

        print("=" * 90)
        print()
        print("  LEGEND")
        print("  B(QQQ) < 0    -> tends to move OPPOSITE to AI bubble (ideal)")
        print("  B(QQQ) 0-0.3  -> largely independent of the AI bubble")
        print("  Bubble score < 30 -> minimal bubble risk")
        print("  AI-Dep score < 40 -> not AI-hype driven in recent news")
        print()
        print("  POSITION DETAIL")
        print("  " + "-" * 80)

        for s, w in zip(self.stocks, self.weights):
            if w < 0.005:
                continue
            beta_tag = ("inverse" if s.beta_vs_qqq < -0.1 else
                        "independent" if s.beta_vs_qqq < 0.3 else "correlated")
            print(f"\n  {s.ticker} ({w * 100:.1f}%)  --  ${w * self.budget:,.0f}")
            print(f"    ${s.current_price:.2f} -> ${s.fair_value:.2f} "
                  f"({s.upside_pct:+.1f}% upside)")
            print(f"    Bubble: {s.bubble_risk_score:.0f}/100  |  "
                  f"Safety: {s.safety_score:.0f}/100  |  "
                  f"B(QQQ): {s.beta_vs_qqq:.3f} [{beta_tag}]")
            print(f"    AI Dep: {s.ai_bubble_dependency_score:.0f}/100  |  "
                  f"News: {s.news_sentiment}  |  "
                  f"Cap: ${s.market_cap_b:.1f}B  |  "
                  f"Industry: {s.industry}")


# ---------------------------------------------------------------------------
# Main scanner class
# ---------------------------------------------------------------------------

class AntiBubbleScanner:
    """
    Full-market anti-bubble scanner using a 4-stage funnel.

    Args:
        max_workers:        Parallel threads (default 16)
        max_stage1_results: Cap survivors after Stage 1 (default 300)
        max_stage2_results: Cap survivors after Stage 2 (default 100)
        ai_dep_threshold:   Max AI dependency score to pass Stage 2 (default 45)
        bubble_threshold:   Max bubble risk score to pass Stage 3 (default 65)
        safety_threshold:   Min safety score to pass Stage 3 (default 30)
        min_upside_pct:     Min valuation upside % to include (default 2)
        max_portfolio_beta: Max portfolio QQQ beta for optimizer (default 0.40)
    """

    def __init__(
        self,
        max_workers: int = 16,
        max_stage1_results: int = 300,
        max_stage2_results: int = 100,
        ai_dep_threshold: float = 45.0,
        bubble_threshold: float = 65.0,
        safety_threshold: float = 30.0,
        min_upside_pct: float = 2.0,
        max_portfolio_beta: float = 0.40,
    ):
        self.max_workers = max_workers
        self.max_stage1_results = max_stage1_results
        self.max_stage2_results = max_stage2_results
        self.ai_dep_threshold = ai_dep_threshold
        self.bubble_threshold = bubble_threshold
        self.safety_threshold = safety_threshold
        self.min_upside_pct = min_upside_pct
        self.max_portfolio_beta = max_portfolio_beta

        self._ir_params = None
        self._risk_free_rate = 0.045
        self._news_cache: Dict[str, Dict] = {}
        self._load_ir_params()

    def _load_ir_params(self) -> None:
        try:
            from data.stock_data import InterestRateDataCollector
            ir = InterestRateDataCollector()
            self._ir_params = ir.calibrate_ir_model()
            self._risk_free_rate = ir.get_current_rates()['risk_free_rate']
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def run(
        self,
        budget: float = 100_000,
        top_n: int = 15,
        verbose: bool = True,
    ) -> PortfolioAllocation:
        """
        Run the full scan and return an optimized portfolio.

        Args:
            budget:  Investment capital in USD
            top_n:   Target number of holdings
            verbose: Print stage-by-stage progress
        """
        t0 = time.time()

        if verbose:
            print("\n" + "=" * 70)
            print("  ANTI-BUBBLE MARKET SCANNER")
            print("=" * 70)

        universe = self._stage0_get_universe(verbose)
        stage1 = self._stage1_metadata_filter(universe, verbose)
        stage2 = self._stage2_news_filter(stage1, verbose)
        candidates = self._stage3_full_analysis(stage2, verbose)

        if not candidates:
            raise RuntimeError(
                "No stocks survived all filters. "
                "Try widening thresholds (ai_dep_threshold, bubble_threshold, safety_threshold)."
            )

        if verbose:
            mins = (time.time() - t0) / 60
            print(f"\n  Scan complete in {mins:.1f} min -- "
                  f"{len(candidates)} candidates for portfolio optimization.")

        return self._optimize_portfolio(candidates, budget, top_n)

    # -------------------------------------------------------------------------
    # Stage 0: Universe
    # -------------------------------------------------------------------------

    def _stage0_get_universe(self, verbose: bool) -> List[str]:
        """
        Collect US-listed tech equity tickers using multiple fallback strategies:
          A) yfinance screener (new versions)
          B) Tech ETF constituent lists (XLK, IGV, SOXX, HACK, WCLD, etc.)
          C) Nasdaq FTP listing files
        """
        if verbose:
            print("\n  [Stage 0] Fetching market universe...")

        import yfinance as yf
        tickers: Set[str] = set()

        # --- Method A: yfinance screener ---
        try:
            screen = yf.screen(
                query={'sector': {'$in': list(TECH_SECTORS)}, 'region': 'us'},
                sortField='marketCap',
                sortType='DESC',
                offset=0,
                size=1500,
            )
            for q in (screen or {}).get('quotes', []):
                sym = (q.get('symbol') or '').replace('-', '.').strip().upper()
                if sym and sym.isalpha() and len(sym) <= 5:
                    tickers.add(sym)
            if verbose and tickers:
                print(f"    yfinance screener: {len(tickers)} tickers")
        except Exception:
            pass

        # --- Method B: ETF holdings ---
        etfs = ['XLK', 'IGV', 'SOXX', 'HACK', 'WCLD', 'FDN', 'SKYY',
                'ROBO', 'PSCT', 'CIBR', 'BUG']
        for etf in etfs:
            try:
                obj = yf.Ticker(etf)
                # Try funds_data attribute (newer yfinance)
                fd = getattr(obj, 'funds_data', None)
                if fd is not None:
                    eh = getattr(fd, 'equity_holdings', None)
                    if eh is not None:
                        for sym in eh.index:
                            s = str(sym).strip().upper()
                            if s.isalpha() and len(s) <= 5:
                                tickers.add(s)
            except Exception:
                pass

        # --- Method C: Nasdaq FTP ---
        if len(tickers) < 200:
            try:
                import urllib.request
                urls = [
                    'ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt',
                    'ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt',
                ]
                for url in urls:
                    try:
                        with urllib.request.urlopen(url, timeout=20) as resp:
                            lines = resp.read().decode('utf-8', errors='replace').split('\n')
                        for line in lines[1:]:
                            parts = line.split('|')
                            if len(parts) >= 2:
                                sym = parts[0].strip().upper()
                                if sym.isalpha() and 1 <= len(sym) <= 5:
                                    tickers.add(sym)
                    except Exception:
                        pass
            except Exception:
                pass

        tickers -= AI_BUBBLE_EXCLUSIONS
        tickers = {t for t in tickers if t.isalpha() and 1 <= len(t) <= 5}

        if verbose:
            print(f"    Universe: {len(tickers)} tickers after exclusions")
        return sorted(tickers)

    # -------------------------------------------------------------------------
    # Stage 1: Cheap metadata filter
    # -------------------------------------------------------------------------

    def _stage1_metadata_filter(self, universe: List[str], verbose: bool) -> List[str]:
        """
        Parallel fetch of yf.Ticker.info (no price history) for fast filtering.
        Criteria: market cap, sector, volume, PE, not in exclusion list.
        """
        if verbose:
            print(f"\n  [Stage 1] Metadata filter -- {len(universe)} tickers "
                  f"with {self.max_workers} workers...")

        passed: List[str] = []
        lock = threading.Lock()
        done = [0]

        def check(ticker: str) -> Optional[str]:
            try:
                import yfinance as yf
                with _yf_semaphore:
                    info = yf.Ticker(ticker).info or {}

                # Market cap filter
                cap_b = (info.get('marketCap') or 0) / 1e9
                if cap_b < 0.15 or cap_b > 50:
                    return None

                # Sector/industry filter
                sector = info.get('sector', '') or ''
                industry = info.get('industry', '') or ''
                if sector not in TECH_SECTORS and industry not in TECH_INDUSTRIES:
                    return None

                # Liquidity
                avg_vol = info.get('averageVolume') or info.get('averageVolume10days') or 0
                if avg_vol < 40_000:
                    return None

                # PE sanity
                pe = info.get('trailingPE') or info.get('forwardPE') or 0
                if pe and pe > 500:
                    return None

                # Price sanity (no penny stocks)
                price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
                if price < 1.0:
                    return None

                if ticker.upper() in AI_BUBBLE_EXCLUSIONS:
                    return None

                return ticker

            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(check, t): t for t in universe}
            for fut in as_completed(futures):
                result = fut.result()
                with lock:
                    done[0] += 1
                    if result:
                        passed.append(result)
                    if verbose and done[0] % 150 == 0:
                        print(f"    {done[0]}/{len(universe)} checked, "
                              f"{len(passed)} passing...")

        passed.sort()
        passed = passed[:self.max_stage1_results]

        if verbose:
            print(f"    Stage 1: {len(passed)} survivors")
        return passed

    # -------------------------------------------------------------------------
    # Stage 2: News AI screen
    # -------------------------------------------------------------------------

    def _stage2_news_filter(self, tickers: List[str], verbose: bool) -> List[str]:
        """
        Fetch news headlines and score AI bubble dependency via the enhanced
        AINewsSentimentAnalyzer. Stocks with ai_dep_score > threshold are dropped.
        """
        if verbose:
            print(f"\n  [Stage 2] News AI screen -- {len(tickers)} tickers...")

        passed: List[str] = []
        lock = threading.Lock()
        done = [0]

        def screen(ticker: str) -> Optional[str]:
            try:
                from data.news_aggregator import NewsAggregator
                from models.ai_sentiment_analyzer import AINewsSentimentAnalyzer

                articles = NewsAggregator().get_articles(ticker, days=14)
                analyzer = AINewsSentimentAnalyzer(use_openai=False)

                if not articles:
                    # No news: assume non-AI-dependent, pass with neutral score
                    with lock:
                        self._news_cache[ticker] = {
                            'ai_bubble_dependency_score': 0.0,
                            'sentiment': 'neutral',
                            'sentiment_score': 0.0,
                        }
                    return ticker

                ai_dep = analyzer.score_ai_bubble_dependency(articles)
                if ai_dep > self.ai_dep_threshold:
                    return None

                sentiment = analyzer._basic_keyword_analysis(articles)
                with lock:
                    self._news_cache[ticker] = {
                        'ai_bubble_dependency_score': ai_dep,
                        'sentiment': sentiment.get('sentiment', 'neutral'),
                        'sentiment_score': sentiment.get('sentiment_score', 0.0),
                    }
                return ticker

            except Exception:
                with lock:
                    self._news_cache[ticker] = {
                        'ai_bubble_dependency_score': 0.0,
                        'sentiment': 'neutral',
                        'sentiment_score': 0.0,
                    }
                return ticker

        with ThreadPoolExecutor(max_workers=min(self.max_workers, 20)) as ex:
            futures = {ex.submit(screen, t): t for t in tickers}
            for fut in as_completed(futures):
                result = fut.result()
                with lock:
                    done[0] += 1
                    if result:
                        passed.append(result)
                if verbose and done[0] % 50 == 0:
                    with lock:
                        p = len(passed)
                    print(f"    {done[0]}/{len(tickers)} analyzed, {p} passing...")

        # Sort: bullish low-AI-dep first
        def news_rank(t: str) -> float:
            nc = self._news_cache.get(t, {})
            bonus = -0.3 if nc.get('sentiment') == 'bullish' else 0.0
            return nc.get('ai_bubble_dependency_score', 0.0) + bonus

        passed.sort(key=news_rank)
        passed = passed[:self.max_stage2_results]

        if verbose:
            print(f"    Stage 2: {len(passed)} survivors")
        return passed

    # -------------------------------------------------------------------------
    # Stage 3: Full analysis
    # -------------------------------------------------------------------------

    def _stage3_full_analysis(self, tickers: List[str], verbose: bool) -> List[CandidateStock]:
        """
        Full pipeline: StockDataCollector + RecommendationEngine + BubbleDetector
        + TechStockSecurityAnalyzer + QQQ beta computation.
        """
        if verbose:
            print(f"\n  [Stage 3] Full analysis -- {len(tickers)} stocks, "
                  f"{self.max_workers} workers...")

        results: List[CandidateStock] = []
        lock = threading.Lock()
        done = [0]

        def analyze(ticker: str) -> Optional[CandidateStock]:
            try:
                import yfinance as yf
                from data.stock_data import StockDataCollector
                from models.recommendation_engine import RecommendationEngine
                from models.bubble_detector import BubbleDetector
                from models.tech_stock_analyzer import TechStockSecurityAnalyzer

                with _yf_semaphore:
                    collector = StockDataCollector(ticker)
                    stock_params = collector.fetch_stock_data()

                if not stock_params or stock_params.current_price <= 0:
                    return None

                with _yf_semaphore:
                    info = yf.Ticker(ticker).info or {}
                sector = info.get('sector', 'Technology') or 'Technology'
                industry = info.get('industry', '') or ''
                market_cap_b = (info.get('marketCap') or 0) / 1e9

                # Bubble detection
                detector = BubbleDetector(stock_params, sector='technology')
                bubble_metrics = detector.detect_bubble()
                if bubble_metrics.bubble_risk_score > self.bubble_threshold:
                    return None

                # Security / quality check
                sec = TechStockSecurityAnalyzer(stock_params)
                sec.analyze()
                safety = sec.get_safety_score()
                if safety < self.safety_threshold:
                    return None

                # Valuation
                if self._ir_params is None:
                    return None
                engine = RecommendationEngine(
                    stock_params, self._ir_params,
                    risk_free_rate=self._risk_free_rate,
                    stock_type='tech',
                )
                rec = engine.generate_recommendation(
                    investor_budget=100_000,
                    investor_risk_tolerance='MODERATE',
                )
                upside = rec.upside_downside_pct
                if upside < self.min_upside_pct:
                    return None

                fair_value = rec.fair_value
                years = max(1.0, rec.recommended_holding_period_months / 12.0)
                exp_return = max(
                    self._risk_free_rate,
                    (upside / 100.0) / years + (stock_params.dividend_yield or 0.0)
                )
                vol = max(0.10, stock_params.volatility or 0.30)

                # QQQ beta
                beta_qqq = self._compute_qqq_beta(ticker)

                # News metadata
                news = self._news_cache.get(ticker, {})
                ai_dep = news.get('ai_bubble_dependency_score', 0.0)
                sentiment = news.get('sentiment', 'neutral')

                prob = min(0.95, max(0.05, 0.5 + upside / 200.0))

                return CandidateStock(
                    ticker=ticker,
                    current_price=stock_params.current_price,
                    fair_value=fair_value,
                    upside_pct=upside,
                    expected_annual_return=exp_return,
                    volatility=vol,
                    beta_vs_qqq=beta_qqq,
                    bubble_risk_score=bubble_metrics.bubble_risk_score,
                    safety_score=safety,
                    ai_bubble_dependency_score=ai_dep,
                    news_sentiment=sentiment,
                    probability_of_profit=prob,
                    market_cap_b=market_cap_b,
                    pe_ratio=stock_params.pe_ratio or 0.0,
                    sector=sector,
                    industry=industry,
                )

            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(analyze, t): t for t in tickers}
            for fut in as_completed(futures):
                result = fut.result()
                with lock:
                    done[0] += 1
                    if result:
                        results.append(result)
                if verbose and result:
                    print(f"    + {result.ticker:<8} upside={result.upside_pct:+.1f}%  "
                          f"bubble={result.bubble_risk_score:.0f}  "
                          f"B(QQQ)={result.beta_vs_qqq:.2f}  "
                          f"AI-dep={result.ai_bubble_dependency_score:.0f}")

        if verbose:
            with lock:
                p = len(results)
                d = done[0]
            print(f"\n    Stage 3: {p} candidates from {d} analyzed")

        # Composite ranking
        def score(s: CandidateStock) -> float:
            return (
                s.expected_annual_return * 40
                - s.bubble_risk_score * 0.30
                - max(0.0, s.beta_vs_qqq) * 10
                + s.safety_score * 0.20
                + s.probability_of_profit * 20
                - s.ai_bubble_dependency_score * 0.20
                + (5.0 if s.news_sentiment == 'bullish' else 0.0)
            )

        results.sort(key=score, reverse=True)
        return results

    # -------------------------------------------------------------------------
    # QQQ beta
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_qqq_beta(ticker: str, period: str = '2y') -> float:
        """
        OLS beta of ticker vs QQQ using weekly returns over given period.
        Negative beta = moves opposite to the AI/tech bubble.
        """
        try:
            import yfinance as yf
            data = yf.download(
                [ticker, 'QQQ'], period=period, interval='1wk',
                auto_adjust=True, progress=False
            )['Close']
            if data.empty or ticker not in data.columns or 'QQQ' not in data.columns:
                return 0.8
            rets = data.pct_change().dropna()
            if len(rets) < 20:
                return 0.8
            sr = rets[ticker].values
            qr = rets['QQQ'].values
            cov = np.cov(sr, qr)
            if cov[1, 1] == 0:
                return 0.8
            return float(cov[0, 1] / cov[1, 1])
        except Exception:
            return 0.8

    # -------------------------------------------------------------------------
    # Portfolio optimizer
    # -------------------------------------------------------------------------

    def _optimize_portfolio(
        self,
        candidates: List[CandidateStock],
        budget: float,
        top_n: int,
        max_position: float = 0.20,
        min_position: float = 0.02,
    ) -> PortfolioAllocation:
        """
        Maximize Sharpe ratio subject to:
          - weights sum to 1
          - min_position <= w_i <= max_position
          - portfolio QQQ beta <= max_portfolio_beta
        Uses scipy SLSQP; falls back to inverse-volatility if it fails.
        """
        from scipy.optimize import minimize

        pool = candidates[:top_n]
        n = len(pool)

        mu = np.array([s.expected_annual_return for s in pool])
        sigma = np.array([s.volatility for s in pool])
        betas = np.array([s.beta_vs_qqq for s in pool])

        # Covariance = idiosyncratic variance + QQQ factor co-movement
        QQQ_VAR = 0.19 ** 2
        cov = np.diag(sigma ** 2) + np.outer(betas, betas) * QQQ_VAR

        rfr = self._risk_free_rate

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = float(w @ mu)
            port_vol = float(np.sqrt(max(float(w @ cov @ w), 1e-10)))
            return -(port_ret - rfr) / port_vol

        constraints = [
            {'type': 'eq',   'fun': lambda w: float(np.sum(w)) - 1.0},
            {'type': 'ineq', 'fun': lambda w: self.max_portfolio_beta - float(w @ betas)},
        ]
        bounds = [(min_position, max_position)] * n
        w0 = np.ones(n) / n

        res = minimize(
            neg_sharpe, w0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

        if not res.success:
            print(f"  Optimizer warning: {res.message}. Using inverse-vol weights.")
            inv_vol = 1.0 / sigma
            w = inv_vol / inv_vol.sum()
        else:
            w = res.x

        w = np.clip(w, 0.0, max_position)
        w /= w.sum()

        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(max(float(w @ cov @ w), 1e-10)))
        sharpe = (port_ret - rfr) / max(port_vol, 1e-6)
        p_beta = float(w @ betas)
        avg_bubble = float(w @ np.array([s.bubble_risk_score for s in pool]))
        avg_ai = float(w @ np.array([s.ai_bubble_dependency_score for s in pool]))

        return PortfolioAllocation(
            stocks=pool,
            weights=w,
            expected_portfolio_return=port_ret,
            expected_portfolio_volatility=port_vol,
            sharpe_ratio=sharpe,
            portfolio_beta_vs_qqq=p_beta,
            avg_bubble_risk=avg_bubble,
            avg_ai_dependency=avg_ai,
            budget=budget,
        )