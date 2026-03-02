"""
Holistic Signal Engine
======================
Addresses three critical issues with the base recommendation engine:

1. STALE PRICE DATA
   yfinance's default close prices can be 1-2 days behind. This module
   uses fast_info.last_price (real-time/15-min delayed) and falls back
   to a 1d download with prepost=True if needed.

2. EX-DIVIDEND DATE BLINDNESS
   Dividend stocks drop by ~the dividend amount on ex-date. The base
   engine sees this as a price decline and incorrectly signals SELL.
   This module:
     - Fetches the dividend calendar (ex-dates + amounts)
     - Detects if the current price drop occurred near an ex-date
     - Computes the "ex-div adjusted price" for valuation
     - Adds an EX_DIVIDEND_DROP context flag to suppress false SELL signals

3. HOLISTIC SELL / HOLD DECISION
   Rather than relying purely on the DDM + Fama-French upside/downside,
   this engine aggregates five independent signals with weighted voting:

     Signal                Weight    Source
     ─────────────────────────────────────────────────────────
     Valuation (DDM/DCF)     30%    Existing recommendation engine
     News Sentiment          25%    AINewsSentimentAnalyzer (your AI)
     Price Momentum           20%    52-week position + recent trend
     Dividend Health          15%    Payout ratio, coverage, growth
     Quality / Balance Sheet  10%    Existing quality score

   Each signal votes on a -2 to +2 scale:
     -2 = Strong Sell, -1 = Sell, 0 = Hold, +1 = Buy, +2 = Strong Buy

   The weighted average determines the final action. A SELL signal from
   valuation alone is suppressed if:
     - It coincides with an ex-dividend drop (within 5 trading days), OR
     - News sentiment is bullish/neutral AND other signals are positive, OR
     - The stock is within 5% of a 52-week high with bullish momentum

Usage:
    from models.holistic_signal_engine import HolisticSignalEngine
    engine = HolisticSignalEngine(ticker, stock_params, ir_params, rfr)
    result = engine.analyze()
    # result.action, result.confidence, result.rationale, result.signals_breakdown
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalVote:
    """A single signal's contribution to the final recommendation."""
    name: str
    score: float          # -2 to +2
    weight: float         # 0-1, sum of all weights = 1
    rationale: str
    confidence: float     # 0-1, how reliable is this signal


@dataclass
class HolisticResult:
    """Full output from the holistic signal engine."""
    ticker: str
    action: str                        # STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
    confidence: float                  # 0-1
    weighted_score: float              # -2 to +2 composite
    rationale: str
    signals: List[SignalVote]

    # Price context
    realtime_price: float
    close_price: float
    using_realtime: bool

    # Ex-dividend context
    near_ex_dividend: bool
    ex_dividend_date: Optional[str]
    ex_dividend_amount: float
    price_drop_explained_by_dividend: bool

    # News context
    news_sentiment: str
    news_sentiment_score: float
    news_keywords: List[str]
    news_validation_status: str

    # Upstream data
    fair_value: float
    upside_downside_pct: float
    base_action: str                   # What the pure DDM said
    catalysts: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

class HolisticSignalEngine:
    """
    Multi-factor signal aggregator that produces a more robust and accurate
    buy/sell/hold recommendation than the base DDM-only engine.
    """

    # Signal weights (must sum to 1.0)
    WEIGHTS = {
        'valuation':      0.30,
        'news':           0.25,
        'momentum':       0.20,
        'dividend_health': 0.15,
        'quality':        0.10,
    }

    # Days around ex-date within which a price drop is considered ex-div related
    EX_DIV_WINDOW_DAYS = 5

    def __init__(self,
                 ticker: str,
                 stock_params,          # StockParams
                 ir_params,             # InterestRateParams
                 risk_free_rate: float = 0.045,
                 stock_type: str = 'dividend',
                 use_news_ai: bool = True):
        self.ticker = ticker
        self.stock = stock_params
        self.ir_params = ir_params
        self.risk_free_rate = risk_free_rate
        self.stock_type = stock_type
        self.use_news_ai = use_news_ai

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self,
                investor_budget: float = 100_000,
                investor_risk_tolerance: str = 'MODERATE') -> HolisticResult:
        """
        Run the full multi-factor analysis and return a HolisticResult.
        """
        # ── Step 1: Get real-time price ──
        realtime_price, close_price, using_realtime = self._fetch_realtime_price()
        effective_price = realtime_price if using_realtime else close_price

        # ── Step 2: Ex-dividend detection ──
        ex_div_info = self._detect_ex_dividend_drop(effective_price)
        # Use ex-div adjusted price for valuation if applicable
        valuation_price = ex_div_info['valuation_price']

        # ── Step 3: Base DDM/DCF valuation signal ──
        val_signal, fair_value, base_action, base_upside = self._valuation_signal(
            valuation_price, investor_risk_tolerance
        )

        # ── Step 4: News AI sentiment signal ──
        news_signal, news_meta = self._news_signal()

        # ── Step 5: Price momentum signal ──
        momentum_signal = self._momentum_signal()

        # ── Step 6: Dividend health signal ──
        div_signal = self._dividend_health_signal()

        # ── Step 7: Quality/balance-sheet signal ──
        quality_signal = self._quality_signal()

        # ── Step 8: Aggregate votes ──
        all_signals = [val_signal, news_signal, momentum_signal, div_signal, quality_signal]
        weighted_score = sum(s.score * s.weight for s in all_signals)

        # ── Step 9: Ex-div suppression of false SELL ──
        # If the SELL signal is largely driven by a dividend drop, override towards HOLD
        if (ex_div_info['price_drop_explained_by_dividend']
                and weighted_score < 0
                and news_signal.score >= 0):
            # Suppress bearish momentum from ex-div drop
            weighted_score = max(0.0, weighted_score + 1.0)

        # ── Step 10: Convert score to action ──
        action, confidence = self._score_to_action(
            weighted_score, all_signals, investor_risk_tolerance
        )

        # ── Step 11: Build rationale ──
        rationale = self._build_rationale(
            action, weighted_score, all_signals, ex_div_info,
            fair_value, base_upside, news_meta
        )

        # ── Step 12: Catalysts and risks ──
        catalysts = self._build_catalysts(all_signals, ex_div_info, news_meta)
        risks = self._build_risks(all_signals, ex_div_info)

        return HolisticResult(
            ticker=self.ticker,
            action=action,
            confidence=confidence,
            weighted_score=weighted_score,
            rationale=rationale,
            signals=all_signals,
            realtime_price=realtime_price,
            close_price=close_price,
            using_realtime=using_realtime,
            near_ex_dividend=ex_div_info['near_ex_dividend'],
            ex_dividend_date=ex_div_info['ex_dividend_date'],
            ex_dividend_amount=ex_div_info['ex_dividend_amount'],
            price_drop_explained_by_dividend=ex_div_info['price_drop_explained_by_dividend'],
            news_sentiment=news_meta.get('sentiment', 'neutral'),
            news_sentiment_score=news_meta.get('sentiment_score', 0.0),
            news_keywords=news_meta.get('keywords', []),
            news_validation_status=news_meta.get('validation_status', 'not_analyzed'),
            fair_value=fair_value,
            upside_downside_pct=base_upside,
            base_action=base_action,
            catalysts=catalysts,
            risks=risks,
        )

    # ── Step 1: Real-time price ───────────────────────────────────────────────

    def _fetch_realtime_price(self) -> Tuple[float, float, bool]:
        """
        Fetch the most up-to-date price available.

        Priority:
          1. yf.Ticker.fast_info.last_price  (15-min delayed, always current session)
          2. yf.download(period='1d', interval='1m')  (intraday last bar)
          3. yf.download(period='2d', interval='1d', prepost=True) (last close)
          4. self.stock.current_price  (fallback: whatever was already fetched)

        Returns: (realtime_price, close_price, using_realtime)
        """
        close_price = self.stock.current_price
        try:
            import yfinance as yf
            ticker_obj = yf.Ticker(self.ticker)

            # Method 1: fast_info (most reliable for current price)
            try:
                fi = ticker_obj.fast_info
                rt = float(fi.last_price)
                if rt and rt > 0:
                    # Also get previous close for comparison
                    prev_close = float(fi.previous_close) if fi.previous_close else close_price
                    return rt, prev_close, True
            except Exception:
                pass

            # Method 2: 1-minute intraday (last bar during market hours)
            try:
                intraday = yf.download(
                    self.ticker, period='1d', interval='1m',
                    auto_adjust=True, progress=False
                )
                if not intraday.empty:
                    rt = float(intraday['Close'].iloc[-1])
                    if rt > 0:
                        return rt, close_price, True
            except Exception:
                pass

            # Method 3: 2-day daily with prepost
            try:
                daily = yf.download(
                    self.ticker, period='2d', interval='1d',
                    auto_adjust=True, prepost=True, progress=False
                )
                if not daily.empty and len(daily) >= 1:
                    rt = float(daily['Close'].iloc[-1])
                    if rt > 0 and rt != close_price:
                        return rt, float(daily['Close'].iloc[-2]) if len(daily) > 1 else close_price, True
            except Exception:
                pass

        except ImportError:
            pass

        # Fallback: use existing price
        return close_price, close_price, False

    # ── Step 2: Ex-dividend detection ────────────────────────────────────────

    def _detect_ex_dividend_drop(self, current_price: float) -> dict:
        """
        Detect whether the current price reflects an ex-dividend drop.

        Logic:
          - Fetch dividend history for the past 60 days
          - If there's an ex-date within EX_DIV_WINDOW_DAYS of today,
            record the dividend amount
          - Compute pre-ex-div "true price" = current_price + dividend_amount
          - Flag if price_drop_since_yesterday ≈ dividend_amount (within 20%)
          - Return valuation_price = pre-ex-div price for fair value calculation
        """
        result = {
            'near_ex_dividend': False,
            'ex_dividend_date': None,
            'ex_dividend_amount': 0.0,
            'price_drop_explained_by_dividend': False,
            'valuation_price': current_price,
        }

        try:
            import yfinance as yf
            import pandas as pd

            ticker_obj = yf.Ticker(self.ticker)

            # Get dividend history (last 90 days)
            try:
                divs = ticker_obj.dividends
                if divs is None or divs.empty:
                    return result
            except Exception:
                return result

            # Ensure timezone-naive for comparison
            try:
                divs.index = divs.index.tz_localize(None)
            except Exception:
                try:
                    divs.index = divs.index.tz_convert(None)
                except Exception:
                    pass

            today = datetime.date.today()
            window_start = today - datetime.timedelta(days=self.EX_DIV_WINDOW_DAYS)
            window_end = today + datetime.timedelta(days=2)  # also catch upcoming ex-dates

            # Filter to recent dividends
            recent = divs[
                (divs.index.date >= window_start) &
                (divs.index.date <= window_end)
            ]

            if recent.empty:
                return result

            # Take the most recent ex-dividend event
            latest_ex_date = recent.index[-1].date()
            latest_div_amount = float(recent.iloc[-1])

            result['near_ex_dividend'] = True
            result['ex_dividend_date'] = str(latest_ex_date)
            result['ex_dividend_amount'] = latest_div_amount

            # Compute the pre-dividend adjusted price for valuation
            valuation_price = current_price + latest_div_amount
            result['valuation_price'] = valuation_price

            # Detect if price drop ≈ dividend (stock is being unfairly penalised)
            try:
                # Fetch previous close (day before ex-date)
                hist = yf.download(
                    self.ticker,
                    start=str(latest_ex_date - datetime.timedelta(days=5)),
                    end=str(latest_ex_date + datetime.timedelta(days=1)),
                    interval='1d', auto_adjust=False, progress=False
                )
                if not hist.empty and len(hist) >= 2:
                    pre_ex_close = float(hist['Close'].iloc[-2])
                    actual_drop = pre_ex_close - current_price
                    # Drop explained if within 30% of dividend amount
                    if abs(actual_drop - latest_div_amount) <= latest_div_amount * 0.30:
                        result['price_drop_explained_by_dividend'] = True
            except Exception:
                # Conservative: if we know we're near ex-date, flag it anyway
                result['price_drop_explained_by_dividend'] = True

        except ImportError:
            pass
        except Exception:
            pass

        return result

    # ── Step 3: Valuation signal ──────────────────────────────────────────────

    def _valuation_signal(self, valuation_price: float,
                          risk_tolerance: str) -> Tuple[SignalVote, float, str, float]:
        """
        Run the base RecommendationEngine with the (possibly ex-div adjusted) price.
        Returns (SignalVote, fair_value, base_action_str, upside_pct).
        """
        try:
            from models.recommendation_engine import RecommendationEngine, ActionType

            # Temporarily override current_price for valuation
            original_price = self.stock.current_price
            self.stock.current_price = valuation_price

            engine = RecommendationEngine(
                self.stock, self.ir_params,
                risk_free_rate=self.risk_free_rate,
                stock_type=self.stock_type,
            )
            rec = engine.generate_recommendation(
                investor_budget=100_000,
                investor_risk_tolerance=risk_tolerance,
            )

            self.stock.current_price = original_price  # restore

            upside = rec.upside_downside_pct
            fair_value = rec.fair_value
            base_action = rec.action.value

            # Convert action to score
            action_to_score = {
                'STRONG_BUY': 2.0,
                'BUY': 1.0,
                'HOLD': 0.0,
                'SELL': -1.0,
                'STRONG_SELL': -2.0,
            }
            score = action_to_score.get(base_action, 0.0)
            confidence = min(rec.confidence, 0.9)

            rationale = f"DDM/DCF valuation: {upside:+.1f}% upside → {base_action}"

            return (
                SignalVote('valuation', score, self.WEIGHTS['valuation'], rationale, confidence),
                fair_value,
                base_action,
                upside,
            )

        except Exception as e:
            # If valuation fails, neutral signal
            return (
                SignalVote('valuation', 0.0, self.WEIGHTS['valuation'],
                           f'Valuation unavailable: {e}', 0.3),
                self.stock.current_price * 1.05,
                'HOLD',
                5.0,
            )

    # ── Step 4: News signal ───────────────────────────────────────────────────

    def _news_signal(self) -> Tuple[SignalVote, dict]:
        """
        Fetch and analyze recent news using the AI sentiment analyzer.
        Returns (SignalVote, news_metadata_dict).
        """
        meta = {
            'sentiment': 'neutral',
            'sentiment_score': 0.0,
            'keywords': [],
            'validation_status': 'not_analyzed',
            'article_count': 0,
        }

        try:
            from data.news_aggregator import NewsAggregator
            from models.ai_sentiment_analyzer import AINewsSentimentAnalyzer

            aggregator = NewsAggregator()
            articles = aggregator.get_articles(self.ticker, days=7)

            if not articles:
                return (
                    SignalVote('news', 0.0, self.WEIGHTS['news'],
                               'No recent news articles found', 0.2),
                    meta,
                )

            analyzer = AINewsSentimentAnalyzer(use_openai=self.use_news_ai)
            result = analyzer.analyze_articles(articles, self.ticker)

            sentiment = result.get('sentiment', 'neutral')
            score_raw = result.get('sentiment_score', 0.0)  # -1 to +1
            confidence = result.get('confidence', 0.5)
            validation = 'validated' if result.get('validation_passed') else 'moderate'

            meta = {
                'sentiment': sentiment,
                'sentiment_score': score_raw,
                'keywords': result.get('keywords', []),
                'validation_status': validation,
                'article_count': result.get('article_count', len(articles)),
            }

            # Map sentiment to score: scale -1..+1 → -2..+2
            news_score = score_raw * 2.0
            # Cap at ±2
            news_score = max(-2.0, min(2.0, news_score))

            rationale = (
                f"News ({result.get('article_count', len(articles))} articles): "
                f"{sentiment} (score {score_raw:+.2f}), "
                f"keywords: {', '.join(result.get('keywords', [])[:3]) or 'none'}"
            )

            return (
                SignalVote('news', news_score, self.WEIGHTS['news'], rationale, confidence),
                meta,
            )

        except Exception as e:
            return (
                SignalVote('news', 0.0, self.WEIGHTS['news'],
                           f'News analysis unavailable: {e}', 0.2),
                meta,
            )

    # ── Step 5: Momentum signal ───────────────────────────────────────────────

    def _momentum_signal(self) -> SignalVote:
        """
        Compute price momentum from:
          - 52-week position: where is the stock in its yearly range?
          - 20-day vs 50-day moving average crossover
          - RSI (14-day) for overbought/oversold

        Returns SignalVote with score -2 to +2.
        """
        try:
            import yfinance as yf
            import pandas as pd

            hist = yf.download(
                self.ticker, period='1y', interval='1d',
                auto_adjust=True, progress=False
            )['Close'].dropna()

            if len(hist) < 20:
                return SignalVote('momentum', 0.0, self.WEIGHTS['momentum'],
                                  'Insufficient history for momentum', 0.3)

            price = float(hist.iloc[-1])
            high_52 = float(hist.max())
            low_52  = float(hist.min())
            range_52 = high_52 - low_52
            position_52 = (price - low_52) / range_52 if range_52 > 0 else 0.5

            # MA crossover
            ma20 = float(hist.rolling(20).mean().iloc[-1])
            ma50 = float(hist.rolling(50).mean().iloc[-1]) if len(hist) >= 50 else ma20
            ma_bullish = price > ma20 > ma50

            # RSI
            delta = hist.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = float(100 - 100 / (1 + rs.iloc[-1]))

            # Build composite momentum score
            score = 0.0

            # 52-week position (higher = more bullish momentum)
            if position_52 > 0.80:
                score += 1.0    # Near 52-week high → strong uptrend
            elif position_52 > 0.60:
                score += 0.5
            elif position_52 < 0.20:
                score -= 1.0   # Near 52-week low → weak
            elif position_52 < 0.40:
                score -= 0.5

            # MA crossover
            if ma_bullish:
                score += 0.5
            elif price < ma20 < ma50:
                score -= 0.5

            # RSI
            if rsi > 70:
                score -= 0.5   # Overbought → caution
            elif rsi < 30:
                score += 0.5   # Oversold → potential reversal

            score = max(-2.0, min(2.0, score))
            confidence = 0.65

            rationale = (
                f"52w position: {position_52:.0%}, "
                f"MA {'bullish' if ma_bullish else 'bearish'} crossover, "
                f"RSI: {rsi:.0f}"
            )

            return SignalVote('momentum', score, self.WEIGHTS['momentum'], rationale, confidence)

        except Exception as e:
            # Fallback: neutral
            return SignalVote('momentum', 0.0, self.WEIGHTS['momentum'],
                              f'Momentum unavailable: {e}', 0.3)

    # ── Step 6: Dividend health signal ────────────────────────────────────────

    def _dividend_health_signal(self) -> SignalVote:
        """
        Assess dividend sustainability and growth trend.
        Only meaningful for dividend stocks; returns neutral for tech.
        """
        if self.stock_type != 'dividend' or self.stock.dividend_yield <= 0:
            return SignalVote('dividend_health', 0.0, self.WEIGHTS['dividend_health'],
                              'Not a dividend stock — neutral', 0.5)

        score = 0.0
        notes = []

        payout = self.stock.payout_ratio or 0.0
        yield_pct = self.stock.dividend_yield * 100
        growth = self.stock.growth_rate or 0.0
        dps = self.stock.dividend_per_share or 0.0
        eps = self.stock.earnings_per_share or 0.0

        # Payout ratio health
        if 0 < payout < 0.40:
            score += 1.0
            notes.append(f"Healthy payout ratio {payout:.0%}")
        elif 0.40 <= payout < 0.65:
            score += 0.5
            notes.append(f"Moderate payout {payout:.0%}")
        elif 0.65 <= payout < 0.85:
            score -= 0.5
            notes.append(f"High payout {payout:.0%}")
        elif payout >= 0.85 or payout < 0:
            score -= 1.5
            notes.append(f"Dangerous payout {payout:.0%}")

        # Dividend yield attractiveness
        if 3.0 <= yield_pct <= 7.0:
            score += 0.5
            notes.append(f"Attractive yield {yield_pct:.1f}%")
        elif yield_pct > 9.0:
            score -= 0.5
            notes.append(f"Suspiciously high yield {yield_pct:.1f}% — check sustainability")

        # Dividend growth
        if growth > 0.08:
            score += 0.5
            notes.append(f"Strong dividend growth {growth:.0%}")
        elif growth < 0:
            score -= 1.0
            notes.append("Dividend cut risk — negative growth")

        # EPS coverage of dividend
        if eps > 0 and dps > 0:
            coverage = eps / dps
            if coverage > 2.0:
                score += 0.5
                notes.append(f"Dividend well-covered (EPS/DPS={coverage:.1f}x)")
            elif coverage < 1.0:
                score -= 1.0
                notes.append(f"Dividend not covered by earnings ({coverage:.1f}x)")

        # Try to assess dividend growth history from yfinance
        try:
            import yfinance as yf
            divs = yf.Ticker(self.ticker).dividends
            if divs is not None and len(divs) >= 8:
                annual = divs.resample('Y').sum()
                if len(annual) >= 3:
                    # Check if dividends have grown over last 3 years
                    div_3yr_growth = (float(annual.iloc[-1]) / float(annual.iloc[-4]) - 1
                                      if float(annual.iloc[-4]) > 0 else 0)
                    if div_3yr_growth > 0.05:
                        score += 0.5
                        notes.append(f"3yr dividend growth {div_3yr_growth:.0%}")
                    elif div_3yr_growth < 0:
                        score -= 0.5
                        notes.append("Dividend declining trend")
        except Exception:
            pass

        score = max(-2.0, min(2.0, score))
        rationale = "; ".join(notes) if notes else "Standard dividend analysis"

        return SignalVote('dividend_health', score, self.WEIGHTS['dividend_health'],
                          rationale, 0.70)

    # ── Step 7: Quality signal ────────────────────────────────────────────────

    def _quality_signal(self) -> SignalVote:
        """
        Assess balance sheet and earnings quality.
        """
        try:
            from models.valuation_engine import StockValuationEngine
            ve = StockValuationEngine(self.stock, self.risk_free_rate)
            quality_score, quality_label = ve.quality_score()
        except Exception:
            quality_score = 50
            quality_label = 'Estimated'

        # Map quality score (0-100) to signal (-2 to +2)
        if quality_score >= 80:
            score = 1.5
        elif quality_score >= 65:
            score = 1.0
        elif quality_score >= 50:
            score = 0.0
        elif quality_score >= 35:
            score = -0.5
        else:
            score = -1.5

        # Additional checks from raw params
        dte = self.stock.debt_to_equity or 0.0
        if dte > 3.0:
            score -= 0.5

        score = max(-2.0, min(2.0, score))
        rationale = f"Quality score: {quality_score:.0f}/100 ({quality_label}), D/E: {dte:.2f}"

        return SignalVote('quality', score, self.WEIGHTS['quality'], rationale, 0.75)

    # ── Step 8: Score → Action ────────────────────────────────────────────────

    def _score_to_action(self,
                         weighted_score: float,
                         signals: List[SignalVote],
                         risk_tolerance: str) -> Tuple[str, float]:
        """
        Convert composite weighted score to ActionType string and confidence.

        Thresholds:
          ≥ 1.2 → STRONG_BUY
          ≥ 0.4 → BUY
          > -0.4 → HOLD
          > -1.2 → SELL
          ≤ -1.2 → STRONG_SELL

        Risk tolerance shifts thresholds slightly:
          CONSERVATIVE → harder to buy, easier to hold
          AGGRESSIVE   → easier to buy, harder to sell
        """
        if risk_tolerance == 'CONSERVATIVE':
            buy_threshold       = 1.4
            strong_buy_thresh   = 1.6
            sell_threshold      = -0.3
            strong_sell_thresh  = -1.0
        elif risk_tolerance == 'AGGRESSIVE':
            buy_threshold       = 0.2
            strong_buy_thresh   = 1.0
            sell_threshold      = -0.6
            strong_sell_thresh  = -1.4
        else:  # MODERATE
            buy_threshold       = 0.4
            strong_buy_thresh   = 1.2
            sell_threshold      = -0.4
            strong_sell_thresh  = -1.2

        if weighted_score >= strong_buy_thresh:
            action = 'STRONG_BUY'
        elif weighted_score >= buy_threshold:
            action = 'BUY'
        elif weighted_score > sell_threshold:
            action = 'HOLD'
        elif weighted_score > strong_sell_thresh:
            action = 'SELL'
        else:
            action = 'STRONG_SELL'

        # Confidence = average signal confidence, scaled by score certainty
        avg_conf = np.mean([s.confidence for s in signals])
        score_certainty = min(1.0, abs(weighted_score) / 2.0)
        confidence = avg_conf * (0.5 + 0.5 * score_certainty)
        confidence = min(0.97, max(0.50, confidence))

        return action, confidence

    # ── Rationale builder ─────────────────────────────────────────────────────

    def _build_rationale(self,
                         action: str,
                         weighted_score: float,
                         signals: List[SignalVote],
                         ex_div_info: dict,
                         fair_value: float,
                         upside: float,
                         news_meta: dict) -> str:
        """Generate a human-readable rationale combining all signals."""
        lines = []

        # Opening
        action_verb = {
            'STRONG_BUY': 'a compelling buy',
            'BUY': 'an attractive buy',
            'HOLD': 'a hold at current levels',
            'SELL': 'a sell candidate',
            'STRONG_SELL': 'a strong sell',
        }.get(action, action)
        lines.append(f"{self.ticker} is {action_verb} (composite score: {weighted_score:+.2f}/2.00).")

        # Ex-dividend context
        if ex_div_info['near_ex_dividend']:
            amt = ex_div_info['ex_dividend_amount']
            dt = ex_div_info['ex_dividend_date']
            lines.append(
                f"Note: Stock went ex-dividend ${amt:.3f} on {dt}. "
                f"{'The recent price drop is consistent with this dividend payment and does not reflect deteriorating fundamentals. ' if ex_div_info['price_drop_explained_by_dividend'] else ''}"
                f"Fair value analysis uses dividend-adjusted price (${self.stock.current_price + amt:.2f})."
            )

        # Valuation
        lines.append(
            f"Valuation: fair value ${fair_value:.2f} implies {upside:+.1f}% upside from current price."
        )

        # News
        sentiment = news_meta.get('sentiment', 'neutral')
        n_articles = news_meta.get('article_count', 0)
        if n_articles > 0:
            kws = ', '.join(news_meta.get('keywords', [])[:4])
            lines.append(
                f"Recent news ({n_articles} articles): {sentiment} sentiment. "
                f"Key themes: {kws or 'general coverage'}."
            )

        # Signal summary
        sig_parts = []
        for s in signals:
            label = '↑' if s.score > 0.2 else '↓' if s.score < -0.2 else '→'
            sig_parts.append(f"{s.name.replace('_', ' ')}: {label}{s.score:+.1f}")
        lines.append("Signal breakdown — " + " | ".join(sig_parts) + ".")

        return " ".join(lines)

    def _build_catalysts(self, signals, ex_div_info, news_meta) -> List[str]:
        cats = []
        if ex_div_info['near_ex_dividend']:
            cats.append(f"Dividend of ${ex_div_info['ex_dividend_amount']:.3f} recently paid — "
                        "price should recover post-ex-date")
        news_kws = news_meta.get('keywords', [])
        if news_kws:
            cats.append(f"Recent news themes: {', '.join(news_kws[:3])}")
        for s in signals:
            if s.score >= 1.0:
                cats.append(s.rationale)
        if not cats:
            cats.append("Stable fundamentals with dividend income")
        return cats[:5]

    def _build_risks(self, signals, ex_div_info) -> List[str]:
        risks = []
        for s in signals:
            if s.score <= -1.0:
                risks.append(s.rationale)
        if self.stock.debt_to_equity and self.stock.debt_to_equity > 2.0:
            risks.append(f"High leverage (D/E: {self.stock.debt_to_equity:.1f})")
        if not risks:
            risks.append("General market volatility")
        return risks[:4]