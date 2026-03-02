"""
Flask Web Server - Main application entry point (SIMPLIFIED FAST MODE)
"""
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import json
import math
import os
import random

from data.stock_data import StockDataCollector, InterestRateDataCollector
from models.recommendation_engine import RecommendationEngine
from models.valuation_engine import StockValuationEngine
from models.dividend_model import DividendGrowthModel

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Create templates directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)


def _clean(val, default=None, digits=None):
    """
    Sanitise a value for JSON. NaN/Infinity -> default.
    Prevents jsonify emitting bare NaN tokens (invalid JSON)
    which causes 'Unexpected end of JSON input' in browsers.
    """
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return round(f, digits) if digits is not None else f
    except (TypeError, ValueError):
        return val


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    """Analyze a stock and return recommendation"""
    try:
        data = request.json
        ticker = data.get('ticker', '').upper()
        budget = float(data.get('budget', 100000))
        risk_tolerance = data.get('risk_tolerance', 'MODERATE').upper()
        
        if not ticker:
            return jsonify({'error': 'Ticker required'}), 400
        
        # Collect stock data
        collector = StockDataCollector(ticker)
        stock_params = collector.fetch_stock_data()
        
        # Get interest rate data
        ir_collector = InterestRateDataCollector()
        ir_params = ir_collector.calibrate_ir_model()
        rates = ir_collector.get_current_rates()
        
        # Generate recommendation
        engine = RecommendationEngine(
            stock_params,
            ir_params,
            risk_free_rate=rates['risk_free_rate']
        )
        
        recommendation = engine.generate_recommendation(
            investor_budget=budget,
            investor_risk_tolerance=risk_tolerance
        )
        
        # Valuation metrics
        valuation = StockValuationEngine(stock_params)
        metrics = valuation.analyze_valuation_ratios()
        quality_score, quality = valuation.quality_score()
        
        # Dividend analysis
        div_model = DividendGrowthModel(stock_params, rates['risk_free_rate'])
        dividend_proj = div_model.multi_stage_dividend_growth()
        payout_ratio, div_safety = div_model.dividend_safety_check()
        yield_analysis = div_model.yield_analysis()
        
        # Format response
        response = {
            'stock_data': {
                'ticker': ticker,
                'current_price': round(stock_params.current_price, 2),
                'dividend_per_share': round(stock_params.dividend_per_share, 3),
                'dividend_yield': f"{stock_params.dividend_yield * 100:.2f}%",
                'pe_ratio': round(stock_params.pe_ratio, 2),
                'earnings_per_share': round(stock_params.earnings_per_share, 2),
                'growth_rate': f"{stock_params.growth_rate * 100:.2f}%",
                'volatility': f"{stock_params.volatility * 100:.2f}%",
                'beta': round(stock_params.beta, 3) if stock_params.beta else None,
                'beta_r2': round(stock_params.beta_r2, 3) if stock_params.beta_r2 else None
            },
            'recommendation': {
                'action': recommendation.action.value,
                'confidence': f"{recommendation.confidence * 100:.1f}%",
                'fair_value': round(recommendation.fair_value, 2),
                'upside_downside': f"{recommendation.upside_downside_pct:.2f}%",
                'holding_period_months': recommendation.recommended_holding_period_months,
                'rationale': recommendation.rationale,
                'alpha': round(recommendation.alpha * 100, 2),
                'beta': round(stock_params.beta, 3) if stock_params.beta else None,
                'beta_r2': round(stock_params.beta_r2, 3) if stock_params.beta_r2 else None
            },
            'position_sizing': {
                'initial_investment': f"${recommendation.position_sizing.initial_investment:,.2f}",
                'position_size_pct': f"{recommendation.position_sizing.position_size_pct:.2f}%",
                'number_of_shares': round(recommendation.position_sizing.number_of_shares, 2),
                'recommended_buy_price': round(recommendation.position_sizing.recommended_buy_price, 2),
                'take_profit_price': round(recommendation.position_sizing.take_profit_price, 2),
                'stop_loss_price': round(recommendation.position_sizing.stop_loss_price, 2),
            },
            'valuation': {
                'fair_value': round(metrics.fair_value, 2),
                'rating': metrics.overall_rating,
                'quality_score': f"{quality_score}/100",
                'quality': quality,
                'pe_ratio': round(metrics.pe_ratio, 2),
                'pb_ratio': round(metrics.pb_ratio, 2),
                'payout_ratio': f"{metrics.payout_ratio * 100:.1f}%",
                'dividend_coverage': metrics.dividend_coverage,
            },
            'dividend_analysis': {
                'current_dividend': round(dividend_proj.current_dividend, 3),
                'payout_ratio': f"{payout_ratio * 100:.1f}%",
                'dividend_safety': div_safety,
                'yield_current': f"{yield_analysis['current_yield']:.2f}%",
                'yield_excess': f"{yield_analysis['excess_yield']:.2f}%",
                'yield_assessment': yield_analysis['assessment'],
            },
            'catalysts': recommendation.key_catalysts,
            'risks': recommendation.risks,
            # Security and bubble analysis for UI
            'security_alerts': [],
            'safety_score': None,
            'bubble_alerts': [],
            'bubble_risk_score': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Enrich response with security and bubble alerts when possible
        try:
            from models.tech_stock_analyzer import TechStockSecurityAnalyzer
            from models.bubble_detector import BubbleDetector
            sec = TechStockSecurityAnalyzer(stock_params)
            is_safe, alerts = sec.analyze()
            safety_score = sec.get_safety_score()
            security_alerts = [
                {
                    'risk_type': getattr(a, 'risk_type', getattr(a, 'type', None)),
                    'severity': getattr(a, 'severity', None),
                    'description': getattr(a, 'description', None),
                    'confidence': getattr(a, 'confidence', None),
                }
                for a in alerts
            ]
        except Exception:
            security_alerts = []
            safety_score = None

        try:
            bmetrics = BubbleDetector(stock_params).detect_bubble()
            bubble_alerts = [
                {
                    'type': getattr(a, 'type', None),
                    'severity': getattr(a, 'severity', None),
                    'description': getattr(a, 'description', None),
                    'confidence': getattr(a, 'confidence', None),
                }
                for a in bmetrics.alerts
            ]
            bubble_risk_score = bmetrics.bubble_risk_score
        except Exception:
            bubble_alerts = []
            bubble_risk_score = None

        response['security_alerts'] = security_alerts
        response['safety_score'] = safety_score
        response['bubble_alerts'] = bubble_alerts
        response['bubble_risk_score'] = bubble_risk_score

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/rates')
def get_rates():
    """Get current interest rates"""
    try:
        collector = InterestRateDataCollector()
        rates = collector.get_current_rates()
        return jsonify(rates)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/rate-forecast', methods=['POST'])
def get_rate_forecast():
    """Get interest rate forecast"""
    try:
        collector = InterestRateDataCollector()
        params = collector.calibrate_ir_model()
        
        from models.interest_rate_model import InterestRateAnalyzer
        analyzer = InterestRateAnalyzer(params)
        projection = analyzer.forecast(T=5)
        scenarios = analyzer.rate_scenario_analysis()
        
        return jsonify({
            'mean_rates': [round(r, 4) for r in projection.mean_rates],
            'expected_final_rate': round(projection.expected_final_rate, 4),
            'confidence_interval': [round(projection.confidence_interval[0], 4), 
                                   round(projection.confidence_interval[1], 4)],
            'scenarios': {
                'base_case': round(scenarios['base_case'], 4),
                'bull_case': round(scenarios['bull_case'], 4),
                'bear_case': round(scenarios['bear_case'], 4),
                'current_rate': round(scenarios['current_rate'], 4),
                'long_term_mean': round(scenarios['long_term_mean'], 4),
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

"""
API endpoints
"""


# ===== MARKET SCAN =====
@app.route('/api/scan-market', methods=['GET'])
def scan_market():
    """Scan market and return stock opportunities.

    The client can request ``?demo=true`` to force the built-in demo dataset
    (useful when yfinance or the web API is unavailable).  By default the
    scanner attempts live data using an HTTP quote fetch and will fall back to
    yfinance if necessary.
    """
    import os

    try:
        limit = request.args.get('limit', 15, type=int)
        demo_mode = request.args.get('demo', '').lower() == 'true'
        if demo_mode:
            os.environ['DEMO_MODE'] = 'true'
        elif 'DEMO_MODE' in os.environ:
            del os.environ['DEMO_MODE']

        from data.cache import StockDataCache

        # shorten cache validity so fetches are fresh
        StockDataCache.CACHE_VALIDITY_HOURS = 0.01
        cache = StockDataCache()
        cache.clear_old_data()

        from models.optimized_market_scanner import OptimizedStockMarketScanner

        def _fmt_opp(opp):
            # convert opportunity dataclass to plain dict for JSON
            result = {
                'ticker': str(opp.ticker),
                'current_price': _clean(getattr(opp, 'current_price', 0), 0, 2),
                'fair_value': _clean(getattr(opp, 'fair_value', 0), 0, 2),
                'upside_downside': f"{_clean(getattr(opp, 'upside_downside_pct', 0),0):.2f}%",
                'dividend_yield': f"{_clean(getattr(opp, 'dividend_yield',0),0):.2f}%",
                'beta': _clean(getattr(opp, 'beta', None), None, 3),
                'beta_r2': _clean(getattr(opp, 'beta_r2', None), None, 3),
                'recommendation': str(getattr(opp, 'recommendation', 'HOLD')),
                'confidence': f"{_clean(getattr(opp, 'confidence',0),0):.0f}%",
                'probability_of_profit': f"{_clean(getattr(opp, 'probability_of_profit',0),0)*100:.1f}%",
                'expected_holding_months': int(getattr(opp, 'expected_holding_months',12) or 12),
                'expected_annual_return': f"{_clean(getattr(opp, 'expected_return',0),0):.2f}%",
                'exit_signal': str(getattr(opp, 'exit_signal','') or ''),
                'key_risk': str(getattr(opp, 'key_risk','') or 'Market volatility'),
                'safety_score': _clean(getattr(opp, 'safety_score'), None, 1),
                # include any security/bubble alerts separately for the UI
                'security_alerts': getattr(opp, 'security_alerts', []) or [],
                'bubble_alerts': getattr(opp, 'bubble_alerts', []) or [],
            }
            return result

        # dividend scan
        dividend_results = []
        print("[scan-market] live dividend scan...")
        div_scanner = OptimizedStockMarketScanner(
            tickers=OptimizedStockMarketScanner.DIVIDEND_STOCKS,
            max_workers=3, stock_type='dividend')
        for opp in div_scanner.get_top_opportunities(n=limit):
            dividend_results.append(_fmt_opp(opp))
        print(f"[scan-market] dividend count {len(dividend_results)}")

        # tech scan
        tech_results = []
        print("[scan-market] live tech scan...")
        tech_scanner = OptimizedStockMarketScanner(
            tickers=OptimizedStockMarketScanner.TECH_GROWTH_STOCKS,
            max_workers=3, stock_type='tech')
        for opp in tech_scanner.get_top_opportunities(n=limit):
            tech_results.append(_fmt_opp(opp))
        print(f"[scan-market] tech count {len(tech_results)}")

        payload = {
            'dividend_stocks': {
                'count': len(dividend_results),
                'opportunities': dividend_results,
                'category': 'High-Quality Dividend Stocks',
            },
            'tech_stocks': {
                'count': len(tech_results),
                'opportunities': tech_results,
                'category': 'Tech Growth Stocks',
            },
            'scenarios': {
                'base_case': 0.045,
                'bull_case': 0.035,
                'bear_case': 0.055,
                'current_rate': 0.045,
                'long_term_mean': 0.045,
            },
            'timestamp': datetime.now().isoformat(),
        }
        raw_json = json.dumps(payload)
        return app.response_class(raw_json, status=200, mimetype='application/json')

    except Exception as e:
        import traceback
        print(f"[scan-market] Error live: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'live scan failed; see server logs'}), 500


@app.route('/api/news-analysis/<ticker>', methods=['GET'])
def news_analysis(ticker):
    """Get AI-powered news sentiment analysis for a stock"""
    try:
        from models.news_enhanced_recommendation import NewsEnhancedRecommendationEngine
        
        days = request.args.get('days', 7, type=int)
        
        engine = NewsEnhancedRecommendationEngine(use_ai=True)
        news_summary = engine.get_news_summary(ticker, days=days)
        
        return jsonify(news_summary)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/scan-anti-bubble', methods=['GET'])
def scan_anti_bubble():
    """Run anti-bubble scan and return outcome for frontend."""
    try:
        budget = float(request.args.get('budget', 100000))
        top = int(request.args.get('top', 15))
        demo_mode = request.args.get('demo', '').lower() == 'true'

        # Demo results for quick testing
        if demo_mode:
            demo_portfolio = [
                {'ticker': 'PINS', 'weight': 0.20, 'current_price': 32.45, 'fair_value': 38.92, 'upside_pct': 20.0, 'bubble_risk_score': 35, 'safety_score': 68, 'ai_bubble_dependency_score': 0.15, 'news_sentiment': 'neutral', 'beta_vs_qqq': -0.15, 'expected_annual_return': 0.25},
                {'ticker': 'SNAP', 'weight': 0.18, 'current_price': 15.30, 'fair_value': 18.76, 'upside_pct': 22.5, 'bubble_risk_score': 42, 'safety_score': 62, 'ai_bubble_dependency_score': 0.20, 'news_sentiment': 'neutral', 'beta_vs_qqq': 0.15, 'expected_annual_return': 0.22},
                {'ticker': 'ROKU', 'weight': 0.16, 'current_price': 68.20, 'fair_value': 79.44, 'upside_pct': 16.5, 'bubble_risk_score': 48, 'safety_score': 55, 'ai_bubble_dependency_score': 0.25, 'news_sentiment': 'mixed', 'beta_vs_qqq': 0.22, 'expected_annual_return': 0.16},
                {'ticker': 'MSTR', 'weight': 0.15, 'current_price': 425.60, 'fair_value': 468.18, 'upside_pct': 10.0, 'bubble_risk_score': 52, 'safety_score': 48, 'ai_bubble_dependency_score': 0.30, 'news_sentiment': 'positive', 'beta_vs_qqq': 0.45, 'expected_annual_return': 0.10},
                {'ticker': 'CEPU', 'weight': 0.14, 'current_price': 24.75, 'fair_value': 26.98, 'upside_pct': 8.9, 'bubble_risk_score': 40, 'safety_score': 71, 'ai_bubble_dependency_score': 0.10, 'news_sentiment': 'neutral', 'beta_vs_qqq': -0.10, 'expected_annual_return': 0.09},
                {'ticker': 'CPRI', 'weight': 0.12, 'current_price': 156.30, 'fair_value': 171.25, 'upside_pct': 9.6, 'bubble_risk_score': 38, 'safety_score': 74, 'ai_bubble_dependency_score': 0.12, 'news_sentiment': 'neutral', 'beta_vs_qqq': -0.08, 'expected_annual_return': 0.09},
                {'ticker': 'EXPR', 'weight': 0.05, 'current_price': 7.40, 'fair_value': 8.28, 'upside_pct': 11.9, 'bubble_risk_score': 45, 'safety_score': 59, 'ai_bubble_dependency_score': 0.18, 'news_sentiment': 'mixed', 'beta_vs_qqq': 0.35, 'expected_annual_return': 0.11},
            ]
            
            # Calculate portfolio statistics
            total_weight = sum(h['weight'] for h in demo_portfolio)
            avg_bubble_risk = sum(h['bubble_risk_score'] * h['weight'] for h in demo_portfolio) / total_weight if total_weight > 0 else 0
            avg_ai_dep = sum(h['ai_bubble_dependency_score'] * h['weight'] for h in demo_portfolio) / total_weight if total_weight > 0 else 0
            portfolio_beta = sum(h['beta_vs_qqq'] * h['weight'] for h in demo_portfolio) if total_weight > 0 else 0
            portfolio_return = sum(h['expected_annual_return'] * h['weight'] for h in demo_portfolio) if total_weight > 0 else 0
            portfolio_volatility = 0.188  # Typical for this mix
            
            # Extended candidate list for the "Stock Candidates" tab
            demo_candidates = demo_portfolio + [
                {'ticker': 'VEEV', 'weight': 0.0, 'current_price': 92.15, 'fair_value': 98.42, 'upside_pct': 6.8, 'bubble_risk_score': 38, 'safety_score': 66, 'ai_bubble_dependency_score': 0.22, 'news_sentiment': 'neutral', 'beta_vs_qqq': 0.28, 'expected_annual_return': 0.067},
                {'ticker': 'DDOG', 'weight': 0.0, 'current_price': 125.30, 'fair_value': 131.80, 'upside_pct': 5.2, 'bubble_risk_score': 42, 'safety_score': 63, 'ai_bubble_dependency_score': 0.26, 'news_sentiment': 'bullish', 'beta_vs_qqq': 0.32, 'expected_annual_return': 0.052},
                {'ticker': 'OKTA', 'weight': 0.0, 'current_price': 73.40, 'fair_value': 76.45, 'upside_pct': 4.2, 'bubble_risk_score': 45, 'safety_score': 58, 'ai_bubble_dependency_score': 0.28, 'news_sentiment': 'neutral', 'beta_vs_qqq': 0.38, 'expected_annual_return': 0.042},
            ]
            
            return jsonify({
                'candidates_found': len(demo_candidates),
                'portfolio': demo_portfolio,
                'all_candidates': demo_candidates,
                'budget': budget,
                'portfolio_stats': {
                    'expected_return': portfolio_return,
                    'expected_volatility': portfolio_volatility,
                    'sharpe_ratio': portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0,
                    'portfolio_beta': portfolio_beta,
                    'avg_bubble_risk': avg_bubble_risk,
                    'avg_ai_dependency': avg_ai_dep
                }
            })

        from models.anti_bubble_scanner import AntiBubbleScanner

        scanner = AntiBubbleScanner(max_workers=8)
        portfolio = scanner.run(budget=budget, top_n=top, verbose=False)

        results = []
        for s, w in zip(portfolio.stocks, portfolio.weights):
            results.append({
                'ticker': s.ticker,
                'weight': w,
                'current_price': s.current_price,
                'fair_value': s.fair_value,
                'upside_pct': s.upside_pct,
                'bubble_risk_score': s.bubble_risk_score,
                'safety_score': s.safety_score,
                'ai_dependency_score': s.ai_bubble_dependency_score,
                'news_sentiment': s.news_sentiment,
            })

        return jsonify({'candidates_found': len(results), 'portfolio': results})
    except Exception as e:
        import traceback
        print(f"[scan-anti-bubble] Error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'anti-bubble scan failed; see server logs'}), 500


@app.route('/api/analyze-with-news', methods=['POST'])
def analyze_stock_with_news():
    """Analyze a stock and return recommendation with news sentiment"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper()
        
        if not ticker:
            return jsonify({'error': 'Ticker required'}), 400
        
        from data.stock_data import StockDataCollector, InterestRateDataCollector
        from models.news_enhanced_recommendation import NewsEnhancedRecommendationEngine
        from models.valuation_engine import StockValuationEngine
        
        # Get stock data
        collector = StockDataCollector(ticker)
        stock_params = collector.fetch_stock_data()
        
        # Generate base recommendation
        ir_collector = InterestRateDataCollector()
        ir_params = ir_collector.calibrate_ir_model()
        rates = ir_collector.get_current_rates()
        
        engine = RecommendationEngine(
            stock_params,
            ir_params,
            risk_free_rate=rates['risk_free_rate']
        )
        recommendation = engine.generate_recommendation()
        
        # Enhance with news sentiment
        news_engine = NewsEnhancedRecommendationEngine(use_ai=True)
        recommendation = news_engine.enhance_recommendation(recommendation, days_lookback=7)
        
        # Get valuation metrics
        valuation_engine = StockValuationEngine(stock_params, risk_free_rate=rates['risk_free_rate'])
        metrics = valuation_engine.analyze_valuation_ratios()
        
        return jsonify({
            'ticker': recommendation.ticker,
            'action': recommendation.action.value,
            'confidence': f"{recommendation.confidence * 100:.1f}%",
            'fair_value': round(recommendation.fair_value, 2),
            'upside_downside': f"{recommendation.upside_downside_pct:.2f}%",
            'news_sentiment': recommendation.news_sentiment,
            'news_sentiment_score': round(recommendation.news_sentiment_score, 2),
            'news_keywords': recommendation.news_keywords,
            'news_validation_status': recommendation.news_validation_status,
            'current_price': round(stock_params.current_price, 2),
            'dividend_yield': f"{stock_params.dividend_yield*100:.2f}%",
            'pe_ratio': round(metrics.pe_ratio, 2),
            'pb_ratio': round(metrics.pb_ratio, 2),
            'key_catalysts': recommendation.key_catalysts,
            'risks': recommendation.risks,
            'rationale': recommendation.rationale,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/allocate-portfolio', methods=['POST'])
def allocate_portfolio():
    """Suggest asset allocation weights for a portfolio"""
    try:
        data = request.json
        recommended_stocks = data.get('stocks', [])
        budget = float(data.get('budget', 100000))
        risk_tolerance = data.get('risk_tolerance', 'MODERATE').upper()
        
        if not recommended_stocks:
            return jsonify({'error': 'No stocks provided for allocation'}), 400
        
        from models.portfolio_optimizer import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer()
        allocation = optimizer.suggest_allocation(
            recommended_stocks,
            budget=budget,
            risk_tolerance=risk_tolerance
        )
        
        result = optimizer.format_allocation_for_json(allocation)
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        print(f"[allocate-portfolio] Error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=3000, threaded=True)
