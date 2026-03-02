"""
Flask Web Server - Main application entry point
"""
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import json
import math
import os

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
            'timestamp': datetime.now().isoformat()
        }
        
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


@app.route('/api/scan-market', methods=['GET'])
def scan_market():
    """Scan market for dividend and tech stock opportunities."""
    try:
        from models.optimized_market_scanner import OptimizedStockMarketScanner
        from data.stock_data import InterestRateDataCollector
        from models.interest_rate_model import InterestRateAnalyzer

        limit = request.args.get('limit', 15, type=int)

        # Interest rate context (non-fatal if it fails)
        try:
            ir_collector = InterestRateDataCollector()
            ir_params = ir_collector.calibrate_ir_model()
            rate_analyzer = InterestRateAnalyzer(ir_params)
            rate_scenarios = rate_analyzer.rate_scenario_analysis()
        except Exception as rate_err:
            print(f"[scan-market] Rate data error (non-fatal): {rate_err}")
            rate_scenarios = {
                'base_case': 0.04, 'bull_case': 0.03, 'bear_case': 0.05,
                'current_rate': 0.04, 'long_term_mean': 0.04,
            }

        def _fmt_opp(opp, include_security=False):
            """Serialise a StockOpportunity to a JSON-safe dict."""
            try:
                d = {
                    'ticker':                 str(opp.ticker or 'N/A'),
                    'current_price':          _clean(getattr(opp, 'current_price', 0), 0, 2),
                    'fair_value':             _clean(getattr(opp, 'fair_value', 0), 0, 2),
                    'upside_downside':        f"{_clean(getattr(opp, 'upside_downside_pct', 0), 0):.2f}%",
                    'dividend_yield':         f"{_clean(getattr(opp, 'dividend_yield', 0), 0):.2f}%",
                    'beta':                   _clean(getattr(opp, 'beta', None), None, 3),
                    'beta_r2':                _clean(getattr(opp, 'beta_r2', None), None, 3),
                    'recommendation':         str(getattr(opp, 'recommendation', 'HOLD') or 'HOLD'),
                    'confidence':             f"{_clean(getattr(opp, 'confidence', 50), 50):.0f}%",
                    'probability_of_profit':  f"{_clean(getattr(opp, 'probability_of_profit', 0.5), 0.5) * 100:.1f}%",
                    'expected_holding_months': int(getattr(opp, 'expected_holding_months', 12) or 12),
                    'expected_annual_return': f"{_clean(getattr(opp, 'expected_return', 0), 0):.2f}%",
                    'exit_signal':            str(getattr(opp, 'exit_signal', '') or ''),
                    'key_risk':               str(getattr(opp, 'key_risk', '') or 'Market volatility'),
                    'safety_score':           _clean(getattr(opp, 'safety_score', None), None, 1),
                }
                if include_security:
                    alerts = getattr(opp, 'security_alerts', None)
                    if alerts and isinstance(alerts, list):
                        safe_alerts = []
                        for a in alerts:
                            try:
                                safe_alerts.append({
                                    'risk_type':   str(a.get('risk_type', '') if isinstance(a, dict) else ''),
                                    'severity':    str(a.get('severity', '') if isinstance(a, dict) else ''),
                                    'description': str(a.get('description', '') if isinstance(a, dict) else ''),
                                    'confidence':  _clean(a.get('confidence') if isinstance(a, dict) else 0, 0, 3),
                                })
                            except Exception:
                                pass
                        if safe_alerts:
                            d['security_alerts'] = safe_alerts
                            bubble = [a for a in safe_alerts if 'bubble' in a.get('risk_type', '')]
                            other  = [a for a in safe_alerts if 'bubble' not in a.get('risk_type', '')]
                            if bubble:
                                d['bubble_alerts'] = bubble
                                d['bubble_risk_level'] = 'CRITICAL' if any(a['severity'] == 'critical' for a in bubble) else 'HIGH'
                            if other:
                                d['fraud_alerts'] = other
                return d
            except Exception as fmt_err:
                print(f"[scan-market] Error formatting opportunity {getattr(opp, 'ticker', '?')}: {fmt_err}")
                return {'ticker': 'ERROR', 'error': str(fmt_err)}

        # Scan dividend stocks only (skip tech for speed)
        dividend_results = []
        try:
            print("[scan-market] Scanning dividend stocks...")
            div_scanner = OptimizedStockMarketScanner(
                tickers=OptimizedStockMarketScanner.DIVIDEND_STOCKS,
                max_workers=4, stock_type='dividend')
            opps = div_scanner.scan_market()
            opps.sort(key=lambda x: _clean(x.probability_of_profit, 0.5), reverse=True)
            for opp in opps[:limit]:
                formatted = _fmt_opp(opp, include_security=False)
                if formatted.get('ticker') != 'ERROR':
                    dividend_results.append(formatted)
            print(f"[scan-market] Dividend done: {len(dividend_results)} results")
        except Exception as div_err:
            import traceback
            print(f"[scan-market] Dividend scan error: {div_err}")
            print(f"[scan-market] Traceback: {traceback.format_exc()[:200]}")

        # Skip tech stocks for now (they require security analysis which can be slow/risky)
        tech_results = []

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
                'base_case':      _clean(rate_scenarios.get('base_case')),
                'bull_case':      _clean(rate_scenarios.get('bull_case')),
                'bear_case':      _clean(rate_scenarios.get('bear_case')),
                'current_rate':   _clean(rate_scenarios.get('current_rate')),
                'long_term_mean': _clean(rate_scenarios.get('long_term_mean')),
            },
            'timestamp': datetime.now().isoformat(),
        }

        # Final safety: encode to string first so any stray NaN raises here not in browser
        try:
            raw_json = json.dumps(payload)
        except TypeError as json_err:
            print(f"[scan-market] JSON serialization failed: {json_err}")
            print(f"[scan-market] Payload keys: {list(payload.keys())}")
            print(f"[scan-market] Dividend count: {len(dividend_results)}, Tech count: {len(tech_results)}")
            return app.response_class(
                json.dumps({'error': f'JSON serialization failed: {str(json_err)}'}), status=500, mimetype='application/json'
            )
        
        return app.response_class(raw_json, status=200, mimetype='application/json')

    except Exception as e:
        import traceback
        print(f"[scan-market] Fatal: {e}\n{traceback.format_exc()}")
        error_response = {'error': str(e)}
        try:
            return app.response_class(
                json.dumps(error_response), status=500, mimetype='application/json'
            )
        except:
            # Last resort: plain text error
            return f"Error: {str(e)}", 500


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


if __name__ == '__main__':
    # Disable debug and reloader in production to avoid slowdowns
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=3000, threaded=True)