"""
Stock Prediction System - Main Execution
Dividend-focused stock recommendation engine with interest rate stochastic modeling
"""
from data.stock_data import StockDataCollector, InterestRateDataCollector, StockScreener
from models.recommendation_engine import RecommendationEngine
from models.valuation_engine import StockValuationEngine
from models.dividend_model import DividendGrowthModel
from models.interest_rate_model import InterestRateAnalyzer
import config
import sys


def analyze_single_stock(ticker: str, budget: float = 100000, risk_tolerance: str = "MODERATE"):
    """
    Analyze a single stock and print comprehensive recommendation
    
    Args:
        ticker: Stock ticker symbol
        budget: Investment budget in dollars
        risk_tolerance: "CONSERVATIVE", "MODERATE", or "AGGRESSIVE"
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING STOCK: {ticker}")
    print(f"{'='*80}\n")
    
    try:
        # Step 1: Collect stock data
        print(f"[1/5] Collecting data for {ticker}...")
        collector = StockDataCollector(ticker)
        stock_params = collector.fetch_stock_data()
        print(f"      [OK] Current Price: ${stock_params.current_price:.2f}")
        print(f"      [OK] Dividend Yield: {stock_params.dividend_yield*100:.2f}%")
        print(f"      [OK] P/E Ratio: {stock_params.pe_ratio:.2f}")
        
        # Step 2: Get interest rate data
        print(f"\n[2/5] Analyzing interest rate environment...")
        ir_collector = InterestRateDataCollector()
        ir_params = ir_collector.calibrate_ir_model()
        rates = ir_collector.get_current_rates()
        print(f"      [OK] Current Risk-Free Rate: {rates['risk_free_rate']*100:.2f}%")
        print(f"      [OK] Long-term Rate: {rates['long_term_rate']*100:.2f}%")
        
        # Step 3: Generate recommendation
        print(f"\n[3/5] Generating recommendation...")
        engine = RecommendationEngine(
            stock_params,
            ir_params,
            risk_free_rate=rates['risk_free_rate']
        )
        recommendation = engine.generate_recommendation(
            investor_budget=budget,
            investor_risk_tolerance=risk_tolerance
        )
        
        # Step 4: Analyze valuation
        print(f"\n[4/5] Analyzing valuation metrics...")
        valuation = StockValuationEngine(stock_params)
        metrics = valuation.analyze_valuation_ratios()
        quality_score, quality = valuation.quality_score()
        
        # Step 5: Dividend analysis
        print(f"\n[5/5] Analyzing dividend sustainability...")
        div_model = DividendGrowthModel(stock_params, rates['risk_free_rate'])
        payout_ratio, div_safety = div_model.dividend_safety_check()
        
        # PRINT RESULTS
        print(f"\n{'='*80}")
        print(f"RECOMMENDATION SUMMARY")
        print(f"{'='*80}\n")
        
        action_str = recommendation.action.value.replace('_', ' ')
        print(f"ACTION: {action_str}")
        print(f"Confidence: {recommendation.confidence*100:.1f}%")
        print(f"\nFair Value: ${recommendation.fair_value:.2f}")
        print(f"Current Price: ${recommendation.current_price:.2f}")
        print(f"Upside/Downside: {recommendation.upside_downside_pct:.2f}%")
        
        print(f"\nHolding Period: {recommendation.recommended_holding_period_months} months")
        print(f"\nRationale:\n{recommendation.rationale}")
        
        print(f"\n{'-'*80}")
        print(f"POSITION SIZING")
        print(f"{'-'*80}")
        print(f"Budget: ${budget:,.2f}")
        print(f"Initial Investment: ${recommendation.position_sizing.initial_investment:,.2f}")
        print(f"Position Size: {recommendation.position_sizing.position_size_pct:.2f}%")
        print(f"Number of Shares: {recommendation.position_sizing.number_of_shares:.2f}")
        print(f"\nRecommended Buy Price: ${recommendation.position_sizing.recommended_buy_price:.2f}")
        print(f"Take Profit Price: ${recommendation.position_sizing.take_profit_price:.2f}")
        print(f"Stop Loss Price: ${recommendation.position_sizing.stop_loss_price:.2f}")
        
        print(f"\n{'-'*80}")
        print(f"VALUATION ANALYSIS")
        print(f"{'-'*80}")
        print(f"Overall Rating: {metrics.overall_rating}")
        print(f"Quality Score: {quality_score}/100 ({quality})")
        print(f"P/E Ratio: {metrics.pe_ratio:.2f}")
        print(f"P/B Ratio: {metrics.pb_ratio:.2f}")
        print(f"Payout Ratio: {metrics.payout_ratio*100:.1f}%")
        print(f"Dividend Coverage: {metrics.dividend_coverage}")
        
        print(f"\n{'-'*80}")
        print(f"DIVIDEND ANALYSIS")
        print(f"{'-'*80}")
        print(f"Payout Ratio: {payout_ratio*100:.1f}%")
        print(f"Dividend Safety: {div_safety}")
        
        print(f"\n{'-'*80}")
        print(f"KEY CATALYSTS")
        print(f"{'-'*80}")
        for i, catalyst in enumerate(recommendation.key_catalysts, 1):
            print(f"{i}. {catalyst}")
        
        print(f"\n{'-'*80}")
        print(f"KEY RISKS")
        print(f"{'-'*80}")
        for i, risk in enumerate(recommendation.risks, 1):
            print(f"{i}. {risk}")
        
        print(f"\n{'='*80}\n")
        
        return recommendation
    
    except Exception as e:
        print(f"✗ Error analyzing {ticker}: {str(e)}")
        return None


def screen_dividend_stocks(tickers: list):
    """
    Screen a list of stocks for dividend investing criteria
    
    Args:
        tickers: List of stock tickers to screen
    """
    print(f"\n{'='*80}")
    print(f"DIVIDEND STOCK SCREENING")
    print(f"{'='*80}\n")
    
    print(f"Screening {len(tickers)} stocks based on dividend criteria...")
    print(f"Minimum Dividend Yield: {config.DIVIDEND_SCREENING['min_dividend_yield']*100:.1f}%")
    print(f"Maximum P/E Ratio: {config.DIVIDEND_SCREENING['max_pe_ratio']}")
    print(f"Maximum Debt/Equity: {config.DIVIDEND_SCREENING['max_debt_to_equity']}\n")
    
    results = StockScreener.screen_stocks(
        tickers,
        min_dividend_yield=config.DIVIDEND_SCREENING['min_dividend_yield'],
        max_pe_ratio=config.DIVIDEND_SCREENING['max_pe_ratio'],
        max_debt_to_equity=config.DIVIDEND_SCREENING['max_debt_to_equity']
    )
    
    if results:
        print(f"Found {len(results)} qualifying stocks:\n")
        for i, stock in enumerate(results, 1):
            print(f"{i}. {stock.ticker}")
            print(f"   Price: ${stock.current_price:.2f}")
            print(f"   Dividend Yield: {stock.dividend_yield*100:.2f}%")
            print(f"   P/E Ratio: {stock.pe_ratio:.2f}")
            print(f"   Dividend Per Share: ${stock.dividend_per_share:.3f}\n")
    else:
        print("No stocks matched the screening criteria.")
    
    return results


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Stock Analysis & Recommendation Engine")
        print("\nUsage:")
        print("  python main.py <ticker>                              - Analyze single stock")
        print("  python main.py <ticker> <budget> <risk_tolerance>    - Analyze with parameters")
        print("  python main.py screen <ticker1> <ticker2> ...        - Screen stocks")
        print("  python main.py scan                                   - Scan market for opportunities")
        print("  python main.py web                                    - Start Flask web server")
        print("\nExamples:")
        print("  python main.py JNJ")
        print("  python main.py PG 50000 CONSERVATIVE")
        print("  python main.py screen JNJ PG KO MCD")
        print("  python main.py scan")
        print("  python main.py web")
        return
    
    if sys.argv[1].lower() == "web":
        # Start Flask web server
        print("Starting Flask web server...")
        from app import app
        app.run(
            debug=config.FLASK_DEBUG,
            use_reloader=config.FLASK_USE_RELOADER,
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            threaded=True
        )
    
    elif sys.argv[1].lower() == "scan":
        # Scan market for profitable stocks
        print("\n" + "="*80)
        print("MARKET SCANNER - Finding Best Dividend Stock Opportunities")
        print("="*80 + "\n")
        print("Scanning market for profitable dividend stocks (parallel processing)...")
        
        from models.optimized_market_scanner import OptimizedStockMarketScanner
        # Use 48 parallel workers for ultra-fast scanning with reduced stock list
        scanner = OptimizedStockMarketScanner(max_workers=48)
        opportunities = scanner.get_top_opportunities(n=20)
        print(scanner.format_results(opportunities))
        print(scanner.get_cache_info())
        scanner.close()
    
    elif sys.argv[1].lower() == "screen":
        # Screen stocks
        if len(sys.argv) < 3:
            print("Please provide at least one ticker to screen")
            return
        
        tickers = sys.argv[2:]
        screen_dividend_stocks(tickers)
    
    else:
        # Analyze single stock
        ticker = sys.argv[1].upper()
        budget = float(sys.argv[2]) if len(sys.argv) > 2 else 100000
        risk_tolerance = sys.argv[3].upper() if len(sys.argv) > 3 else "MODERATE"
        
        analyze_single_stock(ticker, budget, risk_tolerance)


if __name__ == "__main__":
    main()
