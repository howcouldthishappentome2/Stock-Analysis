from models.optimized_market_scanner import OptimizedStockMarketScanner
import time

print("Starting dividend stock scan...")
scanner = OptimizedStockMarketScanner(
    tickers=OptimizedStockMarketScanner.DIVIDEND_STOCKS,
    max_workers=2,  # Reduced for debugging
    stock_type='dividend'
)

print(f"Scanner initialized with {len(scanner.tickers)} stocks")
print(f"Stocks: {scanner.tickers}")

start = time.time()
print("\nCalling scan_market()...")
opportunities = scanner.scan_market(use_tech_stocks=False)
elapsed = time.time() - start

print(f"Scan completed in {elapsed:.1f}s")
print(f"Found {len(opportunities)} opportunities")

if opportunities:
    print("\nTop 3:")
    for i, opp in enumerate(opportunities[:3], 1):
        print(f"{i}. {opp.ticker}: ${opp.current_price:.2f} -> ${opp.fair_value:.2f} ({opp.probability_of_profit*100:.0f}%)")
