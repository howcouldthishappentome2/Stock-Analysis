"""
Stock Data Collection - Fetch stock data from yfinance and calculate metrics
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from models.stock_params import StockParams, InterestRateParams


class StockDataCollector:
    """Collect and calculate stock metrics from yfinance"""
    
    def __init__(self, ticker: str, period: str = "5y"):
        """
        Initialize data collector
        
        Args:
            ticker: Stock ticker symbol
            period: Historical data period ("1y", "5y", "10y", etc.)
        """
        self.ticker = ticker.upper()
        self.period = period
        self.stock = yf.Ticker(self.ticker)
        
    def fetch_stock_data(self) -> StockParams:
        """
        Fetch and calculate all stock parameters.  Network operations are wrapped in a
        short-lived thread with a timeout so the caller can't hang indefinitely.

        Returns:
            StockParams object with complete stock data
        """
        # helper performing the raw data logic
        def _raw_fetch():
            # try to refresh basic quote info using the faster public API before
            # falling back to the full yfinance object.  this is our "google/online"
            # data source mentioned by the user and it keeps prices very up-to-date
            # without holding the yfinance threads open indefinitely.
            try:
                self._refresh_quote_via_http()
            except Exception:
                # if the lightweight request fails we ignore and continue with
                # whatever stock.info already contains (yfinance may have cached
                # data from a previous call).
                pass

            # Get current price (try multiple possible keys)
            current_price = (
                self.stock.info.get('currentPrice')
                or self.stock.info.get('regularMarketPrice')
                or self.stock.info.get('previousClose')
                or 0
            )

            if current_price == 0:
                raise ValueError(f"Could not fetch current price for {self.ticker}")

            # Dividend data
            dividend_per_share = self._get_dividend_per_share()
            dividend_yield = self.stock.info.get('dividendYield', 0) or 0
            # sometimes yfinance omits the yield; compute manually if we have
            # dividend_per_share and price.
            if dividend_yield == 0 and dividend_per_share and current_price:
                try:
                    dividend_yield = dividend_per_share / float(current_price)
                except Exception:
                    pass

            # Earnings and book value
            eps = self.stock.info.get('trailingEps', 0) or 0
            book_value_per_share = self._get_book_value_per_share()

            # Ratios
            pe_ratio = self.stock.info.get('trailingPE', 0) or 0
            payout_ratio = self._calculate_payout_ratio(dividend_per_share, eps)

            # Leverage
            debt_to_equity = self.stock.info.get('debtToEquity', 0) or 0

            # Growth rate (from dividend or earnings history)
            growth_rate = self._calculate_growth_rate()

            # Volatility
            volatility = self._calculate_volatility()

            # Beta estimation vs market (S&P 500)
            beta_info = self._estimate_beta(market_ticker='^GSPC', period=self.period)
            if beta_info is None:
                beta = None
                beta_r2 = None
            else:
                beta, beta_r2 = beta_info

            return StockParams(
                ticker=self.ticker,
                current_price=float(current_price),
                dividend_yield=float(dividend_yield),
                dividend_per_share=float(dividend_per_share),
                growth_rate=float(growth_rate),
                volatility=float(volatility),
                beta=float(beta) if beta is not None else None,
                beta_r2=float(beta_r2) if beta_r2 is not None else None,
                pe_ratio=float(pe_ratio),
                payout_ratio=float(payout_ratio),
                earnings_per_share=float(eps),
                book_value_per_share=float(book_value_per_share),
                debt_to_equity=float(debt_to_equity)
            )

        # execute on separate thread with manual timeout to avoid deadlock
        import threading

        result_container = {}

        def _worker():
            try:
                result_container['result'] = _raw_fetch()
            except Exception as e:
                result_container['error'] = e

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join(timeout=20)
        if thread.is_alive():
            # thread will be killed when program exits; we just give up
            raise ValueError(f"Timeout fetching data for {self.ticker}")
        if 'error' in result_container:
            raise result_container['error']
        return result_container.get('result')
    
    def _get_dividend_per_share(self) -> float:
        """Get annual dividend per share"""
        try:
            # Try from info first (most recent)
            div = self.stock.info.get('trailingAnnualDividendRate', None)
            if div and div > 0:
                return float(div)
            
            # Calculate from dividend history
            dividends = self.stock.dividends
            if len(dividends) > 0:
                # Sum of last 4 quarters
                recent_divs = dividends.tail(4)
                return float(recent_divs.sum())
            
            return 0.0
        
        except Exception:
            return 0.0

    def _refresh_quote_via_http(self) -> None:
        """Attempt to fetch basic quote data using the public Yahoo JSON API.

        This is much lighter than yfinance.history() and rarely hangs, so we
        use it as our primary source for current price / yield information.  If
        it succeeds we merge the returned fields into ``self.stock.info`` so
        the rest of the normal workflow can continue unchanged.
        """

        import requests

        url = f'https://query1.finance.yahoo.com/v7/finance/quote?symbols={self.ticker}'
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if (
            not data
            or 'quoteResponse' not in data
            or not data['quoteResponse'].get('result')
        ):
            raise ValueError('no quote data')

        q = data['quoteResponse']['result'][0]
        # merge useful keys into stock.info (overwrite stale values)
        for k, v in q.items():
            if v is not None:
                self.stock.info[k] = v

    
    def _get_book_value_per_share(self) -> float:
        """Get book value per share"""
        try:
            bvps = self.stock.info.get('bookValue', None)
            if bvps and bvps > 0:
                return float(bvps)
            
            # Try to calculate from balance sheet
            balance_sheet = self.stock.balance_sheet
            if balance_sheet is not None and len(balance_sheet.columns) > 0:
                try:
                    equity = balance_sheet.loc['Stockholders Equity', balance_sheet.columns[0]]
                    shares_outstanding = self.stock.info.get('sharesOutstanding', None)
                    if equity > 0 and shares_outstanding and shares_outstanding > 0:
                        return float(equity / shares_outstanding)
                except:
                    pass
            
            # Estimate from P/E and current price
            eps = self.stock.info.get('trailingEps', 0)
            pe = self.stock.info.get('trailingPE', 0)
            if eps > 0 and pe > 0:
                return float((eps * pe) / 2)  # Conservative estimate
            
            return 0.0
        
        except Exception:
            return 0.0
    
    def _calculate_payout_ratio(self, dividend_per_share: float, eps: float) -> float:
        """Calculate dividend payout ratio"""
        if eps <= 0 or dividend_per_share <= 0:
            return 0.0
        
        payout = dividend_per_share / eps
        return min(1.0, payout)  # Cap at 100%
    
    def _calculate_growth_rate(self) -> float:
        """Calculate historical dividend or earnings growth rate"""
        try:
            # Try dividend history first
            dividends = self.stock.dividends
            if len(dividends) > 4:
                # Calculate CAGR over available history
                div_history = dividends.tail(12)  # Last year monthly
                if len(div_history) > 4:
                    old_div = div_history.iloc[0]
                    new_div = div_history.iloc[-1]
                    if old_div > 0:
                        months = len(div_history)
                        growth = (new_div / old_div) ** (1 / (months / 12)) - 1
                        return max(-0.20, min(0.30, growth))  # Reasonable bounds
            
            # Calculate from earnings history
            earnings_history = self.stock.quarterly_financials
            if earnings_history is not None and len(earnings_history.columns) > 4:
                try:
                    # Get earnings data
                    earnings_key = 'Net Income' if 'Net Income' in earnings_history.index else None
                    if earnings_key:
                        earnings = earnings_history.loc[earnings_key]
                        earnings = earnings[earnings > 0]  # Remove negative values
                        
                        if len(earnings) > 4:
                            old_earn = earnings.iloc[-1]
                            new_earn = earnings.iloc[0]
                            if old_earn > 0:
                                quarters = len(earnings)
                                growth = (new_earn / old_earn) ** (1 / (quarters / 4)) - 1
                                return max(-0.20, min(0.30, growth))
                except:
                    pass
            
            # Default to conservative growth rate
            return 0.05  # 5% default
        
        except Exception:
            return 0.05  # Default to 5%
    
    def _calculate_volatility(self) -> float:
        """Calculate historical volatility"""
        try:
            history = self.stock.history(period=self.period)
            
            if len(history) > 0:
                # Calculate daily returns
                returns = history['Close'].pct_change().dropna()
                
                if len(returns) > 0:
                    # Annualize volatility
                    daily_vol = returns.std()
                    annual_vol = daily_vol * np.sqrt(252)  # 252 trading days
                    return float(annual_vol)
            
            return 0.20  # Default volatility
        
        except Exception:
            return 0.20
    
    def get_dividend_history(self, years: int = 5) -> pd.Series:
        """Get historical dividends"""
        try:
            dividends = self.stock.dividends
            if len(dividends) > 0:
                cutoff_date = datetime.now() - timedelta(days=years * 365)
                return dividends[dividends.index >= cutoff_date]
            return pd.Series()
        except Exception:
            return pd.Series()
    
    def get_price_history(self, period: str = "1y") -> pd.DataFrame:
        """Get historical prices"""
        try:
            return self.stock.history(period=period)
        except Exception:
            return pd.DataFrame()

    def _estimate_beta(self, market_ticker: str = '^GSPC', period: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """
        Estimate beta by regressing stock returns on market returns.

        Args:
            market_ticker: Market index ticker to use as benchmark (default S&P500 '^GSPC')
            period: yfinance history period (e.g., '1y', '5y'); defaults to self.period

        Returns:
            Tuple of (beta, r2) or None if insufficient data
        """
        try:
            use_period = period or self.period
            stock_hist = self.stock.history(period=use_period)[['Close']].rename(columns={'Close': 'stock_close'})

            market = yf.Ticker(market_ticker)
            market_hist = market.history(period=use_period)[['Close']].rename(columns={'Close': 'market_close'})

            if stock_hist.empty or market_hist.empty:
                return None

            # Align by date
            df = stock_hist.join(market_hist, how='inner')
            if df.shape[0] < 60:
                # Not enough data for stable beta
                return None

            # Compute daily returns
            df['r_stock'] = df['stock_close'].pct_change()
            df['r_market'] = df['market_close'].pct_change()
            df = df.dropna()

            if df.shape[0] < 60:
                return None

            x = df['r_market'].values
            y = df['r_stock'].values

            # Use covariance method to compute beta
            cov = np.cov(x, y, ddof=1)
            beta = cov[0, 1] / cov[0, 0] if cov[0, 0] != 0 else None
            if beta is None or np.isnan(beta):
                # Fallback to linear regression
                try:
                    beta = np.polyfit(x, y, 1)[0]
                except Exception:
                    return None

            # Compute R² as correlation squared
            correlation = np.corrcoef(x, y)[0, 1]
            r2 = correlation ** 2 if not np.isnan(correlation) else 0.0

            return (float(beta), float(r2))

        except Exception:
            return None


class InterestRateDataCollector:
    """Collect interest rate data for model calibration"""
    
    def __init__(self):
        """Initialize interest rate data collector"""
        self.treasury_ticker = '^TNX'  # 10-year treasury yield
        self.fed_rate_ticker = '^FVX'  # 5-year treasury yield
    
    def get_current_rates(self) -> Dict:
        """Get current interest rates"""
        try:
            tnx = yf.Ticker(self.treasury_ticker)
            fvx = yf.Ticker(self.fed_rate_ticker)
            
            # Get current values
            tnx_price = tnx.info.get('regularMarketPrice', 2.5) / 100
            fvx_price = fvx.info.get('regularMarketPrice', 2.0) / 100
            
            return {
                'risk_free_rate': float(fvx_price),  # 5-year as risk-free
                'long_term_rate': float(tnx_price),  # 10-year for long-term expectations
                'curve_slope': float(tnx_price - fvx_price)
            }
        except Exception:
            # Return defaults if can't fetch
            return {
                'risk_free_rate': 0.04,
                'long_term_rate': 0.045,
                'curve_slope': 0.005
            }
    
    def calibrate_ir_model(self, model_type: str = "CIR") -> InterestRateParams:
        """
        Calibrate interest rate model parameters from current conditions
        
        Args:
            model_type: "CIR" or "Vasicek"
            
        Returns:
            Calibrated InterestRateParams
        """
        rates = self.get_current_rates()
        current_rate = rates['risk_free_rate']
        long_term_rate = rates['long_term_rate']
        
        # Simple calibration
        kappa = 0.15  # Mean reversion speed (moderate)
        theta = long_term_rate  # Long-term mean equals long rate
        sigma = 0.01  # Volatility (10 bps per year)
        
        return InterestRateParams(
            r0=current_rate,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
            model_type=model_type
        )


class StockScreener:
    """Screen for dividend-paying stocks with desired characteristics"""
    
    @staticmethod
    def is_dividend_stock(ticker: str) -> bool:
        """Check if stock pays dividends"""
        try:
            stock = yf.Ticker(ticker)
            dividend = stock.info.get('trailingAnnualDividendRate', 0)
            return dividend > 0
        except:
            return False
    
    @staticmethod
    def screen_stocks(tickers: list, 
                     min_dividend_yield: float = 0.02,
                     max_pe_ratio: float = 25,
                     min_payout_ratio: float = 0.0,
                     max_debt_to_equity: float = 2.0) -> list:
        """
        Screen stocks based on criteria
        
        Returns:
            List of tickers meeting criteria
        """
        qualifying_stocks = []
        
        for ticker in tickers:
            try:
                collector = StockDataCollector(ticker)
                stock_data = collector.fetch_stock_data()
                
                # Apply filters
                if stock_data.dividend_yield < min_dividend_yield:
                    continue
                if stock_data.pe_ratio > max_pe_ratio and stock_data.pe_ratio > 0:
                    continue
                if stock_data.payout_ratio > 0.95:  # Over-leveraged dividends
                    continue
                if stock_data.debt_to_equity > max_debt_to_equity:
                    continue
                
                qualifying_stocks.append(stock_data)
            
            except Exception as e:
                print(f"Error screening {ticker}: {e}")
                continue
        
        # Sort by yield
        qualifying_stocks.sort(
            key=lambda x: x.dividend_yield,
            reverse=True
        )
        
        return qualifying_stocks
