"""
Stock Data Cache - SQLite-based caching system for fast stock data retrieval
"""
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from models.stock_params import StockParams


class StockDataCache:
    """SQLite-based cache for stock data"""
    
    DB_PATH = Path("stock_cache.db")
    CACHE_VALIDITY_HOURS = 1  # Refresh data older than 1 hour
    
    def __init__(self):
        """Initialize cache database"""
        # check_same_thread=False allows use across threads (but requires careful synchronization)
        self.conn = sqlite3.connect(str(self.DB_PATH), check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Create cache tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Main stock data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                ticker TEXT PRIMARY KEY,
                current_price REAL,
                dividend_yield REAL,
                dividend_per_share REAL,
                growth_rate REAL,
                volatility REAL,
                beta REAL,
                beta_r2 REAL,
                pe_ratio REAL,
                payout_ratio REAL,
                earnings_per_share REAL,
                book_value_per_share REAL,
                debt_to_equity REAL,
                data_json TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Analysis cache table (for scan results)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_cache (
                ticker TEXT PRIMARY KEY,
                fair_value REAL,
                upside_downside REAL,
                recommendation TEXT,
                confidence REAL,
                probability_of_profit REAL,
                holding_months INTEGER,
                expected_return REAL,
                exit_signal TEXT,
                key_risk TEXT,
                analysis_json TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indices for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON stock_data(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON analysis_cache(timestamp)")
        
        self.conn.commit()
    
    def get_stock_data(self, ticker: str) -> Optional[StockParams]:
        """Get cached stock data"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM stock_data WHERE ticker = ?", (ticker.upper(),))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Check if data is still valid
        timestamp = datetime.fromisoformat(row['timestamp'])
        if datetime.now() - timestamp > timedelta(hours=self.CACHE_VALIDITY_HOURS):
            return None  # Data is stale
        
        # Reconstruct StockParams from cached data
        return StockParams(
            ticker=row['ticker'],
            current_price=row['current_price'],
            dividend_yield=row['dividend_yield'],
            dividend_per_share=row['dividend_per_share'],
            growth_rate=row['growth_rate'],
            volatility=row['volatility'],
            beta=row['beta'],
            beta_r2=row['beta_r2'],
            pe_ratio=row['pe_ratio'],
            payout_ratio=row['payout_ratio'],
            earnings_per_share=row['earnings_per_share'],
            book_value_per_share=row['book_value_per_share'],
            debt_to_equity=row['debt_to_equity']
        )
    
    def cache_stock_data(self, stock: StockParams):
        """Cache stock data"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO stock_data 
            (ticker, current_price, dividend_yield, dividend_per_share, growth_rate, 
             volatility, pe_ratio, payout_ratio, earnings_per_share, book_value_per_share, 
             beta, beta_r2, debt_to_equity, data_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stock.ticker,
            stock.current_price,
            stock.dividend_yield,
            stock.dividend_per_share,
            stock.growth_rate,
            stock.volatility,
            stock.pe_ratio,
            stock.payout_ratio,
            stock.earnings_per_share,
            stock.book_value_per_share,
            stock.beta,
            stock.beta_r2,
            stock.debt_to_equity,
            json.dumps(stock.__dict__),
            datetime.now().isoformat()
        ))
        
        self.conn.commit()
    
    def get_analysis(self, ticker: str) -> Optional[Dict]:
        """Get cached analysis results"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM analysis_cache WHERE ticker = ?", (ticker.upper(),))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Check if analysis is still valid
        timestamp = datetime.fromisoformat(row['timestamp'])
        if datetime.now() - timestamp > timedelta(hours=self.CACHE_VALIDITY_HOURS):
            return None
        # Return parsed analysis_json so any extra fields (beta, etc.) are preserved
        try:
            analysis = json.loads(row['analysis_json']) if row['analysis_json'] else {}
            # Ensure ticker present
            analysis.setdefault('ticker', row['ticker'])
            return analysis
        except Exception:
            # Fallback to legacy fields
            return {
                'ticker': row['ticker'],
                'fair_value': row['fair_value'],
                'upside_downside': row['upside_downside'],
                'recommendation': row['recommendation'],
                'confidence': row['confidence'],
                'probability_of_profit': row['probability_of_profit'],
                'holding_months': row['holding_months'],
                'expected_return': row['expected_return'],
                'exit_signal': row['exit_signal'],
                'key_risk': row['key_risk']
            }
    
    def cache_analysis(self, ticker: str, analysis: Dict):
        """Cache analysis results"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO analysis_cache
            (ticker, fair_value, upside_downside, recommendation, confidence,
             probability_of_profit, holding_months, expected_return, exit_signal,
             key_risk, analysis_json, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker.upper(),
            analysis.get('fair_value'),
            analysis.get('upside_downside'),
            analysis.get('recommendation'),
            analysis.get('confidence'),
            analysis.get('probability_of_profit'),
            analysis.get('holding_months'),
            analysis.get('expected_return'),
            analysis.get('exit_signal'),
            analysis.get('key_risk'),
            json.dumps(analysis),
            datetime.now().isoformat()
        ))
        
        self.conn.commit()
    
    def clear_old_data(self, hours: int = 24):
        """Clear cache data older than specified hours"""
        cursor = self.conn.cursor()
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute("DELETE FROM stock_data WHERE timestamp < ?", (cutoff_time,))
        cursor.execute("DELETE FROM analysis_cache WHERE timestamp < ?", (cutoff_time,))
        
        self.conn.commit()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as count FROM stock_data")
        stock_count = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM analysis_cache")
        analysis_count = cursor.fetchone()['count']
        
        cursor.execute("SELECT SUM(size) as db_size FROM (SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size())")
        db_size = cursor.fetchone()['db_size']
        
        return {
            'cached_stocks': stock_count,
            'cached_analyses': analysis_count,
            'db_size_mb': db_size / (1024 * 1024) if db_size else 0
        }
    
    def close(self):
        """Close database connection"""
        self.conn.close()
