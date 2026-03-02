"""
News Aggregator - Fetch articles from certified news sources
Uses NewsAPI for verified, credible sources
"""
import requests
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os

# Certified news sources with high credibility ratings
CERTIFIED_SOURCES = [
    'financial-times', 'reuters', 'bbc-news', 'bloomberg', 'cnbc', 'financial-times',
    'wall-street-journal', 'marketwatch', 'investor-place', 'seeking-alpha'
]

class NewsAggregator:
    """Fetch news articles from certified sources"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize news aggregator
        
        Args:
            api_key: NewsAPI key (optional, can be set via NEWS_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        self.base_url = 'https://newsapi.org/v2'
        self.sources = CERTIFIED_SOURCES
    
    def get_articles(self, ticker: str, days: int = 7) -> List[Dict]:
        """
        Get recent articles about a stock ticker from certified sources
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back (default 7)
            
        Returns:
            List of articles with title, description, source, date
        """
        if not self.api_key:
            print("Warning: NEWS_API_KEY not set. Returning empty articles.")
            return []
        
        articles = []
        
        try:
            # Get company name from ticker
            company_names = self._get_company_name(ticker)
            
            # Search for each company name variant
            for company_name in company_names:
                # Try fetching from everything endpoint with source filter
                params = {
                    'q': f'"{company_name}" OR {ticker}',
                    'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                    'sortBy': 'publishedAt',
                    'language': 'en',
                    'apiKey': self.api_key,
                    'pageSize': 20
                }
                
                response = requests.get(f'{self.base_url}/everything', params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    for article in data.get('articles', []):
                        # Verify source is reputable
                        source = article.get('source', {}).get('name', '').lower()
                        if self._is_certified_source(source):
                            articles.append({
                                'title': article.get('title'),
                                'description': article.get('description'),
                                'source': article.get('source', {}).get('name'),
                                'published_at': article.get('publishedAt'),
                                'url': article.get('url'),
                                'content': article.get('content')
                            })
                
            # Remove duplicates
            unique_articles = {a['url']: a for a in articles}
            return list(unique_articles.values())[:30]  # Top 30 articles
            
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []
    
    def _get_company_name(self, ticker: str) -> List[str]:
        """Map ticker to company name"""
        ticker_map = {
            'AAPL': ['Apple'], 'MSFT': ['Microsoft'], 'GOOGL': ['Google', 'Alphabet'],
            'AMZN': ['Amazon'], 'NVDA': ['NVIDIA'], 'TSLA': ['Tesla'], 
            'META': ['Meta', 'Facebook'], 'NFLX': ['Netflix'], 'INTC': ['Intel'],
            'AMD': ['AMD'], 'CRM': ['Salesforce'], 'ADBE': ['Adobe'],
            'CSCO': ['Cisco'], 'PYPL': ['PayPal'], 'SQ': ['Block', 'Square'],
            'UBER': ['Uber'], 'DASH': ['DoorDash'], 'SPOT': ['Spotify'],
            'COIN': ['Coinbase'], 'PLTR': ['Palantir'], 'ZM': ['Zoom']
        }
        return ticker_map.get(ticker, [ticker])
    
    def _is_certified_source(self, source: str) -> bool:
        """Check if source is from certified list"""
        source_lower = source.lower()
        for certified in CERTIFIED_SOURCES:
            if certified.lower() in source_lower:
                return True
        
        # Additional trusted sources
        trusted_keywords = ['financial times', 'reuters', 'bbc', 'bloomberg', 'cnbc', 
                          'wall street journal', 'marketwatch', 'investor', 'seeking alpha',
                          'yahoo', 'cnbc', 'nasdaq']
        
        for keyword in trusted_keywords:
            if keyword in source_lower:
                return True
        
        return False
