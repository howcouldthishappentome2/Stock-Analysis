"""
News-Enhanced Recommendation Engine
Integrates AI sentiment analysis from news articles with stock recommendations
"""
from typing import Optional, Dict
from data.news_aggregator import NewsAggregator
from models.ai_sentiment_analyzer import AINewsSentimentAnalyzer
from models.recommendation_engine import TradingRecommendation, ActionType

class NewsEnhancedRecommendationEngine:
    """Enhance recommendations with news sentiment analysis"""
    
    def __init__(self, use_ai: bool = True):
        """
        Initialize news-enhanced engine
        
        Args:
            use_ai: Use AI for sentiment analysis (tries OpenAI first, falls back to transformers)
        """
        self.news_aggregator = NewsAggregator()
        self.sentiment_analyzer = AINewsSentimentAnalyzer(use_openai=use_ai)
    
    def enhance_recommendation(self, 
                              recommendation: TradingRecommendation,
                              days_lookback: int = 7) -> TradingRecommendation:
        """
        Enhance existing recommendation with news sentiment analysis
        
        Args:
            recommendation: Original TradingRecommendation
            days_lookback: Number of days to analyze news (default 7)
            
        Returns:
            TradingRecommendation with news sentiment fields populated
        """
        try:
            # Fetch recent articles
            articles = self.news_aggregator.get_articles(recommendation.ticker, days=days_lookback)
            
            if not articles:
                return recommendation
            
            # Analyze sentiment with 3-check validation
            sentiment_result = self.sentiment_analyzer.analyze_articles(articles, recommendation.ticker)
            
            # Update recommendation with news data
            recommendation.news_sentiment = sentiment_result.get('sentiment', 'neutral')
            recommendation.news_sentiment_score = sentiment_result.get('sentiment_score', 0.0)
            recommendation.news_keywords = sentiment_result.get('keywords', [])
            
            # Determine validation status
            confidence = sentiment_result.get('confidence', 0.0)
            if confidence == 1.0:
                recommendation.news_validation_status = 'validated'
            elif confidence >= 0.67:
                recommendation.news_validation_status = 'moderate'
            else:
                recommendation.news_validation_status = 'low'
            
            # Adjust confidence if news sentiment strongly deviates from technical analysis
            self._adjust_confidence_for_news(recommendation, sentiment_result)
            
        except Exception as e:
            print(f"Warning: Could not enhance recommendation with news: {e}")
        
        return recommendation
    
    def _adjust_confidence_for_news(self, 
                                   recommendation: TradingRecommendation,
                                   sentiment_result: Dict) -> None:
        """
        Adjust recommendation confidence based on news sentiment
        
        Args:
            recommendation: TradingRecommendation to adjust
            sentiment_result: Sentiment analysis result
        """
        news_sentiment = sentiment_result.get('sentiment', 'neutral')
        news_score = sentiment_result.get('sentiment_score', 0.0)
        validation_passed = sentiment_result.get('validation_passed', False)
        
        if not validation_passed:
            # Low validation, don't adjust much
            adjustment = 0.0
        else:
            # Check alignment between recommendation and news
            if recommendation.action in [ActionType.STRONG_BUY, ActionType.BUY]:
                # We're bullish, check if news agrees
                if news_sentiment == 'bullish':
                    adjustment = 0.10  # Boost confidence
                elif news_sentiment == 'bearish':
                    adjustment = -0.20  # Reduce confidence
                else:
                    adjustment = 0.0  # Neutral
            
            elif recommendation.action in [ActionType.SELL, ActionType.STRONG_SELL]:
                # We're bearish, check if news agrees
                if news_sentiment == 'bearish':
                    adjustment = 0.10  # Boost confidence
                elif news_sentiment == 'bullish':
                    adjustment = -0.20  # Reduce confidence
                else:
                    adjustment = 0.0
            
            else:  # HOLD
                adjustment = 0.0
        
        # Apply adjustment
        if adjustment != 0:
            recommendation.confidence = max(0.0, min(1.0, recommendation.confidence + adjustment))
    
    def get_news_summary(self, ticker: str, days: int = 7) -> Dict:
        """
        Get news summary for a ticker without full recommendation
        
        Args:
            ticker: Stock ticker
            days: Number of days to analyze
            
        Returns:
            Dictionary with sentiment, keywords, and article count
        """
        articles = self.news_aggregator.get_articles(ticker, days=days)
        
        if not articles:
            return {
                'ticker': ticker,
                'sentiment': 'no_articles',
                'article_count': 0,
                'keywords': []
            }
        
        sentiment_result = self.sentiment_analyzer.analyze_articles(articles, ticker)
        
        return {
            'ticker': ticker,
            'sentiment': sentiment_result.get('sentiment'),
            'sentiment_score': sentiment_result.get('sentiment_score'),
            'confidence': sentiment_result.get('confidence'),
            'validation_passed': sentiment_result.get('validation_passed'),
            'keywords': sentiment_result.get('keywords'),
            'article_count': sentiment_result.get('article_count'),
            'validation_status': 'validated' if sentiment_result.get('validation_passed') else 'moderate'
        }
