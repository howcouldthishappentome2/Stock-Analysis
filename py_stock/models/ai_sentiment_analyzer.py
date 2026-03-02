"""
AI Sentiment Analysis - Extract sentiment and keywords from news articles
Uses OpenAI API with temperature=0 for deterministic results
Validates results with 3-check consistency mechanism

Enhanced with:
  - AI bubble dependency scoring: analyzes whether a company's news
    is heavily focused on AI hype rather than core business fundamentals
  - Anti-bubble screen: returns a 0-100 score where higher = more AI-bubble-dependent
"""
import os
from typing import List, Dict, Optional, Tuple
import json

# Try to import transformers for fallback
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Try to import openai
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ── AI bubble dependency keyword sets ────────────────────────────────────────
# Words that suggest a company is deeply tied to AI hype / bubble
AI_BUBBLE_KEYWORDS = {
    'high': [
        'generative ai', 'large language model', 'llm', 'foundation model',
        'ai infrastructure', 'gpu demand', 'nvidia chips', 'ai datacenter',
        'ai capex', 'ai investment', 'ai revenue', 'ai-driven growth',
        'openai partnership', 'chatgpt integration', 'co-pilot', 'ai adoption',
        'ai monetization', 'ai spending', 'hyperscaler', 'ai workloads',
    ],
    'medium': [
        'artificial intelligence', 'machine learning', 'deep learning',
        'neural network', 'ai product', 'ai feature', 'ai assistant',
        'cloud ai', 'ai platform', 'ai solution', 'ai transformation',
        'digital transformation', 'ai-powered', 'ai-enabled',
    ],
    'low': [
        'automation', 'analytics', 'data science', 'predictive', 'algorithm',
        'smart', 'intelligent', 'ai tool',
    ],
}

# Words that suggest business is driven by NON-AI fundamentals
ANTI_BUBBLE_KEYWORDS = [
    'cybersecurity', 'compliance', 'regulation', 'gdpr', 'hipaa',
    'government contract', 'defense contract', 'federal', 'healthcare it',
    'payment processing', 'embedded system', 'industrial', 'manufacturing',
    'optical', 'broadband', 'grid', 'utility', 'smart meter',
    'property management', 'payroll', 'accounting software', 'erp',
    'recurring revenue', 'subscription', 'saas', 'maintenance contract',
    'dividend', 'buyback', 'cash flow', 'free cash flow',
]


class AINewsSentimentAnalyzer:
    """Analyze news sentiment and extract keywords using AI"""

    def __init__(self, use_openai: bool = True):
        """
        Initialize sentiment analyzer.

        Args:
            use_openai: Use OpenAI API (requires OPENAI_API_KEY env var).
                       Falls back to transformers/keyword analysis if unavailable.
        """
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.use_openai = use_openai and self.openai_key is not None

        if not self.use_openai and HAS_TRANSFORMERS:
            try:
                self.sentiment_pipeline = pipeline(
                    'sentiment-analysis',
                    model='distilbert-base-uncased-finetuned-sst-2-english',
                    device=-1  # CPU
                )
            except Exception as e:
                print(f"Warning: Could not load transformers pipeline: {e}")
                self.sentiment_pipeline = None
        else:
            self.sentiment_pipeline = None

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_articles(self, articles: List[Dict], ticker: str) -> Dict:
        """
        Analyze articles for sentiment and keywords with 3-check validation.

        Returns:
            Dict with sentiment, keywords, validation status, and
            ai_bubble_dependency_score (0-100, higher = more AI-bubble-dependent).
        """
        if not articles:
            return self._empty_result()

        results = []
        for _ in range(3):
            result = self._single_analysis_pass(articles, ticker)
            results.append(result)

        validated = self._validate_results(results, ticker)

        # Add AI bubble dependency score (always computed, fast)
        validated['ai_bubble_dependency_score'] = self._score_ai_bubble_dependency(articles)

        return validated

    def score_ai_bubble_dependency(self, articles: List[Dict]) -> float:
        """
        Standalone method: returns AI bubble dependency score (0-100) from articles.
        0 = no AI bubble exposure, 100 = entirely AI-hype driven.
        Used by the anti-bubble scanner for fast Stage-2 filtering.
        """
        return self._score_ai_bubble_dependency(articles)

    # ── AI bubble scoring ──────────────────────────────────────────────────────

    def _score_ai_bubble_dependency(self, articles: List[Dict]) -> float:
        """
        Score how much of this company's recent news revolves around AI bubble themes
        vs. fundamental business drivers.

        Scoring logic:
          - Each high-signal AI keyword hit:   +8 points
          - Each medium-signal AI keyword hit: +4 points
          - Each low-signal AI keyword hit:    +2 points
          - Each anti-bubble keyword hit:      -3 points
          - Score averaged across articles, capped at 0-100

        A score above 40 indicates meaningful AI-bubble dependency.
        A score above 65 indicates the stock is primarily valued on AI hype.
        """
        if not articles:
            return 0.0

        article_scores = []

        for article in articles:
            text = (
                (article.get('title') or '') + ' ' +
                (article.get('description') or '') + ' ' +
                (article.get('content') or '')
            ).lower()

            score = 0.0

            for kw in AI_BUBBLE_KEYWORDS['high']:
                if kw in text:
                    score += 8.0

            for kw in AI_BUBBLE_KEYWORDS['medium']:
                if kw in text:
                    score += 4.0

            for kw in AI_BUBBLE_KEYWORDS['low']:
                if kw in text:
                    score += 2.0

            for kw in ANTI_BUBBLE_KEYWORDS:
                if kw in text:
                    score -= 3.0

            article_scores.append(score)

        raw = sum(article_scores) / len(article_scores) if article_scores else 0.0
        # Normalize: a raw score of ~30 maps to ~100
        normalized = min(100.0, max(0.0, raw * (100.0 / 30.0)))
        return round(normalized, 1)

    # ── Standard sentiment analysis ────────────────────────────────────────────

    def _single_analysis_pass(self, articles: List[Dict], ticker: str) -> Dict:
        """Single pass of sentiment analysis."""
        if self.use_openai:
            return self._analyze_with_openai(articles, ticker)
        elif HAS_TRANSFORMERS and self.sentiment_pipeline:
            return self._analyze_with_transformers(articles, ticker)
        else:
            return self._basic_keyword_analysis(articles)

    def _analyze_with_openai(self, articles: List[Dict], ticker: str) -> Dict:
        """Analyze using OpenAI API with temperature=0 for determinism."""
        if not HAS_OPENAI:
            return self._analyze_with_transformers(articles, ticker)
        if not self.openai_key:
            return self._analyze_with_transformers(articles, ticker)

        try:
            openai.api_key = self.openai_key

            article_text = "\n\n".join([
                f"Title: {a.get('title', '')}\n"
                f"Description: {a.get('description', '')}\n"
                f"Source: {a.get('source', '')}"
                for a in articles[:10]
            ])

            prompt = f"""Analyze the following news articles about {ticker} and provide:
1. Overall sentiment (bullish/neutral/bearish)
2. Sentiment score (-1.0 to 1.0)
3. Top 5 keywords
4. Key trends
5. Whether the articles suggest the company's revenue/growth is primarily AI-hype driven
   (answer: "ai_dependent", "mixed", or "fundamental")

Articles:
{article_text}

Respond ONLY in this JSON format:
{{
    "sentiment": "bullish|neutral|bearish",
    "sentiment_score": float,
    "keywords": [list of 5 keywords],
    "trends": [list of trends],
    "ai_dependency": "ai_dependent|mixed|fundamental"
}}"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a financial sentiment analyst specializing in identifying "
                            "whether tech companies are genuinely profitable businesses vs. "
                            "AI-bubble-dependent valuation stories. Be precise and consistent."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=300,
            )

            content = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            result = json.loads(content)
            result['method'] = 'openai'
            result['article_count'] = len(articles)
            return result

        except Exception as e:
            print(f"OpenAI error for {ticker}: {e}")
            return self._basic_keyword_analysis(articles)

    def _analyze_with_transformers(self, articles: List[Dict], ticker: str) -> Dict:
        """Analyze using transformers pipeline as fallback."""
        try:
            texts = [
                f"{a.get('title', '')} {a.get('description', '')}"
                for a in articles[:10]
            ]
            texts = [t[:512] for t in texts if t.strip()]

            if not texts:
                return self._basic_keyword_analysis(articles)

            results = self.sentiment_pipeline(texts)
            keywords = set()
            for article in articles:
                text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
                for word in text.split():
                    if len(word) > 4:
                        keywords.add(word.strip('.,;:'))

            scores = []
            for r in results:
                if r['label'] == 'POSITIVE':
                    scores.append(r['score'])
                else:
                    scores.append(-r['score'])

            avg_sentiment = sum(scores) / len(scores) if scores else 0.0
            if avg_sentiment > 0.1:
                sentiment_label = 'bullish'
            elif avg_sentiment < -0.1:
                sentiment_label = 'bearish'
            else:
                sentiment_label = 'neutral'

            return {
                'sentiment': sentiment_label,
                'sentiment_score': float(avg_sentiment),
                'keywords': list(keywords)[:5],
                'trends': [],
                'method': 'transformers',
                'article_count': len(articles),
            }

        except Exception as e:
            print(f"Transformers error: {e}")
            return self._basic_keyword_analysis(articles)

    def _basic_keyword_analysis(self, articles: List[Dict]) -> Dict:
        """Fallback: simple keyword-based sentiment analysis."""
        keywords = set()
        sentiment_indicators = {'positive': 0, 'negative': 0, 'neutral': 0}

        positive_words = ['surge', 'soar', 'gain', 'profit', 'beat', 'strong', 'growth', 'innovation']
        negative_words = ['drop', 'crash', 'loss', 'miss', 'decline', 'weak', 'challenge']

        for article in articles:
            text = (article.get('title', '') + ' ' + article.get('description', '')).lower()

            for word in text.split():
                if len(word) > 4:
                    keywords.add(word.strip('.,;:'))

            pos_score = sum(1 for w in positive_words if w in text)
            neg_score = sum(1 for w in negative_words if w in text)

            if pos_score > neg_score:
                sentiment_indicators['positive'] += 1
            elif neg_score > pos_score:
                sentiment_indicators['negative'] += 1
            else:
                sentiment_indicators['neutral'] += 1

        if sentiment_indicators['positive'] > sentiment_indicators['negative']:
            sentiment = 'bullish'
            score = min(0.5, sentiment_indicators['positive'] / max(1, len(articles)))
        elif sentiment_indicators['negative'] > sentiment_indicators['positive']:
            sentiment = 'bearish'
            score = -min(0.5, sentiment_indicators['negative'] / max(1, len(articles)))
        else:
            sentiment = 'neutral'
            score = 0.0

        return {
            'sentiment': sentiment,
            'sentiment_score': float(score),
            'keywords': list(keywords)[:5],
            'trends': [],
            'method': 'keyword',
            'article_count': len(articles),
        }

    def _validate_results(self, results: List[Dict], ticker: str) -> Dict:
        """
        Validate consistency across 3 runs.
        Returns confident result if all 3 agree, otherwise returns consensus.
        """
        if not results:
            return self._empty_result()

        sentiments = [r['sentiment'] for r in results]
        sentiment_mode = max(set(sentiments), key=sentiments.count)
        sentiment_agreement = sentiments.count(sentiment_mode) / len(sentiments)
        avg_score = sum(r['sentiment_score'] for r in results) / len(results)

        all_keywords = []
        for r in results:
            all_keywords.extend(r.get('keywords', []))
        keyword_counts: Dict[str, int] = {}
        for kw in all_keywords:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
        consistent_keywords = [kw for kw, count in keyword_counts.items() if count >= 2]

        validation_passed = sentiment_agreement == 1.0
        confidence = sentiment_agreement

        return {
            'sentiment': sentiment_mode,
            'sentiment_score': float(avg_score),
            'keywords': consistent_keywords[:5],
            'trends': [],
            'confidence': float(confidence),
            'validation_passed': validation_passed,
            'validation_agreement': f"{sentiment_agreement * 100:.0f}%",
            'article_count': sum(r.get('article_count', 0) for r in results[:1]),
            'analysis_runs': 3,
            'ai_bubble_dependency_score': 0.0,  # Will be overwritten by caller
        }

    @staticmethod
    def _empty_result() -> Dict:
        return {
            'sentiment': 'neutral',
            'sentiment_score': 0.0,
            'keywords': [],
            'trends': [],
            'confidence': 0.0,
            'validation_passed': False,
            'article_count': 0,
            'ai_bubble_dependency_score': 0.0,
        }