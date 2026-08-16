import logging
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SentimentEngine:
    """
    FinBERT NLP Financial Sentiment Analysis Engine.
    Uses HuggingFace ProsusAI/finbert (or yiyanghkust/finbert-tone) to compute
    market sentiment probabilities and generate subjective allocation view vectors.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", use_transformer: bool = True):
        self.model_name = model_name
        self.use_transformer = use_transformer
        self.pipeline = None
        self._init_model()

    def _init_model(self):
        """Initializes HuggingFace FinBERT pipeline if available and enabled."""
        if not self.use_transformer:
            logger.info("Transformer option disabled. Using Financial Lexicon Sentiment Analyzer.")
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            logger.info(f"Loading HuggingFace FinBERT pipeline: {self.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)
            logger.info("FinBERT model initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not load HuggingFace FinBERT transformer ({e}). Defaulting to Lexicon engine.")
            self.pipeline = None

    def analyze_headlines(self, headlines: List[Union[str, Dict[str, str]]]) -> Dict[str, Any]:
        """
        Analyzes a list of news headlines or dicts containing 'headline' keys.
        
        Returns:
            Dict containing individual scores, average positive/negative/neutral probabilities,
            composite sentiment tilt score \in [-1.0, 1.0], and qualitative sentiment label.
        """
        texts = []
        for h in headlines:
            if isinstance(h, dict):
                texts.append(h.get("headline", ""))
            else:
                texts.append(str(h))

        texts = [t.strip() for t in texts if t.strip()]

        if not texts:
            return self._neutral_sentiment_response()

        if self.pipeline is not None:
            return self._analyze_with_finbert(texts)
        else:
            return self._analyze_with_lexicon(texts)

    def _analyze_with_finbert(self, texts: List[str]) -> Dict[str, Any]:
        """Performs batch inference using HuggingFace FinBERT pipeline."""
        try:
            results = self.pipeline(texts)
            parsed_scores = []
            
            pos_sum, neg_sum, neu_sum = 0.0, 0.0, 0.0
            
            for text, res in zip(texts, results):
                # Format: [{'label': 'positive', 'score': 0.9}, ...]
                score_dict = {item['label'].lower(): item['score'] for item in res}
                pos = score_dict.get('positive', 0.0)
                neg = score_dict.get('negative', 0.0)
                neu = score_dict.get('neutral', 0.0)

                pos_sum += pos
                neg_sum += neg
                neu_sum += neu

                # Headline net score
                net_score = pos - neg
                parsed_scores.append({
                    "text": text,
                    "positive": pos,
                    "negative": neg,
                    "neutral": neu,
                    "net_score": net_score
                })

            n = len(texts)
            avg_pos = pos_sum / n
            avg_neg = neg_sum / n
            avg_neu = neu_sum / n
            composite_score = avg_pos - avg_neg  # Range [-1, 1]

            label = "Bullish" if composite_score > 0.15 else ("Bearish" if composite_score < -0.15 else "Neutral")

            return {
                "engine": "FinBERT (HuggingFace)",
                "composite_score": float(composite_score),
                "label": label,
                "avg_positive": float(avg_pos),
                "avg_negative": float(avg_neg),
                "avg_neutral": float(avg_neu),
                "headline_count": n,
                "details": parsed_scores
            }

        except Exception as e:
            logger.error(f"Error during FinBERT inference: {e}")
            return self._analyze_with_lexicon(texts)

    def _analyze_with_lexicon(self, texts: List[str]) -> Dict[str, Any]:
        """Fast financial domain lexicon sentiment analyzer fallback."""
        bullish_words = {
            "surge", "surpasses", "high", "growth", "inflow", "inflows", "bullish", "record",
            "jump", "gain", "gains", "boost", "outperform", "rally", "profit", "uptrend",
            "expansion", "upbeat", "robust", "resilient", "buy"
        }
        bearish_words = {
            "pressure", "slowdown", "spikes", "inflation", "drop", "decline", "fall",
            "concern", "concerns", "bearish", "loss", "slump", "plunge", "risk", "risks",
            "cut", "deficit", "sell", "warns", "warning", "tension", "recession"
        }

        parsed_scores = []
        pos_sum, neg_sum, neu_sum = 0.0, 0.0, 0.0

        for text in texts:
            words = set(re.findall(r'\w+', text.lower()))
            b_count = len(words.intersection(bullish_words))
            r_count = len(words.intersection(bearish_words))
            total = b_count + r_count

            if total > 0:
                pos = b_count / (total + 1.0)
                neg = r_count / (total + 1.0)
                neu = 1.0 - (pos + neg)
            else:
                pos, neg, neu = 0.1, 0.1, 0.8

            pos_sum += pos
            neg_sum += neg
            neu_sum += neu
            parsed_scores.append({
                "text": text,
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "net_score": pos - neg
            })

        n = len(texts)
        avg_pos = pos_sum / n
        avg_neg = neg_sum / n
        avg_neu = neu_sum / n
        composite_score = avg_pos - avg_neg

        label = "Bullish" if composite_score > 0.10 else ("Bearish" if composite_score < -0.10 else "Neutral")

        return {
            "engine": "Financial Lexicon Engine",
            "composite_score": float(composite_score),
            "label": label,
            "avg_positive": float(avg_pos),
            "avg_negative": float(avg_neg),
            "avg_neutral": float(avg_neu),
            "headline_count": n,
            "details": parsed_scores
        }

    def _neutral_sentiment_response(self) -> Dict[str, Any]:
        return {
            "engine": "Neutral Fallback",
            "composite_score": 0.0,
            "label": "Neutral",
            "avg_positive": 0.0,
            "avg_negative": 0.0,
            "avg_neutral": 1.0,
            "headline_count": 0,
            "details": []
        }
