import math
import logging
import re
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """News Sentiment Analyzer using HuggingFace FinBERT and exponential time decay.

    Loads the `ProsusAI/finbert` NLP transformer model to classify financial news headlines
    into positive, negative, and neutral sentiment probabilities. Calculates a composite
    weighted sentiment score using an exponential time-decay mechanism based on headline age.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        use_transformer: bool = True,
    ) -> None:
        """Initializes the NewsSentimentAnalyzer with the specified FinBERT model.

        Args:
            model_name: HuggingFace model identifier (default: 'ProsusAI/finbert').
            use_transformer: Whether to attempt loading the transformer pipeline.
        """
        self.model_name = model_name
        self.use_transformer = use_transformer
        self.pipeline = None
        self._init_model()

    def _init_model(self) -> None:
        """Initializes the HuggingFace text-classification pipeline if available."""
        if not self.use_transformer:
            logger.info("Transformer pipeline disabled. Defaulting to Lexicon Sentiment Engine.")
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            logger.info(f"Loading HuggingFace FinBERT model: {self.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                return_all_scores=True,
            )
            logger.info("FinBERT NLP pipeline initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not load HuggingFace FinBERT pipeline ({e}). Defaulting to Lexicon Engine.")
            self.pipeline = None

    def score_headlines(self, headlines: list[dict], decay_lambda: float = 0.1) -> float:
        """Scores a list of financial news headlines applying exponential time decay weighting.

        Each headline dictionary must contain a 'headline' string and a 'timestamp' string in ISO format.
        FinBERT computes sentiment probabilities. The net sentiment score per headline is:
            score = (+1.0 * prob_positive) + (-1.0 * prob_negative) + (0.0 * prob_neutral)
        Exponential time decay applies weight = exp(-decay_lambda * age_in_days).
        Returns the weighted average sentiment score across all headlines.

        Args:
            headlines: List of dictionaries containing 'headline' and 'timestamp' ISO strings.
            decay_lambda: Time decay parameter lambda (default: 0.1).

        Returns:
            Weighted composite sentiment score as a float in [-1.0, 1.0]. Returns 0.0 if empty or error.
        """
        if not headlines or not isinstance(headlines, list):
            return 0.0

        now = datetime.datetime.now()
        total_weighted_score = 0.0
        total_weight = 0.0

        try:
            for item in headlines:
                if not isinstance(item, dict):
                    continue

                text = item.get("headline", "").strip()
                if not text:
                    continue

                # Parse timestamp and compute age in days
                timestamp_str = item.get("timestamp") or item.get("pub_date") or ""
                age_in_days = self._calculate_age_in_days(timestamp_str, now)

                # Compute unweighted sentiment score
                raw_score = self._predict_single_headline_score(text)

                # Calculate exponential time decay weight: weight = exp(-decay_lambda * age_in_days)
                weight = math.exp(-decay_lambda * age_in_days)

                total_weighted_score += weight * raw_score
                total_weight += weight

            if total_weight <= 0.0:
                return 0.0

            composite_score = float(total_weighted_score / total_weight)
            return float(np.clip(composite_score, -1.0, 1.0))

        except Exception as e:
            logger.error(f"Error scoring headlines in score_headlines: {e}")
            return 0.0

    def analyze_headlines(
        self, headlines: List[Union[str, Dict[str, str]]]
    ) -> Dict[str, Any]:
        """Analyzes a list of news headlines and returns detailed sentiment diagnostics.

        Args:
            headlines: List of string headlines or dictionary objects.

        Returns:
            Dictionary containing composite score, label, probabilities, and headline diagnostics.
        """
        parsed_items = []
        now_str = datetime.datetime.now().isoformat()

        for h in headlines:
            if isinstance(h, dict):
                text = h.get("headline") or h.get("clean_headline") or ""
                ts = h.get("timestamp") or h.get("pub_date") or now_str
                if text.strip():
                    parsed_items.append({"headline": text.strip(), "timestamp": ts})
            elif isinstance(h, str) and h.strip():
                parsed_items.append({"headline": h.strip(), "timestamp": now_str})

        if not parsed_items:
            return self._neutral_sentiment_response()

        composite_score = self.score_headlines(parsed_items, decay_lambda=0.1)

        label = (
            "Bullish"
            if composite_score > 0.10
            else ("Bearish" if composite_score < -0.10 else "Neutral")
        )

        details = []
        pos_sum, neg_sum, neu_sum = 0.0, 0.0, 0.0
        n = len(parsed_items)

        for item in parsed_items:
            text = item["headline"]
            score = self._predict_single_headline_score(text)
            pos = max(0.0, score) if score > 0 else 0.0
            neg = abs(score) if score < 0 else 0.0
            neu = 1.0 - (pos + neg)

            pos_sum += pos
            neg_sum += neg
            neu_sum += neu

            details.append(
                {
                    "text": text,
                    "positive": float(pos),
                    "negative": float(neg),
                    "neutral": float(neu),
                    "net_score": float(score),
                }
            )

        engine_name = (
            "FinBERT (HuggingFace)"
            if self.pipeline is not None
            else "Financial Lexicon Engine"
        )

        return {
            "engine": engine_name,
            "composite_score": float(composite_score),
            "label": label,
            "avg_positive": float(pos_sum / n),
            "avg_negative": float(neg_sum / n),
            "avg_neutral": float(neu_sum / n),
            "headline_count": n,
            "details": details,
        }

    def _predict_single_headline_score(self, text: str) -> float:
        """Predicts net sentiment score for a single headline (+1*pos - 1*neg)."""
        if self.pipeline is not None:
            try:
                res = self.pipeline([text])[0]
                score_dict = {item["label"].lower(): item["score"] for item in res}
                pos = score_dict.get("positive", 0.0)
                neg = score_dict.get("negative", 0.0)
                return float(pos - neg)
            except Exception as e:
                logger.debug(f"Transformer single headline prediction failed ({e}). Falling back to Lexicon.")

        return self._lexicon_single_headline_score(text)

    def _lexicon_single_headline_score(self, text: str) -> float:
        """Lexicon rule-based net sentiment score fallback."""
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

        words = set(re.findall(r"\w+", text.lower()))
        b_count = len(words.intersection(bullish_words))
        r_count = len(words.intersection(bearish_words))
        total = b_count + r_count

        if total > 0:
            pos = b_count / (total + 1.0)
            neg = r_count / (total + 1.0)
            return float(pos - neg)
        return 0.0

    def _calculate_age_in_days(self, timestamp_str: str, current_time: datetime.datetime) -> float:
        """Calculates headline age in days relative to current time."""
        if not timestamp_str:
            return 0.0

        try:
            # Handle ISO format strings and standard date strings
            ts = pd.to_datetime(timestamp_str, errors="coerce")
            if pd.isna(ts):
                return 0.0
            
            # Make naive for uniform subtraction
            if hasattr(ts, "tz_localize") and ts.tzinfo is not None:
                ts = ts.tz_convert(None)

            delta_seconds = (current_time - ts.to_pydatetime()).total_seconds()
            age_days = delta_seconds / 86400.0
            return max(0.0, float(age_days))
        except Exception:
            return 0.0

    def _neutral_sentiment_response(self) -> Dict[str, Any]:
        """Returns default neutral response structure."""
        return {
            "engine": "Neutral Fallback",
            "composite_score": 0.0,
            "label": "Neutral",
            "avg_positive": 0.0,
            "avg_negative": 0.0,
            "avg_neutral": 1.0,
            "headline_count": 0,
            "details": [],
        }


# Class Alias for Backwards Compatibility
SentimentEngine = NewsSentimentAnalyzer

