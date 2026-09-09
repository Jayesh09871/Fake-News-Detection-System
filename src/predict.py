"""
Prediction and Inference Pipeline
Loads trained TF-IDF vectorizer and Logistic Regression classifier to evaluate news articles.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

from src.preprocess import clean_text, combine_title_and_text

logger = logging.getLogger("Predictor")

# Default model directory resolution relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_VECTORIZER_PATH = DEFAULT_MODELS_DIR / "tfidf_vectorizer.joblib"
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "fake_news_model.joblib"


class ModelNotFoundError(FileNotFoundError):
    """Raised when model artifacts are missing from the local models/ directory."""
    pass


class NewsPredictor:
    """
    Singleton-style predictor that caches trained model artifacts in memory.
    """
    _instance: Optional[NewsPredictor] = None

    def __init__(
        self,
        vectorizer_path: Path = DEFAULT_VECTORIZER_PATH,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self.vectorizer_path = Path(vectorizer_path)
        self.model_path = Path(model_path)
        self.vectorizer = None
        self.model = None
        self._is_loaded = False

    @classmethod
    def get_instance(
        cls,
        vectorizer_path: Path = DEFAULT_VECTORIZER_PATH,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> NewsPredictor:
        if cls._instance is None:
            cls._instance = cls(vectorizer_path, model_path)
        return cls._instance

    def are_artifacts_present(self) -> bool:
        """Check if both required joblib artifacts exist locally."""
        return self.vectorizer_path.is_file() and self.model_path.is_file()

    def load(self, force_reload: bool = False) -> None:
        """
        Load vectorizer and model from disk into memory.
        Raises ModelNotFoundError if files do not exist.
        """
        if self._is_loaded and not force_reload:
            return

        if not self.are_artifacts_present():
            missing = []
            if not self.vectorizer_path.is_file():
                missing.append(str(self.vectorizer_path))
            if not self.model_path.is_file():
                missing.append(str(self.model_path))

            raise ModelNotFoundError(
                f"Model artifacts missing: {missing}.\n"
                "Please run cloud training on Kaggle as documented in cloud/README.md "
                "and download 'tfidf_vectorizer.joblib' and 'fake_news_model.joblib' into the 'models/' folder."
            )

        logger.info("Loading TF-IDF vectorizer from %s", self.vectorizer_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

        logger.info("Loading Logistic Regression model from %s", self.model_path)
        self.model = joblib.load(self.model_path)

        self._is_loaded = True

        # Dynamically determine which probability index corresponds to REAL vs FAKE
        # In benchmark news datasets, reputable journalistic wire services (e.g. 'reuters')
        # are strongly associated with the REAL news class.
        self.real_class_idx = 1
        self.fake_class_idx = 0
        if hasattr(self.model, "coef_") and hasattr(self.vectorizer, "vocabulary_"):
            reuters_idx = self.vectorizer.vocabulary_.get("reuters")
            if reuters_idx is not None and self.model.coef_.shape[1] > reuters_idx:
                if self.model.coef_[0][reuters_idx] < 0:
                    self.real_class_idx = 0
                    self.fake_class_idx = 1
                else:
                    self.real_class_idx = 1
                    self.fake_class_idx = 0

        logger.info(
            "NewsPredictor artifacts successfully loaded (REAL=class_%d, FAKE=class_%d).",
            self.real_class_idx,
            self.fake_class_idx,
        )

    def predict(
        self,
        text: str,
        title: Optional[str] = "",
    ) -> Dict[str, Any]:
        """
        Run inference on a given news article.

        Args:
            text: Article body text.
            title: Optional article headline/title.

        Returns:
            Dictionary matching project specification:
            {
                "label": "REAL" | "FAKE",
                "confidence": float,
                "real_probability": float,
                "fake_probability": float,
                "word_count": int,
                "char_count": int,
                "processing_time": float,
                "cleaned_text": str
            }
        """
        start_time = time.perf_counter()

        # Handle raw text metrics before cleaning
        raw_combined = f"{title or ''} {text or ''}".strip()
        word_count = len(raw_combined.split()) if raw_combined else 0
        char_count = len(raw_combined)

        # Preprocessing matching training identically
        combined_cleaned = combine_title_and_text(title, text)

        # Handle empty input gracefully
        if not combined_cleaned.strip():
            elapsed = time.perf_counter() - start_time
            return {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "real_probability": 0.0,
                "fake_probability": 0.0,
                "word_count": word_count,
                "char_count": char_count,
                "processing_time": round(elapsed, 4),
                "cleaned_text": "",
                "error": "Article text is empty after cleaning. Please provide valid text content.",
            }

        # Ensure models are loaded
        if not self._is_loaded:
            self.load()

        # Vectorization
        features = self.vectorizer.transform([combined_cleaned])

        # Prediction probabilities using calibrated class mapping
        probs = self.model.predict_proba(features)[0]
        real_prob = float(probs[self.real_class_idx])
        fake_prob = float(probs[self.fake_class_idx])

        # Classification decision
        if real_prob >= fake_prob:
            label = "REAL"
            confidence = real_prob
        else:
            label = "FAKE"
            confidence = fake_prob

        elapsed = time.perf_counter() - start_time

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "real_probability": round(real_prob, 4),
            "fake_probability": round(fake_prob, 4),
            "word_count": word_count,
            "char_count": char_count,
            "processing_time": round(elapsed, 4),
            "cleaned_text": combined_cleaned,
            "error": None,
        }


def predict_news(
    text: str,
    title: Optional[str] = "",
    vectorizer_path: Path = DEFAULT_VECTORIZER_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Convenience function for standalone prediction.
    """
    predictor = NewsPredictor.get_instance(vectorizer_path, model_path)
    return predictor.predict(text=text, title=title)
