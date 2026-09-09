"""
Automated Test Suite for Fake News Detection System
Verifies preprocessing, inference pipeline, article extraction, observability, and schema integrity.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.article_extractor import validate_url
from src.observability import (
    CSV_HEADERS,
    get_analytics_metrics,
    initialize_log_file,
    load_prediction_logs,
    log_prediction_event,
)
from src.predict import (
    DEFAULT_MODEL_PATH,
    DEFAULT_VECTORIZER_PATH,
    NewsPredictor,
    predict_news,
)
from src.preprocess import clean_text, combine_title_and_text


# =====================================================================
# 1. Model Artifact Verification Tests
# =====================================================================
def test_model_files_exist():
    """Verify that model artifacts exist in the models/ directory."""
    assert DEFAULT_MODEL_PATH.is_file(), f"Missing model artifact: {DEFAULT_MODEL_PATH}"
    assert DEFAULT_VECTORIZER_PATH.is_file(), f"Missing vectorizer artifact: {DEFAULT_VECTORIZER_PATH}"


# =====================================================================
# 2. Text Preprocessing Tests
# =====================================================================
def test_clean_text_basic():
    """Verify lowercase, URL removal, HTML removal, and whitespace normalization."""
    raw = "<p>BREAKING: Check https://example.com/news for details!!!   </p>"
    cleaned = clean_text(raw)
    assert "<p>" not in cleaned
    assert "</p>" not in cleaned
    assert "http" not in cleaned
    assert "breaking check for details" in cleaned
    assert "   " not in cleaned


def test_clean_text_none_and_empty():
    """Verify graceful handling of None and whitespace-only strings."""
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("   \n\t  ") == ""


def test_combine_title_and_text():
    """Verify title and body combination."""
    combined = combine_title_and_text("Breaking Headline", "The full story unfolds.")
    assert combined == "breaking headline the full story unfolds"

    assert combine_title_and_text("Headline Only", None) == "headline only"
    assert combine_title_and_text(None, "Body Only") == "body only"
    assert combine_title_and_text(None, None) == ""


# =====================================================================
# 3. Prediction Pipeline Tests
# =====================================================================
def test_prediction_function_works():
    """Verify that inference runs and outputs a valid result."""
    sample_text = (
        "The Federal Reserve announced steady interest rates following their monthly economic review."
    )
    result = predict_news(text=sample_text, title="Economic Update")

    assert isinstance(result, dict)
    assert result["label"] in ["REAL", "FAKE"]
    assert result["error"] is None


def test_prediction_output_schema():
    """Verify that all required fields are present with correct data types."""
    sample = "Whistleblower reveals top secret documents concerning national infrastructure."
    result = predict_news(text=sample, title="National News")

    required_keys = [
        "label",
        "confidence",
        "real_probability",
        "fake_probability",
        "word_count",
        "char_count",
        "processing_time",
        "cleaned_text",
    ]
    for key in required_keys:
        assert key in result, f"Result dictionary missing required key: {key}"

    assert isinstance(result["label"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["real_probability"], float)
    assert isinstance(result["fake_probability"], float)
    assert isinstance(result["word_count"], int)
    assert isinstance(result["char_count"], int)
    assert isinstance(result["processing_time"], float)


def test_confidence_and_probabilities_boundaries():
    """Verify that confidence and probabilities obey mathematical boundaries: 0 <= P <= 1 and sum ≈ 1."""
    sample = "Scientists at the Antarctic research station report findings on ice sheet melting."
    result = predict_news(text=sample)

    assert 0.0 <= result["confidence"] <= 1.0, "Confidence is outside [0, 1] range."
    assert 0.0 <= result["real_probability"] <= 1.0, "Real probability is outside [0, 1] range."
    assert 0.0 <= result["fake_probability"] <= 1.0, "Fake probability is outside [0, 1] range."

    # Sum of probabilities should equal ~1.0
    total_prob = result["real_probability"] + result["fake_probability"]
    assert math.isclose(total_prob, 1.0, abs_tol=1e-3), f"Probabilities do not sum to 1: {total_prob}"

    # Confidence must equal max(real_probability, fake_probability)
    expected_conf = max(result["real_probability"], result["fake_probability"])
    assert math.isclose(result["confidence"], expected_conf, abs_tol=1e-3)


def test_empty_input_handled_safely():
    """Verify that empty or whitespace-only inputs do not cause crashes."""
    result = predict_news(text="", title="")
    assert result["label"] == "UNKNOWN"
    assert result["confidence"] == 0.0
    assert result["real_probability"] == 0.0
    assert result["fake_probability"] == 0.0
    assert result["error"] is not None


# =====================================================================
# 4. URL Validator Tests
# =====================================================================
def test_url_validation():
    """Verify URL validator handles valid and invalid URLs correctly."""
    valid, _ = validate_url("https://www.reuters.com/world/article")
    assert valid is True

    valid_http, _ = validate_url("http://news.bbc.co.uk/science")
    assert valid_http is True

    invalid_proto, err = validate_url("ftp://ftp.example.com/file")
    assert invalid_proto is False
    assert "http" in err.lower()

    invalid_empty, _ = validate_url("")
    assert invalid_empty is False

    invalid_local, _ = validate_url("http://localhost:8000/news")
    assert invalid_local is False


# =====================================================================
# 5. Observability & Logging Tests
# =====================================================================
def test_observability_logging(tmp_path):
    """Verify logging events, loading logs, and computing analytics."""
    test_log_file = tmp_path / "test_prediction_logs.csv"

    # Initialize log
    initialize_log_file(test_log_file)
    assert test_log_file.is_file()

    # Log two events
    log_prediction_event(
        prediction="REAL",
        confidence=0.95,
        real_probability=0.95,
        fake_probability=0.05,
        word_count=200,
        char_count=1200,
        processing_time=0.045,
        input_source="text",
        log_file=test_log_file,
    )

    log_prediction_event(
        prediction="FAKE",
        confidence=0.88,
        real_probability=0.12,
        fake_probability=0.88,
        word_count=150,
        char_count=900,
        processing_time=0.038,
        input_source="url",
        log_file=test_log_file,
    )

    # Load and verify
    df = load_prediction_logs(test_log_file)
    assert len(df) == 2
    assert list(df.columns) == CSV_HEADERS

    # Check metrics aggregation
    metrics = get_analytics_metrics(test_log_file)
    assert metrics["total_predictions"] == 2
    assert metrics["real_count"] == 1
    assert metrics["fake_count"] == 1
    assert metrics["real_pct"] == 50.0
    assert metrics["fake_pct"] == 50.0
    assert metrics["avg_confidence"] > 0.90
