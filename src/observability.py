"""
Observability and Prediction Logging Module
Records model inference metrics locally without storing sensitive user texts.
Strictly adheres to classical NLP tracking: latency, confidence, word count,
input source, and model version. (No fake LLM token metrics).
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger("Observability")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOGS_DIR / "prediction_logs.csv"

CSV_HEADERS = [
    "timestamp",
    "prediction",
    "confidence",
    "real_probability",
    "fake_probability",
    "word_count",
    "char_count",
    "processing_time",
    "input_source",
    "model_version",
]


def initialize_log_file(log_file: Path = DEFAULT_LOG_FILE) -> None:
    """Ensure the logs directory and CSV file with header exist."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if not log_file.is_file():
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        logger.info("Initialized prediction log file at %s", log_file)


def log_prediction_event(
    prediction: str,
    confidence: float,
    real_probability: float,
    fake_probability: float,
    word_count: int,
    char_count: int,
    processing_time: float,
    input_source: str = "text",
    model_version: str = "v1.0.0-tfidf-logreg",
    log_file: Path = DEFAULT_LOG_FILE,
) -> None:
    """
    Append an anonymized prediction event to the local CSV log.
    Does NOT log article text or any personally identifiable information.

    Args:
        prediction: "REAL" or "FAKE".
        confidence: Model confidence (0.0 to 1.0).
        real_probability: Probability of REAL class.
        fake_probability: Probability of FAKE class.
        word_count: Number of words in the evaluated article.
        char_count: Number of characters in the evaluated article.
        processing_time: Latency in seconds.
        input_source: "text" or "url".
        model_version: Version identifier of the model.
        log_file: Destination CSV path.
    """
    try:
        initialize_log_file(log_file)
        iso_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        row = [
            iso_timestamp,
            prediction,
            f"{confidence:.4f}",
            f"{real_probability:.4f}",
            f"{fake_probability:.4f}",
            word_count,
            char_count,
            f"{processing_time:.4f}",
            input_source,
            model_version,
        ]

        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        logger.debug("Logged prediction event: %s (%s)", prediction, confidence)
    except Exception as e:
        logger.error("Failed to log prediction event: %s", e)


def load_prediction_logs(log_file: Path = DEFAULT_LOG_FILE) -> pd.DataFrame:
    """
    Load logged predictions into a pandas DataFrame.
    Returns empty DataFrame with standard headers if log file doesn't exist.
    """
    if not log_file.is_file():
        return pd.DataFrame(columns=CSV_HEADERS)

    try:
        df = pd.read_csv(log_file)
        # Ensure numerical types
        for col in ["confidence", "real_probability", "fake_probability", "processing_time"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ["word_count", "char_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df
    except Exception as e:
        logger.error("Error reading prediction log file: %s", e)
        return pd.DataFrame(columns=CSV_HEADERS)


def get_analytics_metrics(log_file: Path = DEFAULT_LOG_FILE) -> Dict[str, Any]:
    """
    Aggregate observability metrics from the prediction logs.
    """
    df = load_prediction_logs(log_file)

    if df.empty:
        return {
            "total_predictions": 0,
            "real_count": 0,
            "fake_count": 0,
            "real_pct": 0.0,
            "fake_pct": 0.0,
            "avg_confidence": 0.0,
            "avg_latency": 0.0,
            "avg_word_count": 0,
            "recent_logs": pd.DataFrame(),
        }

    total = len(df)
    real_count = int((df["prediction"] == "REAL").sum())
    fake_count = int((df["prediction"] == "FAKE").sum())

    real_pct = round((real_count / total) * 100, 1) if total > 0 else 0.0
    fake_pct = round((fake_count / total) * 100, 1) if total > 0 else 0.0

    avg_conf = float(df["confidence"].mean()) if not df["confidence"].isna().all() else 0.0
    avg_latency = float(df["processing_time"].mean()) if not df["processing_time"].isna().all() else 0.0
    avg_words = int(df["word_count"].mean()) if not df["word_count"].isna().all() else 0

    # Last 50 predictions sorted newest first
    recent_logs = df.iloc[::-1].head(50)

    return {
        "total_predictions": total,
        "real_count": real_count,
        "fake_count": fake_count,
        "real_pct": real_pct,
        "fake_pct": fake_pct,
        "avg_confidence": round(avg_conf, 4),
        "avg_latency": round(avg_latency, 4),
        "avg_word_count": avg_words,
        "recent_logs": recent_logs,
    }
