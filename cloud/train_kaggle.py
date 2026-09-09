"""
Kaggle Cloud Training Script for Fake News Detection
Dataset: WELFake Dataset (saurabhshahane/fake-news-classification)
Target: 0 = FAKE, 1 = REAL

Architecture:
1. Robustly discovers WELFake CSV in /kaggle/input or custom search paths.
2. Performs deterministic text cleaning, null handling, and deduplication.
3. Stratified Split: 80% Training, 10% Validation, 10% Testing (random_state=42).
4. Strict Featurization: Fits TF-IDF (100,000 features, unigrams+bigrams) on train data only.
5. Trains Logistic Regression (balanced class weights, C=2.0).
6. Evaluates comprehensively on test split (Accuracy, F1, Precision, Recall, Confusion Matrix).
7. Exports artifacts: fake_news_model.joblib, tfidf_vectorizer.joblib, evaluation.json, confusion_matrix.png.
"""

from __future__ import annotations

import html
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless cloud execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("KaggleTrainer")


# =====================================================================
# 1. Deterministic Preprocessing Routines (Self-Contained for Kaggle)
# =====================================================================
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<.*?>", re.DOTALL)
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    """Deterministic string cleaning matching src/preprocess.py identically."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        return ""

    cleaned = html.unescape(text)
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.lower()
    cleaned = SPECIAL_CHAR_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def combine_title_and_text(title: Optional[str] = "", text: Optional[str] = "") -> str:
    """Combine title and article text deterministically."""
    clean_t = clean_text(title)
    clean_b = clean_text(text)
    if clean_t and clean_b:
        return f"{clean_t} {clean_b}"
    elif clean_t:
        return clean_t
    elif clean_b:
        return clean_b
    return ""


# =====================================================================
# 2. Dynamic Dataset Discovery
# =====================================================================
def find_welfake_dataset(base_dirs: Optional[List[str]] = None) -> Path:
    """
    Search candidate directories for a CSV file matching the WELFake schema:
    required columns: 'title', 'text', 'label'.
    """
    if base_dirs is None:
        base_dirs = [
            "/kaggle/input",
            "./data",
            "../data",
            ".",
        ]

    logger.info("Searching for WELFake dataset in candidate directories: %s", base_dirs)
    candidate_files: List[Path] = []

    for base_dir in base_dirs:
        p = Path(base_dir)
        if p.exists():
            # Recursively find all CSV files
            for csv_file in p.glob("**/*.csv"):
                candidate_files.append(csv_file)

    for csv_file in candidate_files:
        try:
            # Peek header to verify columns
            header = pd.read_csv(csv_file, nrows=2).columns.str.strip().str.lower().tolist()
            if "title" in header and "text" in header and "label" in header:
                logger.info("Found matching WELFake dataset: %s", csv_file)
                return csv_file
        except Exception as e:
            logger.debug("Skipping %s due to read error: %s", csv_file, e)

    raise FileNotFoundError(
        "Could not find a CSV dataset with required columns ['title', 'text', 'label']. "
        "Ensure the WELFake dataset (saurabhshahane/fake-news-classification) is attached to the Kaggle session."
    )


# =====================================================================
# 3. Data Ingestion & Cleaning
# =====================================================================
def load_and_preprocess_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load WELFake CSV, clean texts, handle missing values, and deduplicate.
    """
    logger.info("Loading dataset from: %s", csv_path)
    start_time = time.time()
    df = pd.read_csv(csv_path)
    raw_count = len(df)
    logger.info("Raw records loaded: %d (in %.2fs)", raw_count, time.time() - start_time)

    # Normalize column names
    col_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=col_map)

    # Verify required columns
    for col in ["title", "text", "label"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from dataset.")

    # Fill NaNs
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Filter invalid labels (labels must be 0 or 1)
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    logger.info("Applying deterministic text cleaning and combination...")
    clean_start = time.time()
    df["combined_text"] = [
        combine_title_and_text(t, b) for t, b in zip(df["title"], df["text"])
    ]
    logger.info("Text cleaning completed in %.2fs", time.time() - clean_start)

    # Drop empty records
    before_empty = len(df)
    df = df[df["combined_text"].str.strip() != ""].copy()
    empty_dropped = before_empty - len(df)

    # Drop duplicate articles to prevent data leakage across splits
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["combined_text"]).copy()
    duplicates_dropped = before_dedup - len(df)

    fake_count = int((df["label"] == 0).sum())
    real_count = int((df["label"] == 1).sum())

    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info("Total Raw Rows:          %d", raw_count)
    logger.info("Empty Articles Dropped:  %d", empty_dropped)
    logger.info("Duplicates Dropped:      %d", duplicates_dropped)
    logger.info("Cleaned Total Samples:   %d", len(df))
    logger.info("  - FAKE Samples (0):    %d (%.2f%%)", fake_count, (fake_count / len(df)) * 100)
    logger.info("  - REAL Samples (1):    %d (%.2f%%)", real_count, (real_count / len(df)) * 100)
    logger.info("=" * 60)

    return df


# =====================================================================
# 4. Stratified Dataset Split
# =====================================================================
def split_data(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Perform an 80% train / 10% val / 10% test stratified split.
    """
    X = df["combined_text"]
    y = df["label"]

    # First split: 80% train, 20% temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    # Second split: split temp equally into 10% val, 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    logger.info("=" * 60)
    logger.info("DATA SPLIT SUMMARY (Stratified)")
    logger.info("=" * 60)
    logger.info("Training Set:   %d samples (%.1f%%)", len(X_train), (len(X_train) / len(df)) * 100)
    logger.info("Validation Set: %d samples (%.1f%%)", len(X_val), (len(X_val) / len(df)) * 100)
    logger.info("Test Set:       %d samples (%.1f%%)", len(X_test), (len(X_test) / len(df)) * 100)
    logger.info("=" * 60)

    return X_train, X_val, X_test, y_train, y_val, y_test


# =====================================================================
# 5. Training Pipeline: TF-IDF + Logistic Regression
# =====================================================================
def train_pipeline(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    random_state: int = 42,
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """
    Strict featurization ordering: fit TF-IDF ONLY on training data.
    Train Logistic Regression with balanced weights.
    """
    logger.info("Initializing TfidfVectorizer (max_features=100000, ngram_range=(1,2), sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(
        max_features=100000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    t0 = time.time()
    logger.info("Fitting TF-IDF Vectorizer on training set...")
    X_train_vec = vectorizer.fit_transform(X_train)
    logger.info("TF-IDF fit completed in %.2fs. Vocabulary size: %d", time.time() - t0, len(vectorizer.vocabulary_))

    logger.info("Initializing LogisticRegression(C=2.0, class_weight='balanced', max_iter=1000)...")
    model = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1,
    )

    t1 = time.time()
    logger.info("Training Logistic Regression classifier...")
    model.fit(X_train_vec, y_train)
    logger.info("Model training completed in %.2fs", time.time() - t1)

    # Validation check
    logger.info("Evaluating on validation set...")
    X_val_vec = vectorizer.transform(X_val)
    val_preds = model.predict(X_val_vec)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    logger.info("Validation Accuracy: %.4f | Validation F1 (macro): %.4f", val_acc, val_f1)

    return vectorizer, model


# =====================================================================
# 6. Evaluation & Artifact Export
# =====================================================================
def evaluate_and_export(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    X_test: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> Dict:
    """
    Evaluate trained model on the unseen test set and persist artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Transforming test set using fitted vectorizer...")
    t0 = time.time()
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    inference_time = time.time() - t0

    # Calculate overall metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    macro_prec = float(precision_score(y_test, y_pred, average="macro"))
    macro_rec = float(recall_score(y_test, y_pred, average="macro"))

    # Per-class metrics: 0 = FAKE, 1 = REAL
    fake_prec = float(precision_score(y_test, y_pred, pos_label=0))
    fake_rec = float(recall_score(y_test, y_pred, pos_label=0))
    fake_f1 = float(f1_score(y_test, y_pred, pos_label=0))

    real_prec = float(precision_score(y_test, y_pred, pos_label=1))
    real_rec = float(recall_score(y_test, y_pred, pos_label=1))
    real_f1 = float(f1_score(y_test, y_pred, pos_label=1))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    report_str = classification_report(
        y_test, y_pred, target_names=["FAKE (0)", "REAL (1)"], digits=4
    )

    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("Overall Accuracy:       %.4f (%.2f%%)", acc, acc * 100)
    logger.info("Macro F1-Score:         %.4f", macro_f1)
    logger.info("Macro Precision:        %.4f", macro_prec)
    logger.info("Macro Recall:           %.4f", macro_rec)
    logger.info("-" * 60)
    logger.info("FAKE Class (0) -> Precision: %.4f | Recall: %.4f | F1: %.4f", fake_prec, fake_rec, fake_f1)
    logger.info("REAL Class (1) -> Precision: %.4f | Recall: %.4f | F1: %.4f", real_prec, real_rec, real_f1)
    logger.info("-" * 60)
    logger.info("Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d", tn, fp, fn, tp)
    logger.info("\nFull Classification Report:\n%s", report_str)
    logger.info("=" * 60)

    # Structure metrics dictionary
    evaluation_data = {
        "model_name": "TF-IDF + Logistic Regression",
        "dataset": "WELFake Dataset (saurabhshahane/fake-news-classification)",
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_samples": int(len(y_test)),
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "real_class": {
                "label": "REAL",
                "code": 1,
                "precision": round(real_prec, 4),
                "recall": round(real_rec, 4),
                "f1_score": round(real_f1, 4),
                "support": int((y_test == 1).sum()),
            },
            "fake_class": {
                "label": "FAKE",
                "code": 0,
                "precision": round(fake_prec, 4),
                "recall": round(fake_rec, 4),
                "f1_score": round(fake_f1, 4),
                "support": int((y_test == 0).sum()),
            },
        },
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "matrix": [[tn, fp], [fn, tp]],
            "labels": ["FAKE", "REAL"],
        },
        "disclaimer": (
            "These metrics represent statistical classification performance on the historical "
            "WELFake benchmark test split. Predictions reflect learned linguistic patterns and "
            "do not constitute absolute factual truth or external fact-checking verification."
        ),
    }

    # Save evaluation.json
    eval_path = output_dir / "evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=2)
    logger.info("Saved evaluation metrics to: %s", eval_path)

    # Save readable summary report
    summary_path = output_dir / "metrics_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"=== FAKE NEWS DETECTOR MODEL EVALUATION ===\n")
        f.write(f"Evaluated on {len(y_test)} unseen test samples.\n\n")
        f.write(f"Accuracy:        {acc:.4f}\n")
        f.write(f"Macro F1:        {macro_f1:.4f}\n")
        f.write(f"Macro Precision: {macro_prec:.4f}\n")
        f.write(f"Macro Recall:    {macro_rec:.4f}\n\n")
        f.write("Per-Class Performance:\n")
        f.write(f"  FAKE -> Precision: {fake_prec:.4f}, Recall: {fake_rec:.4f}, F1: {fake_f1:.4f}\n")
        f.write(f"  REAL -> Precision: {real_prec:.4f}, Recall: {real_rec:.4f}, F1: {real_f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  [TN={tn}, FP={fp}]\n  [FN={fn}, TP={tp}]\n\n")
        f.write(f"Classification Report:\n{report_str}\n")
    logger.info("Saved readable summary to: %s", summary_path)

    # Generate and save confusion matrix plot
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
        fig.colorbar(cax)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14,
                    weight="bold",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["FAKE", "REAL"], fontsize=11)
        ax.set_yticklabels(["FAKE", "REAL"], fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("WELFake Test Set Confusion Matrix", fontsize=13, pad=15)
        plt.tight_layout()
        cm_path = output_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=200)
        plt.close(fig)
        logger.info("Saved confusion matrix image to: %s", cm_path)
    except Exception as e:
        logger.warning("Could not generate confusion matrix plot: %s", e)

    # Save model artifacts using joblib
    vectorizer_path = output_dir / "tfidf_vectorizer.joblib"
    model_path = output_dir / "fake_news_model.joblib"

    logger.info("Saving TF-IDF Vectorizer to: %s", vectorizer_path)
    joblib.dump(vectorizer, vectorizer_path, compress=3)

    logger.info("Saving Logistic Regression Model to: %s", model_path)
    joblib.dump(model, model_path, compress=3)

    logger.info("All artifacts saved successfully to: %s", output_dir)
    return evaluation_data


# =====================================================================
# Main Execution Entrypoint
# =====================================================================
def main() -> None:
    logger.info("Starting Fake News Detection Cloud Training Pipeline")

    # Determine output directory
    # Kaggle writes outputs to /kaggle/working
    if Path("/kaggle/working").exists():
        output_dir = Path("/kaggle/working")
    else:
        output_dir = Path("./models")

    # 1. Locate dataset
    csv_path = find_welfake_dataset()

    # 2. Ingest and preprocess
    df = load_and_preprocess_dataset(csv_path)

    # 3. Stratified Split (80/10/10)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, random_state=42)

    # 4. Train pipeline
    vectorizer, model = train_pipeline(X_train, y_train, X_val, y_val, random_state=42)

    # 5. Evaluate and save artifacts
    evaluate_and_export(vectorizer, model, X_test, y_test, output_dir)

    logger.info("Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
