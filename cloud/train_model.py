"""
Model Training Pipeline for Fake News Detection (Google Colab, Kaggle, and Local data/ folder)
Dataset: WELFake Dataset (or any news dataset with title, text, label)
Target: 0 = FAKE, 1 = REAL

Execution environments supported:
1. Google Colab (upload dataset to /content or /content/data or data/)
2. Local data folder (place CSV in data/ or pass --data-path data/WELFake_Dataset.csv)
3. Kaggle Cloud (/kaggle/input)

Outputs generated in --output-dir (default 'models/'):
- fake_news_model.joblib
- tfidf_vectorizer.joblib
- evaluation.json
- metrics_summary.txt
- confusion_matrix.png
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ModelTrainer")


# =====================================================================
# 1. Deterministic Preprocessing
# =====================================================================
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<.*?>", re.DOTALL)
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    """Deterministic text cleaning."""
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
    """Combine headline and body deterministically."""
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
# 2. Dataset Discovery
# =====================================================================
def find_dataset(explicit_path: Optional[str] = None) -> Path:
    """Find CSV dataset file across standard search directories."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            logger.info("Using explicitly specified dataset path: %s", p)
            return p
        raise FileNotFoundError(f"Specified dataset file not found: {explicit_path}")

    search_dirs = [
        "./data",
        "data",
        "/content/data",
        "/content",
        "/kaggle/input",
        ".",
    ]

    logger.info("Searching for dataset CSV in directories: %s", search_dirs)
    candidate_files: List[Path] = []

    for d in search_dirs:
        p = Path(d)
        if p.exists():
            for csv_file in p.glob("**/*.csv"):
                # Avoid matching output logs or non-dataset csvs
                if "prediction_logs" not in csv_file.name:
                    candidate_files.append(csv_file)

    for csv_file in candidate_files:
        try:
            cols = pd.read_csv(csv_file, nrows=2).columns.str.strip().str.lower().tolist()
            if "title" in cols and "text" in cols and "label" in cols:
                logger.info("Discovered matching dataset: %s", csv_file)
                return csv_file
        except Exception:
            continue

    raise FileNotFoundError(
        "Could not automatically locate a dataset CSV with columns ['title', 'text', 'label'].\n"
        "Please place your dataset (e.g. WELFake_Dataset.csv) in the 'data/' folder or specify --data-path <path>."
    )


# =====================================================================
# 3. Ingestion & Preprocessing
# =====================================================================
def load_and_preprocess(csv_path: Path) -> pd.DataFrame:
    """Load dataset, clean texts, handle missing values, and deduplicate."""
    logger.info("Loading dataset from: %s", csv_path)
    start_time = time.time()
    df = pd.read_csv(csv_path)
    raw_count = len(df)
    logger.info("Raw rows loaded: %d (in %.2fs)", raw_count, time.time() - start_time)

    # Normalize column names
    col_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=col_map)

    # Validate schema
    for col in ["title", "text", "label"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from dataset.")

    # Fill NaNs
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Ensure valid binary labels (0 = FAKE, 1 = REAL)
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    logger.info("Cleaning and combining title and text fields...")
    t0 = time.time()
    df["combined_text"] = [
        combine_title_and_text(t, b) for t, b in zip(df["title"], df["text"])
    ]
    logger.info("Text cleaning completed in %.2fs", time.time() - t0)

    # Drop empty records
    df = df[df["combined_text"].str.strip() != ""].copy()

    # Drop duplicate articles to prevent data leakage across splits
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["combined_text"]).copy()
    logger.info("Dropped %d duplicate articles to eliminate split leakage.", before_dedup - len(df))

    fake_count = int((df["label"] == 0).sum())
    real_count = int((df["label"] == 1).sum())

    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info("Cleaned Total Articles:  %d", len(df))
    logger.info("  - FAKE Samples (0):    %d (%.2f%%)", fake_count, (fake_count / len(df)) * 100)
    logger.info("  - REAL Samples (1):    %d (%.2f%%)", real_count, (real_count / len(df)) * 100)
    logger.info("=" * 60)

    return df


# =====================================================================
# 4. Stratified Split
# =====================================================================
def split_dataset(
    df: pd.DataFrame, random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Stratified 80% train, 10% validation, 10% test split."""
    X = df["combined_text"]
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    logger.info("Train: %d | Validation: %d | Test: %d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


# =====================================================================
# 5. Training
# =====================================================================
def train_model(
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    random_state: int = 42,
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """Fit TF-IDF on train only, then train Logistic Regression."""
    logger.info("Fitting TfidfVectorizer (max_features=100000, ngram_range=(1,2), sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(
        max_features=100000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    t0 = time.time()
    X_train_vec = vectorizer.fit_transform(X_train)
    logger.info("TF-IDF fit completed in %.2fs. Vocabulary: %d terms", time.time() - t0, len(vectorizer.vocabulary_))

    logger.info("Training LogisticRegression(C=2.0, class_weight='balanced', max_iter=1000)...")
    clf = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1,
    )
    t1 = time.time()
    clf.fit(X_train_vec, y_train)
    logger.info("Classifier training completed in %.2fs", time.time() - t1)

    # Validation evaluation
    X_val_vec = vectorizer.transform(X_val)
    val_preds = clf.predict(X_val_vec)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    logger.info("Validation Accuracy: %.4f | Validation Macro F1: %.4f", val_acc, val_f1)

    return vectorizer, clf


# =====================================================================
# 6. Evaluation & Artifact Export
# =====================================================================
def evaluate_and_save(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    X_test: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> Dict:
    """Evaluate on unseen test set and persist production artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Evaluating on held-out test split (%d samples)...", len(y_test))

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    macro_prec = float(precision_score(y_test, y_pred, average="macro"))
    macro_rec = float(recall_score(y_test, y_pred, average="macro"))

    fake_prec = float(precision_score(y_test, y_pred, pos_label=0))
    fake_rec = float(recall_score(y_test, y_pred, pos_label=0))
    fake_f1 = float(f1_score(y_test, y_pred, pos_label=0))

    real_prec = float(precision_score(y_test, y_pred, pos_label=1))
    real_rec = float(recall_score(y_test, y_pred, pos_label=1))
    real_f1 = float(f1_score(y_test, y_pred, pos_label=1))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    report_str = classification_report(y_test, y_pred, target_names=["FAKE (0)", "REAL (1)"], digits=4)

    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("Accuracy:         %.4f (%.2f%%)", acc, acc * 100)
    logger.info("Macro F1-Score:   %.4f", macro_f1)
    logger.info("Macro Precision:  %.4f", macro_prec)
    logger.info("Macro Recall:     %.4f", macro_rec)
    logger.info("-" * 60)
    logger.info("FAKE Class (0) -> Precision: %.4f | Recall: %.4f | F1: %.4f", fake_prec, fake_rec, fake_f1)
    logger.info("REAL Class (1) -> Precision: %.4f | Recall: %.4f | F1: %.4f", real_prec, real_rec, real_f1)
    logger.info("-" * 60)
    logger.info("\n%s", report_str)
    logger.info("=" * 60)

    evaluation_data = {
        "model_name": "TF-IDF + Logistic Regression",
        "dataset": "WELFake Benchmark Dataset",
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

    # Save JSON report
    with open(output_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=2)

    # Save readable summary
    with open(output_dir / "metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"=== FAKE NEWS DETECTOR EVALUATION REPORT ===\n\n")
        f.write(f"Test Accuracy:    {acc:.4f}\n")
        f.write(f"Macro F1-Score:   {macro_f1:.4f}\n\n")
        f.write(f"FAKE -> Prec: {fake_prec:.4f}, Rec: {fake_rec:.4f}, F1: {fake_f1:.4f}\n")
        f.write(f"REAL -> Prec: {real_prec:.4f}, Rec: {real_rec:.4f}, F1: {real_f1:.4f}\n\n")
        f.write(f"Confusion Matrix: [TN={tn}, FP={fp}] [FN={fn}, TP={tp}]\n\n")
        f.write(report_str)

    # Generate confusion matrix plot
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
        ax.set_title("Test Set Confusion Matrix", fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
        plt.close(fig)
    except Exception as e:
        logger.warning("Could not generate plot: %s", e)

    # Save model artifacts
    logger.info("Serializing model artifacts to %s...", output_dir)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib", compress=3)
    joblib.dump(model, output_dir / "fake_news_model.joblib", compress=3)

    logger.info("SUCCESS: All trained artifacts successfully written to: %s", output_dir.resolve())
    return evaluation_data


def main():
    parser = argparse.ArgumentParser(description="Train Fake News Detection Model")
    parser.add_argument("--data-path", type=str, default=None, help="Direct path to dataset CSV")
    parser.add_argument("--output-dir", type=str, default="models", help="Destination directory for trained artifacts")
    args = parser.parse_args()

    # Determine default output directory based on environment
    if args.output_dir == "models" and Path("/content").exists():
        out_dir = Path("/content/models")
    elif args.output_dir == "models" and Path("/kaggle/working").exists():
        out_dir = Path("/kaggle/working")
    else:
        out_dir = Path(args.output_dir)

    csv_path = find_dataset(args.data_path)
    df = load_and_preprocess(csv_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    vectorizer, clf = train_model(X_train, y_train, X_val, y_val)
    evaluate_and_save(vectorizer, clf, X_test, y_test, out_dir)

    # If running inside Google Colab, display easy download instructions
    try:
        import google.colab  # type: ignore
        print("\n" + "=" * 60)
        print("GOOGLE COLAB RUN DETECTED!")
        print("To download the artifacts directly to your local computer, run:")
        print("from google.colab import files")
        print("files.download('/content/models/fake_news_model.joblib')")
        print("files.download('/content/models/tfidf_vectorizer.joblib')")
        print("files.download('/content/models/evaluation.json')")
        print("=" * 60)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
