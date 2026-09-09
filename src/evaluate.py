"""
Model Evaluation Module
Provides metric computation, confusion matrix formatting, and JSON serialization.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for binary fake news classification.
    Classes: 0 = FAKE, 1 = REAL.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Structured dictionary of evaluation metrics.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro"))
    macro_prec = float(precision_score(y_true_arr, y_pred_arr, average="macro"))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, average="macro"))

    # Per-class metrics
    real_prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    real_rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    real_f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))

    fake_prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    fake_rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))
    fake_f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=0, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    return {
        "model_name": "TF-IDF + Logistic Regression",
        "dataset": "WELFake Dataset (saurabhshahane/fake-news-classification)",
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_samples": int(len(y_true_arr)),
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
                "support": int((y_true_arr == 1).sum()),
            },
            "fake_class": {
                "label": "FAKE",
                "code": 0,
                "precision": round(fake_prec, 4),
                "recall": round(fake_rec, 4),
                "f1_score": round(fake_f1, 4),
                "support": int((y_true_arr == 0).sum()),
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


def save_evaluation_report(metrics_dict: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Save metrics dictionary to a JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    return p


def load_evaluation_report(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load evaluation JSON file if present."""
    p = Path(file_path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
