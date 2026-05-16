from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true: Any, y_pred: Any, y_prob: Any) -> dict[str, Any]:
    """Compute the standard binary classification metrics used in DetectRL."""

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    y_prob_array = np.asarray(y_prob)

    metrics: dict[str, Any] = {
        "f1_macro": float(f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": float(precision_score(y_true_array, y_pred_array, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred_array, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_array, y_pred_array).tolist(),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_prob_array))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


def compute_rrd(f1_seen: float, f1_unseen: float) -> float:
    """Compute Relative Robustness Drop (RRD) as a percentage."""

    if f1_seen == 0:
        return float("nan")
    return float((f1_seen - f1_unseen) / f1_seen * 100.0)
