"""Evaluation metrics for AI-generated text detection.

Implements standard metrics (F1, precision, recall, ROC-AUC) plus low-FPR
TPR, ECE (calibration), and Brier score. All sklearn calls use zero_division=0.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true: Any, y_pred: Any, y_prob: Any) -> dict[str, Any]:
    """Compute the standard binary classification metrics used for RAID."""

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


def compute_ece(
    labels: list[int],
    probs: list[float],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Bins predictions into ``n_bins`` equal-width bins by confidence
    (max softmax probability) and computes |accuracy - confidence|
    weighted by bin size.

    Args:
        labels: Ground-truth labels (0 or 1).
        probs: Predicted probability of class 1.
        n_bins: Number of equal-width bins (default: 10).

    Returns:
        ECE score (float between 0 and 1).
    """
    labels_arr = np.asarray(labels)
    probs_arr = np.asarray(probs)

    # Confidence = probability assigned to the predicted class
    predictions = (probs_arr >= 0.5).astype(np.int64)
    confidences = np.where(predictions == 1, probs_arr, 1.0 - probs_arr)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece: float = 0.0
    total = len(labels_arr)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if i == 0:
            in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = int(in_bin.sum())
        if bin_size == 0:
            continue
        bin_acc = float((predictions[in_bin] == labels_arr[in_bin]).mean())
        bin_conf = float(confidences[in_bin].mean())
        ece += (bin_size / total) * abs(bin_acc - bin_conf)

    return ece


def compute_low_fpr_tpr(
    labels: list[int],
    probs: list[float],
    thresholds: list[float] | None = None,
) -> dict[str, float]:
    """
    Compute True Positive Rate at specified False Positive Rate thresholds.

    Sorts predictions by confidence (probability of class 1) descending,
    then finds the TPR at each FPR threshold by scanning the sorted list.

    Args:
        labels: Ground-truth labels (0 or 1).
        probs: Predicted probability of class 1.
        thresholds: List of FPR thresholds (default: [0.01, 0.05, 0.10]).

    Returns:
        Dict mapping ``"tpr_at_{fpr*100:.0f}pct_fpr"`` to TPR value.
    """
    if thresholds is None:
        thresholds = [0.01, 0.05, 0.10]

    labels_arr = np.asarray(labels)
    probs_arr = np.asarray(probs)

    # Sort descending by predicted probability
    sort_idx = np.argsort(-probs_arr)
    labels_sorted = labels_arr[sort_idx]

    n_pos = int((labels_arr == 1).sum())
    n_neg = int((labels_arr == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return {f"tpr_at_{int(t * 100)}pct_fpr": float("nan") for t in thresholds}

    # Cumulative FP and TP as we walk through sorted predictions
    cum_fp = np.cumsum(labels_sorted == 0)
    cum_tp = np.cumsum(labels_sorted == 1)
    fpr_values = cum_fp / n_neg
    tpr_values = cum_tp / n_pos

    results: dict[str, float] = {}
    for target_fpr in thresholds:
        idx = int(np.searchsorted(fpr_values, target_fpr, side="left"))
        if idx >= len(tpr_values):
            idx = -1
        results[f"tpr_at_{int(target_fpr * 100)}pct_fpr"] = float(tpr_values[idx])

    return results