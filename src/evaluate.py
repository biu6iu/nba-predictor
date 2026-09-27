import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_baseline_metrics(y_true) -> dict:
    """Majority-class baseline: always predict the more frequent class."""
    win_prob = float(np.mean(y_true))
    majority_class = int(win_prob >= 0.5)
    y_pred = np.full(len(y_true), majority_class, dtype=int)
    y_prob = np.full(len(y_true), win_prob, dtype=float)
    return {
        "accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "log_loss":    round(log_loss(y_true, y_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
    }


def compute_metrics(y_true, y_pred_prob, threshold: float) -> dict:
    """Model metrics at a given classification threshold."""
    y_pred = (y_pred_prob >= threshold).astype(int)
    return {
        "accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "precision":   round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":      round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":          round(f1_score(y_true, y_pred, zero_division=0), 4),
        "log_loss":    round(log_loss(y_true, y_pred_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_pred_prob), 4),
        "roc_auc":     round(roc_auc_score(y_true, y_pred_prob), 4),
        "threshold":   round(threshold, 4),
    }


def save_metrics(results: dict, path: Path) -> None:
    """
    Write per-split metrics to a JSON file for dashboard consumption.

    `results` maps a split name ("validation", "test") to
    {"model": {...}, "baseline": {...}}.
    """
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
