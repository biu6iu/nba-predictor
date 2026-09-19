import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
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
        "log_loss":    round(log_loss(y_true, y_pred_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_pred_prob), 4),
        "roc_auc":     round(roc_auc_score(y_true, y_pred_prob), 4),
        "threshold":   round(threshold, 4),
    }


def save_metrics(metrics: dict, baseline: dict, path: str) -> None:
    """Write metrics and baseline to a JSON file for dashboard consumption."""
    payload = {"model": metrics, "baseline": baseline}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def plot_roc_curve(y_true, y_pred_prob) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — XGBoost")
    ax.legend()
    ax.grid(True)
    return fig


def plot_confusion_matrix(y_true, y_pred_prob, threshold: float) -> plt.Figure:
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Loss (0)", "Win (1)"]

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    return fig


def plot_calibration(calibrated_model, X_val, y_val) -> plt.Figure:
    fig, ax = plt.subplots()
    CalibrationDisplay.from_estimator(
        calibrated_model, X_val, y_val, n_bins=10, ax=ax
    )
    ax.set_title("Calibration Curve")
    return fig


def plot_precision_recall(y_true, y_pred_prob, threshold: float) -> plt.Figure:
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)

    fig, ax = plt.subplots()
    ax.plot(thresholds, precision[:-1], label="Precision")
    ax.plot(thresholds, recall[:-1], label="Recall")
    ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall vs Threshold")
    ax.legend()
    return fig
