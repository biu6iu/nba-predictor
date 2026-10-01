import json

import numpy as np
import pytest

from src.evaluate import compute_baseline_metrics, compute_metrics, save_metrics


@pytest.fixture
def labels():
    # 70% home wins
    return np.array([1] * 70 + [0] * 30)


def test_perfect_predictor_scores_auc_one(labels):
    prob = np.where(labels == 1, 0.99, 0.01)
    metrics = compute_metrics(labels, prob, threshold=0.5)
    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_inverted_predictor_scores_auc_zero(labels):
    prob = np.where(labels == 1, 0.01, 0.99)
    assert compute_metrics(labels, prob, threshold=0.5)["roc_auc"] == 0.0


def test_threshold_is_applied_and_reported(labels):
    prob = np.full(len(labels), 0.6)
    assert compute_metrics(labels, prob, threshold=0.5)["recall"] == 1.0
    assert compute_metrics(labels, prob, threshold=0.7)["recall"] == 0.0
    assert compute_metrics(labels, prob, threshold=0.7)["threshold"] == 0.7


def test_baseline_matches_the_base_rate(labels):
    p = labels.mean()
    baseline = compute_baseline_metrics(labels)
    assert baseline["accuracy"] == pytest.approx(p, abs=1e-4)
    assert baseline["brier_score"] == pytest.approx(p * (1 - p), abs=1e-4)
    assert baseline["log_loss"] == pytest.approx(-(p * np.log(p) + (1 - p) * np.log(1 - p)), abs=1e-4)


def test_baseline_predicts_the_majority_class_even_when_it_is_a_loss():
    labels = np.array([1] * 30 + [0] * 70)
    assert compute_baseline_metrics(labels)["accuracy"] == pytest.approx(0.7)


def test_metrics_json_round_trips(labels, tmp_path):
    prob = np.where(labels == 1, 0.8, 0.3)
    results = {
        split: {
            "model": compute_metrics(labels, prob, threshold=0.5),
            "baseline": compute_baseline_metrics(labels),
        }
        for split in ("validation", "test")
    }
    path = tmp_path / "metrics.json"
    save_metrics(results, path)
    assert json.loads(path.read_text()) == results
