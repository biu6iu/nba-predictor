import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_score, recall_score

from src.config import CALIBRATION_TEST_SIZE, TARGET_PRECISION
from src.model import select_threshold, split_fit_calib


@pytest.fixture
def separable():
    """Positives all score higher than negatives, so a perfectly precise threshold exists."""
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    prob = np.concatenate([rng.uniform(0.05, 0.55, 50), rng.uniform(0.45, 0.95, 50)])
    return y, prob


@pytest.fixture
def unreachable():
    """Positives score lowest, so no threshold gets near TARGET_PRECISION (best is the 10% base rate)."""
    y = np.array([1] * 10 + [0] * 90)
    prob = np.linspace(0.05, 0.95, 100)
    return y, prob


def test_threshold_is_a_probability(separable):
    threshold = select_threshold(*separable)
    assert 0 < threshold < 1


def test_threshold_meets_target_precision_and_maximises_recall(separable):
    y, prob = separable
    threshold = select_threshold(y, prob)
    pred = (prob >= threshold).astype(int)
    assert precision_score(y, pred) >= TARGET_PRECISION

    # no other threshold that also meets the target recalls more
    best_recall = max(
        recall_score(y, (prob >= t).astype(int))
        for t in prob
        if precision_score(y, (prob >= t).astype(int), zero_division=0) >= TARGET_PRECISION
    )
    assert recall_score(y, pred) == pytest.approx(best_recall)


def test_unreachable_target_falls_back_with_a_warning_instead_of_raising(unreachable):
    with pytest.warns(UserWarning, match="No threshold reached the target precision"):
        threshold = select_threshold(*unreachable)
    assert 0 < threshold < 1


def test_fit_and_calibration_sets_are_disjoint_and_cover_the_data():
    X = pd.DataFrame({"f": np.arange(100.0)})
    y = pd.Series(np.arange(100) % 2)
    X_fit, X_calib, y_fit, y_calib = split_fit_calib(X, y)

    assert set(X_fit.index).isdisjoint(X_calib.index)
    assert set(X_fit.index) | set(X_calib.index) == set(X.index)
    assert len(X_calib) == pytest.approx(CALIBRATION_TEST_SIZE * len(X), abs=1)
    assert X_fit.index.equals(y_fit.index)
    assert X_calib.index.equals(y_calib.index)


def test_calibration_set_is_the_later_games():
    X = pd.DataFrame({"f": np.arange(100.0)})
    y = pd.Series(np.arange(100) % 2)
    X_fit, X_calib, _, _ = split_fit_calib(X, y)
    assert X_fit.index.max() < X_calib.index.min()
