import logging
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from src.config import (
    ARTEFACTS_DIR,
    BAYES_OPT_INIT_POINTS,
    BAYES_OPT_N_ITER,
    BAYES_OPT_PBOUNDS,
    CALIBRATION_TEST_SIZE,
    EARLY_STOPPING_ROUNDS,
    FEATURE_COLS,
    MODEL_ARTEFACT_KEYS,
    RANDOM_STATE,
    TARGET_COL,
    TARGET_PRECISION,
    TEST_SEASON,
    TRAIN_SEASONS,
    TSCV_N_SPLITS,
    VAL_SEASON,
    split_by_season,
)

logger = logging.getLogger(__name__)


def train(df) -> tuple:
    """
    Run the full training pipeline on the feature-engineered DataFrame.

    Returns
    -------
    calibrated_model : CalibratedClassifierCV
    best_threshold   : float
    best_params      : dict
    """
    # Train / val split
    train_df = df[df["Season"].isin(TRAIN_SEASONS)]
    val_df   = split_by_season(df, VAL_SEASON)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].astype(int)
    X_val   = val_df[FEATURE_COLS]
    y_val   = val_df[TARGET_COL].astype(int)
    logger.info("Train/val split: %d train rows, %d val rows", len(train_df), len(val_df))

    # Bayesian optimisation over XGBoost hyperparameters
    cv = TimeSeriesSplit(n_splits=TSCV_N_SPLITS)

    def xgb_cv(
        max_depth, learning_rate, n_estimators, subsample,
        colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda,
    ):
        model = xgb.XGBClassifier(
            max_depth=int(round(max_depth)),
            learning_rate=learning_rate,
            n_estimators=int(round(n_estimators)),
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            max_delta_step=1,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="logloss",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        scores = []
        for train_idx, fold_val_idx in cv.split(X_train):
            X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[fold_val_idx]
            y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[fold_val_idx]
            model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
            preds = model.predict_proba(X_v)[:, 1]
            scores.append(-log_loss(y_v, preds))
        return np.mean(scores)

    optimizer = BayesianOptimization(
        f=xgb_cv,
        pbounds=BAYES_OPT_PBOUNDS,
        allow_duplicate_points=True,
        random_state=RANDOM_STATE,
        verbose=2,
    )
    t0 = time.perf_counter()
    optimizer.maximize(init_points=BAYES_OPT_INIT_POINTS, n_iter=BAYES_OPT_N_ITER)
    logger.info(
        "Bayesian optimisation finished in %.0fs (%d fits)",
        time.perf_counter() - t0, BAYES_OPT_INIT_POINTS + BAYES_OPT_N_ITER,
    )

    best_params = optimizer.max["params"]
    best_params["max_depth"]    = int(round(best_params["max_depth"]))
    best_params["n_estimators"] = int(round(best_params["n_estimators"]))
    logger.info("Best params: %s", best_params)

    # fit model on 80% of the data, with 20% going to calibration
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train,
        test_size=CALIBRATION_TEST_SIZE,
        shuffle=False,
    )

    model = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        max_delta_step=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_calib, y_calib)], verbose=False)

    # Platt scaling on the held-out later games, FrozenEstimator keeps the model as fitted
    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method="sigmoid",
    )
    calibrated_model.fit(X_calib, y_calib)

    # Threshold optimisation on validation set: target precision >= TARGET_PRECISION, maximise recall
    y_pred_prob = calibrated_model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_prob)
    precision, recall = precision[:-1], recall[:-1]  # drop the extra point precision_recall_curve appends

    valid_idx = np.where(precision >= TARGET_PRECISION)[0]
    if valid_idx.size > 0:
        best_idx = int(valid_idx[np.argmax(recall[valid_idx])])
        best_threshold = float(thresholds[best_idx])
    else:
        # no threshold reaches TARGET_PRECISION so fall back to the threshold that maximises F1
        f1 = np.divide(
            2 * precision * recall, precision + recall,
            out=np.zeros_like(precision), where=(precision + recall) > 0,
        )
        best_idx = int(np.argmax(f1))
        best_threshold = float(thresholds[best_idx])
        warnings.warn(
            f"No threshold reached the target precision of {TARGET_PRECISION:.0%} on the "
            f"validation set (best precision was {precision.max():.1%}). Falling back to the "
            f"max-F1 threshold {best_threshold:.4f} (precision={precision[best_idx]:.1%}, "
            f"recall={recall[best_idx]:.1%}).",
            stacklevel=2,
        )
    logger.info(
        "Threshold=%.4f (val precision=%.1f%%, recall=%.1f%%)",
        best_threshold, 100 * precision[best_idx], 100 * recall[best_idx],
    )

    # Serialise the model together with everything needed to use and audit it
    artefact = {
        "model":           calibrated_model,
        "threshold":       best_threshold,
        "best_params":     dict(best_params),
        "feature_cols":    list(FEATURE_COLS),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
        "trained_at":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_seasons":   list(TRAIN_SEASONS),
    }
    assert set(artefact) == set(MODEL_ARTEFACT_KEYS)

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, ARTEFACTS_DIR / "model.pkl")

    return calibrated_model, best_threshold, best_params
