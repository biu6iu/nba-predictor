import joblib
import numpy as np
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from src.config import (
    ARTEFACTS_DIR,
    BAYES_OPT_INIT_POINTS,
    BAYES_OPT_N_ITER,
    BAYES_OPT_PBOUNDS,
    CALIBRATION_TEST_SIZE,
    FEATURE_COLS,
    TARGET_COL,
    TARGET_PRECISION,
    TEST_SEASON,
    TRAIN_SEASONS,
    TSCV_N_SPLITS,
    VAL_SEASON,
)


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
    val_df   = df[df["Season"] == VAL_SEASON]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].astype(int)
    X_val   = val_df[FEATURE_COLS]
    y_val   = val_df[TARGET_COL].astype(int)

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
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=42,
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
        random_state=42,
        verbose=2,
    )
    optimizer.maximize(init_points=BAYES_OPT_INIT_POINTS, n_iter=BAYES_OPT_N_ITER)

    best_params = optimizer.max["params"]
    best_params["max_depth"]    = int(round(best_params["max_depth"]))
    best_params["n_estimators"] = int(round(best_params["n_estimators"]))

    # fit final model on full training set
    model = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Platt scaling calibration on a held-out split of training data
    _, X_calib, _, y_calib = train_test_split(
        X_train, y_train,
        test_size=CALIBRATION_TEST_SIZE,
        random_state=42,
    )
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv="prefit",
    )
    calibrated_model.fit(X_calib, y_calib)

    # Threshold optimisation on validation set: target precision >= TARGET_PRECISION, maximise recall
    y_pred_prob = calibrated_model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_prob)

    valid_idx = np.where(precision[:-1] >= TARGET_PRECISION)[0]
    best_threshold = float(thresholds[valid_idx[np.argmax(recall[valid_idx])]])

    # Serialise model
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_model, ARTEFACTS_DIR / "model.pkl")

    return calibrated_model, best_threshold, best_params
