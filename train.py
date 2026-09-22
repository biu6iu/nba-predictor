import numpy as np

from src.config import ARTEFACTS_DIR, FEATURE_COLS, TARGET_COL, TEST_SEASON, VAL_SEASON
from src.data_loader import load_match_data, load_team_stats
from src.evaluate import (
    compute_baseline_metrics,
    compute_metrics,
    plot_calibration,
    plot_confusion_matrix,
    plot_precision_recall,
    plot_roc_curve,
    save_metrics,
)
from src.model import train
from src.preprocessor import build_features


def evaluate_split(model, df, season, threshold):
    """
    score a model on one season
    
    returns its metrics (model + baseline), X, y and probabilities
    """
    split = df[df["Season"] == season]
    X = split[FEATURE_COLS]
    y = split[TARGET_COL].astype(int)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "model": compute_metrics(y, y_prob, threshold),
        "baseline": compute_baseline_metrics(y),
    }

    return metrics, X, y, y_prob


def main():
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data
    match_data = load_match_data()
    team_stats = load_team_stats()

    # Feature engineering
    df = build_features(match_data, team_stats)

    # Persist the processed DataFrame for the dashboard
    df.to_parquet(ARTEFACTS_DIR / "processed_df.parquet", index=False)

    # Train
    calibrated_model, best_threshold, best_params = train(df)

    # The threshold was chosen on the validation season, so its metrics are optimistic
    # The test season was never used for any decision and gives the honest numbers
    val_metrics, _, _, _ = evaluate_split(calibrated_model, df, VAL_SEASON, best_threshold)
    test_metrics, X_test, y_test, y_test_prob = evaluate_split(calibrated_model, df, TEST_SEASON, best_threshold)

    save_metrics(
        {"validation": val_metrics, "test": test_metrics},
        ARTEFACTS_DIR / "metrics.json",
    )

    plot_roc_curve(y_test, y_test_prob).savefig(
        ARTEFACTS_DIR / "roc_curve.png", bbox_inches="tight"
    )
    plot_confusion_matrix(y_test, y_test_prob, best_threshold).savefig(
        ARTEFACTS_DIR / "confusion_matrix.png", bbox_inches="tight"
    )
    plot_calibration(calibrated_model, X_test, y_test).savefig(
        ARTEFACTS_DIR / "calibration.png", bbox_inches="tight"
    )
    plot_precision_recall(y_test, y_test_prob, best_threshold).savefig(
        ARTEFACTS_DIR / "precision_recall.png", bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
