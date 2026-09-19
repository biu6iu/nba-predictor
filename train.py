import numpy as np

from src.config import ARTEFACTS_DIR, FEATURE_COLS, TARGET_COL, VAL_SEASON
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

    # Evaluate on validation season
    val_df = df[df["Season"] == VAL_SEASON]
    X_val  = val_df[FEATURE_COLS]
    y_val  = val_df[TARGET_COL].astype(int)

    y_pred_prob = calibrated_model.predict_proba(X_val)[:, 1]

    metrics  = compute_metrics(y_val, y_pred_prob, best_threshold)
    baseline = compute_baseline_metrics(y_val)

    save_metrics(metrics, baseline, ARTEFACTS_DIR / "metrics.json")

    plot_roc_curve(y_val, y_pred_prob).savefig(
        ARTEFACTS_DIR / "roc_curve.png", bbox_inches="tight"
    )
    plot_confusion_matrix(y_val, y_pred_prob, best_threshold).savefig(
        ARTEFACTS_DIR / "confusion_matrix.png", bbox_inches="tight"
    )
    plot_calibration(calibrated_model, X_val, y_val).savefig(
        ARTEFACTS_DIR / "calibration.png", bbox_inches="tight"
    )
    plot_precision_recall(y_val, y_pred_prob, best_threshold).savefig(
        ARTEFACTS_DIR / "precision_recall.png", bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
