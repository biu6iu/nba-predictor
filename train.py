import logging

from src.config import (
    ARTEFACTS_DIR,
    FEATURE_COLS,
    TARGET_COL,
    TEST_SEASON,
    VAL_SEASON,
    split_by_season,
)
from src.data_loader import load_match_data, load_team_stats
from src.evaluate import compute_baseline_metrics, compute_metrics, save_metrics
from src.logging_config import configure_logging
from src.model import train
from src.preprocessor import build_features

logger = logging.getLogger(__name__)


def evaluate_split(model, df, season, threshold):
    """
    score a model on one season
    
    returns its metrics (model + baseline), X, y and probabilities
    """
    split = split_by_season(df, season)
    X = split[FEATURE_COLS]
    y = split[TARGET_COL].astype(int)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "model": compute_metrics(y, y_prob, threshold),
        "baseline": compute_baseline_metrics(y),
    }

    return metrics, X, y, y_prob


def main():
    configure_logging()
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load raw data
    match_data = load_match_data()
    team_stats = load_team_stats()
    logger.info("Loaded %d matches, %d team-season stat rows", len(match_data), len(team_stats))

    # Feature engineering
    df = build_features(match_data, team_stats)
    logger.info("Built features for %d games", len(df))

    # Persist the processed DataFrame for the dashboard
    df.to_parquet(ARTEFACTS_DIR / "processed_df.parquet", index=False)

    # Train
    calibrated_model, best_threshold, best_params = train(df)

    # The threshold was chosen on the validation season, so its metrics are optimistic
    # The test season was never used for any decision and gives the honest numbers
    val_metrics, _, _, _ = evaluate_split(calibrated_model, df, VAL_SEASON, best_threshold)
    test_metrics, _, _, _ = evaluate_split(calibrated_model, df, TEST_SEASON, best_threshold)
    logger.info(
        "Accuracy - validation: %.1f%%, test: %.1f%%",
        100 * val_metrics["model"]["accuracy"],
        100 * test_metrics["model"]["accuracy"],
    )

    save_metrics(
        {"validation": val_metrics, "test": test_metrics},
        ARTEFACTS_DIR / "metrics.json",
    )


if __name__ == "__main__":
    main()
