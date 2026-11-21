import argparse
import json
import logging

from src.config import (
    ARTEFACTS_DIR,
    BEST_PARAMS_PATH,
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


def load_best_params(path=BEST_PARAMS_PATH) -> dict:
    if not path.exists():
        raise SystemExit(f"--skip-tuning needs {path}. Run train.py once without it to create it.")
    return json.loads(path.read_text())


def save_best_params(params: dict, path=BEST_PARAMS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the model and write artefacts/.")
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="reuse the hyperparameters in config/best_params.json instead of tuning",
    )
    args = parser.parse_args(argv)

    configure_logging()
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Fail before loading data if the saved params are missing
    best_params = load_best_params() if args.skip_tuning else None
    if args.skip_tuning:
        logger.info("Skipping tuning, using params from %s", BEST_PARAMS_PATH)

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
    calibrated_model, best_threshold, best_params = train(df, best_params)
    if not args.skip_tuning:
        save_best_params(best_params)
        logger.info("Saved tuned params to %s (commit it for --skip-tuning)", BEST_PARAMS_PATH)

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
