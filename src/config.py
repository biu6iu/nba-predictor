from pathlib import Path

# Paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTEFACTS_DIR = PROJECT_ROOT / "artefacts"

# Seasons

SEASONS = [
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]

TRAIN_SEASONS = ["2021-2022", "2022-2023"]
VAL_SEASON = "2023-2024"
TEST_SEASON = "2024-2025"

# Modelling target

TARGET_COL = "Win"

# Feature list  (28 features, ordered by semantic group)

FEATURE_COLS = [
    # Season-to-date and short-run scoring performance differentials
    "diff_avg_pts_scored",
    "diff_avg_pts_allowed",
    "diff_avg_pts_last5",
    "diff_win_pct_last5",
    "diff_win_pct_last10",
    "diff_pt_diff_last10",

    # Schedule / fatigue
    "diff_days_rest",

    # Venue effects
    "home_home_win_pct_last10",
    "visitor_away_win_pct_last10",
    "home_b2b",
    "visitor_b2b",

    # Overall team strength (previous-season advanced stats differentials)
    "diff_SRS",
    "diff_ORtg",
    "diff_DRtg",
    "diff_NRtg",
    "diff_Pace",

    # Offensive efficiency differentials
    "diff_TS%",
    "diff_eFG%",
    "diff_TOV%",
    "diff_ORB%",
    "diff_FTr",
    "diff_3PAr",

    # Defensive efficiency differentials
    "diff_D_eFG%",
    "diff_D_TOV%",
    "diff_D_DRB%",

    # Momentum (win-rate trajectory)
    "home_form",
    "visitor_form",
    "diff_form",
]

# Bayesian Optimisation settings

BAYES_OPT_PBOUNDS = {
    "max_depth":        (3, 5),
    "learning_rate":    (0.03, 0.08),
    "n_estimators":     (300, 900),
    "subsample":        (0.6, 0.85),
    "colsample_bytree": (0.5, 0.75),
    "min_child_weight": (4, 10),
    "gamma":            (0.5, 3),
    "reg_alpha":        (0.5, 4),
    "reg_lambda":       (1, 8),
}

BAYES_OPT_INIT_POINTS = 20
BAYES_OPT_N_ITER = 60

# Training settings

TSCV_N_SPLITS = 5
CALIBRATION_TEST_SIZE = 0.2
TARGET_PRECISION = 0.65

# Keys stored in artefacts/model.pkl (written by src/model.py, read by the dashboard)

MODEL_ARTEFACT_KEYS = (
    "model",
    "threshold",
    "best_params",
    "feature_cols",
    "sklearn_version",
    "xgboost_version",
    "trained_at",
    "train_seasons",
)
