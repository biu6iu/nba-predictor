import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLS, TARGET_COL
from src.preprocessor import _prev_season, build_differential_features, build_features
from tests.conftest import TEAMS

# Columns that describe the game being predicted, so they can never be features
OUTCOME_COLS = {"Win", "homePTS", "visitorPTS"}


def test_feature_cols_exclude_outcome_columns():
    assert OUTCOME_COLS.isdisjoint(FEATURE_COLS)
    assert TARGET_COL in OUTCOME_COLS


def test_all_feature_cols_are_built(features_df):
    assert set(FEATURE_COLS) <= set(features_df.columns)
    assert len(features_df) > 0


def test_previous_season_stats_are_attached(features_df):
    # every game in the fixture has stats for the season before it, so the merge must not drop them
    assert features_df["diff_SRS"].notna().all()


def test_no_feature_perfectly_tracks_the_outcome(features_df):
    corr = features_df[FEATURE_COLS].corrwith(features_df[TARGET_COL].astype(int)).abs()
    assert (corr.dropna() < 0.99).all(), corr.sort_values(ascending=False).head()


def test_features_ignore_the_game_being_predicted_and_later_games(match_data, team_stats, features_df):
    """
    Rewrite one game's result: no feature of that game, or of any earlier game, may change.
    Catches any rolling feature that is missing its .shift(1).
    """
    pick = 70  # well into the first season, where every rolling window is full
    altered = match_data.copy()
    altered.loc[pick, ["homePTS", "visitorPTS"]] = [
        match_data.loc[pick, "visitorPTS"] + 30,
        match_data.loc[pick, "homePTS"],
    ]
    altered_df = build_features(altered, team_stats)

    cutoff = pd.to_datetime(match_data.loc[pick, "Date"], format="%a %b %d %Y")
    upto = features_df["Date"] <= cutoff
    pd.testing.assert_frame_equal(
        features_df.loc[upto, FEATURE_COLS], altered_df.loc[upto, FEATURE_COLS]
    )

    # the rewrite must reach later games, otherwise the check above proves nothing
    later = features_df["Date"] > cutoff
    assert not features_df.loc[later, FEATURE_COLS].equals(altered_df.loc[later, FEATURE_COLS])


def test_prev_season():
    assert _prev_season("2024-2025") == "2023-2024"
    assert _prev_season("2000-2001") == "1999-2000"


@pytest.mark.parametrize(
    ("date", "season"),
    [
        ("2024-09-30", "2023-2024"),  # day before October
        ("2024-10-01", "2024-2025"),  # first day of October
        ("2024-12-25", "2024-2025"),
        ("2025-01-15", "2024-2025"),  # January belongs to the season that began the previous year
        ("2025-06-15", "2024-2025"),  # Finals
    ],
)
def test_season_label_across_october_boundary(team_stats, date, season):
    game = pd.DataFrame({
        "Date": [pd.Timestamp(date).strftime("%a %b %d %Y")],
        "Visitor": [TEAMS[0]], "visitorPTS": [100],
        "Home": [TEAMS[1]], "homePTS": [110],
    })
    assert build_features(game, team_stats).loc[0, "Season"] == season


def test_differential_features():
    diff_cols = [c for c in FEATURE_COLS if c.startswith("diff_") and c != "diff_form"]
    bases = [c.removeprefix("diff_") for c in diff_cols]
    row = {}
    for base in bases:
        row[f"home_{base}"], row[f"visitor_{base}"] = 10.0, 4.0
    row.update({
        "home_win_pct_last5": 0.8, "home_win_pct_last10": 0.5,
        "visitor_win_pct_last5": 0.4, "visitor_win_pct_last10": 0.6,
    })
    out = build_differential_features(pd.DataFrame([row]))

    for col in diff_cols:
        if col not in ("diff_win_pct_last5", "diff_win_pct_last10"):
            assert out.loc[0, col] == 6.0, col  # home minus visitor
    assert out.loc[0, "home_form"] == pytest.approx(0.3)
    assert out.loc[0, "visitor_form"] == pytest.approx(-0.2)
    assert out.loc[0, "diff_form"] == pytest.approx(0.5)


def test_early_season_features_are_nan_not_zero(features_df):
    first = features_df.sort_values("Date").iloc[0]
    assert np.isnan(first["diff_avg_pts_scored"])
    assert np.isnan(first["diff_win_pct_last10"])
