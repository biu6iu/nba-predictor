"""Small synthetic data shared by the tests. Nothing here reads the real CSVs or a trained model."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessor import build_features

TEAMS = ["Alpha", "Bravo", "Charlie", "Delta"]

# Two seasons of games, plus the previous-season stats each one needs
GAME_SEASONS = {"2023-2024": "2023-10-24", "2024-2025": "2024-10-22"}
STATS_SEASONS = ["2022-2023", "2023-2024"]

# Team-stat columns that build_features turns into diff_ features
STAT_COLS = [
    "SRS", "ORtg", "DRtg", "NRtg", "Pace", "TS%", "eFG%", "TOV%", "ORB%",
    "FTr", "3PAr", "D_eFG%", "D_TOV%", "D_DRB%",
]

# Per day: two games, each pairing (home, visitor) by index into TEAMS
_SCHEDULE = [((0, 1), (2, 3)), ((2, 0), (1, 3)), ((0, 3), (2, 1))]
N_GAME_DAYS = 45


def make_match_data(seed: int = 0) -> pd.DataFrame:
    """Raw results table in the same shape as matchData.csv."""
    rng = np.random.default_rng(seed)
    rows = []
    for start in GAME_SEASONS.values():
        day = pd.Timestamp(start)
        for i in range(N_GAME_DAYS):
            for home, visitor in _SCHEDULE[i % len(_SCHEDULE)]:
                rows.append({
                    "Date": day.strftime("%a %b %d %Y"),
                    "Visitor": TEAMS[visitor],
                    "visitorPTS": int(rng.integers(95, 125)),
                    "Home": TEAMS[home],
                    "homePTS": int(rng.integers(95, 125)),
                })
            day += pd.Timedelta(days=1 + i % 3)  # gaps of 1-3 days, so rest and back-to-backs vary
    match_data = pd.DataFrame(rows)
    # ties don't exist in the NBA
    ties = match_data["homePTS"] == match_data["visitorPTS"]
    match_data.loc[ties, "homePTS"] += 1
    return match_data


def make_team_stats(seed: int = 1) -> pd.DataFrame:
    """One row per team per season, in the same shape as data_loader.load_team_stats()."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in STATS_SEASONS:
        for team in TEAMS:
            values = rng.normal(size=len(STAT_COLS))
            stats = dict(zip(STAT_COLS, values, strict=True))
            rows.append({"Team": team, "season": season, **stats})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def match_data() -> pd.DataFrame:
    return make_match_data()


@pytest.fixture(scope="session")
def team_stats() -> pd.DataFrame:
    return make_team_stats()


@pytest.fixture(scope="session")
def features_df(match_data, team_stats) -> pd.DataFrame:
    return build_features(match_data, team_stats)
