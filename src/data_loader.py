import pandas as pd

from src.config import DATA_DIR, SEASONS


def load_match_data() -> pd.DataFrame:
    """Return the raw match-by-match results table."""
    return pd.read_csv(DATA_DIR / "matchData.csv")


def load_team_stats() -> pd.DataFrame:
    """
    Load, merge, and concatenate per-game and advanced team stats for every
    season defined in config.SEASONS.

    Returns a single DataFrame with one row per team per season and a 'season'
    column containing the season label (e.g. '2022-2023').
    """
    frames = []
    for season in SEASONS:
        year = int(season.split("-")[0])
        per_game = pd.read_csv(DATA_DIR / f"{year}PerGameData.csv")
        advanced = pd.read_csv(DATA_DIR / f"{year}AdvData.csv")

        merged = pd.merge(per_game, advanced, on="Team")
        merged = merged.drop(columns=["G", "MP"])
        merged["season"] = season
        frames.append(merged)

    team_stats = pd.concat(frames, ignore_index=True)
    team_stats["Team"] = (
        team_stats["Team"]
        .str.replace("*", "", regex=False)
        .str.strip()
    )
    return team_stats
