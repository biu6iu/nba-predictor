import pandas as pd


def _prev_season(season_str: str) -> str:
    start, end = season_str.split("-")
    return f"{int(start) - 1}-{int(end) - 1}"


def build_features(match_data: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw match and team-stats DataFrames into a model-ready DataFrame.

    Every rolling/expanding feature is computed with a .shift(1) before
    aggregation to prevent look-ahead leakage.
    """
    df = match_data.copy()

    # Parse dates and derive season labels
    df["Date"] = pd.to_datetime(df["Date"], format="%a %b %d %Y")
    df["Season"] = df["Date"].apply(
        lambda d: f"{d.year}-{d.year + 1}" if d.month >= 10 else f"{d.year - 1}-{d.year}"
    )
    df["totalPTS"] = df["visitorPTS"] + df["homePTS"]

    # Build long-format team_games table (one row per team per game)
    home = df[["Date", "Season", "Home", "homePTS", "visitorPTS"]].copy()
    home.columns = ["Date", "Season", "Team", "PTS_scored", "PTS_allowed"]
    home["is_home"] = 1

    away = df[["Date", "Season", "Visitor", "visitorPTS", "homePTS"]].copy()
    away.columns = ["Date", "Season", "Team", "PTS_scored", "PTS_allowed"]
    away["is_home"] = 0

    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.sort_values(["Season", "Team", "Date"]).reset_index(drop=True)
    team_games["Win"] = (team_games["PTS_scored"] > team_games["PTS_allowed"]).astype(int)

    # Compute rolling features (all shifted to prevent leakage)
    grp = team_games.groupby(["Season", "Team"])

    team_games["avg_pts_scored"] = grp["PTS_scored"].transform(
        lambda x: x.shift().expanding().mean()
    )
    team_games["avg_pts_allowed"] = grp["PTS_allowed"].transform(
        lambda x: x.shift().expanding().mean()
    )
    team_games["avg_pts_last5"] = grp["PTS_scored"].transform(
        lambda x: x.shift().rolling(5).mean()
    )
    team_games["win_pct_last5"] = grp["Win"].transform(
        lambda x: x.shift().rolling(5).mean()
    )
    team_games["win_pct_last10"] = grp["Win"].transform(
        lambda x: x.shift().rolling(10).mean()
    )
    team_games["days_rest"] = grp["Date"].diff().dt.days
    team_games["b2b"] = (team_games["days_rest"] == 1).astype(int)

    team_games["home_win_pct_last10"] = (
        team_games.groupby(["Season", "Team", "is_home"])["Win"]
        .transform(lambda x: x.shift().rolling(10).mean())
    )

    team_games["pt_diff"] = team_games["PTS_scored"] - team_games["PTS_allowed"]
    team_games["pt_diff_last10"] = grp["pt_diff"].transform(
        lambda x: x.shift().rolling(10).mean()
    )

    # Pivot back to match level: separate home and away feature tables
    rolling_cols = [
        "avg_pts_scored", "avg_pts_allowed", "avg_pts_last5",
        "win_pct_last5", "win_pct_last10",
        "days_rest", "b2b", "home_win_pct_last10",
        "pt_diff", "pt_diff_last10",
    ]

    home_features = (
        team_games[team_games["is_home"] == 1][["Date", "Season", "Team"] + rolling_cols]
        .copy()
        .rename(columns={
            "Team": "Home",
            "avg_pts_scored":      "home_avg_pts_scored",
            "avg_pts_allowed":     "home_avg_pts_allowed",
            "avg_pts_last5":       "home_avg_pts_last5",
            "win_pct_last5":       "home_win_pct_last5",
            "win_pct_last10":      "home_win_pct_last10",
            "days_rest":           "home_days_rest",
            "b2b":                 "home_b2b",
            "home_win_pct_last10": "home_home_win_pct_last10",
            "pt_diff":             "home_pt_diff",
            "pt_diff_last10":      "home_pt_diff_last10",
        })
    )

    away_features = (
        team_games[team_games["is_home"] == 0][["Date", "Season", "Team"] + rolling_cols]
        .copy()
        .rename(columns={
            "Team": "Visitor",
            "avg_pts_scored":      "visitor_avg_pts_scored",
            "avg_pts_allowed":     "visitor_avg_pts_allowed",
            "avg_pts_last5":       "visitor_avg_pts_last5",
            "win_pct_last5":       "visitor_win_pct_last5",
            "win_pct_last10":      "visitor_win_pct_last10",
            "days_rest":           "visitor_days_rest",
            "b2b":                 "visitor_b2b",
            "home_win_pct_last10": "visitor_away_win_pct_last10",
            "pt_diff":             "visitor_pt_diff",
            "pt_diff_last10":      "visitor_pt_diff_last10",
        })
    )

    df = df.merge(home_features, on=["Date", "Season", "Home"], how="left")
    df = df.merge(away_features, on=["Date", "Season", "Visitor"], how="left")

    # Attach previous-season advanced team stats
    df["stats_season"] = df["Season"].apply(_prev_season)

    home_stats    = team_stats.add_prefix("home_")
    visitor_stats = team_stats.add_prefix("visitor_")

    df = df.merge(
        home_stats,
        left_on=["Home", "stats_season"],
        right_on=["home_Team", "home_season"],
        how="left",
    )
    df = df.merge(
        visitor_stats,
        left_on=["Visitor", "stats_season"],
        right_on=["visitor_Team", "visitor_season"],
        how="left",
    )

    df = df.drop(columns=["home_Team", "stats_season", "home_season", "visitor_Team", "visitor_season"])

    # Sort, set target, compute differential features
    df = df.sort_values("Date").reset_index(drop=True)
    df["Win"] = df["homePTS"] > df["visitorPTS"]

    df["diff_avg_pts_scored"]  = df["home_avg_pts_scored"]  - df["visitor_avg_pts_scored"]
    df["diff_avg_pts_allowed"] = df["home_avg_pts_allowed"] - df["visitor_avg_pts_allowed"]
    df["diff_avg_pts_last5"]   = df["home_avg_pts_last5"]   - df["visitor_avg_pts_last5"]
    df["diff_win_pct_last5"]   = df["home_win_pct_last5"]   - df["visitor_win_pct_last5"]
    df["diff_win_pct_last10"]  = df["home_win_pct_last10"]  - df["visitor_win_pct_last10"]
    df["diff_days_rest"]       = df["home_days_rest"]       - df["visitor_days_rest"]
    df["diff_pt_diff"]         = df["home_pt_diff"]         - df["visitor_pt_diff"]
    df["diff_pt_diff_last10"]  = df["home_pt_diff_last10"]  - df["visitor_pt_diff_last10"]
    df["diff_SRS"]             = df["home_SRS"]             - df["visitor_SRS"]
    df["diff_ORtg"]            = df["home_ORtg"]            - df["visitor_ORtg"]
    df["diff_DRtg"]            = df["home_DRtg"]            - df["visitor_DRtg"]
    df["diff_NRtg"]            = df["home_NRtg"]            - df["visitor_NRtg"]
    df["diff_Pace"]            = df["home_Pace"]            - df["visitor_Pace"]
    df["diff_TS%"]             = df["home_TS%"]             - df["visitor_TS%"]
    df["diff_eFG%"]            = df["home_eFG%"]            - df["visitor_eFG%"]
    df["diff_TOV%"]            = df["home_TOV%"]            - df["visitor_TOV%"]
    df["diff_ORB%"]            = df["home_ORB%"]            - df["visitor_ORB%"]
    df["diff_FTr"]             = df["home_FTr"]             - df["visitor_FTr"]
    df["diff_3PAr"]            = df["home_3PAr"]            - df["visitor_3PAr"]
    df["diff_FT_FGA"]          = df["home_FT/FGA"]          - df["visitor_FT/FGA"]
    df["diff_D_eFG%"]          = df["home_D_eFG%"]          - df["visitor_D_eFG%"]
    df["diff_D_TOV%"]          = df["home_D_TOV%"]          - df["visitor_D_TOV%"]
    df["diff_D_DRB%"]          = df["home_D_DRB%"]          - df["visitor_D_DRB%"]
    df["diff_D_FT_FGA"]        = df["home_D_FT/FGA"]        - df["visitor_D_FT/FGA"]
    df["home_form"]            = df["home_win_pct_last5"]   - df["home_win_pct_last10"]
    df["visitor_form"]         = df["visitor_win_pct_last5"] - df["visitor_win_pct_last10"]
    df["diff_form"]            = df["home_form"]            - df["visitor_form"]

    return df
