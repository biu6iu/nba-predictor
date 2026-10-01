import pandas as pd
import pytest

from src import data_loader
from src.config import SEASONS

N_TEAMS = 30


def _team_names(playoff_marked: bool = True) -> list[str]:
    # the first 16 teams made the playoffs, which Basketball Reference marks with a trailing *
    return [f"Team {i:02d}{'*' if playoff_marked and i < 16 else ''}" for i in range(N_TEAMS)]


def _write_season(directory, year: int, per_game_teams: list[str], advanced_teams: list[str]) -> None:
    pd.DataFrame({
        "Team": per_game_teams, "G": 82, "MP": 240.0, "PTS": 110.0,
    }).to_csv(directory / f"{year}PerGameData.csv", index=False)
    pd.DataFrame({
        "Team": advanced_teams, "W": 41, "L": 41, "SRS": 0.0,
    }).to_csv(directory / f"{year}AdvData.csv", index=False)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """A temp data directory holding a full set of well-formed CSVs; DATA_DIR is pointed at it."""
    for season in SEASONS:
        _write_season(tmp_path, int(season.split("-")[0]), _team_names(), _team_names())
    monkeypatch.setattr(data_loader, "DATA_DIR", tmp_path)
    return tmp_path


def test_all_30_teams_survive_the_merge_for_every_season(data_dir):
    stats = data_loader.load_team_stats()
    assert len(stats) == N_TEAMS * len(SEASONS)
    assert (stats.groupby("season")["Team"].nunique() == N_TEAMS).all()
    assert set(stats["season"]) == set(SEASONS)


def test_playoff_marker_is_stripped(data_dir):
    stats = data_loader.load_team_stats()
    assert not stats["Team"].str.contains("*", regex=False).any()
    assert set(stats["Team"]) == set(_team_names(playoff_marked=False))


def test_per_game_only_columns_are_dropped(data_dir):
    stats = data_loader.load_team_stats()
    assert {"G", "MP"}.isdisjoint(stats.columns)
    assert {"PTS", "W", "SRS"} <= set(stats.columns)


def test_missing_csv_raises_an_error_naming_the_file(data_dir):
    (data_dir / "2022AdvData.csv").unlink()
    with pytest.raises(FileNotFoundError, match="2022AdvData.csv"):
        data_loader.load_team_stats()


def test_unmatched_team_names_raise_instead_of_silently_dropping_rows(data_dir):
    names = _team_names()
    renamed = names[:-1] + ["Seattle SuperSonics"]
    _write_season(data_dir, 2021, names, renamed)
    with pytest.raises(ValueError, match="didn't merge cleanly") as exc:
        data_loader.load_team_stats()
    assert "Seattle SuperSonics" in str(exc.value)


def test_load_match_data_reads_from_data_dir(data_dir):
    pd.DataFrame({"Date": ["Tue Oct 19 2021"], "Visitor": ["A"], "visitorPTS": [100],
                  "Home": ["B"], "homePTS": [110]}).to_csv(data_dir / "matchData.csv", index=False)
    assert len(data_loader.load_match_data()) == 1


def test_load_match_data_missing_file_raises(data_dir):
    with pytest.raises(FileNotFoundError, match="matchData.csv"):
        data_loader.load_match_data()
