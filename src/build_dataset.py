"""Aggregate raw transfermarkt CSVs into one row per player per PL season.

The target (market value) has to be the value that was *current at the end
of that season* - using today's value would leak information a model at
prediction time could never have had (e.g. training on a 2022 season row
but pricing it with knowledge that the player became a superstar by 2026).
This module's whole job is getting that join right without leaking.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from config import (
    COMPETITION_ID,
    PLAYER_SEASONS_CSV,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEASON_END_DAY,
    SEASON_END_MONTH,
    SEASONS,
)


def _season_end_date(season: int) -> dt.date:
    """The cutoff date treated as 'end of season' for a season starting in `season`.

    E.g. season=2024 (the 2024-25 season) ends 2025-06-30 by this convention.
    See config.py for why that specific date was chosen.
    """
    return dt.date(season + 1, SEASON_END_MONTH, SEASON_END_DAY)


def _load_games(games_path) -> pd.DataFrame:
    games = pd.read_csv(games_path, usecols=["game_id", "competition_id", "season"])
    return games[
        (games["competition_id"] == COMPETITION_ID) & (games["season"].isin(SEASONS))
    ][["game_id", "season"]]


def _load_appearances(appearances_path) -> pd.DataFrame:
    cols = [
        "game_id",
        "player_id",
        "yellow_cards",
        "red_cards",
        "goals",
        "assists",
        "minutes_played",
    ]
    return pd.read_csv(appearances_path, usecols=cols)


def _aggregate_appearances(appearances: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Inner-join appearances onto GB1 games, then sum stats per player per season.

    Joining on game_id (rather than trusting appearances' own competition_id
    column) is what actually restricts this to Premier League matches in
    scope: games.csv is the authoritative source for which competition and
    season a game_id belongs to.
    """
    merged = appearances.merge(games, on="game_id", how="inner")

    agg = merged.groupby(["player_id", "season"]).agg(
        appearances=("game_id", "count"),
        minutes_played=("minutes_played", "sum"),
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
    )
    return agg.reset_index()


def _attach_player_attributes(player_seasons: pd.DataFrame, players_path) -> pd.DataFrame:
    players = pd.read_csv(
        players_path,
        usecols=[
            "player_id",
            "name",
            "date_of_birth",
            "position",
            "sub_position",
            "foot",
            "height_in_cm",
        ],
    )
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    return player_seasons.merge(players, on="player_id", how="left")


def _attach_age(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Age in years at the season's end date, as a float (not integer years).

    Fractional age matters here: two players both "27" by integer age could
    be 27.05 and 27.95, nearly a year apart in actual career stage.
    """
    season_end = player_seasons["season"].map(_season_end_date)
    season_end = pd.to_datetime(season_end)
    age_days = (season_end - player_seasons["date_of_birth"]).dt.days
    player_seasons["age_at_season_end"] = age_days / 365.25
    return player_seasons


def _attach_market_value(player_seasons: pd.DataFrame, valuations_path) -> pd.DataFrame:
    """Join each player-season to the most recent valuation on/before that
    season's end date - i.e. the value that was actually known at the time,
    never a later one. This is the crux of avoiding target leakage.
    """
    valuations = pd.read_csv(
        valuations_path, usecols=["player_id", "date", "market_value_in_eur"]
    )
    valuations["date"] = pd.to_datetime(valuations["date"])
    valuations = valuations.sort_values("date")

    player_seasons = player_seasons.copy()
    player_seasons["cutoff_date"] = pd.to_datetime(
        player_seasons["season"].map(_season_end_date)
    )

    # merge_asof requires both frames sorted by the join key, and matches
    # each left row to the most recent right row with date <= cutoff_date
    # (direction="backward") within the same player_id (by="player_id").
    player_seasons = player_seasons.sort_values("cutoff_date")
    valuations = valuations.sort_values("date")

    merged = pd.merge_asof(
        player_seasons,
        valuations,
        left_on="cutoff_date",
        right_on="date",
        by="player_id",
        direction="backward",
    )
    merged = merged.rename(columns={"market_value_in_eur": "market_value_eur"})
    return merged.drop(columns=["date"])


def build_player_seasons() -> pd.DataFrame:
    """Build the player_seasons table and write it to PLAYER_SEASONS_CSV."""
    games = _load_games(RAW_DATA_DIR / "games.csv")
    appearances = _load_appearances(RAW_DATA_DIR / "appearances.csv")

    player_seasons = _aggregate_appearances(appearances, games)
    player_seasons = _attach_player_attributes(player_seasons, RAW_DATA_DIR / "players.csv")
    player_seasons = _attach_age(player_seasons)
    player_seasons = _attach_market_value(player_seasons, RAW_DATA_DIR / "player_valuations.csv")

    # A row with no valuation on/before its cutoff has no usable target -
    # e.g. the player's transfermarkt valuation history starts after this
    # season ended. Can't train or evaluate on it, so drop it.
    before = len(player_seasons)
    player_seasons = player_seasons.dropna(subset=["market_value_eur", "age_at_season_end"])
    after = len(player_seasons)
    print(f"[build_dataset] dropped {before - after} rows with no valuation/DOB, kept {after}")

    player_seasons = player_seasons.drop(columns=["cutoff_date"])
    player_seasons = player_seasons.sort_values(["season", "player_id"]).reset_index(drop=True)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    player_seasons.to_csv(PLAYER_SEASONS_CSV, index=False)
    print(f"[build_dataset] wrote {len(player_seasons)} rows to {PLAYER_SEASONS_CSV}")
    return player_seasons


if __name__ == "__main__":
    build_player_seasons()
