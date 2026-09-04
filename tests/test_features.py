"""Tests for the feature engineering step (src/features.py)."""

import numpy as np
import pandas as pd
import pytest

from src.features import build_feature_matrix, engineer_features


def _minimal_player_season(**overrides) -> pd.DataFrame:
    row = {
        "player_id": 1,
        "season": 2024,
        "appearances": 20,
        "minutes_played": 1800,  # exactly 20 full matches
        "goals": 9,
        "assists": 4,
        "yellow_cards": 2,
        "red_cards": 0,
        "name": "Test Player",
        "date_of_birth": "1998-01-01",
        "sub_position": "Right Winger",
        "position": "Attack",
        "foot": "right",
        "height_in_cm": 178.0,
        "age_at_season_end": 26.5,
        "market_value_eur": 40_000_000.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_per_90_rates_scale_with_minutes():
    df = engineer_features(_minimal_player_season())
    # 9 goals in 1800 minutes = 20 full matches -> 0.45 goals per 90.
    assert df.loc[0, "goals_per_90"] == pytest.approx(9 / 20)
    assert df.loc[0, "assists_per_90"] == pytest.approx(4 / 20)


def test_age_squared_matches_age():
    df = engineer_features(_minimal_player_season(age_at_season_end=30.0))
    assert df.loc[0, "age_squared"] == pytest.approx(900.0)


def test_engineer_features_does_not_mutate_input():
    original = _minimal_player_season()
    original_copy = original.copy()
    engineer_features(original)
    pd.testing.assert_frame_equal(original, original_copy)


def test_build_feature_matrix_filters_low_minutes():
    df = pd.concat(
        [
            _minimal_player_season(player_id=1, minutes_played=1800),
            _minimal_player_season(player_id=2, minutes_played=45),  # below MIN_MINUTES_PLAYED
        ],
        ignore_index=True,
    )
    X, y, meta = build_feature_matrix(df)
    assert len(X) == 1
    assert meta.loc[0, "player_id"] == 1


def test_build_feature_matrix_targets_log1p_of_value():
    df = _minimal_player_season(market_value_eur=40_000_000.0)
    X, y, meta = build_feature_matrix(df)
    assert y.iloc[0] == pytest.approx(np.log1p(40_000_000.0))
    # meta keeps the raw euro value around for reporting/inverting later.
    assert meta.loc[0, "market_value_eur"] == 40_000_000.0
