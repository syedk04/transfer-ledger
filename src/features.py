"""Turn player_seasons.csv into a model-ready feature matrix.

This module only computes and selects columns - it deliberately does not
scale numeric features or one-hot encode categoricals. That happens inside
the scikit-learn Pipeline in train.py instead, so that the exact same
fitted transform (mean/std from training data, categories seen at fit time)
is reapplied at prediction time. Doing it here with pandas would mean
recomputing statistics on whatever data happens to be loaded, which silently
drifts between training and inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CATEGORICAL_FEATURES,
    ID_COLUMNS,
    LOG_TARGET_COL,
    MIN_MINUTES_PLAYED,
    NUMERIC_FEATURES,
    PLAYER_SEASONS_CSV,
    TARGET_COL,
)


def _per_90(count: pd.Series, minutes_played: pd.Series) -> pd.Series:
    """Rate per 90 minutes played. Safe against division by zero because
    every row here has already been filtered to MIN_MINUTES_PLAYED > 0.
    """
    return count / (minutes_played / 90.0)


def engineer_features(player_seasons: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns to a player_seasons frame. Pure function -
    does not filter rows or touch the target, so it's easy to unit test.
    """
    df = player_seasons.copy()

    df["goals_per_90"] = _per_90(df["goals"], df["minutes_played"])
    df["assists_per_90"] = _per_90(df["assists"], df["minutes_played"])

    # Why age needs a squared term: value doesn't rise or fall linearly with
    # age, it rises through a player's early-to-mid 20s as they establish
    # themselves, peaks, then declines as physical output drops - a parabola,
    # not a line. A plain linear term forces the model to pick one constant
    # slope for a 17-year-old prospect and a 34-year-old veteran, which can't
    # represent a peak at all. Adding age_at_season_end**2 as a second
    # feature lets a *linear* model fit a curve (value ~ b1*age + b2*age^2),
    # which is the standard trick for representing non-monotonic effects
    # without leaving the linear-model framework. The alternative - and the
    # one a tree-based model like the RandomForest gets "for free" without
    # any manual feature engineering - is to let splits handle the
    # non-linearity directly.
    df["age_squared"] = df["age_at_season_end"] ** 2

    return df


def build_feature_matrix(
    player_seasons: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load (or accept) player_seasons, filter, engineer, and split into
    X (raw feature columns, untransformed), y (log1p target), and an id
    frame (player_id/name/season/raw euro value) for reporting.

    Returns
    -------
    X : DataFrame of NUMERIC_FEATURES + CATEGORICAL_FEATURES columns.
    y : Series, log1p(market_value_eur) - see train.py for why we model
        the log rather than raw euros.
    meta : DataFrame with ID_COLUMNS plus the untransformed target, aligned
        index-for-index with X and y, for joining predictions back to names.
    """
    if player_seasons is None:
        player_seasons = pd.read_csv(PLAYER_SEASONS_CSV)

    before = len(player_seasons)
    player_seasons = player_seasons[player_seasons["minutes_played"] >= MIN_MINUTES_PLAYED]
    print(
        f"[features] dropped {before - len(player_seasons)} rows under "
        f"{MIN_MINUTES_PLAYED} minutes played, kept {len(player_seasons)}"
    )

    df = engineer_features(player_seasons)
    df[LOG_TARGET_COL] = np.log1p(df[TARGET_COL])

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].reset_index(drop=True)
    y = df[LOG_TARGET_COL].reset_index(drop=True)
    meta = df[ID_COLUMNS + [TARGET_COL]].reset_index(drop=True)

    return X, y, meta


if __name__ == "__main__":
    X, y, meta = build_feature_matrix()
    print(X.head())
    print(X.describe())
