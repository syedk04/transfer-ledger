"""Tests for the season-end age calculation and season-end date convention
(src/build_dataset.py) - the part of the pipeline most at risk of silently
leaking future information if the cutoff logic is wrong.
"""

import datetime as dt

import pandas as pd
import pytest

from src.build_dataset import _attach_age, _season_end_date


def test_season_end_date_is_june_30_of_the_following_year():
    # Season 2024 == the 2024-25 season, which we treat as "closed" on
    # 2025-06-30 (see config.py for why that specific date was picked).
    assert _season_end_date(2024) == dt.date(2025, 6, 30)
    assert _season_end_date(2022) == dt.date(2023, 6, 30)


def test_attach_age_computes_fractional_years():
    df = pd.DataFrame(
        {
            "season": [2024],
            "date_of_birth": [pd.Timestamp("1998-06-30")],
        }
    )
    result = _attach_age(df)
    # Born 1998-06-30, season 2024 ends 2025-06-30 -> exactly 27 years old.
    assert result.loc[0, "age_at_season_end"] == pytest.approx(27.0, abs=0.01)


def test_attach_age_distinguishes_players_born_in_the_same_calendar_year():
    df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "date_of_birth": [pd.Timestamp("1998-01-01"), pd.Timestamp("1998-12-01")],
        }
    )
    result = _attach_age(df)
    younger, older = result.loc[1, "age_at_season_end"], result.loc[0, "age_at_season_end"]
    # Both are "27" by integer age, but nearly a year apart - this is exactly
    # why we keep age as a float instead of truncating to whole years.
    assert older - younger == pytest.approx(11 / 12, abs=0.05)


def test_attach_age_is_nan_for_missing_date_of_birth():
    df = pd.DataFrame({"season": [2024], "date_of_birth": [pd.NaT]})
    result = _attach_age(df)
    assert pd.isna(result.loc[0, "age_at_season_end"])
