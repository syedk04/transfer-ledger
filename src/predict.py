"""CLI: look up a player's predicted vs actual market value for a season.

    python -m src.predict "Bukayo Saka" --season 2024
    python -m src.predict "Erling Haaland" --season 2025 --model random_forest
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from config import DEFAULT_MODEL, MODEL_PATHS, TARGET_COL
from src.features import build_feature_matrix


def _format_eur(value: float) -> str:
    return f"EUR {value:,.0f}"


def predict_player(name: str, season: int, model_name: str = DEFAULT_MODEL) -> dict:
    """Return predicted value, actual value, and the gap for one player-season.

    Raises ValueError if the model hasn't been trained yet, or if no row
    matches the given name/season (e.g. typo, or the player didn't clear
    MIN_MINUTES_PLAYED that season and was filtered out upstream).
    """
    model_path = MODEL_PATHS.get(model_name)
    if model_path is None or not model_path.exists():
        raise ValueError(
            f"No trained model found for '{model_name}' at {model_path}. "
            "Run `python -m src.train` first."
        )
    pipeline = joblib.load(model_path)

    X, y, meta = build_feature_matrix()

    # Case-insensitive substring match so "saka" or "bukayo saka" both work,
    # but require it be unambiguous - guessing between two matches would be
    # worse than just asking the user to be more specific.
    name_mask = meta["name"].str.contains(name, case=False, na=False)
    season_mask = meta["season"] == season
    matches = meta[name_mask & season_mask]

    if matches.empty:
        available = sorted(meta.loc[name_mask, "season"].unique().tolist())
        if available:
            raise ValueError(
                f"No {season} season row for a player matching '{name}'. "
                f"Seasons available for this player: {available}. "
                "(A player-season is also dropped upstream if they played "
                "fewer than MIN_MINUTES_PLAYED minutes that season.)"
            )
        raise ValueError(f"No player found matching '{name}'.")

    if len(matches) > 1:
        names = matches["name"].unique().tolist()
        raise ValueError(f"'{name}' matches multiple players: {names}. Be more specific.")

    idx = matches.index[0]
    x_row = X.loc[[idx]]
    actual_eur = float(meta.loc[idx, TARGET_COL])

    predicted_log = pipeline.predict(x_row)[0]
    predicted_eur = max(float(np.expm1(predicted_log)), 0.0)

    return {
        "name": meta.loc[idx, "name"],
        "season": int(meta.loc[idx, "season"]),
        "model": model_name,
        "predicted_eur": predicted_eur,
        "actual_eur": actual_eur,
        "gap_eur": predicted_eur - actual_eur,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Player name (or substring), e.g. 'Bukayo Saka'")
    parser.add_argument("--season", type=int, required=True, help="Season start year, e.g. 2024")
    parser.add_argument(
        "--model",
        choices=list(MODEL_PATHS.keys()),
        default=DEFAULT_MODEL,
        help=f"Which trained model to use (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args(argv)

    try:
        result = predict_player(args.name, args.season, args.model)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    gap_pct = 100 * result["gap_eur"] / result["actual_eur"] if result["actual_eur"] else float("nan")
    print(f"{result['name']} - {result['season']} season ({result['model']})")
    print(f"  Predicted value: {_format_eur(result['predicted_eur'])}")
    print(f"  Actual value:    {_format_eur(result['actual_eur'])}")
    print(f"  Gap:             {_format_eur(result['gap_eur'])} ({gap_pct:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
