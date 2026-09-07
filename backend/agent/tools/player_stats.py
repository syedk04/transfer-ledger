"""Tool 1/4: get_player_stats.

Wraps the existing src.features pipeline rather than re-implementing any
lookup logic - this project already has one authoritative row-matching
function (find_player_row, shared by predict.py and explain.py), and the
agent tool layer should be a thin adapter over it, not a second copy of the
matching rules.

Every tool in backend/agent/tools/ returns the same uniform contract:
{"ok": True, "data": ...} or {"ok": False, "error": "<short, LLM-readable
reason>"} - never raises through to the agent loop. This lets the LLM see
a normal tool-result message it can react to (retry with a corrected name,
or degrade confidence and note a caveat) instead of the loop having to
special-case exceptions per tool.
"""

from __future__ import annotations

from typing import Any

from config import SEASONS, TEST_SEASON, TRAIN_SEASONS
from src.features import build_feature_matrix, find_player_row

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_player_stats",
        "description": (
            "Look up a Premier League player's raw season stats (appearances, minutes, "
            "goals, assists, cards, age, height, position) for one season. Use this first "
            "to ground any scouting report in real per-season performance data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "player_name": {
                    "type": "string",
                    "description": "Player name or unambiguous substring, e.g. 'Bukayo Saka'.",
                },
                "season": {
                    "type": "integer",
                    "description": "Season start year, e.g. 2024 for the 2024-25 season.",
                },
            },
            "required": ["player_name", "season"],
        },
    },
}


def _season_hint(season: int) -> str:
    """A one-line hint about why a season might be out of scope, distinct
    from "no data for this player" - the dataset only covers SEASONS at
    all, and TEST_SEASON (2025) is the held-out season the model was
    evaluated on rather than trained on. Getting this distinction right
    matters for the agent's confidence framing, not just the error text.
    """
    if season not in SEASONS:
        return (
            f"Season {season} is outside this dataset's scope entirely "
            f"(covers {min(SEASONS)}-{max(SEASONS)})."
        )
    if season == TEST_SEASON:
        return f"Season {season} is the held-out test season the model was evaluated on, not trained on."
    return f"Season {season} is one of the model's training seasons ({TRAIN_SEASONS})."


def get_player_stats(player_name: str, season: int) -> dict[str, Any]:
    X, _, meta = build_feature_matrix()
    try:
        idx = find_player_row(meta, player_name, season)
    except ValueError as exc:
        return {"ok": False, "error": f"{exc} {_season_hint(season)}"}

    row = meta.loc[idx]
    stats_row = X.loc[idx]

    return {
        "ok": True,
        "data": {
            "player": row["name"],
            "season": int(row["season"]),
            **{
                col: (val.item() if hasattr(val, "item") else val)
                for col, val in stats_row.items()
            },
            "market_value_eur": float(row["market_value_eur"]),
        },
    }
