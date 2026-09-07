"""Tool 2/4: run_valuation.

Thin wrapper over src.predict.predict_player, pinned to SCOUTING_MODEL
(xgboost) - never the caller's choice of model. The agent's later
explain_valuation call assumes the same model produced both the number
and its SHAP explanation (see backend/main.py's /report stub for the bug
this exact mismatch caused before it was pinned there too).
"""

from __future__ import annotations

from typing import Any

from config import SCOUTING_MODEL
from src.predict import predict_player

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_valuation",
        "description": (
            "Run the trained XGBoost model to predict a player's market value (EUR) for a "
            "given season, and compare it to their actual known market value. Always call "
            "get_player_stats first so the valuation can be grounded in real season stats."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "player_name": {"type": "string", "description": "Player name or unambiguous substring."},
                "season": {"type": "integer", "description": "Season start year, e.g. 2024."},
            },
            "required": ["player_name", "season"],
        },
    },
}


def run_valuation(player_name: str, season: int) -> dict[str, Any]:
    try:
        result = predict_player(player_name, season, model_name=SCOUTING_MODEL)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    return {
        "ok": True,
        "data": {
            "player": result["name"],
            "season": result["season"],
            "model": SCOUTING_MODEL,
            "predicted_eur": result["predicted_eur"],
            "actual_eur": result["actual_eur"],
            "gap_eur": result["gap_eur"],
        },
    }
