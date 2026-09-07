"""Tool 3/4: explain_valuation.

Wraps src.explain.explain_prediction. Only ever called against
SCOUTING_MODEL (xgboost) - explain_prediction itself rejects any other
model with a clear ValueError, which this tool surfaces as a normal
{"ok": False, ...} result rather than an unhandled exception reaching the
agent loop.
"""

from __future__ import annotations

from typing import Any

from src.explain import explain_prediction

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "explain_valuation",
        "description": (
            "Get SHAP feature-attribution values explaining WHY the model produced a "
            "particular predicted value for this player-season - which stats pushed the "
            "valuation up or down, and by how much (log-scale, additive). Call this after "
            "run_valuation so key_factors in the final report can cite a real SHAP value "
            "instead of a generic claim."
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


def explain_valuation(player_name: str, season: int) -> dict[str, Any]:
    try:
        result = explain_prediction(player_name, season)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    return {"ok": True, "data": result}
