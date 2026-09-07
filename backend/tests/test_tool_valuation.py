"""Tests for backend/agent/tools/valuation.py.

Run against the real trained xgboost model, same rationale as
test_tool_player_stats.py: this is a thin wrapper over src.predict, so
testing it for real is what proves the wrapper is correct.
"""

from backend.agent.tools.valuation import run_valuation


def test_run_valuation_known_player():
    result = run_valuation("Erling Haaland", 2024)
    assert result["ok"] is True
    assert result["data"]["model"] == "xgboost"
    assert result["data"]["predicted_eur"] > 0
    assert result["data"]["actual_eur"] > 0
    assert result["data"]["gap_eur"] == result["data"]["predicted_eur"] - result["data"]["actual_eur"]


def test_run_valuation_unknown_player_returns_ok_false_not_raise():
    result = run_valuation("Definitely Not A Real Player Xyz", 2024)
    assert result["ok"] is False
    assert "No player found" in result["error"]
