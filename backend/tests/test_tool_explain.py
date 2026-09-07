"""Tests for backend/agent/tools/explain.py.

Run against the real trained xgboost model, same rationale as the other
tool test files.
"""

from backend.agent.tools.explain import explain_valuation


def test_explain_valuation_known_player():
    result = explain_valuation("Erling Haaland", 2024)
    assert result["ok"] is True
    assert result["data"]["model"] == "xgboost"
    assert len(result["data"]["contributions"]) > 0
    assert "predicted_eur" in result["data"]


def test_explain_valuation_unknown_player_returns_ok_false_not_raise():
    result = explain_valuation("Definitely Not A Real Player Xyz", 2024)
    assert result["ok"] is False
    assert "No player found" in result["error"]
