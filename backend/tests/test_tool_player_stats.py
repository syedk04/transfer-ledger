"""Tests for backend/agent/tools/player_stats.py, plus the tool registry.

Run against the real trained model + real dataset, same rationale as
backend/tests/test_main.py: this tool is a thin wrapper over src.features,
so testing it against real data is what proves the wrapper (argument
names, the {"ok"/"error"} contract, error-message plumbing) is correct - a
mocked src.features would only prove the plumbing compiles.
"""

from backend.agent.tools import TOOL_REGISTRY, call_tool, tool_schemas
from backend.agent.tools.player_stats import get_player_stats


def test_get_player_stats_known_player():
    result = get_player_stats("Bukayo Saka", 2024)
    assert result["ok"] is True
    assert result["data"]["player"] == "Bukayo Saka"
    assert result["data"]["season"] == 2024
    assert result["data"]["market_value_eur"] > 0
    assert "goals_per_90" in result["data"]


def test_get_player_stats_unknown_player_returns_ok_false_not_raise():
    result = get_player_stats("Definitely Not A Real Player Xyz", 2024)
    assert result["ok"] is False
    assert "No player found" in result["error"]


def test_get_player_stats_season_outside_dataset_gets_a_distinct_hint():
    result = get_player_stats("Bukayo Saka", 1999)
    assert result["ok"] is False
    assert "outside this dataset's scope" in result["error"]


# --- registry / dispatch ----------------------------------------------------


def test_tool_schemas_covers_all_registered_tools():
    schemas = tool_schemas()
    assert len(schemas) == len(TOOL_REGISTRY)
    names = {schema["function"]["name"] for schema in schemas}
    assert names == set(TOOL_REGISTRY.keys())


def test_call_tool_dispatches_by_name():
    result = call_tool("get_player_stats", {"player_name": "Erling Haaland", "season": 2024})
    assert result["ok"] is True
    assert result["data"]["player"] == "Erling Haaland"


def test_call_tool_unknown_name_returns_ok_false_not_raise():
    result = call_tool("not_a_real_tool", {})
    assert result["ok"] is False
    assert "Unknown tool" in result["error"]
