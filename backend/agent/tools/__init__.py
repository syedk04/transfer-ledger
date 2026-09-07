"""Tool registry the agent loop dispatches against.

Each entry maps a tool's name (as the LLM will name it in a tool_call) to
its OpenAI-compatible JSON schema (for the Groq request's `tools` param)
and the actual Python callable to invoke. Centralizing this here means
backend/agent/loop.py never needs an if/elif chain over tool names - it
just looks the name up.

Tools are added here one at a time as they're built (get_player_stats
first, then run_valuation, explain_valuation, search_news).
"""

from __future__ import annotations

from backend.agent.tools.player_stats import TOOL_SCHEMA as PLAYER_STATS_SCHEMA
from backend.agent.tools.player_stats import get_player_stats
from backend.agent.tools.valuation import TOOL_SCHEMA as VALUATION_SCHEMA
from backend.agent.tools.valuation import run_valuation

TOOL_REGISTRY = {
    "get_player_stats": {"schema": PLAYER_STATS_SCHEMA, "callable": get_player_stats},
    "run_valuation": {"schema": VALUATION_SCHEMA, "callable": run_valuation},
}


def tool_schemas() -> list[dict]:
    """The `tools` list to pass straight into a Groq chat-completions call."""
    return [entry["schema"] for entry in TOOL_REGISTRY.values()]


def call_tool(name: str, arguments: dict) -> dict:
    """Dispatch a tool call by name. Returns the same {"ok"/"error"} shape
    every tool callable returns; an unknown tool name (the LLM hallucinated
    a tool that doesn't exist) is itself just another {"ok": False} result
    the loop can feed back, not a crash.
    """
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return {"ok": False, "error": f"Unknown tool '{name}'."}
    return entry["callable"](**arguments)
