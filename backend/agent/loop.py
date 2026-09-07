"""The hand-rolled tool-use agent loop.

Deliberately not a framework (no LangGraph/CrewAI): this is a plain loop
over Groq's OpenAI-compatible chat-completions API, following the general
agent-engineering guidance to start with the simplest thing that works
and only add structure once that's demonstrably insufficient - four tools
and one fixed synthesis step never got there.

Design choices worth calling out:
- The LLM is NEVER trusted to invent the numbers in the final report.
  predicted_value_eur/model_used come straight from run_valuation's tool
  result; key_factors/news_context are only populated from an LLM-written
  explanation sentence GROUNDED to a feature name or article url a tool
  call actually returned (see _assemble_report) - a hallucinated feature
  name or url is silently dropped, never passed through. Only the
  qualitative parts (confidence level + its reasoning, explanatory
  sentences, caveats) are genuinely the LLM's own synthesis.
- Max MAX_TOOL_TURNS tool-calling turns, then the loop forces a final
  synthesis call regardless - this bounds worst-case Groq/NewsData usage
  per report to a small, fixed number of calls, well under free-tier
  rate limits.
- If the model's synthesis reply isn't valid JSON, one repair pass is
  attempted (feeding the parse error back) before falling back to an
  empty synthesis dict, which _assemble_report handles gracefully (real
  SHAP ranking used directly, confidence defaults to "low" with a caveat
  explaining why) - the endpoint should never 500 just because the LLM's
  JSON was malformed twice in a row.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from backend.agent.client import get_client, get_model
from backend.agent.prompts import SYNTHESIS_INSTRUCTIONS, SYSTEM_PROMPT
from backend.agent.tools import call_tool, tool_schemas
from backend.schemas import KeyFactor, NewsCitation, ScoutingReport

MAX_TOOL_TURNS = 3
TOP_N_KEY_FACTORS = 5


class AgentError(Exception):
    """Raised only when no report can be produced at all (e.g. the player
    was never found, so there is no real predicted_value_eur to report) -
    every other failure (SHAP unavailable, news unavailable, malformed
    LLM JSON) degrades into a caveat instead of raising.
    """


def run_agent(player_name: str, season: int) -> ScoutingReport:
    client = get_client()
    model = get_model()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Produce a scouting report for {player_name}, season {season}."
            ),
        },
    ]

    tool_results: dict[str, dict] = {}

    for _turn in range(MAX_TOOL_TURNS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas(),
            tool_choice="auto",
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            break

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tool_call in tool_calls:
            name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                result = {"ok": False, "error": "Malformed tool-call arguments (not valid JSON)."}
            else:
                result = call_tool(name, arguments)
            tool_results[name] = result
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
            )

    valuation = tool_results.get("run_valuation")
    if not valuation or not valuation.get("ok"):
        raise AgentError(
            valuation["error"] if valuation else
            f"The agent never successfully called run_valuation for '{player_name}' (season {season})."
        )

    explanation = tool_results.get("explain_valuation")
    news = tool_results.get("search_news")

    messages.append({"role": "user", "content": SYNTHESIS_INSTRUCTIONS})
    synthesis_response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    raw_content = synthesis_response.choices[0].message.content or ""

    try:
        synthesis = json.loads(raw_content)
    except json.JSONDecodeError:
        synthesis = _repair_json(client, model, messages, raw_content)

    return _assemble_report(player_name, season, valuation, explanation, news, synthesis)


def _repair_json(client, model: str, messages: list[dict], bad_content: str) -> dict:
    """One repair attempt: show the model its own malformed output and ask
    again for strict JSON. If this also fails, return {} - _assemble_report
    treats a missing/empty synthesis as "the LLM contributed nothing",
    not as a crash.
    """
    repair_messages = messages + [
        {"role": "assistant", "content": bad_content},
        {
            "role": "user",
            "content": "That was not valid JSON. Reply again with ONLY a single valid JSON object matching the requested schema, no other text.",
        },
    ]
    response = client.chat.completions.create(
        model=model, messages=repair_messages, response_format={"type": "json_object"}
    )
    try:
        return json.loads(response.choices[0].message.content or "")
    except json.JSONDecodeError:
        return {}


def _assemble_report(
    player_name: str,
    season: int,
    valuation: dict,
    explanation: dict | None,
    news: dict | None,
    synthesis: dict,
) -> ScoutingReport:
    data = valuation["data"]

    real_contributions = (
        explanation["data"]["contributions"] if explanation and explanation.get("ok") else []
    )
    real_by_feature = {c["feature"]: c for c in real_contributions}

    key_factors: list[KeyFactor] = []
    for rank, item in enumerate(synthesis.get("key_factors", []) or [], start=1):
        real = real_by_feature.get(item.get("feature"))
        if real is None:
            continue  # LLM cited a feature no tool actually returned - drop, don't fabricate
        key_factors.append(
            KeyFactor(
                feature=real["feature"],
                direction=real["direction"],
                magnitude_rank=len(key_factors) + 1,
                explanation=item.get("explanation")
                or f"{real['feature']} was a {real['direction']} driver of this valuation.",
            )
        )
        if len(key_factors) >= TOP_N_KEY_FACTORS:
            break

    if not key_factors and real_contributions:
        # Synthesis referenced no real feature at all (empty/malformed
        # synthesis) - fall back to the raw SHAP ranking directly rather
        # than shipping an empty key_factors list when real data exists.
        for rank, c in enumerate(real_contributions[:TOP_N_KEY_FACTORS], start=1):
            key_factors.append(
                KeyFactor(
                    feature=c["feature"],
                    direction=c["direction"],
                    magnitude_rank=rank,
                    explanation=f"{c['feature']} was a {c['direction']} driver of this valuation.",
                )
            )

    real_articles = news["data"]["articles"] if news and news.get("ok") else []
    real_by_url = {a["url"]: a for a in real_articles}
    news_context: list[NewsCitation] = []
    for item in synthesis.get("news_context", []) or []:
        matched = real_by_url.get(item.get("url"))
        if matched is None:
            continue  # can't cite an article no tool call actually returned
        news_context.append(NewsCitation(**matched))

    caveats: list[str] = list(synthesis.get("caveats", []) or [])
    if not explanation or not explanation.get("ok"):
        caveats.append(
            f"SHAP explanation was unavailable: {(explanation or {}).get('error', 'not called')}"
        )
    if not news or not news.get("ok"):
        caveats.append(f"News search was unavailable: {(news or {}).get('error', 'not called')}")
    elif not real_articles:
        caveats.append("No recent news articles were found for this player.")

    confidence = synthesis.get("confidence")
    confidence_reasoning = synthesis.get("confidence_reasoning")
    if confidence not in ("low", "medium", "high"):
        confidence = "low"
        confidence_reasoning = (
            "Defaulted to 'low': the agent did not return a usable confidence assessment."
        )

    return ScoutingReport(
        player=data["player"],
        season=data["season"],
        predicted_value_eur=data["predicted_eur"],
        model_used=data["model"],
        confidence=confidence,
        confidence_reasoning=confidence_reasoning or "No reasoning was provided.",
        key_factors=key_factors,
        news_context=news_context,
        caveats=caveats,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
