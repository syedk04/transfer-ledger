"""End-to-end test of the agent loop with a MOCKED Groq client - this is
the one test in the suite that fakes an LLM response rather than running
against something real, since there is no live GROQ_API_KEY in CI and
burning real API calls in a test suite would be wasteful even with one.
The four tools themselves are real (real trained model, real dataset),
consistent with every other tool test in this suite - only the LLM is a
fixture.

This test exercises the full contract the loop promises: tool-call turn
-> synthesis turn -> grounding (a hallucinated feature/url is dropped,
not passed through) -> a valid ScoutingReport Pydantic object out the
other end.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from backend.agent.loop import run_agent
from backend.schemas import ScoutingReport


def _tool_call(call_id, name, arguments):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _response_with_tool_calls(tool_calls):
    message = MagicMock()
    message.content = None
    message.tool_calls = tool_calls
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


def _response_with_content(content: str):
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


def test_agent_loop_produces_valid_grounded_report():
    player, season = "Erling Haaland", 2024

    tool_call_turn = _response_with_tool_calls(
        [
            _tool_call("c1", "get_player_stats", {"player_name": player, "season": season}),
            _tool_call("c2", "run_valuation", {"player_name": player, "season": season}),
            _tool_call("c3", "explain_valuation", {"player_name": player, "season": season}),
            _tool_call("c4", "search_news", {"player_name": player}),
        ]
    )
    no_more_tool_calls_turn = _response_with_content(None)

    # The synthesis turn cites one real feature (goals - which explain_
    # valuation will genuinely return for Haaland) and one HALLUCINATED
    # feature name + one hallucinated news url that no tool ever returned -
    # both must be silently dropped by _assemble_report's grounding check.
    synthesis_json = json.dumps(
        {
            "confidence": "high",
            "confidence_reasoning": "Large sample of minutes and a clear top SHAP driver.",
            "key_factors": [
                {"feature": "goals", "explanation": "Goals were a major driver of value."},
                {"feature": "totally_made_up_feature", "explanation": "This should be dropped."},
            ],
            "news_context": [
                {"url": "https://not-a-real-article.example.com", "claim": "This should be dropped."}
            ],
            "caveats": ["Model has no feature for reputation/marketability premium."],
        }
    )
    synthesis_turn = _response_with_content(synthesis_json)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        tool_call_turn,
        no_more_tool_calls_turn,
        synthesis_turn,
    ]

    with patch("backend.agent.loop.get_client", return_value=fake_client), patch(
        "backend.agent.loop.get_model", return_value="llama-3.3-70b-versatile"
    ):
        report = run_agent(player, season)

    assert isinstance(report, ScoutingReport)
    assert report.player == player
    assert report.season == season
    assert report.model_used == "xgboost"
    assert report.predicted_value_eur > 0
    assert report.confidence == "high"

    # The real, tool-returned feature made it through...
    factor_features = {f.feature for f in report.key_factors}
    assert "goals" in factor_features
    # ...but the hallucinated one did not.
    assert "totally_made_up_feature" not in factor_features

    # The hallucinated news citation was dropped (search_news mocked tool
    # is never actually hit with a fake key here, so it legitimately
    # returns ok:false - either way, nothing hallucinated survives).
    for citation in report.news_context:
        assert citation.url != "https://not-a-real-article.example.com"


def test_agent_loop_raises_agent_error_when_run_valuation_never_succeeds():
    from backend.agent.loop import AgentError

    tool_call_turn = _response_with_tool_calls(
        [_tool_call("c1", "get_player_stats", {"player_name": "Nobody Real", "season": 2024})]
    )
    no_more_tool_calls_turn = _response_with_content(None)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [tool_call_turn, no_more_tool_calls_turn]

    with patch("backend.agent.loop.get_client", return_value=fake_client), patch(
        "backend.agent.loop.get_model", return_value="llama-3.3-70b-versatile"
    ):
        try:
            run_agent("Nobody Real", 2024)
            assert False, "expected AgentError"
        except AgentError:
            pass
