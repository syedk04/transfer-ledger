"""Pydantic (v2) response/request models for the scouting-report API.

Rationale: these schemas are the contract between the ML pipeline (src/),
the (not-yet-built) agent that will narrate a scouting report, and whatever
frontend consumes /report. Defining them up front - before the agent exists -
lets the API be genuinely testable now (backend/main.py's stub endpoint
below returns real, valid instances of these models) and lets the agent be
built later against a fixed, already-agreed shape instead of improvising one
alongside the prompt engineering.

Nothing here talks to Groq or NewsData yet. `NewsCitation` and the
`news_context` field exist so the shape is settled ahead of time, but the
stub endpoint always returns an empty list for it - real citations are wired
up in a later commit.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Three tiers rather than a raw probability/score: the confidence an LLM
# agent can honestly report about its own valuation reasoning is qualitative
# ("the model had plenty of comparable data and the SHAP drivers are
# unsurprising" vs "this is a low-minutes rookie season with a thin
# comparison set") - a fake-precise numeric score would overstate exactly
# the kind of certainty this project's own docs (see explain.py) are
# careful to avoid.
Confidence = Literal["low", "medium", "high"]


class KeyFactor(BaseModel):
    """One SHAP feature contribution, reshaped for a human-facing report.

    This is a direct, minimally-transformed view of one entry from
    src/explain.py's `contributions` list (see explain_prediction's
    docstring for what `feature`/`direction` mean there) - deliberately not
    a paraphrase, so every claim in a report can be traced back to an exact
    SHAP value. `explanation` is the one field with room for narrative: for
    now (this commit) it's a placeholder sentence built mechanically from
    the feature name and direction; a later commit lets the LLM agent write
    a genuinely explanatory sentence in its place without changing the
    schema.
    """

    feature: str = Field(..., description="Human-readable feature name, e.g. 'goals_per_90'.")
    direction: Literal["positive", "negative"] = Field(
        ..., description="Whether this feature pushed the predicted value up or down."
    )
    magnitude_rank: int = Field(
        ...,
        ge=1,
        description="1 = the single biggest driver of this prediction, 2 = second biggest, etc. "
        "Mirrors the sort order of explain_prediction()'s contributions list.",
    )
    explanation: str = Field(
        ..., description="Human-readable sentence describing this factor's effect on the valuation."
    )


class NewsCitation(BaseModel):
    """One news article an agent is expected to cite when it makes a claim
    in `news_context` (e.g. an injury, a hot scoring streak, transfer
    speculation). Every field is required except `published_at`, because a
    citation with no title/source/url/snippet isn't actually checkable by a
    reader - if the agent (a later commit) can't produce all four, it
    should leave the claim out of the report rather than cite a fragment.
    """

    title: str
    source: str
    url: str
    snippet: str
    published_at: str | None = Field(
        default=None, description="ISO date string if the source provided one, else None."
    )


class ScoutingReport(BaseModel):
    """The full response shape for POST /report.

    This is the one artifact every later commit (the agent, the frontend)
    is built against, so its fields are deliberately conservative and
    traceable: `predicted_value_eur`/`model_used` come straight from
    src/predict.py, `key_factors` from src/explain.py, and `news_context`
    will come from the NewsData-backed retrieval step once it exists. Until
    then, `confidence`/`confidence_reasoning`/`news_context`/`caveats` are
    populated with honest placeholders (see backend/main.py) rather than
    fabricated content, so a client can already tell a "stub" report from a
    fully agent-generated one by reading `caveats`.
    """

    player: str
    season: int
    predicted_value_eur: float
    model_used: str
    confidence: Confidence
    confidence_reasoning: str = Field(
        ..., description="Why this confidence level was assigned, not just what it is."
    )
    key_factors: list[KeyFactor] = Field(
        default_factory=list,
        description="Top SHAP-driven factors behind the prediction, ranked by magnitude.",
    )
    news_context: list[NewsCitation] = Field(
        default_factory=list,
        description="Cited articles backing any real-world claims in the report. Empty until "
        "the news-retrieval step (a later commit) is wired up.",
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Honest limitations of this specific report (e.g. 'stub response', "
        "'thin sample of minutes this season').",
    )
    generated_at: str = Field(..., description="ISO 8601 timestamp of when this report was built.")


class ReportRequest(BaseModel):
    """Request body for POST /report. Mirrors the CLI args of src/predict.py
    and src/explain.py (player name/substring + season) so the same
    find_player_row() matching rules and error messages apply here too.
    """

    player_name: str
    season: int
