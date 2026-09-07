"""FastAPI app exposing the transfer-value pipeline as a scouting-report API.

Run from the repo root, exactly like the existing `python -m src.train` /
`python -m src.predict` CLIs are - this is a deliberate convention, not an
oversight: `from config import ...` and `from src.explain import ...` below
only resolve because the process's working directory (and therefore the
first entry on sys.path) is the repo root, not backend/. Concretely:

    uvicorn backend.main:app --reload

POST /report runs the full agent (backend.agent.loop.run_agent) when
GROQ_API_KEY is configured. Without one (e.g. CI, or a fresh local
checkout before the reader has signed up for a free Groq key), it
gracefully degrades to a deterministic stub wired directly to
src.predict/src.explain - same model, same SHAP values, just without an
LLM's narrative synthesis - rather than making the whole API unusable
without external credentials. Every returned report's `caveats` says
plainly which path produced it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.agent.loop import AgentError, run_agent
from backend.cache import get_cached_report, store_report
from backend.schemas import KeyFactor, ReportRequest, ScoutingReport
from config import SCOUTING_MODEL
from src.explain import explain_prediction
from src.predict import predict_player

# Loaded here (rather than relying on the shell) so `uvicorn backend.main:app`
# picks up GROQ_API_KEY / NEWSDATA_API_KEY from a repo-root .env.
load_dotenv()

app = FastAPI(title="Transfer Value Predictor - Scouting Report API")

# TODO: allow_origins=["*"] is intentionally permissive for local
# development only. Once the frontend is deployed to Cloudflare Pages,
# this must be tightened to that exact origin - a wildcard origin combined
# with a public API is fine for a read-only demo, but should not ship as
# the permanent config.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# How many SHAP contributions to surface as `key_factors` in the stub path.
# explain_prediction returns every transformed feature's contribution sorted
# by magnitude; a scouting report only needs the handful of biggest drivers.
TOP_N_FACTORS = 5


@app.get("/health")
def health() -> dict:
    """Liveness check. No dependency on the model/data being loadable, so
    this stays fast and always-200 as long as the process is up."""
    return {"status": "ok"}


@app.post("/report", response_model=ScoutingReport)
def report(request: ReportRequest) -> ScoutingReport:
    """Full scouting-report endpoint: agent-synthesized when Groq is
    configured, a deterministic model+SHAP stub otherwise.

    find_player_row() (shared by predict_player/explain_prediction, and
    by every agent tool) raises ValueError for an unmatched or ambiguous
    name/season - that's a client input error, not a server fault, so
    it's translated to a 404 here instead of propagating into an
    unhandled 500. AgentError is the equivalent case surfaced through the
    agent path (e.g. the model never resolved a real player despite
    retrying) and is handled the same way.
    """
    cached = get_cached_report(request.player_name, request.season, SCOUTING_MODEL)
    if cached is not None:
        return ScoutingReport(**cached)

    try:
        scouting_report = run_agent(request.player_name, request.season)
    except RuntimeError:
        # GROQ_API_KEY not configured - fall back to the deterministic stub
        # rather than making the whole API unusable without a Groq account.
        scouting_report = _stub_report(request.player_name, request.season)
    except AgentError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    store_report(
        scouting_report.player, scouting_report.season, SCOUTING_MODEL, scouting_report.model_dump()
    )
    return scouting_report


def _stub_report(player_name: str, season: int) -> ScoutingReport:
    """Deterministic fallback used when no Groq key is configured: real
    model + real SHAP values, honest placeholders for the parts only an
    LLM can synthesize (confidence reasoning, news citations).
    """
    try:
        # Explicitly SCOUTING_MODEL, not predict_player's own default (ridge,
        # the interpretable baseline predict.py's CLI favors) - explain_
        # prediction() only supports SCOUTING_MODEL (SHAP needs a tree
        # model), so the predicted value and its SHAP explanation must come
        # from the same model or predicted_value_eur/key_factors would
        # silently describe two different predictions.
        prediction = predict_player(player_name, season, model_name=SCOUTING_MODEL)
        explanation = explain_prediction(player_name, season)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    key_factors = [
        KeyFactor(
            feature=contribution["feature"],
            direction=contribution["direction"],
            magnitude_rank=rank,
            explanation=(
                f"{contribution['feature']} was a {contribution['direction']} driver of this "
                "valuation (mechanical description - no GROQ_API_KEY configured, so no LLM "
                "narrative synthesis ran for this report)."
            ),
        )
        for rank, contribution in enumerate(explanation["contributions"][:TOP_N_FACTORS], start=1)
    ]

    return ScoutingReport(
        player=prediction["name"],
        season=prediction["season"],
        predicted_value_eur=prediction["predicted_eur"],
        model_used=explanation["model"],
        confidence="medium",
        confidence_reasoning=(
            "No GROQ_API_KEY configured - this is a deterministic stub report, not an "
            "agent-assessed confidence level."
        ),
        key_factors=key_factors,
        news_context=[],
        caveats=[
            "GROQ_API_KEY is not configured - this report is a deterministic stub wired "
            "directly to the model+SHAP pipeline, with no LLM synthesis or news search."
        ],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
