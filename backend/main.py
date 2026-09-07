"""FastAPI app exposing the transfer-value pipeline as a scouting-report API.

Run from the repo root, exactly like the existing `python -m src.train` /
`python -m src.predict` CLIs are - this is a deliberate convention, not an
oversight: `from config import ...` and `from src.explain import ...` below
only resolve because the process's working directory (and therefore the
first entry on sys.path) is the repo root, not backend/. Concretely:

    uvicorn backend.main:app --reload

This commit only wires the pipeline directly into a stub /report endpoint -
there is no agent/LLM/news logic yet (that's a later commit). The stub is
still a real, working call into src.predict / src.explain against the
actual trained model, not a hardcoded fixture.
"""

from __future__ import annotations

from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.cache import get_cached_report, store_report
from backend.schemas import KeyFactor, ReportRequest, ScoutingReport
from config import SCOUTING_MODEL
from src.explain import explain_prediction
from src.predict import predict_player

# Loaded here (rather than relying on the shell) so `uvicorn backend.main:app`
# picks up GROQ_API_KEY / NEWSDATA_API_KEY from a repo-root .env the moment
# those are introduced in a later commit - this commit doesn't read either
# var yet, but the loading needs to happen at process startup either way.
load_dotenv()

app = FastAPI(title="Transfer Value Predictor - Scouting Report API")

# TODO: allow_origins=["*"] is intentionally permissive for local
# development only. Once the frontend is deployed to Cloudflare Pages,
# this must be tightened to that exact origin - a wildcard origin combined
# with a public API is fine for a read-only stub, but should not ship as
# the permanent config.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# How many SHAP contributions to surface as `key_factors`. explain_prediction
# returns every transformed feature's contribution sorted by magnitude; a
# scouting report only needs the handful of biggest drivers, not the full
# (often ~10-15 item, one-hot-expanded) list.
TOP_N_FACTORS = 5


@app.get("/health")
def health() -> dict:
    """Liveness check. No dependency on the model/data being loadable, so
    this stays fast and always-200 as long as the process is up."""
    return {"status": "ok"}


@app.post("/report", response_model=ScoutingReport)
def report(request: ReportRequest) -> ScoutingReport:
    """Stub scouting-report endpoint.

    Populates predicted_value_eur/model_used/key_factors from the real
    model + SHAP pipeline (src.predict / src.explain). confidence,
    confidence_reasoning, news_context, and caveats are honest placeholders
    - narrative confidence reasoning and news citation come from the agent
    built in a later commit, not from this endpoint.

    find_player_row() (shared by predict_player/explain_prediction) raises
    ValueError for an unmatched or ambiguous name/season - that's a client
    input error, not a server fault, so it's translated to a 404 here
    instead of propagating into an unhandled 500.

    Checked against backend.cache before doing any real work: repeat
    requests for the same (player, season, model) are the expected common
    case (someone re-opening the dashboard, or the agent's own retries in
    a later commit), and re-running the pipeline for those would be pure
    waste even though this stub doesn't yet spend any paid/rate-limited
    credit - the cache is wired in now so later commits (Groq, NewsData)
    get it for free instead of needing their own follow-up commit.
    """
    cached = get_cached_report(request.player_name, request.season, SCOUTING_MODEL)
    if cached is not None:
        return ScoutingReport(**cached)

    try:
        # Explicitly SCOUTING_MODEL, not predict_player's own default (ridge,
        # the interpretable baseline predict.py's CLI favors) - explain_
        # prediction() only supports SCOUTING_MODEL (SHAP needs a tree
        # model), so the predicted value and its SHAP explanation must come
        # from the same model or predicted_value_eur/key_factors would
        # silently describe two different predictions.
        prediction = predict_player(request.player_name, request.season, model_name=SCOUTING_MODEL)
        explanation = explain_prediction(request.player_name, request.season)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    key_factors = [
        KeyFactor(
            feature=contribution["feature"],
            direction=contribution["direction"],
            magnitude_rank=rank,
            explanation=(
                f"{contribution['feature']} was a "
                f"{'positive' if contribution['direction'] == 'positive' else 'negative'} "
                "driver of this valuation (placeholder - narrative explanation not yet "
                "generated by an agent)."
            ),
        )
        for rank, contribution in enumerate(explanation["contributions"][:TOP_N_FACTORS], start=1)
    ]

    scouting_report = ScoutingReport(
        player=prediction["name"],
        season=prediction["season"],
        predicted_value_eur=prediction["predicted_eur"],
        model_used=explanation["model"],
        confidence="medium",
        confidence_reasoning="placeholder — full agent reasoning not yet wired up",
        key_factors=key_factors,
        news_context=[],
        caveats=[
            "news/LLM synthesis not yet implemented — this is a stub response wired "
            "directly to the model+SHAP pipeline"
        ],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    store_report(prediction["name"], prediction["season"], SCOUTING_MODEL, scouting_report.model_dump())
    return scouting_report
