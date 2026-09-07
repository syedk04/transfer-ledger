"""Tests for backend/main.py's stub /report endpoint.

Runs against the real trained model + real dataset (models/xgboost.joblib,
data/processed/player_seasons.csv) rather than mocking src.predict /
src.explain - this commit's whole point is that the stub endpoint is a
genuinely working call into the ML pipeline, not a fixture. A fully mocked
version of this test would only prove the FastAPI wiring is correct, not
that the endpoint actually works end to end.
"""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_report_returns_real_prediction_for_known_player():
    response = client.post("/report", json={"player_name": "Bukayo Saka", "season": 2024})
    assert response.status_code == 200

    body = response.json()
    assert body["player"] == "Bukayo Saka"
    assert body["season"] == 2024
    assert body["model_used"] == "xgboost"
    assert body["predicted_value_eur"] > 0

    # Stub-specific: confidence/news/caveats are honest placeholders, not
    # fabricated agent output, until the agent is built in a later commit.
    assert body["confidence"] == "medium"
    assert body["news_context"] == []
    assert len(body["caveats"]) >= 1

    # key_factors come straight from SHAP, ranked by magnitude.
    assert len(body["key_factors"]) == 5
    ranks = [factor["magnitude_rank"] for factor in body["key_factors"]]
    assert ranks == sorted(ranks)


def test_report_unknown_player_returns_404_not_500():
    response = client.post(
        "/report", json={"player_name": "Definitely Not A Real Player Xyz", "season": 2024}
    )
    assert response.status_code == 404
    assert "No player found" in response.json()["detail"]
