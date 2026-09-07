"""Tests for backend/agent/tools/news.py.

Unlike the other three tools, this one genuinely depends on an external,
rate/credit-limited API - so every test here mocks httpx.get rather than
hitting NewsData.io for real (consistent with the project's own stated
testing approach: "unit tests for each tool, mock the external APIs").
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import backend.cache as cache_module
from backend.agent.tools.news import search_news


def _fake_response(results):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"results": results}
    return response


def test_search_news_no_api_key_returns_ok_false(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_path / "cache.db")
    monkeypatch.delenv("NEWSDATA_API_KEY", raising=False)

    result = search_news("Bukayo Saka")
    assert result["ok"] is False
    assert "NEWSDATA_API_KEY" in result["error"]


def test_search_news_parses_articles_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_path / "cache.db")
    monkeypatch.setenv("NEWSDATA_API_KEY", "fake-key")

    raw = [
        {
            "title": "Saka returns to training",
            "source_id": "bbc",
            "link": "https://example.com/1",
            "description": "Saka is back in training after injury.",
            "pubDate": "2026-09-01 10:00:00",
        }
    ]
    with patch("httpx.get", return_value=_fake_response(raw)) as mock_get:
        result = search_news("Bukayo Saka")

    assert mock_get.call_count == 1
    assert result["ok"] is True
    assert result["data"]["source"] == "live"
    articles = result["data"]["articles"]
    assert len(articles) == 1
    assert articles[0]["title"] == "Saka returns to training"
    assert articles[0]["source"] == "bbc"
    assert articles[0]["url"] == "https://example.com/1"

    # Second call within the same ISO week hits the cache, no second HTTP call.
    with patch("httpx.get", return_value=_fake_response(raw)) as mock_get_2:
        result_2 = search_news("Bukayo Saka")
    assert mock_get_2.call_count == 0
    assert result_2["data"]["source"] == "cache"
    assert result_2["data"]["articles"] == articles


def test_search_news_zero_results_returns_empty_list_not_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_path / "cache.db")
    monkeypatch.setenv("NEWSDATA_API_KEY", "fake-key")

    with patch("httpx.get", return_value=_fake_response([])):
        result = search_news("Some Obscure Player")

    assert result["ok"] is True
    assert result["data"]["articles"] == []


def test_search_news_http_error_returns_ok_false(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_path / "cache.db")
    monkeypatch.setenv("NEWSDATA_API_KEY", "fake-key")

    import httpx

    with patch("httpx.get", side_effect=httpx.ConnectTimeout("timed out")):
        result = search_news("Bukayo Saka")

    assert result["ok"] is False
    assert "NewsData.io request failed" in result["error"]


def test_search_news_refuses_once_daily_credit_budget_reached(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_path / "cache.db")
    monkeypatch.setenv("NEWSDATA_API_KEY", "fake-key")

    from backend.agent.tools import news as news_module

    monkeypatch.setattr(news_module, "NEWSDATA_DAILY_CREDIT_BUDGET", 1)
    raw = [
        {"title": "A", "source_id": "s", "link": "u", "description": "d", "pubDate": "2026-09-01"}
    ]

    with patch("httpx.get", return_value=_fake_response(raw)):
        first = search_news("Player One")
        second = search_news("Player Two")  # different player -> not a cache hit

    assert first["ok"] is True
    assert second["ok"] is False
    assert "credit budget" in second["error"]
