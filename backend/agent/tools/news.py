"""Tool 4/4: search_news.

Hits NewsData.io's free tier (200 credits/day, 10 articles/request, the
/latest endpoint covering only the last 48 hours - there is no from_date/
to_date range filter or historical archive access on the free plan, so
this tool can only ever surface very recent news, not a player's full
history; that's an explicit, documented limitation, not a bug). Verified
against NewsData.io's own docs/blog at build time (2026-09) - re-verify
if this stops working, since free-tier limits and field names are exactly
the kind of thing that drifts.

Query construction: a bare player-name query returns a lot of noise (name
collisions, transfer-rumor chatter unrelated to valuation, articles about
a same-named person in an unrelated field). Scoping the query to
"<name>" AND (transfer OR contract OR injury OR performance) keeps
results relevant to what a scouting report actually needs, at the cost of
sometimes returning zero results for a quiet week - which is the correct,
honest outcome (see search_news's "no ok:false, just an empty list"
behavior below), not a bug to work around by loosening the query.

Credit budget: every call here is cached per (player_name, ISO week) in
backend.cache, and a simple daily counter (also in backend.cache) refuses
new NewsData calls once NEWSDATA_DAILY_CREDIT_BUDGET is reached for the
day, falling back to a clear "budget reached" result instead of silently
degrading or crashing. This matters because the credit budget is shared
across every visitor to a deployed demo, not per-user - a handful of
people trying several players each could otherwise exhaust it.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from backend.cache import (
    get_cached_news,
    get_credits_used_today,
    record_credit_usage,
    store_news,
)

NEWSDATA_BASE_URL = "https://newsdata.io/api/1/latest"
NEWSDATA_DAILY_CREDIT_BUDGET = 180  # leave headroom under the 200/day free cap
MAX_ARTICLES = 5

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_news",
        "description": (
            "Search for recent (last 48 hours) news articles about a player - form, injuries, "
            "contract situation, transfer speculation. Use this to ground any real-world claims "
            "in the report with a citable source. If it returns zero articles, do not invent "
            "one - report that no recent news was found."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "player_name": {"type": "string", "description": "Full player name, e.g. 'Bukayo Saka'."},
            },
            "required": ["player_name"],
        },
    },
}


def _week_key(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%G-W%V")  # ISO year-week, e.g. "2026-W36"


def _today_key(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d")


def _build_query(player_name: str) -> str:
    return f'"{player_name}" AND (transfer OR contract OR injury OR performance)'


def _parse_articles(raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map NewsData.io's response fields to this project's citation shape
    (title/source/url/snippet/published_at - matching backend.schemas.
    NewsCitation). Field names (title/link/source_id/description/pubDate)
    are NewsData's documented response shape as of build time; accessed
    defensively (.get with a fallback) since a live key wasn't available to
    verify a real response against - see module docstring.
    """
    articles = []
    for item in raw_results[:MAX_ARTICLES]:
        articles.append(
            {
                "title": item.get("title", "Untitled"),
                "source": item.get("source_id") or item.get("source_name") or "Unknown source",
                "url": item.get("link", ""),
                "snippet": (item.get("description") or item.get("content") or "")[:500],
                "published_at": item.get("pubDate"),
            }
        )
    return articles


def search_news(player_name: str) -> dict[str, Any]:
    week_key = _week_key()
    cached = get_cached_news(player_name, week_key)
    if cached is not None:
        return {"ok": True, "data": {"articles": cached, "source": "cache"}}

    today = _today_key()
    if get_credits_used_today(today) >= NEWSDATA_DAILY_CREDIT_BUDGET:
        return {
            "ok": False,
            "error": (
                "NewsData.io daily credit budget reached for today. News context is "
                "unavailable until it resets - proceed without it and note the gap as a caveat."
            ),
        }

    api_key = os.environ.get("NEWSDATA_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": "NEWSDATA_API_KEY is not configured - news search is unavailable.",
        }

    try:
        response = httpx.get(
            NEWSDATA_BASE_URL,
            params={
                "apikey": api_key,
                "q": _build_query(player_name),
                "language": "en",
                "size": MAX_ARTICLES,
            },
            timeout=10.0,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"NewsData.io request failed: {exc}"}

    record_credit_usage(today, 1)

    articles = _parse_articles(payload.get("results") or [])
    store_news(player_name, week_key, articles)

    return {"ok": True, "data": {"articles": articles, "source": "live"}}
