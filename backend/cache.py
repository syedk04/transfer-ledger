"""SQLite-backed cache for scouting-report runs.

Why SQLite instead of Supabase/Postgres for a single-user demo: the only
job of this cache is avoiding re-spending Groq/NewsData credits on a
repeat (player, season, model) request. That's a single-process,
low-concurrency workload - exactly SQLite's sweet spot - with none of the
durability requirements that would justify a hosted Postgres. The
honest tradeoff (documented in the README): Render's free tier gives the
backend an ephemeral filesystem, so this cache is wiped on every
redeploy/restart, not durable across them. A Supabase project would
survive that, but free-tier Supabase projects auto-pause after 7 days of
inactivity and need a manual resume - for a portfolio demo a recruiter
might open sporadically, "cache is empty after a while" is a strictly
better failure mode than "first request after a week hard-fails on a
paused database." If this were a real product with real durability
needs, this is the first piece that would move to Postgres/Redis.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

CACHE_DB_PATH = Path(__file__).resolve().parent / "cache.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            player_name TEXT NOT NULL,
            season INTEGER NOT NULL,
            model TEXT NOT NULL,
            report_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (player_name, season, model)
        )
        """
    )
    return conn


def _key(player_name: str, season: int, model: str) -> tuple[str, int, str]:
    # Lowercased so "Bukayo Saka" and "bukayo saka" share a cache entry -
    # find_player_row()'s own matching is already case-insensitive, so two
    # requests that resolve to the same player shouldn't be cached separately
    # just because of input casing.
    return (player_name.strip().lower(), season, model)


def get_cached_report(player_name: str, season: int, model: str) -> dict[str, Any] | None:
    """Return a previously cached report dict, or None on a cache miss.

    A miss is the normal, expected case for any new (player, season, model)
    combination - callers should fall through to running the real pipeline,
    not treat None as an error.
    """
    key = _key(player_name, season, model)
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT report_json FROM runs WHERE player_name = ? AND season = ? AND model = ?",
            key,
        ).fetchone()
    finally:
        conn.close()
    return json.loads(row[0]) if row else None


def store_report(player_name: str, season: int, model: str, report: dict[str, Any]) -> None:
    """Persist a report, keyed by the resolved (player, season, model).

    INSERT OR REPLACE rather than INSERT: a re-run of the same key (e.g.
    after a model retrain, or a repeated request before eviction is ever
    implemented) should overwrite the stale cached report, not fail on a
    duplicate primary key.
    """
    key = _key(player_name, season, model)
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO runs (player_name, season, model, report_json, created_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            (*key, json.dumps(report)),
        )
        conn.commit()
    finally:
        conn.close()
