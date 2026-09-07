"""Tests for backend/cache.py, against a temporary on-disk SQLite file so
tests never touch (or depend on) a real backend/cache.db.
"""

from __future__ import annotations

import backend.cache as cache_module


def test_cache_miss_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_path / "cache.db")

    assert cache_module.get_cached_report("Bukayo Saka", 2024, "xgboost") is None


def test_store_then_get_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_path / "cache.db")

    report = {"predicted_value_eur": 62_000_000, "player": "Bukayo Saka"}
    cache_module.store_report("Bukayo Saka", 2024, "xgboost", report)

    assert cache_module.get_cached_report("Bukayo Saka", 2024, "xgboost") == report


def test_lookup_is_case_and_whitespace_insensitive(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_path / "cache.db")

    cache_module.store_report("Bukayo Saka", 2024, "xgboost", {"ok": True})

    assert cache_module.get_cached_report("  bukayo saka  ", 2024, "xgboost") == {"ok": True}


def test_different_model_is_a_separate_cache_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_path / "cache.db")

    cache_module.store_report("Bukayo Saka", 2024, "xgboost", {"model": "xgboost"})
    cache_module.store_report("Bukayo Saka", 2024, "ridge", {"model": "ridge"})

    assert cache_module.get_cached_report("Bukayo Saka", 2024, "xgboost") == {"model": "xgboost"}
    assert cache_module.get_cached_report("Bukayo Saka", 2024, "ridge") == {"model": "ridge"}


def test_store_report_overwrites_existing_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DB_PATH", tmp_path / "cache.db")

    cache_module.store_report("Bukayo Saka", 2024, "xgboost", {"version": 1})
    cache_module.store_report("Bukayo Saka", 2024, "xgboost", {"version": 2})

    assert cache_module.get_cached_report("Bukayo Saka", 2024, "xgboost") == {"version": 2}
