"""
tests/test_llm_client.py: Tests for LLMClient in dry-run mode and SQLite audit logging.
"""

import os
import sqlite3
import pytest
from underspec_sim.llm.llm_client import LLMClient


def test_dry_run_completion_and_sqlite_logging(tmp_path):
    db_file = str(tmp_path / "test_runs.db")
    client = LLMClient(model="claude-sonnet-4-6", db_path=db_file, dry_run=True)

    resp = client.complete(
        user_prompt="Explain testing requirements for kappa=0.3",
        metadata={"role": "simulated_user", "kappa": 0.3, "m_target": 7},
    )

    assert resp.is_dry_run is True
    assert resp.model == "claude-sonnet-4-6"
    assert "Specification" in resp.text
    assert resp.latency_seconds >= 0.0

    # Verify SQLite logging
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT id, model, is_dry_run, user_prompt, response_text FROM llm_runs")
    rows = c.fetchall()
    assert len(rows) == 1
    assert rows[0][1] == "claude-sonnet-4-6"
    assert rows[0][2] == 1
    assert "Explain testing requirements" in rows[0][3]
    conn.close()


def test_env_api_key_safety(monkeypatch):
    """Confirm client never hardcodes or exposes API key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(dry_run=True)
    assert client.api_key is None
    # Still succeeds cleanly because dry_run=True
    resp = client.complete("Hello assistant")
    assert resp.is_dry_run is True
