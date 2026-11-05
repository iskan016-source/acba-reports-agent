"""Offline tests for the calculation tool and the API's security behavior."""

import time

from fastapi.testclient import TestClient

from app import RATE_LIMIT, app, recent
from bank_agent.agent import compare_profits

client = TestClient(app)


# ── compare_profits ──
def test_compare_profits_math():
    result = compare_profits(55.0, 42.5)
    assert result["status"] == "success"
    assert result["difference_billion_amd"] == 12.5
    assert result["percent_difference"] == 29.41


def test_compare_profits_zero():
    result = compare_profits(100, 0)
    assert result["status"] == "error"


# ── API security ──
def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_ask_without_key():
    # No X-API-Key header: the protected endpoint must refuse.
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 401


def test_ask_unknown_session():
    # Valid key but a session that was never created: a clean 404, not a crash.
    r = client.post(
        "/ask",
        headers={"x-api-key": "test-key"},  # matches conftest.py
        json={"question": "hi", "session_id": "does-not-exist"},
    )
    assert r.status_code == 404


def test_chat_rate_limit():
    # Use a dedicated IP so this test doesn't share counters with others, and
    # pre-fill it to the limit so the request is refused before Gemini is called.
    recent["9.9.9.9"].extend([time.time()] * RATE_LIMIT)

    r = client.post("/chat", json={"question": "hi"}, headers={"x-forwarded-for": "9.9.9.9"})
    assert r.status_code == 429
