from fastapi.testclient import TestClient
from bank_agent.agent import compare_profits
from app import app
from app import RATE_LIMIT


client = TestClient(app)   # a fake browser that calls your API in memory, no server needed


# ── compare_profits ──
def test_compare_profits_math():
    result = compare_profits(55.0, 42.5)
    assert result["status"] == "success"
    assert result["difference_billion_amd"] == 12.5
    # TODO 1: assert the percent difference is 29.41
    assert result["percent_difference"] == 29.41


def test_compare_profits_zero():
    # TODO 2: call compare_profits with profit_b = 0 and assert the status is "error"
    result = compare_profits(100 , 0)
    assert result["status"] == "error"

# ── API security ──
def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_ask_without_key():
    # Send a question with NO x-api-key header -> the API must refuse
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 401


def test_ask_unknown_session():
    # Correct key, but a session_id that was never created -> polite 404, not a crash
    r = client.post(
        "/ask",
        headers={"x-api-key": "test-key"},          # the fake key from conftest.py
        json={"question": "hi", "session_id": "does-not-exist"},
    )
    assert r.status_code == 404

def test_chat_rate_limit():
    # A made-up IP so this test doesn't share counters with other tests.
    # Pre-fill its counter as if it already asked RATE_LIMIT questions,
    # so the next one must be refused (and Gemini is never called).
    from app import recent
    import time
    recent["9.9.9.9"].extend([time.time()] * RATE_LIMIT)

    r = client.post("/chat", json={"question": "hi"}, headers={"x-forwarded-for": "9.9.9.9"})
    assert r.status_code == 429