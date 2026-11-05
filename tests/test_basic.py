from fastapi.testclient import TestClient
from bank_agent.agent import compare_profits
from app import app

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