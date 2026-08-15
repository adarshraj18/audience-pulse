"""End-to-end API tests against the real trained model. No TensorFlow
needed; this exercises the same NumPy path the deployed app uses."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_serves_frontend():
    res = client.get("/")
    assert res.status_code == 200
    assert "Audience Pulse" in res.text


def test_analyze_rejects_empty_text():
    res = client.post("/api/analyze", json={"text": "   "})
    assert res.status_code == 400


def test_analyze_returns_pulse_report():
    payload = {
        "text": (
            "This was a fantastic movie, I loved every second of it.\n\n"
            "Terrible film, a complete waste of time.\n\n"
            "It was an average watch, nothing special."
        )
    }
    res = client.post("/api/analyze", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert len(body["reviews"]) == 3
    assert body["positive_count"] + body["negative_count"] == 3
    assert 0 <= body["pulse_score"] <= 100
    assert len(body["harshest"]) <= 5
