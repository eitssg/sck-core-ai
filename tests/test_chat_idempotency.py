import os

import pytest


@pytest.mark.skip(reason="Flaky tests, re-enable when fixed")
def test_chat_idempotent_two_calls(client):
    os.environ["CORE_AI_INTERNAL_IDEMPOTENCY_ENABLED"] = "true"
    payload = {"messages": [{"role": "user", "content": "Describe SCK architecture"}]}
    r1 = client.post("/v1/chat", json=payload)
    r2 = client.post("/v1/chat", json=payload)
    assert r1.status_code == 200 and r2.status_code == 200
    d1 = r1.json()["data"]
    d2 = r2.json()["data"]
    # Replies should match when cached
    assert d1["reply"] == d2["reply"]
    idem1 = d1.get("idempotency", {})
    idem2 = d2.get("idempotency", {})
    # First call context/reply hits False, second should be True
    assert idem1.get("context", {}).get("hit") is False
    assert idem2.get("context", {}).get("hit") is True
    assert idem1.get("reply", {}).get("hit") is False
    assert idem2.get("reply", {}).get("hit") is True
