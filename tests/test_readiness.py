def test_readiness_unavailable(client):
    """Simulate Langflow outage and assert /ready returns 503 error envelope.

    We directly null out the global langflow_client so the readiness probe
    exercises the strict failure branch without needing to modify the fixture.
    """
    from core_ai import server

    # Simulate outage
    server.langflow_client = None
    r = client.get("/ready")
    body = r.json()
    assert r.status_code == 200  # envelope itself succeeds
    assert body["status"] == "error"
    assert body["code"] == 503
    assert body["data"]["ai_engine"]["available"] is False
