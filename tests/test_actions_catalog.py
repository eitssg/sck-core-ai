"""Basic tests for actions catalog endpoints.

These tests intentionally avoid deep structural assertions – they simply
verify that the endpoints respond with the standard envelope and that
search filtering returns a subset when a query is applied (if any items
exist). If core_execute is not installed the catalog may be empty; tests
handle that gracefully.
"""

from fastapi.testclient import TestClient

from core_ai.server import app, create_envelope_response  # noqa: F401 (import side effects)


client = TestClient(app)


def test_actions_catalog_envelope():
    resp = client.get("/v1/catalog/actions")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "success"
    assert isinstance(payload.get("data", {}).get("items"), list)


def test_actions_catalog_search_subset():
    base = client.get("/v1/catalog/actions").json()["data"]["items"]
    if not base:  # nothing to assert further – environment missing actions
        return
    first_id = base[0]["id"]
    resp = client.get(f"/v1/catalog/actions/search?q={first_id[:4]}")
    assert resp.status_code == 200
    data = resp.json()["data"]["items"]
    # Should contain at least one item matching partial
    assert any(first_id == i["id"] for i in data)
