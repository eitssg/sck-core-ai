import os


def test_list_tools_endpoint_basic(client):
    r = client.get("/v1/tools")
    assert r.status_code == 200
    data = r.json()["data"]
    assert "tools" in data
    names = {t["name"] for t in data["tools"]}
    # Core tools always present
    assert {"lint_yaml", "validate_cloudformation", "suggest_completion"}.issubset(names)


def test_invoke_lint_yaml_tool(client):
    payload = {"content": "Resources:\n"}
    r = client.post("/v1/tools/lint_yaml/invoke", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert "result" in body["data"]


def test_chat_endpoint_minimal(client):
    req = {"messages": [{"role": "user", "content": "Explain what this service does."}]}
    r = client.post("/v1/chat", json=req)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    data = body["data"]
    assert "reply" in data
    assert "context_used" in data


def test_chat_stream_sse_events(client):
    req = {"messages": [{"role": "user", "content": "What is indexing?"}]}
    with client.stream("POST", "/v1/chat/stream", json=req) as s:
        events = []
        for line in s.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            if line.startswith("event: end"):
                break
    assert {"start", "answer", "end"}.issubset(set(events))


def test_tool_not_found(client):
    r = client.post("/v1/tools/no_such_tool/invoke", json={})
    assert r.status_code == 404


def test_chat_validation_error(client):
    r = client.post("/v1/chat", json={"messages": []})
    # FastAPI model validation should reject empty list (custom logic raises 400)
    assert r.status_code in (400, 422)
