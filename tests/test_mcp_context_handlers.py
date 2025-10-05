import json
import pytest

from core_ai.core_server import (
    handle_get_context_for_query,
    handle_get_indexing_stats,
    handle_initialize_indexes,
)


def _parse_textcontent_list(result):
    assert isinstance(result, (list, tuple)) and result, "Expected non-empty TextContent list"
    tc = result[0]
    # TextContent from mcp.types
    assert hasattr(tc, "text"), "Expected TextContent with a text field"
    return json.loads(tc.text)


@pytest.mark.asyncio
async def test_initialize_indexes_envelope_smoke():
    # Do not force rebuild to keep it fast
    res = await handle_initialize_indexes(force_rebuild=False)
    data = _parse_textcontent_list(res)
    assert data.get("status") == "success"
    assert "data" in data
    payload = data["data"]
    assert isinstance(payload, dict)
    assert "results" in payload
    # Results should contain keys for all sources
    results = payload["results"]
    for key in ("documentation", "codebase", "consumables", "actions"):
        assert key in results


@pytest.mark.asyncio
async def test_get_context_for_query_envelope_smoke():
    # Small query to exercise read path without heavy work
    res = await handle_get_context_for_query(query="cloudformation", strategy="balanced", max_results=4, fused=True)
    data = _parse_textcontent_list(res)
    assert data.get("status") == "success"
    assert "data" in data
    context = data["data"]
    # Context structure sanity checks
    for key in ("query", "strategy", "documentation", "codebase", "consumables", "actions", "summary"):
        assert key in context
    summary = context["summary"]
    for key in ("doc_results", "code_results", "consumables_results", "actions_results", "total_results"):
        assert key in summary


@pytest.mark.asyncio
async def test_get_indexing_stats_envelope_smoke():
    res = await handle_get_indexing_stats()
    data = _parse_textcontent_list(res)
    assert data.get("status") == "success"
    assert "data" in data
    stats = data["data"]
    for key in ("documentation", "codebase", "vector_store", "consumables", "actions"):
        assert key in stats
