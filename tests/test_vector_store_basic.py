"""Basic tests for SimpleVectorStore add/search/clear operations.

These focus on exercising success paths without hitting persistence side effects.
"""

import pytest

from core_ai.indexing.simple_vector_store import SimpleVectorStore, SKLEARN_AVAILABLE

pytestmark = pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn / sentence-transformers not installed")


def test_add_and_search_documents(tmp_path):
    store = SimpleVectorStore(persist_directory=str(tmp_path))
    ok = store.add_documents(
        collection_name="codebase",
        documents=["alpha function", "beta class"],
        metadatas=[
            {"source": "test", "path": "a.py"},
            {"source": "test", "path": "b.py"},
        ],
        ids=["alpha", "beta"],
    )
    assert ok is True
    stats = store.get_collection_stats()
    assert "codebase" in stats and stats["codebase"]["document_count"] == 2

    results = store.search_documents("codebase", query="alpha", n_results=1)
    assert results and results[0]["id"] == "alpha"


def test_clear_collection(tmp_path):
    store = SimpleVectorStore(persist_directory=str(tmp_path))
    store.add_documents(
        collection_name="codebase",
        documents=["gamma content"],
        metadatas=[{"source": "test", "path": "g.py"}],
        ids=["gamma"],
    )
    assert store.clear_collection("codebase") is True
    assert store.get_collection_stats()["codebase"]["document_count"] == 0
