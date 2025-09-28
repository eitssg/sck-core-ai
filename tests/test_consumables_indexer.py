import pytest

from core_ai.indexing.consumables_indexer import ConsumablesIndexer


def test_consumables_indexer_build_index_basic():
    indexer = ConsumablesIndexer()
    index = indexer.build_index()
    # We expect at least one AWS resource style id containing '::'
    assert any(
        "::" in entry["id"] for entry in index
    ), "No resource ids discovered in consumables index"
    # Spot check shape
    sample = index[0]
    assert {
        "id",
        "category",
        "source_spec_key",
        "properties",
        "enum_index",
        "doc_urls",
    }.issubset(sample.keys())
    assert isinstance(sample["properties"], list)
    # Properties entries have at minimum a path
    if sample["properties"]:
        assert "path" in sample["properties"][0]
