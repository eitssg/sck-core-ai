"""Tests for error and edge paths in service layer."""

from unittest.mock import patch, MagicMock

from core_framework.ai.contracts import (
    TemplateGenerateRequest,
    SearchDocsRequest,
    SearchSymbolsRequest,
    OptimizeCloudFormationRequest,
)

from core_ai import service


def test_generate_idempotent_hit(monkeypatch):
    req = TemplateGenerateRequest(
        prompt="hello",
        tenant_client="t1",
        client_id="c1",
        previous_dsl=None,
    )
    with patch("core_ai.service.LangflowClient") as mock_client_cls:
        inst = MagicMock()
        inst.process_sync.return_value = {"status": "success", "outputs": []}
        mock_client_cls.return_value = inst
        # First call (miss)
        r1 = service.generate(req, langflow=inst)
        assert r1.metrics["idempotent_hit"] is False
        # Second call (hit)
        r2 = service.generate(req, langflow=inst)
        assert r2.metrics["idempotent_hit"] is True


def test_generate_error_path(monkeypatch):
    req = TemplateGenerateRequest(
        prompt="err",
        tenant_client="t2",
        client_id="c1",
        previous_dsl=None,
    )
    with patch("core_ai.service.run_idempotent", side_effect=RuntimeError("boom")):
        try:
            service.generate(req, langflow=None)
        except RuntimeError as e:
            assert "boom" in str(e)


def test_search_docs_no_store(monkeypatch):
    # Force vector store init failure
    with patch("core_ai.service._SVS", side_effect=Exception("init fail")):
        resp = service.search_docs(SearchDocsRequest(query="abc", top_k=3))
        assert resp.total_indexed == 0 and resp.hits == []


def test_search_symbols_no_store(monkeypatch):
    with patch("core_ai.service._SVS", side_effect=Exception("init fail")):
        resp = service.search_symbols(SearchSymbolsRequest(query="abc", top_k=3))
        assert resp.hits == []


def test_optimize_cloudformation_basic():
    resp = service.optimize_cloudformation(
        OptimizeCloudFormationRequest(
            cloudformation="Resources:\n  Bucket: {Type: AWS::S3::Bucket}"
        )
    )
    assert resp.recommendations and isinstance(resp.recommendations, list)
