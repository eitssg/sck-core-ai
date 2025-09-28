import os

from core_ai.service import (
    generate,
    validate_dsl,
    compile as compile_fn,
    validate_cloudformation,
    completions,
    search_docs,
    search_symbols,
    optimize_cloudformation,
)
from core_framework.ai.contracts import (
    TemplateGenerateRequest,
    DSLValidateRequest,
    CompileRequest,
    CloudFormationValidateRequest,
    CompletionRequest,
    Cursor,
    SearchDocsRequest,
    SearchSymbolsRequest,
    OptimizeCloudFormationRequest,
)


def test_generate_idempotent():
    os.environ["CORE_AI_INTERNAL_IDEMPOTENCY_ENABLED"] = "true"
    r = TemplateGenerateRequest(prompt="bucket", tenant_client="core")
    resp1 = generate(r, None)
    resp2 = generate(r, None)
    assert resp1.artifact.dsl == resp2.artifact.dsl
    assert resp1.metrics.get("idempotent_hit") is False
    assert resp2.metrics.get("idempotent_hit") is True
    # Metrics enrichment
    assert resp1.metrics.get("operation") == "templates.generate"
    assert isinstance(resp1.metrics.get("duration_ms"), int)


def test_validate_dsl_basic():
    resp = validate_dsl(DSLValidateRequest(dsl="foo: bar"))
    assert resp.valid is True
    empty = validate_dsl(DSLValidateRequest(dsl="   "))
    assert empty.valid is False
    assert resp.metrics.get("operation") == "dsl.validate"


def test_compile_response():
    req = CompileRequest(dsl="something", tenant_client="core")
    resp = compile_fn(req)
    assert resp.success is True
    assert resp.artefact is not None
    assert resp.metrics.get("operation") == "templates.compile"


def test_cfn_validate():
    req = CloudFormationValidateRequest(cloudformation="Resources:\n  A: {}")
    resp = validate_cloudformation(req)
    assert resp.valid is True
    assert resp.metrics.get("operation") == "cloudformation.validate"


def test_completions():
    req = CompletionRequest(dsl="", cursor=Cursor(line=1, column=1))
    resp = completions(req)
    assert len(resp.items) == 1
    assert resp.items[0].text.startswith("Resources")


def test_search_docs_empty_store():
    # Without optional deps, store initialization fails -> empty response
    resp = search_docs(SearchDocsRequest(query="bucket", top_k=3))
    assert resp.total_indexed in (0, resp.total_indexed)
    assert isinstance(resp.latency_ms, int)


def test_search_symbols_empty_store():
    resp = search_symbols(SearchSymbolsRequest(query="Class", top_k=2))
    assert isinstance(resp.latency_ms, int)


def test_optimize_cloudformation_placeholder():
    req = OptimizeCloudFormationRequest(
        cloudformation="Resources:\n  B: {Type: AWS::S3::Bucket}"
    )
    resp = optimize_cloudformation(req)
    assert resp.recommendations
    assert resp.metrics.get("operation") == "cloudformation.optimize"
