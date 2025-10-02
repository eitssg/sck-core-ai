"""Domain service layer for sck-core-ai.

Each function accepts a shared contract request model instance and returns the
corresponding response model. Langflow integration is isolated so future
backends (direct LLM, rule engine) can slot in without route changes.
"""

from __future__ import annotations

import time

import core_logging as log

from core_framework.ai.contracts import (
    TemplateGenerateRequest,
    TemplateGenerateResponse,
    GeneratedTemplateArtifact,
    DSLValidateRequest,
    DSLValidateResponse,
    CompileRequest,
    CompileResponse,
    CompiledArtefact,
    CloudFormationValidateRequest,
    CloudFormationValidateResponse,
    CompletionRequest,
    CompletionResponse,
    CompletionItem,
    SearchDocsRequest,
    SearchDocsResponse,
    SearchSymbolsRequest,
    SearchSymbolsResponse,
    SearchHit,
    OptimizeCloudFormationRequest,
    OptimizeCloudFormationResponse,
)

from .langflow.client import LangflowClient
from .cache import run_idempotent
from .indexing.simple_vector_store import SimpleVectorStore as _SVS  # type: ignore

_VECTOR_STORE: _SVS | None = None


def _vs() -> _SVS | None:
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        try:
            _VECTOR_STORE = _SVS()
        except Exception as e:  # pragma: no cover
            log.warning("Vector store init failed", error=str(e))
            return None
    return _VECTOR_STORE


def generate(req: TemplateGenerateRequest, langflow: LangflowClient | None) -> TemplateGenerateResponse:
    start = time.time()

    def _produce() -> dict:
        # For now delegate fully to Langflow mockable response. Placeholder logic.
        if langflow:
            _ = langflow.process_sync({"input_value": req.prompt})  # ignoring raw envelope
        artefact = {
            "dsl": req.previous_dsl or f"generated_dsl_for:{req.prompt}",
            "rationale": "heuristic draft",
            "assumptions": ["mock"],
            "warnings": [],
        }
        return {
            "artifact": artefact,
            "issues": [],
            "metrics": {"mock": True},
        }

    result_dict, meta = run_idempotent(
        operation="templates.generate",
        payload=req.model_dump(),
        scope=req.tenant_client,
        producer=_produce,
    )
    resp = TemplateGenerateResponse.model_validate(result_dict)
    resp.metrics.update(
        {
            "idempotent_hit": meta["hit"],
            "operation": "templates.generate",
            "duration_ms": int((time.time() - start) * 1000),
        }
    )
    return resp


def validate_dsl(req: DSLValidateRequest) -> DSLValidateResponse:
    start = time.time()
    valid = bool(req.dsl.strip())
    return DSLValidateResponse(
        valid=valid,
        errors=[] if valid else [],
        warnings=[],
        suggestions=[],
        inferred_metadata={"length": len(req.dsl)},
        metrics={
            "mock": True,
            "operation": "dsl.validate",
            "duration_ms": int((time.time() - start) * 1000),
        },
    )


def compile(req: CompileRequest) -> CompileResponse:
    start = time.time()
    artefact = CompiledArtefact(
        cloudformation=f"Resources:\n  Mock: {{Type: AWS::S3::Bucket}}\n# source hash:{hash(req.dsl) & 0xffff:x}",
        artefact_id="mock-artefact",
        resources_count=1,
        metadata={"mock": True},
    )
    return CompileResponse(
        success=True,
        artefact=artefact,
        issues=[],
        metrics={
            "compile_ms": 5,
            "operation": "templates.compile",
            "duration_ms": int((time.time() - start) * 1000),
        },
    )


def validate_cloudformation(
    req: CloudFormationValidateRequest,
) -> CloudFormationValidateResponse:
    start = time.time()
    txt = req.cloudformation
    valid = "Resources" in txt
    return CloudFormationValidateResponse(
        valid=valid,
        errors=[],
        warnings=[],
        suggestions=[],
        risk_summary={"mock": True},
        metrics={
            "size_bytes": len(txt),
            "operation": "cloudformation.validate",
            "duration_ms": int((time.time() - start) * 1000),
        },
    )


def completions(req: CompletionRequest) -> CompletionResponse:
    start = time.time()
    item = CompletionItem(
        text="Resources:",
        label="Resources section",
        detail="CloudFormation root Resources",
    )
    duration_ms = int((time.time() - start) * 1000)
    return CompletionResponse(
        items=[item],
        truncated=False,
        generation_ms=duration_ms,
    )


def search_docs(req: SearchDocsRequest) -> SearchDocsResponse:
    start = time.time()
    store = _vs()
    if not store:
        return SearchDocsResponse(hits=[], total_indexed=0, latency_ms=0)
    results = store.search_documents("docs", req.query, n_results=req.top_k)
    hits: list[SearchHit] = []
    for r in results:
        hits.append(
            SearchHit(
                hit_type="documentation",  # type: ignore
                title=r.get("id", "doc"),
                snippet=r.get("document", "")[:240],
                score=float(r.get("similarity", 0.0)),
                source_id=r.get("id", "unknown"),
                metadata={"distance": r.get("distance")},
            )
        )
    total = store.get_collection_stats().get("docs", {}).get("document_count", 0)
    return SearchDocsResponse(
        hits=hits,
        total_indexed=total,
        latency_ms=int((time.time() - start) * 1000),
    )


def search_symbols(req: SearchSymbolsRequest) -> SearchSymbolsResponse:
    start = time.time()
    store = _vs()
    if not store:
        return SearchSymbolsResponse(hits=[], latency_ms=0)
    results = store.search_documents("symbols", req.query, n_results=req.top_k)
    hits: list[SearchHit] = []
    for r in results:
        hits.append(
            SearchHit(
                hit_type="symbol",  # type: ignore
                title=r.get("id", "symbol"),
                snippet=r.get("document", "")[:240],
                score=float(r.get("similarity", 0.0)),
                source_id=r.get("id", "unknown"),
                metadata={"distance": r.get("distance")},
            )
        )
    return SearchSymbolsResponse(
        hits=hits,
        latency_ms=int((time.time() - start) * 1000),
    )


def index_status() -> dict:
    store = _vs()
    if not store:
        return {"collections": {}, "enabled": False}
    return {"collections": store.get_collection_stats(), "enabled": True}


def optimize_cloudformation(
    req: OptimizeCloudFormationRequest, langflow: LangflowClient | None = None
) -> OptimizeCloudFormationResponse:
    """Return placeholder optimization recommendations.

    This is intentionally lightweight until a Langflow / rule-engine pipeline
    is implemented. It parses the template size and produces heuristic
    recommendation stubs to exercise contract plumbing end-to-end.
    """
    start = time.time()
    lines = req.cloudformation.splitlines()
    recs: list[str] = []
    if len(lines) > 200:
        recs.append("Consider modularizing large template (>200 lines).")
    if "AWS::S3::Bucket" in req.cloudformation and "BucketEncryption" not in req.cloudformation:
        recs.append("Add BucketEncryption for S3 buckets (security).")
    if not recs:
        recs.append("Template appears minimal; no major optimization opportunities detected.")
    diff_preview = "# diff preview not yet implemented"
    return OptimizeCloudFormationResponse(
        recommendations=recs,
        diff_preview=diff_preview,
        metrics={
            "operation": "cloudformation.optimize",
            "duration_ms": int((time.time() - start) * 1000),
            "line_count": len(lines),
        },
    )
