"""SCK Core AI FastAPI server.

Provides REST API endpoints for YAML / CloudFormation processing, contract
operations, catalog queries, tool invocation, and chat orchestration.

The module intentionally keeps all logic in a single file (per project rules)
to avoid accidental architectural sprawl. Future structural changes should be
explicitly approved before introducing new modules/packages.
"""

import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, AsyncGenerator
import asyncio
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mangum import Mangum
from pydantic import BaseModel, Field

import core_logging as log

from core_framework.ai.contracts import (
    TemplateGenerateRequest,
    TemplateGenerateResponse,
    DSLValidateRequest,
    DSLValidateResponse,
    CloudFormationValidateRequest as CFNContractRequest,
    CloudFormationValidateResponse as CFNContractResponse,
    CompileRequest,
    CompileResponse,
    CompletionRequest,
    CompletionResponse,
    SearchDocsRequest,
    SearchDocsResponse,
    SearchSymbolsRequest,
    SearchSymbolsResponse,
    OptimizeCloudFormationRequest,
    OptimizeCloudFormationResponse,
)

# NOTE: Avoid importing core_api to keep this package decoupled. Lambda
# integration (ProxyEvent adaptation) should occur in sck-core-api layer.

from .langflow.client import LangflowClient
from . import service as ai_service
from .tools.registry import list_all_tool_specs
from .indexing import ContextManager, ConsumablesIndexer
from . import cache as ai_cache

import os


# API Models
class YamlLintRequest(BaseModel):
    """Request model for YAML linting.

    Args:
        content: Raw YAML source to lint.
        options: Optional linting directives (e.g. validation mode).
    """

    content: str = Field(..., description="YAML content to lint")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Linting options (strict, schema, etc.)")


class CloudFormationValidateRequest(BaseModel):
    """Request model for CloudFormation validation.

    Args:
        template: CloudFormation template (dict form) to validate.
        region: AWS region used for region‑specific validation rules.
        strict: Enable strict validation (schema + semantic checks).
    """

    template: Dict[str, Any] = Field(..., description="CloudFormation template")
    region: str = Field(default="us-east-1", description="AWS region")
    strict: bool = Field(default=True, description="Enable strict validation")


class CodeCompletionRequest(BaseModel):
    """Request model for code completion.

    Args:
        content: Partial YAML / CloudFormation content preceding the cursor.
        cursor_position: Line / column (1-based logical positions) where the
            completion is requested.
        context: Optional additional caller‑supplied context.
    """

    content: str = Field(..., description="Partial YAML/CloudFormation content")
    cursor_position: Dict[str, int] = Field(..., description="Cursor line/column")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context for completion")


class ApiResponse(BaseModel):
    """Standard SCK API response envelope.

    This mirrors the cross‑service envelope contract used by non‑OAuth
    endpoints. For error paths, `status` should be ``error`` and an HTTP
    status code reflecting the failure is set in ``code``.
    """

    status: str = Field(..., description="Response status (success/error)")
    code: int = Field(..., description="HTTP status code")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Response data")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata")
    message: Optional[str] = Field(default=None, description="Human-readable message")
    errors: Optional[List[Dict[str, Any]]] = Field(default=None, description="Error details")


class ChatMessage(BaseModel):
    """Single chat message within a conversation.

    Args:
        role: One of system | user | assistant.
        content: Natural language message text.
    """

    role: str = Field(..., description="Message role: system|user|assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request with conversation history.

    Args:
        messages: Ordered list of conversation messages.
        strategy: Retrieval strategy for contextual augmentation.
        max_context: Maximum total context chunks (documentation + code).
        temperature: LLM sampling temperature.
    """

    messages: List[ChatMessage] = Field(..., description="Ordered chat history")
    strategy: str = Field(
        default="balanced",
        description="Context retrieval strategy (balanced, documentation_focused, code_focused, documentation_only, code_only)",
    )
    max_context: int = Field(default=12, description="Max total context chunks (doc+code) to include")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="LLM temperature override")


class ChatResponse(BaseModel):
    """Chat response payload.

    Attributes:
        reply: Model response text.
        context_used: Trimmed context returned to caller (subset of internal).
        model: Identifier for underlying flow / model.
        usage: Reserved for token accounting (populated later).
    """

    reply: str
    context_used: Dict[str, Any]
    model: str = "langflow-flow"
    usage: Dict[str, Any] = Field(default_factory=dict)


# Global singletons (initialized in lifespan)
langflow_client: Optional[LangflowClient] = None
context_manager: Optional[ContextManager] = None
# Background reconnect task handle
_langflow_reconnect_task = None  # type: ignore

# Standard user-facing reply when AI engine is unavailable (strict mode)
AI_ENGINE_UNAVAILABLE_REPLY = "I'm sorry, I am unable to process your request at this time."


# ---------------------------------------------------------------------------
# Small utility helpers (local only – avoids module proliferation)
# ---------------------------------------------------------------------------


def _truthy_env(name: str, default: bool = False) -> bool:
    """Return boolean interpretation of an environment variable.

    Args:
        name: Environment variable name.
        default: Default value if unset.

    Returns:
        Boolean representation treating ``1,true,yes,on`` (case‑insensitive) as
        True.
    """
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def _floaty_env(name: str, default: float = 0.0) -> float:
    """Return float interpretation of an environment variable.

    Args:
        name: Environment variable name.
        default: Default value if unset or invalid.

    Returns:
        Float representation of the environment variable, or the default if
        unset or conversion fails.
    """
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


ENDPOINT_LIST: List[str] = [
    "/health",
    "/ready",
    "/api/v1/lint/yaml",
    "/api/v1/validate/cloudformation",
    "/api/v1/complete",
    "/v1/tools",
    "/v1/tools/{name}/invoke",
    "/v1/mcp/sse",
    "/v1/chat",
    "/v1/chat/stream",
    "/v1/catalog/consumables",
    "/v1/catalog/actions",
    "/v1/catalog/consumables/search",
    "/v1/catalog/actions/search",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Responsibilities:
      * Initialize Langflow client (strict – failures surface via /ready 503)
      * Start optional auto‑reconnect loop
      * Initialize documentation & code indexing (best effort, background)
      * Build consumables & actions catalogs (best effort)

    Readiness Contract:
        /ready returns 503 while Langflow is unavailable; we never provide
        degraded / fallback answers.
    """
    global context_manager, _langflow_reconnect_task

    auto_reconnect = _truthy_env("CORE_AI_LANGFLOW_AUTO_RECONNECT", True)
    reconnect_interval = _floaty_env("CORE_AI_LANGFLOW_RECONNECT_INTERVAL_SECONDS", 30.0)
    health_recreate = _truthy_env("CORE_AI_LANGFLOW_RECREATE_ON_UNHEALTHY", True)
    auto_index_init = _truthy_env("CORE_AI_AUTO_INDEX_INIT", True)

    def _init_langflow_once():  # internal helper
        global langflow_client
        try:
            base_url = os.getenv("LANGFLOW_BASE_URL", "http://localhost:7860")
            flow_id = os.getenv("LANGFLOW_FLOW_ID", "yaml-cf-ai-agent-v1").strip().strip('"').strip("'")
            api_key = os.getenv("LANGFLOW_API_KEY") or None
            langflow_client_local = LangflowClient(
                base_url=base_url,
                flow_id=flow_id,
                api_key=api_key,
            )
            langflow_client = langflow_client_local
            log.info(
                "Langflow client initialized successfully",
                base_url=base_url,
                flow_id=flow_id,
                api_key_present=bool(api_key),
            )
        except Exception as e:  # pragma: no cover
            log.error(
                "Langflow client initialization failed - service considered DOWN for chat ops",
                error=str(e),
            )
            langflow_client = None

    _init_langflow_once()

    async def _reconnect_loop():  # pragma: no cover (timing dependent)

        while True:
            try:
                if not langflow_client:
                    _init_langflow_once()
                elif health_recreate:
                    try:
                        hc = langflow_client.health_check()  # type: ignore[attr-defined]
                        if not hc.get("available"):
                            log.warning(
                                "Langflow unhealthy - attempting re-create",
                                status=hc.get("status"),
                            )
                            _init_langflow_once()
                    except Exception as he:
                        log.warning(
                            "Langflow health check failed - forcing re-init",
                            error=str(he),
                        )
                        _init_langflow_once()
            except Exception as loop_e:
                log.error("Reconnect loop iteration failed", error=str(loop_e))
            await asyncio.sleep(reconnect_interval)

    if auto_reconnect and _langflow_reconnect_task is None:
        _langflow_reconnect_task = asyncio.create_task(_reconnect_loop())
        log.info(
            "Langflow auto-reconnect loop started",
            interval_seconds=reconnect_interval,
            recreate_on_unhealthy=health_recreate,
        )

    # Optional indexing subsystem
    try:  # pragma: no cover - optional path
        import os

        # Allow explicit override (useful in container or dev mismatch)
        override = os.getenv("SCK_WORKSPACE_ROOT")
        if override and os.path.isdir(override):
            root = os.path.abspath(override)
            log.info("Workspace root override in use", root=root)
        else:
            # Correct monorepo root is two levels up from core_ai (core_ai -> sck-core-ai -> simple-cloud-kit)
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        build_dir = os.path.join(root, "sck-core-docs", "build")
        if not os.path.isdir(build_dir):
            log.warning(
                "Docs build directory not found (skipping initial doc index)",
                build_dir=build_dir,
            )
        context_manager = ContextManager(build_directory=build_dir, workspace_root=root)
        log.info("ContextManager initialized", workspace_root=root, build_directory=build_dir)
    except Exception as e:  # pragma: no cover
        context_manager = None
        log.warning("Indexing subsystem unavailable", error=str(e))

    # Attempt auto-initialization of indexes if present but empty (non-blocking)
    if context_manager and auto_index_init:
        try:  # pragma: no cover - best effort
            stats = context_manager.get_system_stats()
            needs_doc = stats.get("documentation", {}).get("total_chunks", 0) == 0
            needs_code = stats.get("codebase", {}).get("total_elements", 0) == 0
            if needs_doc or needs_code:
                import threading

                def _init():  # background thread
                    if not context_manager:  # safety
                        return
                    try:
                        log.info("Auto-initializing indexes (background thread)")
                        context_manager.initialize_indexes()
                        log.info("Index auto-initialization complete")
                    except Exception as e:  # pragma: no cover
                        log.warning("Auto index init failed", error=str(e))

                threading.Thread(target=_init, daemon=True).start()
        except Exception as e:  # pragma: no cover
            log.debug("Skipping auto index init check", error=str(e))

    # Build catalogs (best-effort, non-fatal)
    try:  # pragma: no cover
        consumables_indexer = ConsumablesIndexer()
        app.state.consumables_index = consumables_indexer.build_index()
        log.info("Consumables index built", entries=len(app.state.consumables_index))
    except Exception as e:  # pragma: no cover
        app.state.consumables_index = []  # type: ignore[attr-defined]
        log.warning("Failed to build consumables index", error=str(e))

    try:  # pragma: no cover
        from .indexing.action_indexer import ActionIndexer

        action_indexer = ActionIndexer()
        app.state.actions_index = action_indexer.build_index()
        log.info("Actions index built", entries=len(app.state.actions_index))
    except Exception as e:  # pragma: no cover
        app.state.actions_index = []  # type: ignore[attr-defined]
        log.warning("Failed to build actions index", error=str(e))

    yield

    # Cleanup
    log.info("Shutting down SCK Core AI server")
    if _langflow_reconnect_task is not None:  # pragma: no cover
        try:
            _langflow_reconnect_task.cancel()
        except Exception:
            pass


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI instance with CORS and lifespan manager.
    """

    app = FastAPI(
        title="SCK Core AI",
        description="AI-powered YAML and CloudFormation processing agent",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


# Create app instance
app = create_app()


def create_envelope_response(
    status: str = "success",
    code: int = 200,
    data: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create SCK-standard API response envelope.

    Args:
        status: ``success`` or ``error``.
        code: HTTP status code mirrored in envelope for UI alignment.
        data: Primary response payload (dict form).
        message: Human readable message.
        errors: Optional structured error list.
        metadata: Optional metadata (pagination, counts, etc.).

    Returns:
        Dict representing the full envelope structure.
    """
    response = {"status": status, "code": code}

    if data is not None:
        response["data"] = data
    if message is not None:
        response["message"] = message
    if errors is not None:
        response["errors"] = errors
    if metadata is not None:
        response["metadata"] = metadata

    return response


# Health Check Endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Liveness probe.

    Returns quickly without enforcing Langflow readiness. Used for container
    liveness / basic uptime checks.
    """
    ai_status = {
        "available": bool(langflow_client),
        "status": "up" if langflow_client else "down",
    }
    data = {"status": "healthy", "service": "sck-core-ai", "ai_engine": ai_status}
    return create_envelope_response(data=data, message="Service liveness OK")


@app.get("/ready", tags=["health"])  # readiness (strict)
async def readiness_check():
    """Readiness probe.

    Returns 503 until Langflow is fully initialized (no degraded mode
    permitted). Suitable for ECS / ALB target health checks.
    """
    if not langflow_client:
        return create_envelope_response(
            status="error",
            code=503,
            message="AI engine not initialized",
            data={"ai_engine": {"available": False, "status": "down"}},
        )
    return create_envelope_response(
        data={
            "status": "ready",
            "service": "sck-core-ai",
            "ai_engine": {"available": True, "status": "up"},
        },
        message="Service is fully ready",
    )


@app.get("/", tags=["root"])
async def root():
    """Root index endpoint."""
    return create_envelope_response(
        data={
            "service": "SCK Core AI",
            "version": "0.1.0",
            "endpoints": ENDPOINT_LIST,
        },
        message="Welcome to SCK Core AI Agent",
    )


@app.post("/api/v1/lint/yaml", tags=["linting"], response_model=ApiResponse)
async def lint_yaml(request: YamlLintRequest):
    """Lint YAML via Langflow.

    Returns a 503 envelope (not HTTP raise) when Langflow is unavailable to
    preserve uniform envelope contract for UI consumers.
    """
    try:
        log.info("Processing YAML lint request", content_length=len(request.content))
        if not langflow_client:
            return create_envelope_response(
                status="error",
                code=503,
                message="AI engine unavailable",
                data={
                    "reason": "langflow_unavailable",
                    "reply": AI_ENGINE_UNAVAILABLE_REPLY,
                },
            )
        result = langflow_client.process_sync(
            {
                "input_value": request.content,
                "tweaks": {
                    "yaml-parser": {"validation_mode": (request.options or {}).get("mode", "syntax")},
                    "response-formatter": {"format_type": "sck_envelope"},
                },
            }
        )
        if isinstance(result, dict) and result.get("data"):
            data = result["data"]
        else:  # minimal fallback structure
            data = {"valid": True, "errors": [], "warnings": []}
        return create_envelope_response(data=data)
    except Exception as e:  # pragma: no cover
        log.error("YAML lint failed", error=str(e))
        raise HTTPException(status_code=500, detail="lint_failed")


@app.post("/api/v1/validate/cloudformation", tags=["validation"], response_model=ApiResponse)
async def validate_cloudformation_endpoint(request: CloudFormationValidateRequest):
    """Validate a CloudFormation template via Langflow (strict engine check)."""
    try:
        log.info(
            "Processing CloudFormation validation",
            template_keys=list(request.template.keys()),
        )
        if not langflow_client:
            return create_envelope_response(
                status="error",
                code=503,
                message="AI engine unavailable",
                data={
                    "reason": "langflow_unavailable",
                    "reply": AI_ENGINE_UNAVAILABLE_REPLY,
                },
            )
        template_yaml = json.dumps(request.template)
        result = langflow_client.process_sync(
            {
                "input_value": template_yaml,
                "tweaks": {
                    "cf-validator": {
                        "region": request.region,
                        "strict_mode": request.strict,
                    },
                    "response-formatter": {"format_type": "sck_envelope"},
                },
            }
        )
        if isinstance(result, dict) and result.get("data"):
            data = result["data"]
        else:
            tpl = request.template
            data = {
                "valid": True,
                "template_info": {
                    "resources": len(tpl.get("Resources", {})),
                    "parameters": len(tpl.get("Parameters", {})),
                    "outputs": len(tpl.get("Outputs", {})),
                },
                "errors": [],
                "warnings": [],
            }
        return create_envelope_response(data=data)
    except Exception as e:  # pragma: no cover
        log.error("CloudFormation validation failed", error=str(e))
        raise HTTPException(status_code=500, detail="cfn_validate_failed")


@app.post("/api/v1/complete", tags=["completion"], response_model=ApiResponse)
async def code_completion(request: CodeCompletionRequest):
    """Provide AI-powered code completion suggestions.

    The underlying Langflow flow produces a list of suggestion objects with
    fields: text, description, insertText, kind.
    """
    try:
        log.info(
            "Processing code completion request",
            cursor_line=request.cursor_position.get("line", 0),
        )

        if not langflow_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Langflow client not available",
            )

        # Process through specialized completion workflow
        result = langflow_client.process_sync(
            {
                "input_value": request.content,
                "tweaks": {
                    "ai-analyzer": {
                        "system_message": f"""Provide code completion suggestions for the given YAML/CloudFormation content.

Cursor position: Line {request.cursor_position.get('line', 0)}, Column {request.cursor_position.get('column', 0)}

Focus on:
1. Valid CloudFormation resources and properties
2. Proper YAML syntax and indentation
3. AWS best practices and security
4. Common configuration patterns

Return suggestions as a JSON array with: text, description, insertText, kind."""
                    },
                    "response-formatter": {"format_type": "sck_envelope"},
                },
            }
        )

        # Extract completion suggestions
        if isinstance(result, dict) and "data" in result:
            response_data = result["data"]
        else:
            response_data = {
                "suggestions": [
                    {
                        "text": "Resources:",
                        "description": "CloudFormation Resources section",
                        "insertText": "Resources:\n  ",
                        "kind": "keyword",
                    }
                ],
                "context": request.context,
            }

        return create_envelope_response(data=response_data, message="Code completion suggestions generated")

    except Exception as e:
        log.error("Code completion failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Code completion failed: {str(e)}",
        )


# ---------------------------------------------------------------------------
# New contract-aligned endpoints using service layer
# ---------------------------------------------------------------------------


@app.post("/v1/templates/generate", response_model=ApiResponse, tags=["ai-contracts"])
async def generate_templates_contract(
    req: TemplateGenerateRequest,
):  # FastAPI parses JSON into model
    try:
        resp: TemplateGenerateResponse = ai_service.generate(req, langflow_client)
        return create_envelope_response(data=resp.model_dump())
    except Exception as e:  # pragma: no cover - fallback path
        log.error("generate_templates_contract failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/dsl/validate", response_model=ApiResponse, tags=["ai-contracts"])
async def validate_dsl_contract(req: DSLValidateRequest):
    resp = ai_service.validate_dsl(req)
    return create_envelope_response(data=resp.model_dump())


@app.post("/v1/templates/compile", response_model=ApiResponse, tags=["ai-contracts"])
async def compile_contract(req: CompileRequest):
    resp = ai_service.compile(req)
    return create_envelope_response(data=resp.model_dump())


@app.post("/v1/cloudformation/validate", response_model=ApiResponse, tags=["ai-contracts"])
async def cfn_validate_contract(req: CFNContractRequest):
    resp = ai_service.validate_cloudformation(req)
    return create_envelope_response(data=resp.model_dump())


@app.post("/v1/completions", response_model=ApiResponse, tags=["ai-contracts"])
async def completions_contract(req: CompletionRequest):
    resp = ai_service.completions(req)
    return create_envelope_response(data=resp.model_dump())


@app.post("/v1/search/docs", response_model=ApiResponse, tags=["ai-contracts"])
async def search_docs_contract(req: SearchDocsRequest):
    resp = ai_service.search_docs(req)
    return create_envelope_response(data=resp.model_dump())


@app.post("/v1/search/symbols", response_model=ApiResponse, tags=["ai-contracts"])
async def search_symbols_contract(req: SearchSymbolsRequest):
    resp = ai_service.search_symbols(req)
    return create_envelope_response(data=resp.model_dump())


@app.get("/v1/index/status", response_model=ApiResponse, tags=["ai-contracts"])
async def index_status_contract():
    data = ai_service.index_status()
    return create_envelope_response(data=data)


@app.post(
    "/v1/cloudformation/optimize",
    response_model=ApiResponse,
    tags=["ai-contracts"],
)
async def optimize_cfn_contract(req: OptimizeCloudFormationRequest):
    resp: OptimizeCloudFormationResponse = ai_service.optimize_cloudformation(req, langflow_client)
    return create_envelope_response(data=resp.model_dump())


# AWS Lambda Handler (using Mangum)
def lambda_handler(event, context):  # pragma: no cover
    """AWS Lambda adapter entrypoint.

    Delegates ASGI handling to Mangum. API Gateway event normalization lives
    in ``sck-core-api``; this layer is intentionally thin.
    """
    asgi_handler = Mangum(app, lifespan="off")
    return asgi_handler(event, context)


# ---------------------------------------------------------------------------
# Tool Registry (HTTP surface of MCP tools)
# ---------------------------------------------------------------------------


@app.get("/v1/tools", tags=["tools"], response_model=ApiResponse)
async def list_tools_endpoint():
    include_indexing = context_manager is not None
    specs = [{"name": t.name, "description": t.description, "schema": t.schema} for t in list_all_tool_specs(include_indexing)]
    return create_envelope_response(data={"tools": specs})


def _run_langflow_sync(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run Langflow synchronously (threadpool wrapper will call)."""
    if not langflow_client:
        raise RuntimeError("Langflow client not initialized")
    return langflow_client.process_sync(payload)


async def _invoke_base_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    import asyncio

    loop = asyncio.get_event_loop()
    if name == "lint_yaml":
        req = {
            "input_value": arguments.get("content", ""),
            "tweaks": {
                "yaml-parser": {"validation_mode": arguments.get("mode", "syntax")},
                "response-formatter": {"format_type": "sck_envelope"},
            },
        }
        return await loop.run_in_executor(None, _run_langflow_sync, req)
    if name == "validate_cloudformation":
        req = {
            "input_value": arguments.get("template", ""),
            "tweaks": {
                "cf-validator": {
                    "region": arguments.get("region", "us-east-1"),
                    "strict_mode": arguments.get("strict", True),
                },
                "response-formatter": {"format_type": "sck_envelope"},
            },
        }
        return await loop.run_in_executor(None, _run_langflow_sync, req)
    if name == "suggest_completion":
        content = arguments.get("content", "")
        prompt = (
            f"Provide code completion suggestions for the following content.\n"
            f"Cursor line: {arguments.get('cursor_line')} column: {arguments.get('cursor_column')}\n\n{content}\n"
        )
        req = {
            "input_value": prompt,
            "tweaks": {
                "ai-analyzer": {
                    "system_message": "You are a YAML/CloudFormation completion assistant.",
                    "temperature": 0.1,
                },
                "response-formatter": {"format_type": "sck_envelope"},
            },
        }
        return await loop.run_in_executor(None, _run_langflow_sync, req)
    if name == "analyze_template":
        analysis_type = arguments.get("analysis_type", "comprehensive")
        analysis_prompts = {
            "security": "Focus on security vulnerabilities, IAM policies, encryption, and access controls.",
            "cost": "Analyze cost optimization opportunities, resource sizing, and billing implications.",
            "best_practices": "Evaluate adherence to AWS Well-Architected Framework principles.",
            "comprehensive": "Perform complete analysis covering security, cost, performance, and best practices.",
        }
        system_message = analysis_prompts.get(analysis_type, analysis_prompts["comprehensive"])
        req = {
            "input_value": arguments.get("template", ""),
            "tweaks": {
                "ai-analyzer": {"system_message": system_message, "temperature": 0.1},
                "cf-validator": {
                    "region": arguments.get("region", "us-east-1"),
                    "strict_mode": True,
                },
                "response-formatter": {"format_type": "sck_envelope"},
            },
        }
        return await loop.run_in_executor(None, _run_langflow_sync, req)
    raise ValueError(f"Unknown base tool: {name}")


async def _invoke_indexing_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if not context_manager:
        raise RuntimeError("Indexing subsystem not available")
    import asyncio

    loop = asyncio.get_event_loop()
    if name == "search_documentation":
        results = await loop.run_in_executor(
            None,
            context_manager.doc_indexer.search_documentation,
            arguments.get("query", ""),
            arguments.get("section"),
            arguments.get("max_results", 5),
        )
        return {"status": "success", "data": {"results": results}}
    if name == "search_codebase":
        results = await loop.run_in_executor(
            None,
            context_manager.code_indexer.search_codebase,
            arguments.get("query", ""),
            arguments.get("project"),
            arguments.get("element_type"),
            arguments.get("max_results", 5),
        )
        return {"status": "success", "data": {"results": results}}
    if name == "get_context_for_query":
        context = await loop.run_in_executor(
            None,
            context_manager.get_context_for_query,
            arguments.get("query", ""),
            arguments.get("strategy", "balanced"),
            arguments.get("max_results", 10),
            True,
        )
        return {"status": "success", "data": context}
    if name == "initialize_indexes":
        results = await loop.run_in_executor(
            None,
            context_manager.initialize_indexes,
            arguments.get("force_rebuild", False),
        )
        return {"status": "success", "data": results}
    if name == "get_indexing_stats":
        stats = await loop.run_in_executor(None, context_manager.get_system_stats)
        return {"status": "success", "data": stats}
    raise ValueError(f"Unknown indexing tool: {name}")


@app.post("/v1/tools/{name}/invoke", tags=["tools"], response_model=ApiResponse)
async def invoke_tool_endpoint(name: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    include_indexing = context_manager is not None
    available = {t.name for t in list_all_tool_specs(include_indexing)}
    if name not in available:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    try:
        if name in {
            "lint_yaml",
            "validate_cloudformation",
            "suggest_completion",
            "analyze_template",
        }:
            result = await _invoke_base_tool(name, body)
        else:
            result = await _invoke_indexing_tool(name, body)
        return create_envelope_response(data={"result": result})
    except Exception as e:  # pragma: no cover - error path
        log.error("Tool invocation failed", tool=name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# SSE Endpoint (Phase 3) - minimal streaming wrapper around a tool invocation
# ---------------------------------------------------------------------------


async def _sse_event_generator(payload: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
    import asyncio

    yield b"event: start\n" + b'data: {"status": "starting"}\n\n'
    await asyncio.sleep(0)
    data_json = json.dumps(payload)
    yield b"event: result\n" + f"data: {data_json}\n\n".encode("utf-8")
    yield b"event: end\n" + b'data: {"status": "complete"}\n\n'


@app.post("/v1/mcp/sse", tags=["tools"])
async def mcp_sse_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    name = body.get("tool")
    args = body.get("arguments", {})
    if not name:
        raise HTTPException(status_code=400, detail="Missing 'tool' field")
    include_indexing = context_manager is not None
    available = {t.name for t in list_all_tool_specs(include_indexing)}
    if name not in available:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    try:
        if name in {
            "lint_yaml",
            "validate_cloudformation",
            "suggest_completion",
            "analyze_template",
        }:
            result = await _invoke_base_tool(name, args)
        else:
            result = await _invoke_indexing_tool(name, args)
    except Exception as e:  # pragma: no cover
        result = {"status": "error", "message": str(e)}
    return StreamingResponse(
        _sse_event_generator(result),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Chat Orchestration (Option C Phase 4)
# ---------------------------------------------------------------------------


def _build_chat_prompt(context: Dict[str, Any], messages: List[ChatMessage]) -> str:
    """Construct chat prompt string.

    Includes recent conversation turns (last 10) and a bounded subset of
    documentation / code snippets to keep overall prompt size controlled.
    """
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
    doc_snips = []
    code_snips = []
    for item in context.get("documentation", [])[:5]:
        snippet = item.get("content", "")[:600]
        doc_snips.append(f"[DOC:{item.get('source', '?')}]\n{snippet}")
    for item in context.get("codebase", [])[:5]:
        snippet = item.get("content", "")[:600]
        code_snips.append(f"[CODE:{item.get('location', '?')}]\n{snippet}")
    convo_lines = [f"{m.role.upper()}: {m.content}" for m in messages[-10:]]
    prompt = (
        "You are the SCK Core AI assistant. Use provided context when helpful.\n"
        "Answer succinctly, citing file names or doc sections when referencing context.\n\n"
        "# Conversation\n" + "\n".join(convo_lines) + "\n\n"
        "# Documentation Context\n" + ("\n\n".join(doc_snips) or "<none>") + "\n\n"
        "# Code Context\n" + ("\n\n".join(code_snips) or "<none>") + "\n\n"
        f"# Latest User Query\n{last_user}\n\n"
        "Respond with a helpful answer."
    )
    return prompt


def _detect_catalog_intent(user_text: str) -> Dict[str, Any]:
    """Lightweight heuristic intent detection for catalog lookups.

    Recognizes phrases like:
      * "show me the consumable ..." / "consumable template for <X>"
      * "action spec for <X>" / "describe action <X>"

    Returns a dict with keys: kind ('consumable'|'action') and token (lookup id)
    if a match is detected; otherwise empty dict.
    """
    import re

    text = user_text.lower()
    # Consumable patterns
    cons_pat = re.search(r"consumable (?:template|spec)? for ([a-z0-9_.-]+)", text)
    if not cons_pat:
        cons_pat = re.search(r"show me (?:the )?consumable ([a-z0-9_.-]+)", text)
    if cons_pat:
        return {"kind": "consumable", "token": cons_pat.group(1)}
    act_pat = re.search(r"action (?:spec|definition)? for ([a-z0-9_.-]+)", text)
    if not act_pat:
        act_pat = re.search(r"describe action ([a-z0-9_.-]+)", text)
    if act_pat:
        return {"kind": "action", "token": act_pat.group(1)}
    return {}


def _lookup_catalog_entry(intent: Dict[str, Any]) -> Optional[str]:
    """Return formatted catalog entry snippet for inclusion in prompt.

    Args:
        intent: Intent dict from _detect_catalog_intent.
    """
    if not intent:
        return None
    kind = intent.get("kind")
    token = intent.get("token")
    try:
        if kind == "consumable":
            items = getattr(app.state, "consumables_index", [])  # type: ignore[attr-defined]
            for item in items:
                rid = str(item.get("id") or item.get("name"))
                if rid and token and token in rid.lower():
                    props = item.get("properties", [])
                    prop_lines = []
                    for p in props[:30]:  # cap
                        prop_lines.append(f" - {p.get('path')}: type={p.get('type')} required={p.get('required')}")
                    return (
                        f"[CATALOG:CONSUMABLE {rid}]\nCategory: {item.get('category')}\n" + "Properties:\n" + "\n".join(prop_lines)
                    )
        elif kind == "action":
            items = getattr(app.state, "actions_index", [])  # type: ignore[attr-defined]
            for item in items:
                aid = str(item.get("id"))
                if aid and token and token in aid.lower():
                    params = item.get("params", [])
                    param_lines = []
                    for p in params[:40]:
                        default = p.get("default")
                        default_repr = "" if default is None else f" default={default!r}"
                        param_lines.append(f" - {p.get('name')}: {p.get('type')} required={p.get('required')}{default_repr}")
                    return (
                        f"[CATALOG:ACTION {aid}]\nModule: {item.get('module')} Class: {item.get('class_name')}\n"
                        f"Summary: {item.get('summary')}\nParams:\n" + "\n".join(param_lines)
                    )
    except Exception as e:  # pragma: no cover
        log.debug("Catalog lookup failed", error=str(e))
    return None


async def _invoke_chat_llm(prompt: str, temperature: float) -> str:
    """Invoke Langflow to get chat completion (STRICT: raise on failure)."""
    import asyncio

    if not langflow_client:
        raise RuntimeError("Langflow client unavailable")

    loop = asyncio.get_event_loop()

    def _call():
        if not langflow_client:
            raise RuntimeError("Langflow client became unavailable")
        req = {
            "input_value": prompt,
            "tweaks": {
                "ai-analyzer": {
                    "system_message": "You are the SCK Core AI assistant.",
                    "temperature": temperature,
                },
                "response-formatter": {"format_type": "sck_envelope"},
            },
        }
        result = langflow_client.process_sync(req)
        if not isinstance(result, dict):
            return str(result)[:4000]
        if result.get("status") == "error":
            raise RuntimeError(result.get("message", "Langflow processing error"))
        data = result.get("data") or result.get("result") or result
        if isinstance(data, dict):
            for k in ("text", "content", "result"):
                if k in data:
                    return str(data[k])[:4000]
        return str(result)[:4000]

    return await loop.run_in_executor(None, _call)


@app.post("/v1/chat", tags=["chat"], response_model=ApiResponse)
async def chat_endpoint(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")
    # Build canonical payload for idempotency (last 10 messages + strategy + temp)
    idem_payload = {
        "messages": [m.model_dump() for m in req.messages[-10:]],
        "strategy": req.strategy,
        "max_context": req.max_context,
        "temperature": req.temperature,
    }

    def _producer():  # synchronous for cache wrapper
        ctx: Dict[str, Any] = {}
        if context_manager:
            try:
                ctx_local = context_manager.get_context_for_query(
                    query=req.messages[-1].content,
                    strategy=req.strategy,
                    max_results=req.max_context,
                    include_metadata=True,
                )
                ctx.update(ctx_local)
            except Exception as e:  # pragma: no cover
                log.warning("Context retrieval failed", error=str(e))
        # Lightweight catalog intent augmentation (non-cached dynamic snippet)
        user_text = req.messages[-1].content
        intent = _detect_catalog_intent(user_text)
        if intent:
            snippet = _lookup_catalog_entry(intent)
            if snippet:
                # Inject into documentation context section for prompt assembly
                docs = ctx.setdefault("documentation", [])
                docs.insert(0, {"source": f"catalog:{intent['kind']}", "content": snippet})
        prompt = _build_chat_prompt(ctx, req.messages)
        # Synchronous wait via event loop blocking call not ideal; we separate reply outside
        return {"ctx": ctx, "prompt": prompt}

    # First stage: context + prompt generation idempotent
    prompt_packet, meta = ai_cache.run_idempotent(operation="chat.context", payload=idem_payload, producer=_producer)

    # Second stage: model reply idempotent on prompt packet hash
    async def _reply_stage():
        reply_payload = {
            "prompt_hash": hash(prompt_packet["prompt"]),
            "temperature": req.temperature,
        }
        text = await _invoke_chat_llm(prompt_packet["prompt"], req.temperature)
        reply_result, reply_meta = ai_cache.run_idempotent(
            operation="chat.reply",
            payload=reply_payload,
            producer=lambda: {"reply": text},
        )
        return reply_result, reply_meta

    try:
        reply_result, reply_meta = await _reply_stage()
    except Exception as e:
        log.error("Chat generation failed", error=str(e))
        return create_envelope_response(
            status="error",
            code=503,
            message="AI engine unavailable",
            data={
                "reason": str(e),
                "reply": AI_ENGINE_UNAVAILABLE_REPLY,
            },
        )

    ctx = prompt_packet.get("ctx", {})
    trimmed_ctx = {
        "summary": ctx.get("summary", {}),
        "documentation": ctx.get("documentation", [])[:3],
        "codebase": ctx.get("codebase", [])[:3],
    }
    response = ChatResponse(reply=reply_result["reply"], context_used=trimmed_ctx, usage={})
    envelope = response.model_dump()
    envelope["idempotency"] = {"context": meta, "reply": reply_meta}
    return create_envelope_response(data=envelope)


@app.post("/v1/chat/stream", tags=["chat"])
async def chat_stream_endpoint(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    ctx: Dict[str, Any] = {}
    if context_manager:
        try:
            ctx = context_manager.get_context_for_query(
                query=req.messages[-1].content,
                strategy=req.strategy,
                max_results=req.max_context,
                include_metadata=True,
            )
        except Exception as e:  # pragma: no cover
            ctx = {"error": str(e)}

    prompt = _build_chat_prompt(ctx, req.messages)

    async def gen() -> AsyncGenerator[bytes, None]:
        import json as _json

        yield b"event: start\n" + b'data: {"status": "starting"}\n\n'
        yield b"event: context\n" + f"data: {_json.dumps({'summary': ctx.get('summary', {})})}\n\n".encode()
        try:
            reply = await _invoke_chat_llm(prompt, req.temperature)
            chunk = {"reply": reply[:4000]}
            yield b"event: answer\n" + f"data: {_json.dumps(chunk)}\n\n".encode()
            yield b"event: end\n" + b'data: {"status": "complete"}\n\n'
        except Exception as e:
            err = {
                "status": "error",
                "message": "AI engine unavailable",
                "reason": str(e),
                "reply": AI_ENGINE_UNAVAILABLE_REPLY,
            }
            yield b"event: error\n" + f"data: {_json.dumps(err)}\n\n".encode()
            yield b"event: end\n" + b'data: {"status": "failed"}\n\n'

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def main():
    """Local development entrypoint.

    Launches uvicorn on port 8200 (default) – separated for clarity and to
    avoid collision with local DynamoDB default (8000).
    """
    import uvicorn

    # Use 8200 to avoid collision with local DynamoDB (8000)
    import os

    port = int(os.getenv("SCK_AI_PORT", "8200"))
    uvicorn.run("core_ai.server:app", host="0.0.0.0", port=port, reload=True, log_level="info")


if __name__ == "__main__":
    main()


# Catalog endpoints (placed at end to minimize merge conflicts)
@app.get("/v1/catalog/consumables", tags=["catalog"])
def catalog_consumables():
    data = app.state.consumables_index or []  # type: ignore[attr-defined]
    return create_envelope_response(
        data={"items": data},
        metadata={"count": len(data)},
    )


@app.get("/v1/catalog/actions", tags=["catalog"])
def catalog_actions():
    data = app.state.actions_index or []  # type: ignore[attr-defined]
    return create_envelope_response(
        data={"items": data},
        metadata={"count": len(data)},
    )


@app.get("/v1/catalog/consumables/search", tags=["catalog"], response_model=ApiResponse)
def catalog_consumables_search(q: Optional[str] = None, limit: int = 50, offset: int = 0):
    """Search consumables catalog (simple substring match)."""
    items = list(getattr(app.state, "consumables_index", []))  # type: ignore[attr-defined]
    if q:
        ql = q.lower()
        items = [i for i in items if ql in str(i.get("id", "")).lower() or ql in str(i.get("category", "")).lower()]
    total = len(items)
    sliced = items[offset : offset + limit]
    return create_envelope_response(
        data={"items": sliced},
        metadata={
            "count": len(sliced),
            "total": total,
            "offset": offset,
            "limit": limit,
            "query": q,
        },
    )


@app.get("/v1/catalog/actions/search", tags=["catalog"], response_model=ApiResponse)
def catalog_actions_search(q: Optional[str] = None, limit: int = 50, offset: int = 0):
    """Search actions catalog (simple substring match)."""
    items = list(getattr(app.state, "actions_index", []))  # type: ignore[attr-defined]
    if q:
        ql = q.lower()
        items = [i for i in items if ql in str(i.get("id", "")).lower() or ql in str(i.get("summary", "")).lower()]
    total = len(items)
    sliced = items[offset : offset + limit]
    return create_envelope_response(
        data={"items": sliced},
        metadata={
            "count": len(sliced),
            "total": total,
            "offset": offset,
            "limit": limit,
            "query": q,
        },
    )
