"""
MCP (Model Context Protocol) server for SCK Core AI agent.

Provides MCP tool interfaces for AI assistants to interact with
YAML/CloudFormation linting and validation capabilities.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Sequence
import asyncio
import os
import json
import time
import platform

import sys

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ToolAnnotations

from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
env_file = find_dotenv()
if env_file:
    load_dotenv(env_file)
    # MCP protocol: No stdout output allowed - using stderr for debug
    print(f"Loaded environment from: {env_file}", file=sys.stderr)
else:
    load_dotenv()  # Load from current directory or environment
    print("Loaded environment from current directory or system environment", file=sys.stderr)

from core_ai.core_server import (
    handle_analyze_template,
    handle_get_context_for_query,
    handle_get_indexing_stats,
    handle_initialize_indexes,
    handle_lint_yaml,
    handle_search_codebase,
    handle_search_documentation,
    handle_suggest_completion,
    handle_validate_cloudformation,
    process_exception,
    context_manager,
)

# Import SCK framework components (when available)
import core_logging as log  # noqa: E402

from core_ai.tools.registry import AnalysisTypes, CodeElementType, ContextStrategy, DocNames, LintModes  # noqa: E402
from core_ai.indexing import get_availability_status  # noqa: E402

# Unconditional imports for version reporting (no try/except around imports)
import core_ai  # noqa: E402
from importlib.metadata import version as pkg_version  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastMCP) -> AsyncIterator[None]:
    """Manage server lifecycle. Optionally index on startup based on environment flags.

    Env flags:
      - CORE_AI_MCP_INDEX_ON_START: "true" | "false" (default false)
      - CORE_AI_MCP_FORCE_REBUILD: "true" | "false" (default false)
    """

    index_on_start = os.getenv("CORE_AI_MCP_INDEX_ON_START", "false").lower() == "true"
    force_rebuild = os.getenv("CORE_AI_MCP_FORCE_REBUILD", "false").lower() == "true"

    start_ts = time.monotonic()
    log.info(
        "MCP server starting",
        index_on_start=index_on_start,
        force_rebuild=force_rebuild,
        pid=os.getpid(),
        cwd=os.getcwd(),
        python=sys.version.split(" ")[0],
    )

    if index_on_start:
        # Perform indexing in background so startup isn't blocked
        loop = asyncio.get_event_loop()

        async def _do_index():
            try:
                t0 = time.monotonic()
                log.info("Startup indexing started", force_rebuild=force_rebuild)
                await loop.run_in_executor(None, context_manager.initialize_indexes, force_rebuild)
                dur = time.monotonic() - t0
                log.info("Startup indexing completed", seconds=round(dur, 3))
                print("Initial indexing completed.", file=sys.stderr)
            except Exception as e:
                log.error("Startup indexing failed", error=str(e))

        try:
            app._startup_index_task = asyncio.create_task(_do_index())  # type: ignore[attr-defined]
            log.debug("Startup indexing task scheduled")
        except Exception:
            # Fallback if task attachment fails
            asyncio.create_task(_do_index())
            log.debug("Startup indexing task scheduled (fallback)")

    # Final startup confirmation before entering main loop
    ready_dur = time.monotonic() - start_ts
    log.info(
        "MCP server started successfully",
        service_name="simple-cloud-kit-mcp",
        startup_seconds=round(ready_dur, 3),
    )

    yield  # Server runs here

    # Shutdown: Optional cleanup
    print("Server shutdown.", file=sys.stderr)
    log.info("MCP server shutdown")


mcp = FastMCP(name="simple-cloud-kit-mcp", lifespan=lifespan)


@mcp.tool(name="check_availability")
async def availability_tool() -> Dict[str, Any]:
    """Check availability of resources."""
    return get_availability_status()


@mcp.tool(
    name="ping",
    description="Simple health check",
    title="Ping",
    annotations=ToolAnnotations(title="Ping"),
)
async def ping_tool() -> Sequence[TextContent]:
    """Return pong with a timestamp for quick health checks."""
    try:
        return [TextContent(type="text", text=json.dumps({"status": "ok", "message": "pong"}))]
    except Exception as e:
        return process_exception(e)


@mcp.tool(
    name="version",
    description="Return MCP server name and version information",
    title="Version",
    annotations=ToolAnnotations(title="Version"),
)
async def version_tool() -> Sequence[TextContent]:
    """Report package version and runtime info."""
    try:
        ver = getattr(core_ai, "__version__", None) or pkg_version("sck-core-ai")
        payload = {
            "name": "simple-cloud-kit-mcp",
            "version": ver,
            "python": sys.version.split(" ")[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        }
        return [TextContent(type="text", text=json.dumps(payload, indent=2))]
    except Exception as e:
        return process_exception(e)


@mcp.tool(
    name="get_capabilities",
    description="List supported enums and configuration flags",
    title="Capabilities",
    annotations=ToolAnnotations(title="Capabilities"),
)
async def get_capabilities_tool() -> Sequence[TextContent]:
    """Return supported enum values and selected env flags."""
    try:
        caps = {
            "analysis_types": [a.value for a in AnalysisTypes],
            "code_element_types": [e.value for e in CodeElementType],
            "context_strategies": [s.value for s in ContextStrategy],
            "doc_sections": [d.value for d in DocNames],
            "lint_modes": [m.value for m in LintModes],
            "env": {
                "CORE_AI_MCP_INDEX_ON_START": os.getenv("CORE_AI_MCP_INDEX_ON_START", "false"),
                "CORE_AI_MCP_FORCE_REBUILD": os.getenv("CORE_AI_MCP_FORCE_REBUILD", "false"),
            },
        }
        return [TextContent(type="text", text=json.dumps({"status": "success", "data": caps}, indent=2))]
    except Exception as e:
        return process_exception(e)


@mcp.tool(
    name="lint_yaml",
    description="Lint and validate YAML content with AI-powered suggestions",
    title="YAML Lint Tool",
    annotations=ToolAnnotations(title="YAML Lint Tool"),
    structured_output=False,
)
async def lint_yaml_tool(content: str, mode: LintModes = LintModes.syntax, strict: bool = True) -> Sequence[TextContent]:
    """Lint YAML documents."""
    try:
        log.info("Linting YAML content")
        result = await handle_lint_yaml(content=content, mode=mode, strict=strict)
    except Exception as e:
        log.error(f"YAML linting tool failed: {e}")
        result = process_exception(e)
    return result


@mcp.tool(
    name="validate_cloudformation",
    description="Validate CloudFormation templates with comprehensive analysis",
    title="CloudFormation Lint Tool",
    annotations=ToolAnnotations(title="CloudFormation Lint Tool"),
    structured_output=False,
)
async def lint_cloudformation_tool(template: str, region: str = "us-east-1", strict: bool = True) -> Sequence[TextContent]:
    """Lint CloudFormation templates."""
    try:
        log.info("Validating CloudFormation template")
        result = await handle_validate_cloudformation(template=template, region=region, strict=strict)
    except Exception as e:
        log.error(f"CloudFormation linting tool failed: {e}")
        result = process_exception(e)
    return result


@mcp.tool(
    name="suggest_completion",
    description="Get AI-powered code completion suggestions for YAML/CloudFormation",
    title="Content Completion",
    annotations=ToolAnnotations(title="Suggestions for completions"),
    structured_output=False,
)
async def suggest_completion(
    content: str, cursor_line: int, cursor_column: int, context_type: str = "auto"
) -> Sequence[TextContent]:
    """Get code completion suggestions."""
    try:
        log.info("Getting code completion suggestions")
        result = await handle_suggest_completion(
            content=content, cursor_line=cursor_line, cursor_column=cursor_column, context_type=context_type
        )
    except Exception as e:
        log.error(f"Code completion suggestion tool failed: {e}")
        result = process_exception(e)
    return result


@mcp.tool(
    name="analyze_template",
    description="Perform deep analysis of CloudFormation templates for security, cost, and best practices",
    title="Analyze SCK Template",
    annotations=ToolAnnotations(title="Analyze Core SCK template"),
    structured_output=False,
)
async def analyze_template(
    template: str, analysis_type: AnalysisTypes = AnalysisTypes.comprehensive, region: str = "us-east-1"
) -> Sequence[TextContent]:
    """Analyze CloudFormation templates."""
    try:
        log.info("Analyzing CloudFormation template")
        result = await handle_analyze_template(template=template, analysis_type=analysis_type, region=region)
    except Exception as e:
        log.error(f"CloudFormation template analysis tool failed: {e}")
        result = process_exception(e)
    return result


@mcp.tool(
    name="search_documentation",
    description="Search SCK documentation",
    title="Search Documentation",
    annotations=ToolAnnotations(title="Search SCK documentation"),
    structured_output=False,
)
async def search_documentation(
    query: str, section: DocNames = DocNames.technical_reference, max_results: int = 5
) -> Sequence[TextContent]:
    """Search documentation."""
    try:
        log.info("Searching documentation")
        result = await handle_search_documentation(query=query, section=section, max_results=max_results)
    except Exception as e:
        log.error(f"Documentation search tool failed: {e}")
        result = process_exception(e)
    return result


@mcp.tool(
    name="search_codebase",
    description="Search SCK codebase",
    title="Search Codebase",
    annotations=ToolAnnotations(title="Search SCK codebase"),
    structured_output=False,
)
async def search_codebase(query: str, project: str = "", element_type: str = "", max_results: int = 5) -> Sequence[TextContent]:
    """Search codebase."""
    try:
        log.info("Searching codebase")
        result = await handle_search_codebase(query=query, project=project, element_type=element_type, max_results=max_results)
    except Exception as e:
        log.error(f"Codebase search tool failed: {e}")
        result = process_exception(e)
    return result


@mcp.tool(
    name="get_context_for_query",
    description="Get comprehensive context from both documentation and codebase for development assistance",
    title="Get Context for Query",
    annotations=ToolAnnotations(title="Get Context for Query"),
    structured_output=False,
)
async def get_context_for_query(
    query: str,
    strategy: ContextStrategy,
    max_results: int = 5,
    score_mode: str = "auto",
    fused: bool = True,
    token_budget: int | None = None,
) -> Sequence[TextContent]:
    """Get context for query."""
    try:
        log.info("Getting context for query")
        result = await handle_get_context_for_query(
            query=query,
            strategy=strategy,
            max_results=max_results,
            # Pass-through new options to handlers via kwargs (they will forward to ContextManager)
            score_mode=score_mode,
            fused=fused,
            token_budget=token_budget,
        )
    except Exception as e:
        log.error(f"Get context for query tool failed: {e}")
        result = process_exception(e)
    return result


@mcp.tool(
    name="initialize_indexes",
    description="Initialize or rebuild documentation and codebase indexes",
    title="Initialize All Indexes",
    annotations=ToolAnnotations(title="Initialize Indexes"),
    structured_output=False,
)
async def initialize_indexes(force_rebuild: bool = False) -> Sequence[TextContent]:
    """Initialize or rebuild indexes."""
    try:
        log.info("Initializing indexes")
        result = await handle_initialize_indexes(force_rebuild=force_rebuild)
    except Exception as e:
        log.error(f"Initialize indexes tool failed: {e}")
        result = process_exception(e)
    return result


@mcp.tool(
    name="get_indexing_stats",
    description="Get statistics about indexed documentation and codebase",
    structured_output=False,
)
async def get_indexing_status() -> Sequence[TextContent]:
    """Get indexing statistics."""
    try:
        log.info("Getting indexing stats")
        result = await handle_get_indexing_stats()
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], TextContent):
            # Parse JSON from TextContent
            return result
        else:
            return process_exception(message="Unexpected response format")
    except Exception as e:
        log.error(f"Get indexing stats tool failed: {e}")
        return process_exception(e)


if __name__ == "__main__":
    # Run the async main method
    log.info("Starting MCP stdio transport")
    mcp.run(transport="stdio")
