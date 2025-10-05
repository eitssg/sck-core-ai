import asyncio
import json
from typing import Any, Dict, Sequence

import core_logging as log
from mcp.types import TextContent

from core_ai.indexing.context_manager import ContextManager
from core_ai.langflow import LangflowClient


# We use langflow to communicate to the LLM
langflow_client = LangflowClient()


# Create the context manager and staartup all the indexing services
context_manager = ContextManager()


def _process_exception_data(e: Exception | None, code=500, message: str = "Tool execution failed") -> Dict[str, Any]:
    rv = {
        "status": "error",
        "code": code,
    }
    rv["message"] = message if e is None else f"{message}: {str(e)}"
    return rv


def process_exception(e: Exception | None = None, code: int = 500, message: str = "Tool execution failed") -> Sequence[TextContent]:
    return [
        TextContent(type="text", text=json.dumps(_process_exception_data(e, code, message), indent=2)),
    ]


def _process_success(code: int = 200, data: dict | None = None) -> Sequence[TextContent]:
    rv = {"status": "success", "code": code}
    if data is not None:
        rv["data"] = data
    return [TextContent(type="text", text=json.dumps(rv, indent=2))]


def _process_langflow_sync(request: Dict[str, Any]) -> Dict[str, Any]:

    try:
        result = langflow_client.process_sync(request)
        return result
    except Exception as e:
        log.error(f"Langflow processing failed: {e}")
        return _process_exception_data(e, message="AI processing failed")


async def handle_lint_yaml(**kwargs) -> Sequence[TextContent]:
    """Handle YAML linting requests."""
    content = kwargs.get("content", "")
    mode = kwargs.get("mode", "syntax")

    # Process through Langflow in thread pool (sync to async)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        _process_langflow_sync,
        {
            "input_value": content,
            "tweaks": {
                "yaml-parser": {"validation_mode": mode},
                "response-formatter": {"format_type": "mcp_response"},
            },
        },
    )

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_validate_cloudformation(**kwargs) -> Sequence[TextContent]:
    """Handle CloudFormation validation requests."""

    template = kwargs.get("template", "")
    region = kwargs.get("region", "us-east-1")
    strict = kwargs.get("strict", True)

    # Process through Langflow
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        _process_langflow_sync,
        {
            "input_value": template,
            "tweaks": {
                "cf-validator": {"region": region, "strict_mode": strict},
                "response-formatter": {"format_type": "mcp_response"},
            },
        },
    )

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_suggest_completion(**kwargs) -> Sequence[TextContent]:
    """Handle code completion requests."""

    content = kwargs.get("content", "")
    cursor_line = kwargs.get("cursor_line", 1)
    cursor_column = kwargs.get("cursor_column", 1)
    context_type = kwargs.get("context_type", "auto")

    # Prepare AI prompt for completion
    completion_prompt = f"""Provide code completion suggestions for the following content.

Content type: {context_type}
Cursor position: Line {cursor_line}, Column {cursor_column}

Content:
{content}

Return JSON array of suggestions with: text, description, insertText, kind."""

    # Process through Langflow
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        _process_langflow_sync,
        {
            "input_value": completion_prompt,
            "tweaks": {
                "ai-analyzer": {
                    "system_message": "You are a YAML/CloudFormation completion assistant. Provide helpful, contextually relevant suggestions.",
                    "temperature": 0.1,
                },
                "response-formatter": {"format_type": "mcp_response"},
            },
        },
    )

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_analyze_template(**kwargs) -> Sequence[TextContent]:
    """Handle deep template analysis requests."""
    template = kwargs.get("template", "")
    analysis_type = kwargs.get("analysis_type", "comprehensive")
    region = kwargs.get("region", "us-east-1")

    # Prepare analysis prompt
    analysis_prompts = {
        "security": "Focus on security vulnerabilities, IAM policies, encryption, and access controls.",
        "cost": "Analyze cost optimization opportunities, resource sizing, and billing implications.",
        "best_practices": "Evaluate adherence to AWS Well-Architected Framework principles.",
        "comprehensive": "Perform complete analysis covering security, cost, performance, and best practices.",
    }

    system_message = f"""You are an expert AWS CloudFormation analyst. {analysis_prompts.get(analysis_type, analysis_prompts['comprehensive'])}

Analyze the provided CloudFormation template and provide detailed findings with:
1. Specific issues and recommendations
2. Risk levels (high/medium/low)
3. Implementation guidance
4. References to AWS documentation"""

    # Process through Langflow
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        _process_langflow_sync,
        {
            "input_value": template,
            "tweaks": {
                "ai-analyzer": {
                    "system_message": system_message,
                    "temperature": 0.1,
                },
                "cf-validator": {"region": region, "strict_mode": True},
                "response-formatter": {"format_type": "mcp_response"},
            },
        },
    )

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_search_documentation(**kwargs) -> Sequence[TextContent]:
    """Handle documentation search requests."""
    if not context_manager:
        return process_exception(code=503, message="Indexing system not available")

    query = kwargs.get("query", "")
    section = kwargs.get("section")
    max_results = kwargs.get("max_results", 5)

    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            context_manager.doc_indexer.search_documentation,
            query,
            section,
            max_results,
        )

        return _process_success(
            data={
                "query": query,
                "section": section,
                "results": results,
                "count": len(results),
            },
        )

    except Exception as e:
        log.error(f"Documentation search failed: {e}")
        return process_exception(e, code=500, message="Documentation search failed")


async def handle_search_codebase(**kwargs) -> Sequence[TextContent]:
    """Handle codebase search requests."""
    if not context_manager:
        return process_exception(code=503, message="Indexing system not available")

    query = kwargs.get("query", "")
    project = kwargs.get("project")
    element_type = kwargs.get("element_type")
    max_results = kwargs.get("max_results", 5)

    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            context_manager.code_indexer.search_codebase,
            query,
            project,
            element_type,
            max_results,
        )

        return _process_success(
            data={
                "query": query,
                "project": project,
                "element_type": element_type,
                "results": results,
                "count": len(results),
            },
        )

    except Exception as e:
        log.error(f"Codebase search failed: {e}")
        return process_exception(e, code=500, message="Codebase search failed")


async def handle_get_context_for_query(**kwargs) -> Sequence[TextContent]:
    """Handle comprehensive context retrieval requests."""
    if not context_manager:
        return process_exception(code=503, message="Indexing system not available")

    query = kwargs.get("query", "")
    strategy = kwargs.get("strategy", "balanced")
    max_results = kwargs.get("max_results", 10)
    score_mode = kwargs.get("score_mode", "auto")
    fused = kwargs.get("fused", True)
    token_budget = kwargs.get("token_budget")

    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        context = await loop.run_in_executor(
            None,
            context_manager.get_context_for_query,
            query,
            strategy,
            max_results,
            True,  # include_metadata
            score_mode,
            fused,
            token_budget,
        )

        return _process_success(data=context)

    except Exception as e:
        log.error(f"Context retrieval failed: {e}")
        return process_exception(e, code=500, message="Context retrieval failed")


async def handle_initialize_indexes(**kwargs) -> Sequence[TextContent]:
    """Handle index initialization requests."""
    if not context_manager:
        return process_exception(code=503, message="Indexing system not available")

    force_rebuild = kwargs.get("force_rebuild", False)

    try:
        # Run in thread pool as this can take a while
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, context_manager.initialize_indexes, force_rebuild)

        return _process_success(
            data={
                "force_rebuild": force_rebuild,
                "results": results,
            },
        )

    except Exception as e:
        log.error(f"Index initialization failed: {e}")
        return process_exception(e, code=500, message="Index initialization failed")


async def handle_get_indexing_stats(**kwargs) -> Sequence[TextContent]:
    """Handle indexing statistics requests."""
    if not context_manager:
        return process_exception(code=503, message="Indexing system not available")

    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, context_manager.get_system_stats)

        return _process_success(data=stats)

    except Exception as e:
        log.error(f"Getting indexing stats failed: {e}")
        return process_exception(e, code=500, message="Getting indexing stats failed")
