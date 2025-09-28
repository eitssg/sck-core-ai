"""Shared tool registry definitions for SCK Core AI.

This module extracts the MCP tool metadata definitions so they can be:
1. Re-used by the stdio MCP server implementation
2. Exposed over HTTP (REST) for /v1/tools and /v1/tools/{name}/invoke
3. Streamed over Server-Sent Events (SSE) for MCP-over-HTTP compatibility (phase 3)

The data structures here are intentionally framework-agnostic (no FastAPI imports)
so they can be imported by both the FastAPI layer and the stdio MCP server
without creating circular dependencies.
"""

from __future__ import annotations

from typing import Dict, Any, List, Callable, Awaitable, Optional
from dataclasses import dataclass


@dataclass
class ToolSpec:
    """Specification for an AI tool.

    Attributes:
        name: Unique tool name
        description: Human readable description
        schema: JSON schema (dict) describing input payload
    """

    name: str
    description: str
    schema: Dict[str, Any]


# Base (non-indexing) tools that are always available
BASE_TOOL_SPECS: List[ToolSpec] = [
    ToolSpec(
        name="lint_yaml",
        description="Lint and validate YAML content with AI-powered suggestions",
        schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "YAML content to lint and validate",
                },
                "mode": {
                    "type": "string",
                    "enum": ["syntax", "schema", "best_practices", "security"],
                    "default": "syntax",
                    "description": "Validation mode",
                },
                "strict": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable strict validation rules",
                },
            },
            "required": ["content"],
        },
    ),
    ToolSpec(
        name="validate_cloudformation",
        description="Validate CloudFormation templates with comprehensive analysis",
        schema={
            "type": "object",
            "properties": {
                "template": {
                    "type": "string",
                    "description": "CloudFormation template (YAML or JSON format)",
                },
                "region": {
                    "type": "string",
                    "default": "us-east-1",
                    "description": "AWS region for validation context",
                },
                "strict": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable strict validation rules",
                },
            },
            "required": ["template"],
        },
    ),
    ToolSpec(
        name="suggest_completion",
        description="Get AI-powered code completion suggestions for YAML/CloudFormation",
        schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Partial YAML/CloudFormation content",
                },
                "cursor_line": {
                    "type": "integer",
                    "description": "Cursor line number (1-based)",
                },
                "cursor_column": {
                    "type": "integer",
                    "description": "Cursor column number (1-based)",
                },
                "context_type": {
                    "type": "string",
                    "enum": ["yaml", "cloudformation", "auto"],
                    "default": "auto",
                    "description": "Content type context",
                },
            },
            "required": ["content", "cursor_line", "cursor_column"],
        },
    ),
    ToolSpec(
        name="analyze_template",
        description="Perform deep analysis of CloudFormation templates for security, cost, and best practices",
        schema={
            "type": "object",
            "properties": {
                "template": {
                    "type": "string",
                    "description": "CloudFormation template (YAML or JSON format)",
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["security", "cost", "best_practices", "comprehensive"],
                    "default": "comprehensive",
                    "description": "Type of analysis to perform",
                },
                "region": {
                    "type": "string",
                    "default": "us-east-1",
                    "description": "AWS region context",
                },
            },
            "required": ["template"],
        },
    ),
]

# Indexing-related tools (optional, only if indexing subsystem is available)
INDEXING_TOOL_SPECS: List[ToolSpec] = [
    ToolSpec(
        name="search_documentation",
        description="Search SCK documentation for relevant information",
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for documentation",
                },
                "section": {
                    "type": "string",
                    "enum": ["technical_reference", "developer_guide", "user_guide"],
                    "description": "Optional documentation section filter",
                },
                "max_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of results to return",
                },
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="search_codebase",
        description="Search SCK codebase for functions, classes, and code patterns",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for codebase"},
                "project": {
                    "type": "string",
                    "description": "Optional project filter (e.g., sck-core-api)",
                },
                "element_type": {
                    "type": "string",
                    "enum": ["class", "function", "method", "module"],
                    "description": "Optional element type filter",
                },
                "max_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of results to return",
                },
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="get_context_for_query",
        description="Get comprehensive context from both documentation and codebase for development assistance",
        schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Development query or task description",
                },
                "strategy": {
                    "type": "string",
                    "enum": [
                        "balanced",
                        "documentation_focused",
                        "code_focused",
                        "documentation_only",
                        "code_only",
                    ],
                    "default": "balanced",
                    "description": "Context retrieval strategy",
                },
                "max_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum total results to return",
                },
            },
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="initialize_indexes",
        description="Initialize or rebuild documentation and codebase indexes",
        schema={
            "type": "object",
            "properties": {
                "force_rebuild": {
                    "type": "boolean",
                    "default": False,
                    "description": "Force rebuild of existing indexes",
                }
            },
        },
    ),
    ToolSpec(
        name="get_indexing_stats",
        description="Get statistics about indexed documentation and codebase",
        schema={"type": "object", "properties": {}},
    ),
]


def list_all_tool_specs(include_indexing: bool) -> List[ToolSpec]:
    """Return tool specs depending on indexing availability."""
    tools = list(BASE_TOOL_SPECS)
    if include_indexing:
        tools.extend(INDEXING_TOOL_SPECS)
    return tools
