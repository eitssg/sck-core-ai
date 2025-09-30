"""
MCP (Model Context Protocol) server for SCK Core AI agent.

Provides MCP tool interfaces for AI assistants to interact with
YAML/CloudFormation linting and validation capabilities.
"""

from typing import Any, Dict, List, Optional, Sequence

import asyncio
import json
import os

from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
env_file = find_dotenv()
if env_file:
    load_dotenv(env_file)
    print(f"Loaded environment from: {env_file}")
else:
    load_dotenv()  # Load from current directory or environment
    print("Loaded environment from current directory or system environment")

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server


# Import SCK framework components (when available)
import core_logging as logger

from .langflow.client import LangflowClient

from .indexing import ContextManager
from .indexing import VectorStore
from .indexing import get_availability_status


class SCKCoreAIMCPServer:
    """MCP server for SCK Core AI agent."""

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        build_directory: Optional[str] = None,
    ):
        """
        Initialize MCP server.

        Args:
            workspace_root: Path to workspace root (auto-detected if None)
            build_directory: Path to documentation build directory (auto-detected if None)
        """
        # Log startup and environment info
        logger.info("Initializing SCK Core AI MCP Server")
        logger.info(f"LOCAL_MODE: {os.getenv('LOCAL_MODE', 'Not set')}")
        logger.info(f"LOG_DIR: {os.getenv('LOG_DIR', 'Not set')}")
        logger.info(f"VOLUME: {os.getenv('VOLUME', 'Not set')}")
        logger.info(f"CLIENT: {os.getenv('CLIENT', 'Not set')}")
        logger.info(f"LOG_LEVEL: {os.getenv('LOG_LEVEL', 'Not set')}")

        self.server = Server("sck-core-ai")
        self.langflow_client: Optional[LangflowClient] = None

        # Initialize indexing system if available
        self.context_manager: Optional[ContextManager] = None

        try:
            # Auto-detect paths if not provided
            if workspace_root is None:
                # Assume we're in sck-core-ai, go up to workspace root
                workspace_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../../../")
                )

            if build_directory is None:
                build_directory = os.path.join(workspace_root, "sck-core-docs", "build")

            self.context_manager = ContextManager(
                build_directory=build_directory, workspace_root=workspace_root
            )
            logger.info(f"Indexing system initialized with workspace: {workspace_root}")

        except Exception as e:
            logger.warning(f"Failed to initialize indexing system: {e}")
            self.context_manager = None

        self._setup_tools()

    def _setup_tools(self):
        """Set up MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            """List available AI agent tools."""
            return [
                types.Tool(
                    name="lint_yaml",
                    description="Lint and validate YAML content with AI-powered suggestions",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "YAML content to lint and validate",
                            },
                            "mode": {
                                "type": "string",
                                "enum": [
                                    "syntax",
                                    "schema",
                                    "best_practices",
                                    "security",
                                ],
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
                types.Tool(
                    name="validate_cloudformation",
                    description="Validate CloudFormation templates with comprehensive analysis",
                    inputSchema={
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
                types.Tool(
                    name="suggest_completion",
                    description="Get AI-powered code completion suggestions for YAML/CloudFormation",
                    inputSchema={
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
                types.Tool(
                    name="analyze_template",
                    description="Perform deep analysis of CloudFormation templates for security, cost, and best practices",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "template": {
                                "type": "string",
                                "description": "CloudFormation template (YAML or JSON format)",
                            },
                            "analysis_type": {
                                "type": "string",
                                "enum": [
                                    "security",
                                    "cost",
                                    "best_practices",
                                    "comprehensive",
                                ],
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

            # Define base tools list
            base_tools = [
                types.Tool(
                    name="lint_yaml",
                    description="Lint and validate YAML content with AI-powered suggestions",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "YAML content to lint and validate",
                            },
                            "mode": {
                                "type": "string",
                                "enum": [
                                    "syntax",
                                    "schema",
                                    "best_practices",
                                    "security",
                                ],
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
                types.Tool(
                    name="validate_cloudformation",
                    description="Validate CloudFormation templates with comprehensive analysis",
                    inputSchema={
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
                types.Tool(
                    name="suggest_completion",
                    description="Get AI-powered code completion suggestions for YAML/CloudFormation",
                    inputSchema={
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
                types.Tool(
                    name="analyze_template",
                    description="Perform deep analysis of CloudFormation templates for security, cost, and best practices",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "template": {
                                "type": "string",
                                "description": "CloudFormation template (YAML or JSON format)",
                            },
                            "analysis_type": {
                                "type": "string",
                                "enum": [
                                    "security",
                                    "cost",
                                    "best_practices",
                                    "comprehensive",
                                ],
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

            # Add indexing tools if available
            if self.context_manager:
                indexing_tools = [
                    types.Tool(
                        name="search_documentation",
                        description="Search SCK documentation for relevant information",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query for documentation",
                                },
                                "section": {
                                    "type": "string",
                                    "enum": [
                                        "technical_reference",
                                        "developer_guide",
                                        "user_guide",
                                    ],
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
                    types.Tool(
                        name="search_codebase",
                        description="Search SCK codebase for functions, classes, and code patterns",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query for codebase",
                                },
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
                    types.Tool(
                        name="get_context_for_query",
                        description="Get comprehensive context from both documentation and codebase for development assistance",
                        inputSchema={
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
                    types.Tool(
                        name="initialize_indexes",
                        description="Initialize or rebuild documentation and codebase indexes",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "force_rebuild": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Force rebuild of existing indexes",
                                },
                            },
                        },
                    ),
                    types.Tool(
                        name="get_indexing_stats",
                        description="Get statistics about indexed documentation and codebase",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                        },
                    ),
                ]

                return base_tools + indexing_tools

            return base_tools

        @self.server.call_tool()
        async def call_tool(
            name: str, arguments: Dict[str, Any]
        ) -> Sequence[types.TextContent]:
            """Handle tool calls."""
            logger.info(f"MCP tool called: {name}")

            try:
                # Original YAML/CloudFormation tools
                if name == "lint_yaml":
                    return await self._handle_lint_yaml(arguments)
                elif name == "validate_cloudformation":
                    return await self._handle_validate_cloudformation(arguments)
                elif name == "suggest_completion":
                    return await self._handle_suggest_completion(arguments)
                elif name == "analyze_template":
                    return await self._handle_analyze_template(arguments)

                # Indexing tools
                elif name == "search_documentation":
                    return await self._handle_search_documentation(arguments)
                elif name == "search_codebase":
                    return await self._handle_search_codebase(arguments)
                elif name == "get_context_for_query":
                    return await self._handle_get_context_for_query(arguments)
                elif name == "initialize_indexes":
                    return await self._handle_initialize_indexes(arguments)
                elif name == "get_indexing_stats":
                    return await self._handle_get_indexing_stats(arguments)

                else:
                    raise ValueError(f"Unknown tool: {name}")

            except Exception as e:
                logger.error(f"Tool execution failed: {name}", error=str(e))
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "status": "error",
                                "code": 500,
                                "message": f"Tool execution failed: {str(e)}",
                            },
                            indent=2,
                        ),
                    )
                ]

    async def _handle_lint_yaml(
        self, arguments: Dict[str, Any]
    ) -> Sequence[types.TextContent]:
        """Handle YAML linting requests."""
        content = arguments.get("content", "")
        mode = arguments.get("mode", "syntax")
        strict = arguments.get("strict", True)

        # Process through Langflow in thread pool (sync to async)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._process_langflow_sync,
            {
                "input_value": content,
                "tweaks": {
                    "yaml-parser": {"validation_mode": mode},
                    "response-formatter": {"format_type": "mcp_response"},
                },
            },
        )

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_validate_cloudformation(
        self, arguments: Dict[str, Any]
    ) -> Sequence[types.TextContent]:
        """Handle CloudFormation validation requests."""
        template = arguments.get("template", "")
        region = arguments.get("region", "us-east-1")
        strict = arguments.get("strict", True)

        # Process through Langflow
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._process_langflow_sync,
            {
                "input_value": template,
                "tweaks": {
                    "cf-validator": {"region": region, "strict_mode": strict},
                    "response-formatter": {"format_type": "mcp_response"},
                },
            },
        )

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_suggest_completion(
        self, arguments: Dict[str, Any]
    ) -> Sequence[types.TextContent]:
        """Handle code completion requests."""
        content = arguments.get("content", "")
        cursor_line = arguments.get("cursor_line", 1)
        cursor_column = arguments.get("cursor_column", 1)
        context_type = arguments.get("context_type", "auto")

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
            self._process_langflow_sync,
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

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_analyze_template(
        self, arguments: Dict[str, Any]
    ) -> Sequence[types.TextContent]:
        """Handle deep template analysis requests."""
        template = arguments.get("template", "")
        analysis_type = arguments.get("analysis_type", "comprehensive")
        region = arguments.get("region", "us-east-1")

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
            self._process_langflow_sync,
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

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    def _process_langflow_sync(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request through Langflow synchronously."""
        if not self.langflow_client:
            # Initialize Langflow client if not already done
            self.langflow_client = LangflowClient()

        try:
            result = self.langflow_client.process_sync(request)
            return result
        except Exception as e:
            logger.error(f"Langflow processing failed: {e}")
            return {
                "status": "error",
                "code": 500,
                "message": f"AI processing failed: {str(e)}",
                "data": None,
            }

    # Indexing tool handlers
    async def _handle_search_documentation(self, arguments: Dict[str, Any]):
        """Handle documentation search requests."""
        if not self.context_manager:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "error",
                            "code": 503,
                            "message": "Indexing system not available",
                        },
                        indent=2,
                    ),
                )
            ]

        query = arguments.get("query", "")
        section = arguments.get("section")
        max_results = arguments.get("max_results", 5)

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.context_manager.doc_indexer.search_documentation,
                query,
                section,
                max_results,
            )

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "success",
                            "code": 200,
                            "data": {
                                "query": query,
                                "section": section,
                                "results": results,
                                "count": len(results),
                            },
                        },
                        indent=2,
                    ),
                )
            ]

        except Exception as e:
            logger.error(f"Documentation search failed: {e}")
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "error",
                            "code": 500,
                            "message": f"Documentation search failed: {str(e)}",
                        },
                        indent=2,
                    ),
                )
            ]

    async def _handle_search_codebase(self, arguments: Dict[str, Any]):
        """Handle codebase search requests."""
        if not self.context_manager:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "error",
                            "code": 503,
                            "message": "Indexing system not available",
                        },
                        indent=2,
                    ),
                )
            ]

        query = arguments.get("query", "")
        project = arguments.get("project")
        element_type = arguments.get("element_type")
        max_results = arguments.get("max_results", 5)

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.context_manager.code_indexer.search_codebase,
                query,
                project,
                element_type,
                max_results,
            )

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "success",
                            "code": 200,
                            "data": {
                                "query": query,
                                "project": project,
                                "element_type": element_type,
                                "results": results,
                                "count": len(results),
                            },
                        },
                        indent=2,
                    ),
                )
            ]

        except Exception as e:
            logger.error(f"Codebase search failed: {e}")
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "error",
                            "code": 500,
                            "message": f"Codebase search failed: {str(e)}",
                        },
                        indent=2,
                    ),
                )
            ]

    async def _handle_get_context_for_query(self, arguments: Dict[str, Any]):
        """Handle comprehensive context retrieval requests."""
        if not self.context_manager:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "error",
                            "code": 503,
                            "message": "Indexing system not available",
                        },
                        indent=2,
                    ),
                )
            ]

        query = arguments.get("query", "")
        strategy = arguments.get("strategy", "balanced")
        max_results = arguments.get("max_results", 10)

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            context = await loop.run_in_executor(
                None,
                self.context_manager.get_context_for_query,
                query,
                strategy,
                max_results,
                True,
            )

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"status": "success", "code": 200, "data": context}, indent=2
                    ),
                )
            ]

        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "error",
                            "code": 500,
                            "message": f"Context retrieval failed: {str(e)}",
                        },
                        indent=2,
                    ),
                )
            ]

    async def _handle_initialize_indexes(self, arguments: Dict[str, Any]):
        """Handle index initialization requests."""
        if not self.context_manager:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "error",
                            "code": 503,
                            "message": "Indexing system not available",
                        },
                        indent=2,
                    ),
                )
            ]

        force_rebuild = arguments.get("force_rebuild", False)

        try:
            # Run in thread pool as this can take a while
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, self.context_manager.initialize_indexes, force_rebuild
            )

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "success",
                            "code": 200,
                            "data": {
                                "force_rebuild": force_rebuild,
                                "results": results,
                            },
                        },
                        indent=2,
                    ),
                )
            ]

        except Exception as e:
            logger.error(f"Index initialization failed: {e}")
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "error",
                            "code": 500,
                            "message": f"Index initialization failed: {str(e)}",
                        },
                        indent=2,
                    ),
                )
            ]

    async def _handle_get_indexing_stats(self, arguments: Dict[str, Any]):
        """Handle indexing statistics requests."""
        if not self.context_manager:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "error",
                            "code": 503,
                            "message": "Indexing system not available",
                        },
                        indent=2,
                    ),
                )
            ]

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None, self.context_manager.get_system_stats
            )

            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"status": "success", "code": 200, "data": stats}, indent=2
                    ),
                )
            ]

        except Exception as e:
            logger.error(f"Getting indexing stats failed: {e}")
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "status": "error",
                            "code": 500,
                            "message": f"Getting indexing stats failed: {str(e)}",
                        },
                        indent=2,
                    ),
                )
            ]

    async def run(self):
        """Run the MCP server."""
        logger.info("Starting SCK Core AI MCP server")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )


def create_mcp_server() -> SCKCoreAIMCPServer:
    """Create MCP server instance."""
    return SCKCoreAIMCPServer()


async def main():
    """Main entry point for MCP server."""
    server = create_mcp_server()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
