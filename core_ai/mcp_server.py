"""
MCP (Model Context Protocol) server for SCK Core AI agent.

Provides MCP tool interfaces for AI assistants to interact with
YAML/CloudFormation linting and validation capabilities.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

# MCP server imports
try:
    from mcp import types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    MCP_AVAILABLE = True
except ImportError:
    # Mock for development without MCP
    MCP_AVAILABLE = False
    types = None
    Server = None

# Import SCK framework components (when available)
try:
    import core_logging as log

    SCK_AVAILABLE = True
except ImportError:
    import logging as log

    SCK_AVAILABLE = False

# Import Langflow client
try:
    from .langflow.client import LangflowClient
except ImportError:
    # Mock for development
    class LangflowClient:
        def __init__(self, *args, **kwargs):
            pass

        def process_sync(self, *args, **kwargs):
            return {"status": "success", "message": "Mock response"}


# Configure logging
if SCK_AVAILABLE:
    logger = log.get_logger(__name__)
else:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class SCKCoreAIMCPServer:
    """MCP server for SCK Core AI agent."""

    def __init__(self):
        """Initialize MCP server."""
        if not MCP_AVAILABLE:
            raise ImportError("MCP not available - install with: uv add mcp")

        self.server = Server("sck-core-ai")
        self.langflow_client: Optional[LangflowClient] = None
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

        @self.server.call_tool()
        async def call_tool(
            name: str, arguments: Dict[str, Any]
        ) -> Sequence[types.TextContent]:
            """Handle tool calls."""
            logger.info(f"MCP tool called: {name}")

            try:
                if name == "lint_yaml":
                    return await self._handle_lint_yaml(arguments)
                elif name == "validate_cloudformation":
                    return await self._handle_validate_cloudformation(arguments)
                elif name == "suggest_completion":
                    return await self._handle_suggest_completion(arguments)
                elif name == "analyze_template":
                    return await self._handle_analyze_template(arguments)
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
            # Initialize client if needed
            try:
                self.langflow_client = LangflowClient(
                    base_url="http://localhost:7860", flow_id="yaml-cf-ai-agent-v1"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Langflow client: {e}")
                self.langflow_client = LangflowClient()  # Mock client

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

    async def run(self):
        """Run the MCP server."""
        if not MCP_AVAILABLE:
            raise RuntimeError("MCP not available")

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
