"""
SCK MCP Tools for Langflow Integration
Connects Langflow Agent to the SCK MCP Server for documentation and code search
"""

from typing import List
import httpx
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class MCPSearchInput(BaseModel):
    """Input schema for MCP search tools."""

    query: str = Field(description="The search query for documentation or code")


class SCKDocumentationSearchTool(BaseTool):
    """Tool to search SCK documentation via MCP server."""

    name: str = "search_sck_documentation"
    description: str = """
    Search the Simple Cloud Kit (SCK) documentation for information about:
    - Architecture patterns and components
    - Development guidelines and best practices  
    - API documentation and usage examples
    - Configuration and setup instructions
    - Multi-tenancy and authentication patterns
    
    Use this when users ask about SCK concepts, patterns, or implementation details.
    """
    args_schema: type[BaseModel] = MCPSearchInput
    mcp_server_url: str = (
        "http://localhost:8200"  # Default MCP server URL (changed from 8000 to avoid conflicts)
    )

    def _run(self, query: str) -> str:
        """Search SCK documentation using the MCP server."""
        try:
            # Prepare MCP request
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search_documentation",
                    "arguments": {"query": query},
                },
            }

            # Call MCP server
            with httpx.Client() as client:
                response = client.post(
                    f"{self.mcp_server_url}/mcp", json=mcp_request, timeout=30.0
                )
                response.raise_for_status()

            # Parse response
            result = response.json()
            if "result" in result and "content" in result["result"]:
                content = result["result"]["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", "No documentation found")
                elif isinstance(content, str):
                    return content
                else:
                    return "No documentation found"
            else:
                return f"MCP server error: {result.get('error', 'Unknown error')}"

        except Exception as e:
            return f"Error searching documentation: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version of the search."""
        return self._run(query)


class SCKCodeSearchTool(BaseTool):
    """Tool to search SCK codebase via MCP server."""

    name: str = "search_sck_codebase"
    description: str = """
    Search the Simple Cloud Kit (SCK) codebase for:
    - Function and class implementations
    - Code examples and patterns
    - Module structure and dependencies
    - Import statements and usage
    - Lambda handlers and API endpoints
    
    Use this when users need to see actual code implementations or examples.
    """
    args_schema: type[BaseModel] = MCPSearchInput
    mcp_server_url: str = "http://localhost:8200"

    def _run(self, query: str) -> str:
        """Search SCK codebase using the MCP server."""
        try:
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "search_codebase", "arguments": {"query": query}},
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.mcp_server_url}/mcp", json=mcp_request, timeout=30.0
                )
                response.raise_for_status()

            result = response.json()
            if "result" in result and "content" in result["result"]:
                content = result["result"]["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", "No code found")
                elif isinstance(content, str):
                    return content
                else:
                    return "No code found"
            else:
                return f"MCP server error: {result.get('error', 'Unknown error')}"

        except Exception as e:
            return f"Error searching codebase: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version of the search."""
        return self._run(query)


class SCKCloudFormationTool(BaseTool):
    """Tool to validate and analyze CloudFormation templates via MCP server."""

    name: str = "validate_cloudformation"
    description: str = """
    Validate and analyze CloudFormation templates using SCK AI tools:
    - Syntax validation and error checking
    - Security analysis and recommendations
    - Cost optimization suggestions
    - Best practices compliance
    - Resource configuration review
    
    Use this when users provide CloudFormation templates or ask about AWS infrastructure.
    """
    args_schema: type[BaseModel] = MCPSearchInput  # Reuse for template content
    mcp_server_url: str = "http://localhost:8200"

    def _run(self, query: str) -> str:
        """Validate CloudFormation template using MCP server."""
        try:
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "validate_cloudformation",
                    "arguments": {"template": query},
                },
            }

            with httpx.Client() as client:
                response = client.post(
                    f"{self.mcp_server_url}/mcp", json=mcp_request, timeout=30.0
                )
                response.raise_for_status()

            result = response.json()
            if "result" in result and "content" in result["result"]:
                content = result["result"]["content"]
                if isinstance(content, list) and len(content) > 0:
                    return content[0].get("text", "Validation failed")
                elif isinstance(content, str):
                    return content
                else:
                    return "Validation failed"
            else:
                return f"MCP server error: {result.get('error', 'Unknown error')}"

        except Exception as e:
            return f"Error validating CloudFormation: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async version of the validation."""
        return self._run(query)


# Tool registry for easy import
SCK_MCP_TOOLS = [
    SCKDocumentationSearchTool(),
    SCKCodeSearchTool(),
    SCKCloudFormationTool(),
]


def get_sck_mcp_tools(mcp_server_url: str = "http://localhost:8200") -> List[BaseTool]:
    """Get all SCK MCP tools with custom server URL."""
    tools = []
    for tool_class in [
        SCKDocumentationSearchTool,
        SCKCodeSearchTool,
        SCKCloudFormationTool,
    ]:
        tool = tool_class()
        tool.mcp_server_url = mcp_server_url
        tools.append(tool)
    return tools
