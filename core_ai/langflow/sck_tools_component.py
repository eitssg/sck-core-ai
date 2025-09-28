"""
SCK Custom Tool Component for Langflow
This component provides SCK-specific search and architecture tools
"""

from langflow.base.langchain_utilities.model import LCToolComponent
from langflow.inputs import StrInput
from langflow.schema import Data
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional, List
import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import our tools
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from simple_sck_tools import (
        SCKDocumentationTool,
        SCKCodeSearchTool,
        SCKArchitectureTool,
    )
except ImportError:
    # Fallback if import fails
    class SCKDocumentationTool:
        @staticmethod
        def search_documentation(query: str) -> str:
            return f"Documentation search not available. Searched for: {query}"

    class SCKCodeSearchTool:
        @staticmethod
        def search_codebase(query: str) -> str:
            return f"Codebase search not available. Searched for: {query}"

    class SCKArchitectureTool:
        @staticmethod
        def get_architecture_info(component: str = "") -> str:
            return f"Architecture info not available. Requested: {component}"


class DocumentationSearchInput(BaseModel):
    """Input for documentation search."""

    query: str = Field(description="Search query for SCK documentation")


class CodeSearchInput(BaseModel):
    """Input for codebase search."""

    query: str = Field(description="Search query for SCK codebase")


class ArchitectureInput(BaseModel):
    """Input for architecture information."""

    component: str = Field(
        default="",
        description="SCK component to get info about (overview, imports, lambda, s3)",
    )


class SCKToolsComponent(LCToolComponent):
    display_name = "SCK Expert Tools"
    description = "Search SCK documentation, codebase, and get architecture guidance"
    documentation: str = "https://docs.langflow.org/components-tools"
    name = "SCKTools"

    inputs = [
        StrInput(
            name="tool_description",
            display_name="Tool Description",
            value="SCK Expert Tools for documentation, code search, and architecture guidance",
            advanced=True,
        ),
    ]

    def build_tool(self) -> List[StructuredTool]:
        """Build the SCK tools."""

        def search_documentation_wrapper(query: str) -> str:
            """Search SCK documentation."""
            try:
                result = SCKDocumentationTool.search_documentation(query)
                return result
            except Exception as e:
                return f"Error searching documentation: {str(e)}"

        def search_codebase_wrapper(query: str) -> str:
            """Search SCK codebase."""
            try:
                result = SCKCodeSearchTool.search_codebase(query)
                return result
            except Exception as e:
                return f"Error searching codebase: {str(e)}"

        def get_architecture_wrapper(component: str = "") -> str:
            """Get SCK architecture information."""
            try:
                result = SCKArchitectureTool.get_architecture_info(component)
                return result
            except Exception as e:
                return f"Error getting architecture info: {str(e)}"

        # Create the structured tools
        tools = [
            StructuredTool(
                name="search_sck_documentation",
                description="Search the Simple Cloud Kit (SCK) documentation for patterns, guidelines, and examples. Use this when users ask about SCK concepts, configurations, or best practices.",
                func=search_documentation_wrapper,
                args_schema=DocumentationSearchInput,
            ),
            StructuredTool(
                name="search_sck_codebase",
                description="Search the SCK codebase for implementations, code examples, and patterns. Use this when users need to see actual code like ProxyEvent usage, core_logging patterns, or Lambda handlers.",
                func=search_codebase_wrapper,
                args_schema=CodeSearchInput,
            ),
            StructuredTool(
                name="get_sck_architecture",
                description="Get SCK architecture information and patterns. Use 'overview' for general info, 'imports' for import patterns, 'lambda' for handler patterns, 's3' for S3 operations. Leave empty for overview.",
                func=get_architecture_wrapper,
                args_schema=ArchitectureInput,
            ),
        ]

        return tools
