from typing import Any

from .sck_arch_tools import SCKArchitectureTool
from .sck_docs_tools import SCKDocumentationTool
from .sck_code_tools import SCKCodeSearchTool


# Export tools for easy access
def get_sck_tools() -> dict[str, Any]:
    """Get all SCK tools."""
    return {
        "search_documentation": SCKDocumentationTool.search_documentation,
        "search_codebase": SCKCodeSearchTool.search_codebase,
        "get_architecture": SCKArchitectureTool.get_architecture_info,
    }


# Test the tools
def show_tools():
    tools = get_sck_tools()

    print("\nTesting SCK Documentation Search...")
    result = tools["search_documentation"]("architecture")
    print(result[:200] + "..." if len(result) > 200 else result)

    print("\nTesting SCK Codebase Search...")
    result = tools["search_codebase"]("ProxyEvent")
    print(result[:200] + "..." if len(result) > 200 else result)

    print("\nTesting SCK Architecture Info...")
    result = tools["get_architecture"]("lambda")
    print(result)
