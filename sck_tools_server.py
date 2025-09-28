"""
Simple SCK Tools HTTP Server for Langflow Integration
Provides HTTP endpoints for SCK documentation, codebase search, and architecture info
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
from pathlib import Path

# Import our SCK tools
try:
    # Add the langflow directory to the path
    langflow_dir = Path(__file__).parent / "langflow"
    sys.path.insert(0, str(langflow_dir))
    from simple_sck_tools import (
        SCKDocumentationTool,
        SCKCodeSearchTool,
        SCKArchitectureTool,
    )

    print("✅ Successfully imported SCK tools")
except ImportError as e:
    print(f"Warning: Could not import SCK tools: {e}. Using fallback implementations.")

    # Fallback implementations
    class SCKDocumentationTool:
        @staticmethod
        def search_documentation(query: str) -> str:
            return f"Documentation search for '{query}' - service not available"

    class SCKCodeSearchTool:
        @staticmethod
        def search_codebase(query: str) -> str:
            return f"Codebase search for '{query}' - service not available"

    class SCKArchitectureTool:
        @staticmethod
        def get_architecture_info(component: str = "") -> str:
            return f"Architecture info for '{component}' - service not available"


app = FastAPI(
    title="SCK Tools API",
    description="Simple Cloud Kit documentation, codebase search, and architecture tools",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str


class ArchitectureRequest(BaseModel):
    component: Optional[str] = ""


class ToolResponse(BaseModel):
    result: str
    success: bool = True
    error: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SCK Tools API",
        "status": "running",
        "tools": [
            "/search-docs - Search SCK documentation",
            "/search-code - Search SCK codebase",
            "/architecture - Get SCK architecture info",
        ],
    }


@app.post("/search-docs", response_model=ToolResponse)
async def search_documentation(request: SearchRequest):
    """Search SCK documentation."""
    try:
        result = SCKDocumentationTool.search_documentation(request.query)
        return ToolResponse(result=result, success=True)
    except Exception as e:
        return ToolResponse(
            result="", success=False, error=f"Documentation search failed: {str(e)}"
        )


@app.post("/search-code", response_model=ToolResponse)
async def search_codebase(request: SearchRequest):
    """Search SCK codebase."""
    try:
        result = SCKCodeSearchTool.search_codebase(request.query)
        return ToolResponse(result=result, success=True)
    except Exception as e:
        return ToolResponse(
            result="", success=False, error=f"Codebase search failed: {str(e)}"
        )


@app.post("/architecture", response_model=ToolResponse)
async def get_architecture(request: ArchitectureRequest):
    """Get SCK architecture information."""
    try:
        result = SCKArchitectureTool.get_architecture_info(request.component)
        return ToolResponse(result=result, success=True)
    except Exception as e:
        return ToolResponse(
            result="", success=False, error=f"Architecture info failed: {str(e)}"
        )


# Simple GET endpoints for testing
@app.get("/search-docs/{query}")
async def search_docs_get(query: str):
    """GET version of documentation search."""
    try:
        result = SCKDocumentationTool.search_documentation(query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search-code/{query}")
async def search_code_get(query: str):
    """GET version of codebase search."""
    try:
        result = SCKCodeSearchTool.search_codebase(query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/architecture/{component}")
async def get_architecture_get(component: str = ""):
    """GET version of architecture info."""
    try:
        result = SCKArchitectureTool.get_architecture_info(component)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🚀 Starting SCK Tools API on http://localhost:8002")
    print("📚 Available endpoints:")
    print("  GET  /search-docs/{query}")
    print("  GET  /search-code/{query}")
    print("  GET  /architecture/{component}")
    print("  POST /search-docs")
    print("  POST /search-code")
    print("  POST /architecture")

    uvicorn.run(app, host="0.0.0.0", port=8002)
