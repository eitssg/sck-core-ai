"""
HTTP bridge for MCP server to work with Langflow
This creates an HTTP endpoint that Langflow can call, which then communicates with the MCP server via stdio
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="SCK MCP HTTP Bridge", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MCPClient:
    """Simple MCP client that communicates via subprocess."""

    def __init__(self):
        self.process = None
        self.initialized = False

    async def start(self):
        """Start the MCP server process."""
        if self.process is not None:
            return

        cwd = Path("D:/Development/simple-cloud-kit-oss/simple-cloud-kit/sck-core-ai")

        self.process = subprocess.Popen(
            [sys.executable, "-m", "core_ai.mcp_server"],
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
        )

        # Initialize the MCP server
        await self.initialize()

    async def initialize(self):
        """Initialize the MCP server."""
        if self.initialized:
            return

        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "http-bridge", "version": "1.0.0"},
            },
        }

        response = await self.send_request(init_request)
        if response and "result" in response:
            self.initialized = True
            print("✅ MCP server initialized successfully")
        else:
            raise Exception(f"Failed to initialize MCP server: {response}")

    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the MCP server and get response."""
        if self.process is None:
            await self.start()

        try:
            # Send request
            request_line = json.dumps(request) + "\n"
            self.process.stdin.write(request_line)
            self.process.stdin.flush()

            # Read response
            response_line = self.process.stdout.readline()
            if response_line:
                return json.loads(response_line.strip())
            else:
                return {"error": "No response from MCP server"}

        except Exception as e:
            return {"error": f"Communication error: {str(e)}"}

    def stop(self):
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None
            self.initialized = False


# Global MCP client
mcp_client = MCPClient()


@app.on_event("startup")
async def startup():
    """Start the MCP client on server startup."""
    await mcp_client.start()


@app.on_event("shutdown")
async def shutdown():
    """Stop the MCP client on server shutdown."""
    mcp_client.stop()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "SCK MCP HTTP Bridge", "status": "running"}


@app.get("/tools")
async def list_tools():
    """List available MCP tools."""
    request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

    response = await mcp_client.send_request(request)
    if "result" in response:
        return response["result"]["tools"]
    else:
        raise HTTPException(status_code=500, detail=response.get("error", "Unknown error"))


@app.post("/search-documentation")
async def search_documentation(query: str):
    """Search SCK documentation."""
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "search_documentation", "arguments": {"query": query}},
    }

    response = await mcp_client.send_request(request)
    if "result" in response:
        content = response["result"].get("content", [])
        if content and isinstance(content, list) and len(content) > 0:
            return {"result": content[0].get("text", "No content found")}
        else:
            return {"result": "No documentation found"}
    else:
        raise HTTPException(status_code=500, detail=response.get("error", "Search failed"))


@app.post("/search-codebase")
async def search_codebase(query: str):
    """Search SCK codebase."""
    request = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "search_codebase", "arguments": {"query": query}},
    }

    response = await mcp_client.send_request(request)
    if "result" in response:
        content = response["result"].get("content", [])
        if content and isinstance(content, list) and len(content) > 0:
            return {"result": content[0].get("text", "No content found")}
        else:
            return {"result": "No code found"}
    else:
        raise HTTPException(status_code=500, detail=response.get("error", "Search failed"))


@app.post("/validate-cloudformation")
async def validate_cloudformation(template: str):
    """Validate CloudFormation template."""
    request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "validate_cloudformation",
            "arguments": {"template": template},
        },
    }

    response = await mcp_client.send_request(request)
    if "result" in response:
        content = response["result"].get("content", [])
        if content and isinstance(content, list) and len(content) > 0:
            return {"result": content[0].get("text", "Validation completed")}
        else:
            return {"result": "Validation completed with no output"}
    else:
        raise HTTPException(status_code=500, detail=response.get("error", "Validation failed"))


if __name__ == "__main__":
    print("🚀 Starting SCK MCP HTTP Bridge on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
