#!/usr/bin/env python3
"""
Test MCP server connection and tools
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path


async def test_mcp_server():
    """Test the MCP server by running it and sending commands."""

    print("🔧 Testing SCK MCP Server...")

    # Change to the sck-core-ai directory
    cwd = Path("D:/Development/simple-cloud-kit-oss/simple-cloud-kit/sck-core-ai")

    try:
        # Start MCP server process
        process = subprocess.Popen(
            [sys.executable, "-m", "core_ai.mcp_server"],
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Send initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        print("📤 Sending initialization request...")
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()

        # Read response
        response_line = process.stdout.readline()
        if response_line:
            try:
                response = json.loads(response_line.strip())
                print("✅ Initialization successful!")
                print(
                    f"📋 Server capabilities: {response.get('result', {}).get('capabilities', {})}"
                )
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse initialization response: {response_line}")
                print(f"Error: {e}")

        # Send tools list request
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        print("📤 Requesting tools list...")
        process.stdin.write(json.dumps(tools_request) + "\n")
        process.stdin.flush()

        # Read tools response
        tools_response_line = process.stdout.readline()
        if tools_response_line:
            try:
                tools_response = json.loads(tools_response_line.strip())
                tools = tools_response.get("result", {}).get("tools", [])
                print(f"🛠️  Available tools ({len(tools)}):")
                for tool in tools:
                    print(
                        f"   - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}"
                    )
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse tools response: {tools_response_line}")
                print(f"Error: {e}")

        # Test documentation search
        search_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_documentation",
                "arguments": {"query": "SCK architecture"},
            },
        }

        print("📤 Testing documentation search...")
        process.stdin.write(json.dumps(search_request) + "\n")
        process.stdin.flush()

        # Read search response
        search_response_line = process.stdout.readline()
        if search_response_line:
            try:
                search_response = json.loads(search_response_line.strip())
                content = search_response.get("result", {}).get("content", [])
                if content:
                    print("📚 Documentation search successful!")
                    print(f"   Found {len(content)} results")
                    if content and isinstance(content, list) and len(content) > 0:
                        first_result = content[0].get("text", "")[:200]
                        print(f"   Preview: {first_result}...")
                else:
                    print("📚 Documentation search returned no results")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse search response: {search_response_line}")
                print(f"Error: {e}")

        # Clean up
        process.terminate()
        process.wait(timeout=5)

        print("✅ MCP server test completed!")

    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        if "process" in locals():
            process.terminate()


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
