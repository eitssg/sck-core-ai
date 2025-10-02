# Windows PowerShell: Redirect stderr to see debug output
# uv run mcp dev core_ai/mcp_server.py 2>&1

# npx @modelcontextprotocol/inspector python core_ai/mcp_server.pyv run mcp dev core_ai/mcp_server.py 2>&1

# npx @modelcontextprotocol/inspector python core_ai/mcp_server.py

# Show stderr in red text (PowerShell feature)
uv run mcp dev core_ai/mcp_server.py 2>&1 | ForEach-Object {
    if ($_ -is [System.Management.Automation.ErrorRecord]) {
        Write-Host $_ -ForegroundColor Red
    }
    else {
        Write-Host $_
    }
}

