#!/usr/bin/env pwsh

# Start script for sck-core-ai MCP server (stdio)

param()

$ErrorActionPreference = 'Stop'

# Prefer copy to avoid hardlink warnings in some envs
$env:UV_LINK_MODE = 'copy'

# Ensure .env file exists (dotenv is loaded inside the app)
if (-not (Test-Path '.env')) {
    if (Test-Path '.env.example') {
        Write-Host 'Creating .env file from template...' -ForegroundColor Yellow
        Copy-Item '.env.example' '.env'
        Write-Host 'Please edit .env file with your configuration' -ForegroundColor Cyan
    }
}

# Show MCP indexing behavior if set
$idx = $env:CORE_AI_MCP_INDEX_ON_START
$frc = $env:CORE_AI_MCP_FORCE_REBUILD
if ($idx -or $frc) {
    Write-Host "MCP indexing on start: $idx (force_rebuild=$frc)" -ForegroundColor DarkGray
}

try {
    Write-Host 'Starting MCP server (stdio)...' -ForegroundColor Green
    & uv run -m core_ai.mcp_server
}
catch {
    Write-Host "Server startup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}