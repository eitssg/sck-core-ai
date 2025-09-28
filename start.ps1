#!/usr/bin/env pwsh

# Start script for sck-core-ai local development server

param(
    [string]$Mode = "api",  # api, mcp, or both
    [int]$Port = 8200,
    [string]$ServerHost = "0.0.0.0",
    [switch]$NoReload = $false
)

$ErrorActionPreference = "Stop"

# Set UV configuration to avoid hardlink warnings
$env:UV_LINK_MODE = "copy"

# Check if uv is available
$uvAvailable = $false
try {
    $uvVersion = uv --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        $uvAvailable = $true
        Write-Host "Using uv: $uvVersion" -ForegroundColor Cyan
    }
}
catch {
    Write-Host "uv not found, using direct commands" -ForegroundColor Yellow
}

$runPrefix = if ($uvAvailable) { "uv run" } else { "" }

# Ensure .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "Please edit .env file with your configuration" -ForegroundColor Cyan
}

# Load .env values (simple parser) to allow overriding defaults without changing script
$envFile = Get-Content .env | Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' }
foreach ($line in $envFile) {
    $k, $v = $line -split '=', 2
    if (-not [string]::IsNullOrWhiteSpace($k)) {
        # Only set if not already defined in the process environment
        if (-not (Test-Path env:$k)) {
            Set-Item -Path env:$k -Value $v
        }
    }
}

# Port precedence: explicit -Port parameter > SCK_AI_PORT env > SERVER_PORT env > script default
if ($PSBoundParameters.ContainsKey('Port') -eq $false) {
    if ($env:SCK_AI_PORT) { $Port = [int]$env:SCK_AI_PORT }
    elseif ($env:SERVER_PORT) { $Port = [int]$env:SERVER_PORT }
}

try {
    switch ($Mode.ToLower()) {
        "api" {
            Write-Host "Starting FastAPI development server..." -ForegroundColor Green
            Write-Host "Server will be available at: http://$ServerHost`:$Port" -ForegroundColor Cyan
            Write-Host "API documentation: http://$ServerHost`:$Port/docs" -ForegroundColor Cyan
            
            if (-not $NoReload) {
                Invoke-Expression "$runPrefix uvicorn core_ai.server:app --host $ServerHost --port $Port --reload"
            }
            else {
                Invoke-Expression "$runPrefix uvicorn core_ai.server:app --host $ServerHost --port $Port"
            }
        }
        
        "mcp" {
            Write-Host "Starting MCP server..." -ForegroundColor Green
            Write-Host "MCP server will run on stdio interface" -ForegroundColor Cyan
            
            Invoke-Expression "$runPrefix python -m core_ai.mcp_server"
        }
        
        "both" {
            Write-Host "Starting both API and MCP servers..." -ForegroundColor Green
            Write-Host "This will start API server in background and MCP server in foreground" -ForegroundColor Yellow
            
            # Start API server in background
            Start-Job -ScriptBlock {
                param($runPrefix, $ServerHost, $Port)
                Invoke-Expression "$runPrefix uvicorn core_ai.server:app --host $ServerHost --port $Port"
            } -ArgumentList $runPrefix, $ServerHost, $Port -Name "sck-ai-api"
            
            Write-Host "API server started in background (Job: sck-ai-api)" -ForegroundColor Cyan
            Write-Host "Starting MCP server..." -ForegroundColor Cyan
            
            Invoke-Expression "$runPrefix python -m core_ai.mcp_server"
        }
        
        default {
            throw "Invalid mode: $Mode. Use 'api', 'mcp', or 'both'"
        }
    }
    
}
catch {
    Write-Host "Server startup failed: $($_.Exception.Message)" -ForegroundColor Red
    
    # Clean up background job if it exists
    if (Get-Job -Name "sck-ai-api" -ErrorAction SilentlyContinue) {
        Remove-Job -Name "sck-ai-api" -Force
        Write-Host "Cleaned up background API server job" -ForegroundColor Yellow
    }
    
    exit 1
}