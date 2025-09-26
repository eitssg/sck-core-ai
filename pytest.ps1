#!/usr/bin/env pwsh

# Test runner script for sck-core-ai
# Follows SCK testing patterns

param(
    [string]$Target = "tests/",
    [switch]$NoCoverage = $false,
    [switch]$Verbose = $false,
    [string]$Pattern = ""
)

$ErrorActionPreference = "Stop"

# Set UV configuration to avoid hardlink warnings
$env:UV_LINK_MODE = "copy"

Write-Host "Running tests for sck-core-ai..." -ForegroundColor Green

# Create .env file if it doesn't exist (for testing)
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file for testing..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env" -ErrorAction SilentlyContinue
    
    # Add test-specific environment variables
    @"
# Auto-generated test environment
LOCAL_MODE=True
CLIENT=test-client
LOG_DIR=./logs
CONSOLE=interactive
LOG_LEVEL=DEBUG
LANGFLOW_HOST=localhost
LANGFLOW_PORT=7860
LANGFLOW_FLOW_ID=yaml-cf-ai-agent-v1
"@ | Out-File -FilePath ".env" -Append -Encoding UTF8
}

# Check if uv is available
$uvAvailable = $false
try {
    $uvVersion = uv --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        $uvAvailable = $true
        Write-Host "Using uv: $uvVersion" -ForegroundColor Cyan
    }
} catch {
    Write-Host "uv not found, using direct pytest" -ForegroundColor Yellow
}

# Build pytest command
$pytestArgs = @()

if (-not $NoCoverage) {
    $pytestArgs += "--cov=core_ai"
    $pytestArgs += "--cov-report=html"
    $pytestArgs += "--cov-report=term-missing"
}

if ($Verbose) {
    $pytestArgs += "-v"
}

if ($Pattern) {
    $pytestArgs += "-k", $Pattern
}

$pytestArgs += $Target

$runPrefix = if ($uvAvailable) { "uv run" } else { "" }
$command = "$runPrefix pytest " + ($pytestArgs -join " ")

try {
    Write-Host "Running: $command" -ForegroundColor Blue
    Invoke-Expression $command
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "All tests passed!" -ForegroundColor Green
        
        if ((-not $NoCoverage) -and (Test-Path "htmlcov/index.html")) {
            Write-Host "`nCoverage report generated: htmlcov/index.html" -ForegroundColor Cyan
        }
    } else {
        Write-Host "Some tests failed!" -ForegroundColor Red
        exit 1
    }
    
} catch {
    Write-Host "Test execution failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}