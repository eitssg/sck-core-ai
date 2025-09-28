#!/usr/bin/env pwsh

<#!
.SYNOPSIS
  Coverage helper for sck-core-ai.

.DESCRIPTION
  Runs pytest with coverage instrumentation honoring the module's .coveragerc.
  Provides switches for fail-under threshold, XML report (for CI), and opening the
  HTML report after generation.

.PARAMETER FailUnder
  Optional integer threshold (percentage) that will cause the run to fail if overall
  coverage is below this value. (Pass 0 to disable.)

.PARAMETER Xml
  Emit coverage XML (coverage.xml) in addition to terminal + HTML reports.

.PARAMETER OpenHtml
  Open the generated HTML coverage report (htmlcov/index.html) on success.

.PARAMETER Pattern
  Optional pytest -k expression to filter tests.

.PARAMETER Target
  Target test path (default: tests/).

.EXAMPLE
  # Run full suite with coverage + HTML only
  ./coverage.ps1

.EXAMPLE
  # Enforce minimum coverage of 28%
  ./coverage.ps1 -FailUnder 28

.EXAMPLE
  # Generate XML for CI tooling & open HTML report
  ./coverage.ps1 -Xml -OpenHtml

.NOTES
  Uses uv if available; falls back to direct pytest.
  Creates a .env (mirroring pytest.ps1) if missing so LOCAL_MODE tests work.
!#>

param(
    [int]$FailUnder = 0,
    [switch]$Xml = $false,
    [switch]$OpenHtml = $false,
    [string]$Pattern = "",
    [string]$Target = "tests/"
)

$ErrorActionPreference = "Stop"

Write-Host "Running coverage for sck-core-ai..." -ForegroundColor Green

# Ensure test .env exists (reuse logic from pytest.ps1 but simpler)
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file for testing (auto)" -ForegroundColor Yellow
    @(
        'LOCAL_MODE=True'
        'CLIENT=test-client'
        'LOG_DIR=./logs'
        'CONSOLE=interactive'
        'LOG_LEVEL=DEBUG'
        'LANGFLOW_HOST=localhost'
        'LANGFLOW_PORT=7860'
        'LANGFLOW_FLOW_ID=yaml-cf-ai-agent-v1'
    ) | Set-Content .env -Encoding UTF8
}

# Prefer uv if present
$uvAvailable = $false
try {
    $uvVersion = uv --version 2>$null
    if ($LASTEXITCODE -eq 0) { $uvAvailable = $true; Write-Host "Using uv: $uvVersion" -ForegroundColor Cyan }
}
catch { Write-Host "uv not found; using system pytest" -ForegroundColor Yellow }

$pytestArgs = @(
    "--cov=core_ai",
    "--cov-report=term-missing",
    "--cov-report=html"
)

if ($Xml) { $pytestArgs += "--cov-report=xml" }
if ($FailUnder -gt 0) { $pytestArgs += "--cov-fail-under=$FailUnder" }
if ($Pattern) { $pytestArgs += "-k"; $pytestArgs += $Pattern }
$pytestArgs += $Target

$runPrefix = if ($uvAvailable) { "uv run" } else { "" }
$command = "${runPrefix} pytest " + ($pytestArgs -join ' ')

Write-Host "Command: $command" -ForegroundColor Blue

try {
    Invoke-Expression $command
    if ($LASTEXITCODE -ne 0) { throw "pytest exited with code $LASTEXITCODE" }

    Write-Host "\nCoverage run successful." -ForegroundColor Green

    if (Test-Path "htmlcov/index.html") {
        Write-Host "HTML report: htmlcov/index.html" -ForegroundColor Cyan
        if ($OpenHtml) {
            try { Start-Process "$(Resolve-Path htmlcov/index.html)" } catch { Write-Host "Could not open HTML report: $($_.Exception.Message)" -ForegroundColor Yellow }
        }
    }
    else {
        Write-Host "HTML coverage report not found (expected htmlcov/index.html)" -ForegroundColor Yellow
    }

    if ($Xml -and (Test-Path "coverage.xml")) {
        Write-Host "XML report: coverage.xml" -ForegroundColor Cyan
    }

}
catch {
    Write-Host "Coverage run failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
