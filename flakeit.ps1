#!/usr/bin/env pwsh

# Code quality script for sck-core-ai
# Runs formatting and linting following SCK patterns

param(
    [switch]$Fix = $false,
    [switch]$Check = $false,
    [string]$Target = "core_ai/"
)

$ErrorActionPreference = "Stop"

Write-Host "Running code quality checks for sck-core-ai..." -ForegroundColor Green

# Check if uv is available
$uvAvailable = $false
try {
    $uvVersion = uv --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        $uvAvailable = $true
        Write-Host "Using uv: $uvVersion" -ForegroundColor Cyan
    }
} catch {
    Write-Host "uv not found, using direct commands" -ForegroundColor Yellow
}

$runPrefix = if ($uvAvailable) { "uv run" } else { "" }

try {
    if ($Fix) {
        Write-Host "Formatting code with black..." -ForegroundColor Blue
        Invoke-Expression "$runPrefix black $Target"
        if ($LASTEXITCODE -ne 0) { throw "Black formatting failed" }
        
        Write-Host "Sorting imports with isort..." -ForegroundColor Blue
        Invoke-Expression "$runPrefix isort $Target"
        if ($LASTEXITCODE -ne 0) { throw "Import sorting failed" }
        
        Write-Host "Code formatting completed!" -ForegroundColor Green
    } else {
        # Check mode (default)
        $hasErrors = $false
        
        Write-Host "Checking code format with black..." -ForegroundColor Blue
        Invoke-Expression "$runPrefix black --check $Target"
        if ($LASTEXITCODE -ne 0) { 
            $hasErrors = $true
            Write-Host "Black formatting issues found" -ForegroundColor Yellow
        }
        
        Write-Host "Checking import order with isort..." -ForegroundColor Blue
        Invoke-Expression "$runPrefix isort --check-only $Target"
        if ($LASTEXITCODE -ne 0) { 
            $hasErrors = $true
            Write-Host "Import order issues found" -ForegroundColor Yellow
        }
        
        Write-Host "Running flake8 linting..." -ForegroundColor Blue
        Invoke-Expression "$runPrefix flake8 $Target"
        if ($LASTEXITCODE -ne 0) { 
            $hasErrors = $true
            Write-Host "Flake8 issues found" -ForegroundColor Yellow
        }
        
        Write-Host "Running mypy type checking..." -ForegroundColor Blue
        Invoke-Expression "$runPrefix mypy $Target"
        if ($LASTEXITCODE -ne 0) { 
            $hasErrors = $true
            Write-Host "MyPy type issues found" -ForegroundColor Yellow
        }
        
        if ($hasErrors) {
            Write-Host "`nCode quality issues found. Run with -Fix to auto-fix formatting issues." -ForegroundColor Red
            exit 1
        } else {
            Write-Host "All code quality checks passed!" -ForegroundColor Green
        }
    }
    
} catch {
    Write-Host "Code quality check failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}