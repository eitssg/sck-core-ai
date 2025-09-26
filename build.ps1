#!/usr/bin/env pwsh

# Build script for sck-core-ai (PowerShell)
# Follows SCK framework build patterns

param(
    [switch]$Clean = $false,
    [switch]$Test = $false,
    [switch]$Verbose = $false
)

$ErrorActionPreference = "Stop"

Write-Host "Building sck-core-ai..." -ForegroundColor Green

# Clean previous build if requested
if ($Clean) {
    Write-Host "Cleaning previous build artifacts..." -ForegroundColor Yellow
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    if (Test-Path "*.egg-info") { Remove-Item -Recurse -Force "*.egg-info" }
    if (Test-Path "htmlcov") { Remove-Item -Recurse -Force "htmlcov" }
    if (Test-Path ".pytest_cache") { Remove-Item -Recurse -Force ".pytest_cache" }
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
    Write-Host "uv not found, falling back to pip/poetry" -ForegroundColor Yellow
}

# Install dependencies and build
try {
    if ($uvAvailable) {
        Write-Host "Installing dependencies with uv..." -ForegroundColor Blue
        uv sync --dev
        if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }
        
        Write-Host "Building package with uv..." -ForegroundColor Blue  
        uv build
        if ($LASTEXITCODE -ne 0) { throw "uv build failed" }
    } else {
        # Fallback to poetry if available
        $poetryAvailable = $false
        try {
            $poetryVersion = poetry --version 2>$null
            if ($LASTEXITCODE -eq 0) {
                $poetryAvailable = $true
                Write-Host "Using poetry: $poetryVersion" -ForegroundColor Cyan
            }
        } catch {
            Write-Host "Poetry not found either" -ForegroundColor Red
        }
        
        if ($poetryAvailable) {
            Write-Host "Installing dependencies with poetry..." -ForegroundColor Blue
            poetry install --with dev
            if ($LASTEXITCODE -ne 0) { throw "poetry install failed" }
            
            Write-Host "Building package with poetry..." -ForegroundColor Blue
            poetry build
            if ($LASTEXITCODE -ne 0) { throw "poetry build failed" }
        } else {
            throw "Neither uv nor poetry available for building"
        }
    }
    
    # Run tests if requested
    if ($Test) {
        Write-Host "Running tests..." -ForegroundColor Blue
        
        if ($uvAvailable) {
            uv run pytest
            if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
        } elseif ($poetryAvailable) {
            poetry run pytest
            if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
        } else {
            pytest
            if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
        }
    }
    
    Write-Host "Build completed successfully!" -ForegroundColor Green
    
    # Show build artifacts
    if (Test-Path "dist") {
        Write-Host "`nBuild artifacts:" -ForegroundColor Cyan
        Get-ChildItem "dist" | ForEach-Object {
            Write-Host "  $($_.Name)" -ForegroundColor Gray
        }
    }
    
} catch {
    Write-Host "Build failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}