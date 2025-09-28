# SCK Langflow Setup Script
# Creates necessary directories and starts Langflow with persistent storage

Write-Host "🚀 Setting up SCK Langflow environment..." -ForegroundColor Green

# Create data directories
$dataDir = "data"
$subdirs = @("langflow-data", "langflow-config", "langflow-cache", "langflow-components")

if (-not (Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir -Force
    Write-Host "✅ Created data directory" -ForegroundColor Green
}

foreach ($subdir in $subdirs) {
    $path = Join-Path $dataDir $subdir
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force
        Write-Host "✅ Created $path" -ForegroundColor Green
    } else {
        Write-Host "📁 $path already exists" -ForegroundColor Yellow
    }
}

# Create langflow directory if it doesn't exist
if (-not (Test-Path "langflow")) {
    New-Item -ItemType Directory -Path "langflow" -Force
    Write-Host "✅ Created langflow directory" -ForegroundColor Green
}

# Set permissions (Windows specific)
try {
    $acl = Get-Acl $dataDir
    $accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule("Everyone", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow")
    $acl.SetAccessRule($accessRule)
    Set-Acl -Path $dataDir -AclObject $acl
    Write-Host "✅ Set directory permissions" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Could not set permissions: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎯 Setup complete! Next steps:" -ForegroundColor Cyan
Write-Host "1. Start Langflow: docker-compose up -d" -ForegroundColor White
Write-Host "2. Access at: http://localhost:7860" -ForegroundColor White  
Write-Host "3. Login with: admin / admin123" -ForegroundColor White
Write-Host "4. Import flow: langflow/sck-documentation-chat.json" -ForegroundColor White
Write-Host ""