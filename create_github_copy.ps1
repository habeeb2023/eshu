# create_github_copy.ps1 — Creates a clean copy of eshu for GitHub upload
# Run from project root: .\create_github_copy.ps1
# Output: ..\eshu_github\ (sibling folder, no data, no .env)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$dest = Join-Path (Split-Path $root -Parent) "eshu_github"

Write-Host "Creating clean GitHub copy at: $dest" -ForegroundColor Cyan

# Remove existing if present
if (Test-Path $dest) {
    Remove-Item $dest -Recurse -Force
}
New-Item -ItemType Directory -Path $dest -Force | Out-Null

# Copy code & config (exclude data, .env, caches)
$include = @(
    "app",
    "ui",
    "logo",
    "tests",
    "README.md",
    "requirements.txt",
    "conftest.py",
    "pytest.ini",
    "check_db.py",
    "start_api.bat",
    "start_ui.bat",
    ".gitignore",
    ".env.example"
)

foreach ($item in $include) {
    $srcPath = Join-Path $root $item
    if (Test-Path $srcPath) {
        $destPath = Join-Path $dest $item
        if (Test-Path $srcPath -PathType Container) {
            Copy-Item -Path $srcPath -Destination $destPath -Recurse -Force
        } else {
            Copy-Item -Path $srcPath -Destination $destPath -Force
        }
        Write-Host "  + $item" -ForegroundColor Green
    }
}

# Remove __pycache__ from copied folders
Get-ChildItem -Path $dest -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
Get-ChildItem -Path $dest -Recurse -Directory -Filter ".pytest_cache" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

# Create empty data placeholder so structure is clear
$dataDir = Join-Path $dest "data"
New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
"# Created at runtime. Vault database and metadata go here." | Out-File (Join-Path $dataDir ".gitkeep") -Encoding utf8

Write-Host ""
Write-Host "Done. Clean copy ready at: $dest" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. cd $dest"
Write-Host "  2. git init"
Write-Host "  3. git add ."
Write-Host "  4. git commit -m 'Initial commit: eshu'"
Write-Host "  5. Create repo on GitHub, then: git remote add origin <url> && git push -u origin main"
