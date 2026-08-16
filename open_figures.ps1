# Open the generated figure gallery in the default browser (Windows).
$figDir = Join-Path $PSScriptRoot "..\data\processed\figures"
$html = Join-Path $figDir "index.html"
if (-not (Test-Path $html)) {
    Write-Host "Run scripts/10_make_figures.py first. Missing: $html"
    exit 1
}
Start-Process $html
