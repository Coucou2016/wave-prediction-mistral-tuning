# Local LLM zero-shot (Ollama by default) + refresh figures.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

$env:MISTRAL_BACKEND = "ollama"
if (-not $env:OLLAMA_MODEL) {
    $env:OLLAMA_MODEL = "deepseek-r1:14b"
}

Write-Host "Backend=ollama model=$env:OLLAMA_MODEL (set OLLAMA_MODEL or pull mistral first)"

if (Get-Command conda -ErrorAction SilentlyContinue) {
    conda run -n wave_llm python scripts\07c_mistral_zero_shot_infer.py @args
    conda run -n wave_llm python scripts\10_make_figures.py
} else {
    python scripts\07c_mistral_zero_shot_infer.py @args
    python scripts\10_make_figures.py
}

$html = Join-Path (Get-Location) "data\processed\figures\index.html"
if (Test-Path $html) { Start-Process $html }
