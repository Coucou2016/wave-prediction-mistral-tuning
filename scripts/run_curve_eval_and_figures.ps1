# Run after curve LoRA training finishes (adapter must exist).
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) ".."))
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUTF8 = "1"
$py = "D:\miniforge3\envs\wave_llm\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$adapter = "data\processed\mistral\curve_lora_run\adapter"
if (-not (Test-Path $adapter)) {
    Write-Error "Missing $adapter — wait for 07b_train_mistral_curve_lora.py to finish."
}
& $py scripts\08b_eval_mistral_curve.py
& $py scripts\10_make_figures.py
Write-Host "Done. See data\processed\figures\forecast_panel_mistral_lora_*.png"
