# Mistral curve-forecast JSONL + LoRA + eval + figures
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUTF8 = "1"

Write-Host @"
=== Mistral curve forecast (hs_forecast_m JSON) ===
Requires: panel_hourly.parquet, local Mistral-7B-Instruct weights
Config: configs/model_config.yaml -> mistral_curve.* (default adapter dir: curve_lora_run_v2)
"@

$py = "D:\miniforge3\envs\wave_llm\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$weights = Join-Path (Get-Location) "models\mistral\Mistral-7B-Instruct-v0.3\model-00003-of-00003.safetensors"
if (-not (Test-Path $weights)) {
    Write-Host "Mistral weights not found — downloading ..."
    & $py scripts\00_download_mistral_weights.py
    if ($LASTEXITCODE -ne 0) { throw "Mistral download failed." }
}

& $py -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
if (-not (Test-Path "data\processed\predictions\test_predictions_lgbm.parquet")) {
    Write-Host "Numeric baselines missing — running 05_train_numeric_baselines.py ..."
    & $py scripts\05_train_numeric_baselines.py
}
& $py scripts\06d_export_mistral_curve_jsonl.py --horizon 24 --history-hours 168
& $py scripts\07b_train_mistral_curve_lora.py
& $py scripts\08b_eval_mistral_curve.py
& $py scripts\10_make_figures.py

$html = Join-Path (Get-Location) "data\processed\figures\mistral_index.html"
if (Test-Path $html) { Start-Process $html }
