# Mistral 官方预训练 + LoRA 微调 + base vs LoRA 评估 + 出图
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUTF8 = "1"

Write-Host @"
=== Mistral LoRA 流程 ===
1) 若权重已在本地，请在 configs/model_config.yaml 设置:
     lora.local_model_path: '你的Mistral-7B-Instruct目录'
   或环境变量: `$env:MISTRAL_LOCAL_PATH = '...'
2) 若从 Hub 首次下载且模型需许可，请设置: `$env:HF_TOKEN = '...'
3) 建议 GPU；CPU 请把 lora.max_train_samples 调小 (如 64)
"@

$py = "D:\miniforge3\envs\wave_llm\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$weights = Join-Path (Get-Location) "models\mistral\Mistral-7B-Instruct-v0.3\model-00003-of-00003.safetensors"
if (-not (Test-Path $weights)) {
    Write-Host "Mistral weights not found — downloading to models/mistral/ ..."
    & $py scripts\00_download_mistral_weights.py
    if ($LASTEXITCODE -ne 0) { throw "Mistral download failed. Set HF_TOKEN if the model is gated." }
}

& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — re-run setup_conda.ps1'"
& $py scripts\06_export_llm_jsonl.py
& $py scripts\06c_build_holdout_test_jsonl.py
& $py scripts\06b_split_llm_jsonl.py
& $py scripts\07_train_mistral_lora.py
& $py scripts\08_eval_base_vs_lora.py
& $py scripts\10_make_figures.py

$html = Join-Path (Get-Location) "data\processed\figures\mistral_index.html"
if (Test-Path $html) { Start-Process $html }
