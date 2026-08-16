$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not $env:CONDA_DEFAULT_ENV) {
    Write-Host "Run: conda activate wave_llm   (or execute scripts via conda run)"
}

$py = "conda"
$argsRun = @("run", "-n", "wave_llm", "python")
if (Get-Command conda -ErrorAction SilentlyContinue) {
    $runner = { param($script) & conda run -n wave_llm python $script }
} else {
    Write-Warning "conda not found; using python on PATH"
    $runner = { param($script) & python $script }
}

$scripts = @(
    "scripts\01_download_cmems_wave.py",
    "scripts\02_download_ndbc_wave.py",
    "scripts\02b_download_cdip_wave.py",
    "scripts\03_build_station_panel.py",
    "scripts\04_build_windows.py",
    "scripts\05_train_numeric_baselines.py",
    "scripts\05b_chronos_forecast.py",
    "scripts\06_export_llm_jsonl.py",
    "scripts\07_train_mistral_lora.py",
    "scripts\08_eval_base_vs_lora.py",
    "scripts\09_train_ast_spectrogram.py",
    "scripts\10_make_figures.py"
)

foreach ($s in $scripts) {
    Write-Host "==== $s ===="
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        conda run -n wave_llm python $s
    } else {
        python $s
    }
}

if ($env:WAVE_LLM_INFER -eq "1") {
    Write-Host "==== scripts\07c_mistral_zero_shot_infer.py (WAVE_LLM_INFER=1) ===="
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        conda run -n wave_llm python scripts\07c_mistral_zero_shot_infer.py
    } else {
        python scripts\07c_mistral_zero_shot_infer.py
    }
} else {
    Write-Host "Skip Mistral zero-shot (set WAVE_LLM_INFER=1 to run scripts\07c_mistral_zero_shot_infer.py)"
}

Write-Host "Pipeline finished."
