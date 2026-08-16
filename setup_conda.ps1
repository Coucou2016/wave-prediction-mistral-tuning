param(
    [string]$EnvName = "wave_llm"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$env:PIP_INDEX_URL = "https://pypi.org/simple"

Write-Host "Creating/updating conda env $EnvName from environment.yml ..."
if (conda env list | Select-String -Pattern "^$EnvName\s") {
    conda env update -f environment.yml -n $EnvName --prune
} else {
    conda env create -f environment.yml -n $EnvName
}
conda run -n $EnvName python -m pip install --upgrade pip
Write-Host "Installing PyTorch + CUDA 12.4 (conda, RTX 4090) ..."
conda install -n $EnvName -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
Write-Host "Installing requirements.txt (PyPI, keep conda torch) ..."
conda run -n $EnvName python -m pip install -r requirements.txt --no-deps
conda run -n $EnvName python -m pip install peft datasets sentencepiece huggingface_hub safetensors regex requests tqdm pyyaml packaging filelock
Write-Host "Editable install wave_llm ..."
conda run -n $EnvName python -m pip install -e .

Write-Host "Done. Activate with: conda activate $EnvName"
