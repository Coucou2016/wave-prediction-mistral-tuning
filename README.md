# Wave buoy: baselines, regimes, and LLM exports

Research pipeline for global buoy wave observations: real downloads (NDBC, optional Copernicus Marine / CDIP), QC and resampling, physics features, baseline–residual framing, windowed samples, and JSONL for instruction tuning.

## Quick start (Windows, Conda)

```powershell
cd d:\Projects\wave-prediction-mistral-tuning
.\setup_conda.ps1
conda activate wave_llm
.\run_pipeline.ps1
```

## Copernicus Marine (optional)

Set environment variables (see [toolbox docs](https://toolbox-docs.marine.copernicus.eu/en/stable/usage/environment-variables.html)):

- `COPERNICUSMARINE_SERVICE_USERNAME`
- `COPERNICUSMARINE_SERVICE_PASSWORD`

Then run `python scripts\01_download_cmems_wave.py` or the full pipeline.

## Figures (after step 10)

Outputs live under `data/processed/figures/` (PNGs + `index.html` gallery).

```powershell
.\open_figures.ps1
# or: start data\processed\figures\index.html
```

### What the forecast panels show

- **Black**: NDBC observed $H_s$ (truth) on the hold-out tail from `05_train_numeric_baselines.py`.
- **Grey / blue**: real model outputs — persistence and **LightGBM** trained on the same real panel (not synthetic waves).
- **Purple (optional)**: **Amazon Chronos-T5** downloaded from Hugging Face (`amazon/chronos-t5-*`), zero-shot on the real past context (`scripts/05b_chronos_forecast.py`). This is a **pretrained time-series foundation model** line for continuous $H_s$; it is the practical way to keep curves strictly from model forward passes without inventing numbers.
- **Mistral curve LoRA** (`scripts/06d_export_mistral_curve_jsonl.py`, `07b_train_mistral_curve_lora.py`, `08b_eval_mistral_curve.py`) fine-tunes Instruct weights to emit JSON `hs_forecast_m` hourly curves; panels compare Truth / Persistence / LightGBM† / Chronos† / Mistral Base+LoRA (`forecast_panel_mistral_lora_*.png`). Classification LoRA remains a separate task.

### Mistral 官方预训练 + LoRA 微调（推荐主线）

与 LightGBM / Chronos 一样：**先微调，再评估**。

1. 把已下载的官方权重目录写到 `configs/model_config.yaml`：

```yaml
lora:
  local_model_path: "D:/你的路径/Mistral-7B-Instruct-v0.3"
  local_files_only: true
  max_train_samples: 256   # GPU 可加大；CPU 先用 64
```

或环境变量：`MISTRAL_LOCAL_PATH`、`HF_LOCAL_ONLY=1`  
**本地已有权重时一般不需要 HF_TOKEN**；只有首次从 Hub 拉取且模型 gated 时才需要。

2. 一键流程：

```powershell
.\run_mistral_lora.ps1
```

步骤：`06` 导出 JSONL → `06b` 划分 train/val/test → `07` LoRA 训练 → `08` base vs LoRA 对比 → `10` 出图。

产物：
- `data/processed/mistral/lora_run/adapter/` — LoRA 权重
- `data/processed/mistral/metrics_base.json` / `metrics_lora.json`
- `figures/mistral_base_regime_confusion.png` / `mistral_lora_regime_confusion.png`

**需要 NVIDIA GPU + 约 12GB+ 显存**（`use_4bit: true`）。当前环境若无 CUDA，请在本机 GPU 上运行上述脚本。

### Mistral curve LoRA（Hs JSON 曲线预报）

与分类 LoRA 分开。导出严格 issue-time 的 `hs_forecast_m` JSONL，再 LoRA 微调，再与 Persistence / LightGBM / Chronos 对齐评估。

```powershell
.\run_mistral_curve.ps1
# 或分步：
python scripts\06d_export_mistral_curve_jsonl.py --horizon 24 --history-hours 168
python scripts\07b_train_mistral_curve_lora.py
python scripts\08b_eval_mistral_curve.py
python scripts\10_make_figures.py
```

配置：`configs/model_config.yaml` → `mistral_curve.*`
- 当前默认输出目录：`data/processed/mistral/curve_lora_run_v2`（pilot 仍保留在 `curve_lora_run`）
- 输入：完整 `history_hs_m` + 压缩 `features`（Hs 统计；Tp/wind 的 mean/trend/last_6），**不**塞入 Tp/wind 168 点全序列
- `max_seq_length: 2048` + `gradient_checkpointing`（避免旧 3120-token 灾难配置）
- `06d` 命名：导出曲线 JSONL（与分类用的 `06`/`06b`/`06c` 并列）

产物：adapter、`curve_metrics_*.json`、`forecast_panel_mistral_lora_*.png`、`model_rmse_comparison_by_lead.png`、`curve_method_rmse_summary.png`

### Mistral / Ministral 结果图（零样本 / Ollama）

1. 先完成 `06` 导出 `data/processed/llm/train_mistral.jsonl`。
2. 若模型在 Hugging Face 上需许可，请先设置 **`HF_TOKEN`**（或 `HUGGING_FACE_HUB_TOKEN`）。
3. 运行（会下载权重并推理，CPU 上可能较慢，可先改小 `mistral_infer.max_samples`）：

```powershell
.\run_mistral_infer.ps1
# 或手动：
conda activate wave_llm
python scripts\07c_mistral_zero_shot_infer.py --max-samples 8
python scripts\10_make_figures.py
```

生成 `data/processed/mistral/zero_shot_results.jsonl`、`metrics.json`，并在 `figures/` 下增加 **`mistral_regime_confusion.png`**、**`mistral_predictability_accuracy.png`**（需先有 `metrics.json`）。

可用环境变量 **`MISTRAL_MODEL_ID`** 覆盖 `configs/model_config.yaml` 里的 `mistral_infer.model_id`。

## Layout

- `configs/` — data sources, station panel, model settings
- `src/wave_llm/` — library code
- `scripts/` — numbered pipeline steps
- `data/raw|interim|processed/` — downloaded and derived data (gitignored)
