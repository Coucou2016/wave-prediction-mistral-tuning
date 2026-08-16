# Figure & table plan (paper)

**Note:** A parallel agent is restyling plots (SciencePlots + Times New Roman). This file only plans **content and filenames**. Do **not** edit `scripts/10_make_figures.py` here.

**Gallery index:** `data/processed/figures/index.html`  
**ChatGPT thread (context):** https://chatgpt.com/c/6a812828-a690-83ea-a218-25721d148a25

Status legend: **EXISTING** = file already under `data/processed/figures/` · **TO GENERATE** = needed for paper, not yet a dedicated asset · **OPTIONAL** = SI only.

---

## Main-text figures

| ID | Caption intent | Source file(s) | Status | Maps to section |
|----|----------------|----------------|--------|-----------------|
| **Fig. 1** | Study stations on coastlines (NDBC panel) | `station_map.png` | EXISTING | Data |
| **Fig. 2** | End-to-end methods: numeric baselines + Chronos + Mistral class/curve LoRA | `mistral_methods_summary.png` | EXISTING | Methods |
| **Fig. 3** | Numeric skill: RMSE / skill vs persistence by lead | `baseline_rmse_skill.png` (+ optional `model_rmse_comparison_by_lead.png`) | EXISTING | Results §numeric |
| **Fig. 4** | Curve-method RMSE summary: Persist / LGBM / Chronos / Mistral Base / LoRA | `curve_method_rmse_summary.png` | EXISTING | Results §curve |
| **Fig. 5** | Multi-model forecast panels (pick 2 stations × lead 24 h) | `forecast_panel_*_lead24h.png` and/or `forecast_panel_mistral_lora_*.png` | EXISTING | Results §cases |
| **Fig. 6** | Regime confusion: Base vs LoRA | `mistral_base_regime_confusion.png`, `mistral_lora_regime_confusion.png` | EXISTING | Results §classification |
| **Fig. 7** | Predictability accuracy Base vs LoRA (+ vs LGBM error if space) | `mistral_predictability_accuracy.png`, `mistral_lora_predictability_accuracy.png`, `mistral_predictability_vs_lgbm_error.png` | EXISTING | Results §classification |

**Recommended main-text budget:** Fig. 1–6 core; Fig. 7 → main or SI depending on page limit.

---

## Supplementary figures (SI)

| ID | Caption intent | Source file(s) | Status |
|----|----------------|----------------|--------|
| Fig. S1 | Raw / multivar series by station | `series_*.png`, `multivar_*.png` | EXISTING |
| Fig. S2 | Hs distribution by station | `hs_boxplot_by_station.png` | EXISTING |
| Fig. S3 | Regime counts / by station | `regime_counts.png`, `regime_by_station.png` | EXISTING |
| Fig. S4 | Lead-6h forecast panels | `forecast_panel_*_lead6h.png` | EXISTING |
| Fig. S5 | Mistral overlay TS + regime timeline | `mistral_forecast_ts_*`, `mistral_regime_timeline_*` | EXISTING |
| Fig. S6 | Scatter lead-24h / eval snapshots | `mistral_forecast_scatter_lead24h.png`, `mistral_eval_snapshots.png` | EXISTING |
| Fig. S7 | Extra Mistral LoRA panels (42040, 44013) | `forecast_panel_mistral_lora_42040.png`, `..._44013.png` | EXISTING |
| Fig. S8 | Zero-shot / other confusion if used | `mistral_regime_confusion.png` | EXISTING (task-dependent) |

---

## Tables

| ID | Content | Data source | Status |
|----|---------|-------------|--------|
| **Table 1** | Dataset: stations, years, Δt, window length, n_train/val/test | configs + build logs / JSONL counts | TO GENERATE (compile from configs + scripts) |
| **Table 2** | Numeric RMSE/MAE/skill by lead | `data/processed/metrics/numeric_baselines.json` | TO GENERATE (from JSON) |
| **Table 3** | Curve metrics Base vs LoRA + Persist/LGBM/Chronos means | `curve_metrics_*.json`, `curve_compare_base_lora.json` | TO GENERATE |
| **Table 4** | Classification accuracy Base vs LoRA | `metrics_base.json`, `metrics_lora.json`, `compare_base_lora.json` | TO GENERATE |
| **Table 5** | Hyperparameters (LoRA r, α, lr, max_steps, seq len) | `configs/model_config.yaml` | TO GENERATE |
| Table S1 | Per-sample curve RMSE list | `per_sample_rmse` in curve metrics JSON | OPTIONAL |
| Table S2 | RMSE by forecast step h=1…24 | `rmse_by_forecast_step_h` | OPTIONAL |

---

## Filename ↔ paper ID quick map (existing PNGs)

```
station_map.png                          → Fig. 1
mistral_methods_summary.png              → Fig. 2
baseline_rmse_skill.png                  → Fig. 3
model_rmse_comparison_by_lead.png        → Fig. 3 alt / SI
curve_method_rmse_summary.png            → Fig. 4
forecast_panel_41010_lead24h.png         → Fig. 5a
forecast_panel_46047_lead24h.png         → Fig. 5b
forecast_panel_mistral_lora_41010.png    → Fig. 5 alt
mistral_base_regime_confusion.png        → Fig. 6a
mistral_lora_regime_confusion.png        → Fig. 6b
mistral_predictability_accuracy.png      → Fig. 7a
mistral_lora_predictability_accuracy.png → Fig. 7b
mistral_predictability_vs_lgbm_error.png → Fig. 7c / SI
```

---

## TO GENERATE (content only; plotting deferred to figure agent)

1. **Table 1–5** Markdown/LaTeX from metrics JSON (no new PNG required).  
2. Optional **Fig. schema** redraw if `mistral_methods_summary.png` is too busy after SciencePlots restyle—same filename preferred to avoid outline churn.  
3. Optional **qualitative reason box** figure (screenshot of JSON output)—**TO GENERATE** only if ethics/privacy OK (no secrets).

---

## Metric freeze for captions (do not invent)

- Curve: Base mean RMSE **1.271**, LoRA **0.699**, Persist **0.688**, LGBM **0.698**, Chronos **0.951** (n=24).  
- Class: regime **0.042→0.417**, predictability **0.375→0.250** (n=24).  
- Numeric LGBM skill positive from lead **12–72 h** (`numeric_baselines.json`).
