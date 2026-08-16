# Paper outline — Mistral LoRA for buoy Hs forecasting

**nature-writing routing (stated for correction):**  
`task=manuscript` · `paper_type=methods` · `journal=generic` (target: *Ocean Engineering*; Nat-family claim discipline only) · sections: full scaffold · language: outline in EN with ZH notes where useful.

**One-sentence argument**  
In NDBC buoy significant-wave-height forecasting, we show that instruction-tuned Mistral-7B-Instruct-v0.3 with LoRA can emit structured JSON `hs_forecast_m` curves together with sea-state regime / predictability labels and textual reasons, supported by base→LoRA improvements and fair persistence / LightGBM / Chronos baselines, with the boundary that pilot RMSE does **not** beat specialized numeric forecasters.

**Primary reader question to lead with:** *novelty of capability* (structured + explainable multi-task LLM interface for wave ops), then *trust* (fair baselines + honest gaps).

**Terminology ledger (canonical forms)**  
Hs / significant wave height · NDBC · persistence · LightGBM · Chronos-T5 · Mistral-7B-Instruct-v0.3 · LoRA · `hs_forecast_m` · regime · predictability · JSON validity · RMSE / MAE · skill vs persistence.

---

## 1. Title candidates

1. **Structured wave forecasting with instruction-tuned LLMs: JSON Hs curves, regime labels, and predictability rationales from NDBC buoys**  
2. **From buoy windows to parseable forecasts: LoRA fine-tuning of Mistral for significant wave height sequences and sea-state explanations**  
3. **Do not claim RMSE supremacy: evaluating Mistral-LoRA as a structured, explainable companion to LightGBM and Chronos for Hs prediction**

Preferred for OE: **(1)**; keep **(3)** as internal honesty check on the Abstract.

---

## 2. Target venue

| Priority | Journal | Why |
|----------|---------|-----|
| 1 | *Ocean Engineering* | Hs ML tradition; engineering motivation; IMRaD + tables |
| 2 | *Applied Ocean Research* | Applied coastal/ocean ML alternate |
| 3 (later) | *Artificial Intelligence for the Earth Systems* | If ML-methods audience preferred (cf. Chaichitehrani 2024) |
| Not yet | *Nature Communications* | Needs larger n, clearer geophysical impact |

---

## 3. Contribution statements (draft; methods-paper style)

We make three **bounded** contributions:

1. **Pipeline + schema.** An end-to-end NDBC→QC→window→JSONL→LoRA recipe that trains Mistral to output machine-parseable `hs_forecast_m` hourly sequences (JSON validity = 1.0 on the pilot curve eval).  
2. **Multi-task language interface.** The same Instruct backbone family supports (a) curve generation and (b) regime + predictability classification with free-text reasons—linking forecast *numbers* to *decision labels*.  
3. **Fair comparative evaluation.** We report Mistral Base vs LoRA against persistence, LightGBM, and Chronos on shared windows, and **explicitly show** where LoRA helps (vs Base) and where it still trails numeric baselines.

---

## 4. Section-by-section outline

### Abstract (~200–250 words for OE)

| Job | Content | Evidence source |
|-----|---------|-----------------|
| Context | Operational need for Hs + situational labels | Engineering stake |
| Gap | Numeric models lack natural-language rationale; LLMs lack ocean schema | Related work |
| Approach | Mistral LoRA → JSON curves + regime/predictability | Methods |
| Result | Curve RMSE Base 1.510 → LoRA 0.977 (n=12); regime acc 0.042 → 0.417 (n=24); numeric LGBM/Chronos still competitive | Local metrics JSON |
| Boundary | Pilot scale; LoRA curve RMSE ≈ LGBM, > persistence | Honest limit |

### 1. Introduction

Paragraph map (one job each):

1. **Context** — buoy Hs for navigation / coastal ops.  
2. **Prior numeric ML** — LSTM / boosting / ensembles (Fan 2020; Domala 2022; Chaichitehrani 2024).  
3. **TS foundation models** — PatchTST, Chronos, TimeGPT.  
4. **LLM-for-TS** — LLMTime, Time-LLM; **counterpoint** Tan et al. 2024.  
5. **Gap** — missing *structured ocean JSON + explainable multi-task* evaluation against strong numeric baselines.  
6. **This work** — state one-sentence argument + 3 contributions.  
7. **Paper roadmap**.

**Figures:** none (or cite Fig. 1 pipeline forward).

### 2. Related work

Clusters (not paper dump):

- Hs prediction with ML / DL  
- Time-series Transformers & foundation models  
- LLMs for time series (incl. skepticism papers)  
- PEFT / LoRA for domain adaptation  

### 3. Data and problem setup

| Point | Detail | Local source |
|-------|--------|--------------|
| Stations / panel | NDBC IDs used in figures (41010, 46047, 46246, …) | `configs/`, figures |
| Resample | `1h`; history / leads from config | `model_config.yaml` |
| Targets | Curve horizon 24 h; classification predictability lead 24 h | config |
| Splits | seeds `split_seed: 42`; train/val/test fracs | config + scripts 06b/06d |
| QC | pipeline steps 02–04 | README / scripts |

**Fig. 1** station map · **Fig. 2** example Hs series / multivar · **Table 1** dataset summary.

### 4. Methods

#### 4.1 Numeric baselines
Persistence; LightGBM; Chronos-T5 zero-shot (`scripts/05`, `05b`).

#### 4.2 Classification LoRA
JSONL export (`06`); train (`07`); eval Base vs LoRA (`08`) — regime + predictability + reason.

#### 4.3 Curve LoRA
Export (`06d`); train (`07b`); eval (`08b`) — schema `{hs_forecast_m, uncertainty?, reason?}`.

#### 4.4 Evaluation protocol
RMSE/MAE by step; JSON validity; accuracy; **same-window** baseline means in `curve_metrics_*.json`.

**Fig. 3** methods schematic (`mistral_methods_summary.png`).

### 5. Experiments / Results

Evidence ladder:

1. **Numeric skill by lead** — LGBM vs persistence.  
   - Data: `numeric_baselines.json`  
   - Fig: `baseline_rmse_skill.png`, `model_rmse_comparison_by_lead.png`  
   - Table 2: RMSE @ 6/12/24/48/72 h  

2. **Curve generation Base vs LoRA**  
   - Data: `curve_metrics_*.json`, `curve_compare_base_lora.json`  
   - Key numbers: mean RMSE 1.510 → 0.977; JSON valid 1.0; baselines persist 0.884 / LGBM 0.977 / Chronos 0.965  
   - Fig: `curve_method_rmse_summary.png`, `forecast_panel_mistral_lora_*.png`  

3. **Classification Base vs LoRA**  
   - Data: `metrics_*.json`, `compare_base_lora.json`  
   - Regime 0.042 → 0.417; predictability 0.375 → 0.25 (**report regression honestly**)  
   - Fig: confusion matrices, predictability bars  

4. **Case studies** — station forecast panels (truth / persist / LGBM / Chronos / Mistral).  

5. **Interpretability samples** — qualitative reason text (not quantitative fidelity yet).

### 6. Discussion

| Theme | Message |
|-------|---------|
| What improved | LoRA teaches schema + ocean vocabulary; large Base→LoRA curve RMSE drop |
| What did not | Predictability accuracy dropped; LoRA mean RMSE still > persistence; ≈ LGBM on pilot |
| Positioning vs LLM-TS literature | Align with Tan et al.: do not sell LLM as RMSE engine |
| Operational value | Parseable JSON + human-readable reason for watchstander workflows |
| Threats to validity | Small n; class imbalance (mostly `storm_growth`); single Instruct model; compute |

### 7. Limitations (explicit)

- Curve eval **n=12**; classification **n=24**.  
- LoRA curve mean RMSE **0.977** vs persistence **0.884**, Chronos **0.965**, LGBM **0.977**.  
- Predictability LoRA **worse** than Base on pilot.  
- Figures/metrics may be mid-refresh (SciencePlots restyle by parallel agent)—freeze numbers from JSON files above.  
- No claim of spatial generalization beyond panel stations.

### 8. Conclusions

Restate contributions; point to larger multi-basin eval + calibrated uncertainty + human rating of reasons as next steps.

---

## 5. Innovation argumentation logic (how to *write* novelty)

```
Need: ops want Hs numbers AND situational labels/explanations
  → Gap: numeric SOTA ≠ language interface; LLM-TS papers chase RMSE
    → Move: Instruct+LoRA → JSON hs_forecast_m + regime/predictability + reason
      → Evidence: Base→LoRA gains; 100% JSON validity; fair triad baselines
        → Boundary: pilot n; RMSE not superior to persist/Chronos/LGBM
```

**Three innovation bullets for cover letter / highlights:**

1. Schema-constrained Hs **sequence generation** (`hs_forecast_m` JSON) under the same chronological windows as Persist/LGBM/Chronos—not grid SWH estimation (≠ Orca CIKM’24) and not Chronos-only numeric SWH (≠ Zhai OE’25).  
2. **Joint** forecast + regime + exploratory predictability labels + **textual rationale** (word carefully: not “explainability” / not calibrated UQ without coverage/CRPS).  
3. Transparent **capability boundary**: Base→LoRA gains with **honest** non-superiority vs persistence/Chronos/LGBM and predictability drop.

**Mandatory differentiation sentences (for Intro / Related):**

- vs **Orca**: buoy→grid estimation with spatiotemporal LLM encoding; we study **future point trajectories** + parseable JSON under classic lead-time verification.  
- vs **Zhai Chronos-SWH**: they optimize Chronos for Hs RMSE; we ask what an **instruction LLM** adds as a structured generator *beside* Chronos as a baseline.  
- vs **Aurora**: global Earth-system FM; out of scope as competitor—cite as field context only.

---

## 6. Metrics snapshot (authoritative)

### Numeric baselines (`numeric_baselines.json`)

| lead_h | rmse_persist | rmse_lgbm | skill_vs_persist |
|--------|--------------|-----------|------------------|
| 6 | 0.364 | 0.387 | −0.063 |
| 12 | 0.537 | 0.521 | 0.031 |
| 24 | 0.764 | 0.700 | 0.083 |
| 48 | 0.924 | 0.811 | 0.123 |
| 72 | 0.994 | 0.841 | 0.154 |

### Curve Mistral (`curve_compare` + baselines in curve metrics)

| Model | mean_rmse | notes |
|-------|-----------|-------|
| Mistral Base | 1.510 | n=12, json_valid=1.0 |
| Mistral LoRA | 0.977 | n=12, json_valid=1.0 |
| Persistence (same windows) | 0.884 | |
| LightGBM @ numeric leads | 0.977 | |
| Chronos @ numeric leads | 0.965 | |

### Classification (`compare_base_lora.json`)

| | regime_acc | predictability_acc | n |
|--|------------|--------------------|---|
| Base | 0.042 | 0.375 | 24 |
| LoRA | 0.417 | 0.250 | 24 |

---

## 7. Assumptions / missing inputs (nature-writing gate)

- Final station list & years for Table 1 not fully frozen in this outline → pull from panel build logs when drafting.  
- Uncertainty field in JSON may be qualitative; calibration metrics TBD.  
- Parallel agent may restyle figures; **do not edit plotting code here**—only plan names.
