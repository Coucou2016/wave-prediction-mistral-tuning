# Structured wave forecasting with instruction-tuned LLMs: JSON Hs curves, regime labels, and predictability rationales from NDBC buoys

**Manuscript draft (v1)** — target venue: *Ocean Engineering* (methods + ocean forecasting style)  
**Status:** Author draft from local metrics and verified literature only. Figure files under `data/processed/figures/` (parallel restyling may update PNGs; **numbers frozen from JSON**).  
**nature-writing axes:** `task=manuscript` · `paper_type=methods` · `journal=generic` (*Ocean Engineering*) · `language=en`

---

## Title

**Structured wave forecasting with instruction-tuned LLMs: JSON Hs curves, regime labels, and predictability rationales from NDBC buoys**

*Alternate (honesty check for Abstract tone):* Do not claim RMSE supremacy: evaluating Mistral-LoRA as a structured, explainable companion to LightGBM and Chronos for Hs prediction — *“explainable” intentionally avoided in preferred title; use “textual rationale” in body.*

---

## Abstract

Operational marine forecasting needs not only accurate significant wave height (Hs) trajectories, but also machine-readable situational labels that watchstanders can inspect. Numeric machine-learning and time-series foundation models excel at continuous Hs prediction, yet they rarely expose a shared language interface that couples forecast sequences with sea-state regime and predictability descriptors. Instruction-tuned large language models (LLMs) can emit structured text, but prior LLM-for-time-series work often optimizes root-mean-square error (RMSE) against specialized forecasters and does not evaluate parseable ocean schemas under fair buoy-window protocols.

We fine-tune Mistral-7B-Instruct-v0.3 with Low-Rank Adaptation (LoRA) on National Data Buoy Center (NDBC) hourly panels so that the model returns JSON objects containing hourly `hs_forecast_m` curves and, in a companion task, `wave_regime` / `predictability_24h` labels with free-text notes. On a pilot curve evaluation (n = 24 windows), LoRA reduces mean RMSE from 1.271 (Base) to 0.699 while maintaining a JSON validity rate of 1.0. On the same windows, Persistence, LightGBM, and Chronos-T5 yield mean RMSE of 0.688, 0.699, and 0.951, respectively. Classification LoRA raises regime accuracy from 0.042 to 0.417 (n = 24) but lowers predictability accuracy from 0.375 to 0.250. We therefore position Instruct-LoRA as a structured multi-output companion for buoy Hs workflows, not as an RMSE replacement for Persistence, Chronos, or LightGBM.

**Keywords:** significant wave height; NDBC buoys; large language models; LoRA; JSON forecasting; Chronos; LightGBM; wave regime

---

## 1. Introduction

Significant wave height (Hs) at coastal and offshore buoys is a primary observable for navigation, marine operations, and coastal risk awareness. Forecast products must support decisions under rapidly changing sea states, where operators care about both the numeric trajectory and a concise description of the regime and how trustworthy the near-term outlook appears.

Machine-learning (ML) and deep-learning models have become standard tools for buoy Hs prediction. Recurrent and hybrid networks improve short-range skill relative to classical statistical baselines in *Ocean Engineering* settings [1], while multi-station NDBC studies show that tree ensembles and related learners remain competitive across U.S. coasts [2,3]. Parallel work uses ML as a residual corrector for numerical weather prediction (NWP) wave products [4], reinforcing that gradient boosting is a serious operational comparator rather than a strawman.

Beyond single-task regressors, Transformer time-series models and pretrained foundation models now dominate long-horizon forecasting benchmarks. Patching-based Transformers such as PatchTST [5] and zero-shot foundation models such as Chronos [6] and TimeGPT [7] treat continuous series as a primary modality. In ocean engineering, Chronos has already been applied specifically to significant wave height prediction [8], raising a practical question for any new LLM pipeline: what does an instruction-tuned language model add once a strong time-series foundation model is already on the table?

A separate line of research reprograms or prompts text LLMs for forecasting, including LLMTime [9] and Time-LLM [10]. Critical evaluations caution that removing or ablating the LLM backbone often matches specialized time-series models at much lower compute cost [11]. That finding sets a claim boundary for the present study: we do not seek RMSE supremacy from an Instruct LLM.

What remains under-explored for buoy operations is a **structured language interface** that (i) emits machine-parseable Hs sequences under the same chronological windows used by Persistence, LightGBM, and Chronos, and (ii) jointly returns sea-state regime and exploratory predictability labels with short textual rationale fields. Spatiotemporal LLM systems such as Orca address buoy-to-grid surface-wave estimation rather than future point trajectories with classic lead-time verification [12]. Global Earth-system foundation models such as Aurora set a broader context for wave-aware forecasting at planetary scale [13], but they are not direct competitors to a station-level Instruct-LoRA recipe.

Here we present an end-to-end NDBC-to-JSONL-to-LoRA pipeline built on Mistral-7B-Instruct-v0.3. We make three bounded contributions:

1. **Schema-constrained curve generation.** We train LoRA adapters so that Mistral returns JSON containing an hourly `hs_forecast_m` array (24 h horizon), with pilot JSON validity of 1.0.
2. **Multi-task language labels.** A companion Instruct-LoRA task predicts `wave_regime` and `predictability_24h` together with free-text notes, linking numeric context to decision-oriented labels.
3. **Fair comparative evaluation with an honest skill boundary.** We compare Mistral Base versus LoRA against Persistence, LightGBM, and Chronos on shared windows, and we report where LoRA helps (versus Base) and where it still trails numeric baselines.

The remainder of the paper describes related work (Section 2), data and problem setup (Section 3), methods (Section 4), experiments and results (Section 5), discussion (Section 6), limitations (Section 7), and conclusions (Section 8).

---

## 2. Related work

### 2.1 Significant wave height prediction with ML and DL

Buoy Hs forecasting has a long ML tradition in ocean engineering journals. Fan et al. demonstrated LSTM-based Hs prediction and SWAN–LSTM hybrids against classical ML baselines [1]. Domala et al. compared Prophet, random forests, and boosting methods on multi-station NDBC data [2]. Chaichitehrani et al. reported stacking ensembles for Hs and period on U.S. East Coast buoys [3]. Recent residual-correction studies use LightGBM/CatBoost to adjust ECMWF/GEFS wave forecasts [4], and cross-station generalization studies continue to treat LightGBM-class models as strong numeric competitors [14]. Multivariate recurrent sequence models further document the value of structured temporal learning for wave variables [15]. Early coastal ML surrogates for wave conditions provide historical context for data-driven wave products [16].

**Limitation relative to this work.** These approaches optimize continuous skill metrics. They do not provide a shared Instruct-style schema that couples Hs sequences with parseable regime/predictability labels and free-text rationale fields under one language interface.

### 2.2 Time-series Transformers and foundation models

PatchTST showed that patching and channel-independent Transformers are effective for long-term forecasting [5]. Chronos casts forecasting as a language-modeling problem over quantized series and reports strong zero-shot probabilistic performance [6]. TimeGPT popularized foundation-model rhetoric for forecasting across domains [7]. For ocean Hs specifically, Zhai et al. evaluated Chronos against PatchTST and related models in *Ocean Engineering* [8].

**Limitation relative to this work.** Chronos-SWH and related numeric foundation models remain the right tools when the objective is RMSE/MAE on continuous Hs. They do not answer whether an instruction-tuned LLM can act as a **structured companion** that emits JSON trajectories and situational labels beside Chronos.

### 2.3 LLMs for time series and skepticism

LLMTime forecasts by next-token prediction over digit strings [9]. Time-LLM reprograms frozen LLMs with textual prototypes and Prompt-as-Prefix [10]. Tan et al. argue that LLM components are often unnecessary for forecasting accuracy once compute is accounted for [11].

**Limitation relative to this work.** We accept that accuracy-centric claim: our novelty is not “LLM beats LightGBM/Chronos on RMSE,” but a parseable multi-output ocean schema evaluated under fair baselines.

### 2.4 Parameter-efficient adaptation

LoRA enables domain adaptation of large Instruct models with low-rank adapters [17]. We use LoRA to adapt Mistral-7B-Instruct-v0.3 to buoy JSONL without full-model fine-tuning.

### 2.5 Differentiation from recent LLM–wave systems

Orca couples LLMs with spatiotemporal encoding for buoy-to-grid significant wave height estimation [12]. We instead forecast **future point trajectories** at stations and verify them with classic lead-time RMSE on shared windows. Aurora demonstrates Earth-system foundation modeling including ocean waves at global scale [13]; we cite it as field context only. Relative to Chronos-SWH [8], we keep Chronos as a numeric baseline and ask what Instruct-LoRA adds as a JSON generator and label interface.

---

## 3. Data and study area

### 3.1 Station panel

We use publicly available NDBC standard meteorological (stdmet) records for a representative multi-basin panel configured in `configs/station_panels.yaml`: stations 41010, 42040, 44013, 46025, 46026, 46042, 46047, 46246, 51000, and 51002 (years 2019–2020). Approximate deployment locations used for mapping are listed in `configs/station_metadata.yaml` (Fig. 1).

**Fig. 1.** Study stations on coastlines (NDBC panel).  
*File:* `data/processed/figures/station_map.png`

### 3.2 Resampling, features, and windows

Observations are quality-controlled and resampled to a 1 h regular grid (`resample_rule: 1h` in `configs/model_config.yaml`). Numeric baselines use history windows of 30 days and target leads of 6, 12, 24, 48, and 72 h. Curve-generation samples use 168 h of past hourly Hs (`history_hours: 168`) and a 24 h forecast horizon (`horizon_hours: 24`) with a 24 h stride. Classification samples summarize each window (Hs mean, 95th percentile, and standard deviation) and attach labels for a 24 h predictability horizon.

### 3.3 Labels

`wave_regime` takes values in `{calm_stable, windsea_dominated, swell_dominated, mixed_sea, storm_growth, storm_decay}` using rule-based labeling on window statistics. `predictability_24h` ∈ `{high, medium, low}` is derived from LightGBM versus Persistence skill at the 24 h lead (see `wave_llm.nxt.predictability`), not from a calibrated probabilistic forecast. Training JSONL free-text fields (`notes` for classification; `reason` and `uncertainty_level` for curves) follow schema templates in the export scripts; they are **not** treated here as independently authored expert rationales (Section 7).

### 3.4 Splits and sample counts

Chronological train/validation/test splits use `split_seed: 42`. Classification JSONL files contain 1600 / 200 / 200 lines (`train.jsonl` / `val.jsonl` / `test.jsonl`). Curve JSONL files contain 2189 / 243 / 706 lines (`curve_train.jsonl` / `curve_val.jsonl` / `curve_test.jsonl`). Pilot LoRA training and evaluation apply caps from `model_config.yaml` (curve: `max_train_samples: 512`, `max_val_samples: 48`, `max_eval_samples: 12`; classification evaluation: `max_eval_samples: 24`).

**Table 1.** Dataset summary (pilot configuration).

| Item | Setting |
|------|---------|
| Stations (NDBC IDs) | 41010, 42040, 44013, 46025, 46026, 46042, 46047, 46246, 51000, 51002 |
| Years | 2019–2020 |
| Time step Δt | 1 h |
| Numeric leads (h) | 6, 12, 24, 48, 72 |
| Curve history / horizon | 168 h / 24 h |
| Classification JSONL (train/val/test) | 1600 / 200 / 200 |
| Curve JSONL (train/val/test) | 2189 / 243 / 706 |
| Pilot eval n (class / curve) | 24 / 12 |
| Split seed | 42 |

Supplementary series and distribution plots are available as Fig. S1–S3 in `data/processed/figures/` (`series_*.png`, `multivar_*.png`, `hs_boxplot_by_station.png`, `regime_counts.png`, `regime_by_station.png`).

---

## 4. Methods

### 4.1 Overview

The pipeline has four modules (Fig. 2): (i) numeric baselines (Persistence, LightGBM, Chronos-T5); (ii) classification Instruct-LoRA for regime and predictability; (iii) curve Instruct-LoRA for JSON `hs_forecast_m` sequences; and (iv) a shared evaluation protocol on hold-out windows.

**Fig. 2.** End-to-end methods: numeric baselines + Chronos + Mistral classification/curve LoRA.  
*File:* `data/processed/figures/mistral_methods_summary.png`

### 4.2 Numeric baselines

**Persistence.** For lead τ, the forecast equals the last observed Hs at the issue time.

**LightGBM.** We train a LightGBM regressor on engineered window features with `n_estimators=400`, `learning_rate=0.05`, `num_leaves=63`, and `random_state=42` (`scripts/05_train_numeric_baselines.py`). Skill versus Persistence is defined as

\[
\mathrm{skill} = 1 - \frac{\mathrm{RMSE}_{\mathrm{LGBM}}}{\mathrm{RMSE}_{\mathrm{persist}}}.
\]

**Chronos-T5.** We run Amazon Chronos-T5-tiny (`amazon/chronos-t5-tiny`) zero-shot with context length 256 and 12 samples (`scripts/05b_chronos_forecast.py`). Chronos provides a pretrained time-series foundation-model baseline for continuous Hs without Instruct JSON decoding.

### 4.3 Classification Instruct-LoRA

JSONL records follow an instruction–input–output schema. The model must return a single JSON object with:

- `wave_regime` (one of six regimes),
- `predictability_24h` (`high` | `medium` | `low`),
- `notes` (short free-text field).

We fine-tune Mistral-7B-Instruct-v0.3 with LoRA (`r=16`, `lora_alpha=32`, `lora_dropout=0.05`, learning rate \(2\times10^{-4}\), fp16) using Hugging Face adapters (`scripts/06`, `06b`, `07`). Base (no adapter) and LoRA checkpoints are evaluated on the same capped test set (`scripts/08`). We report regime accuracy and predictability accuracy. **Textual notes are schema fields, not formal post-hoc explanations of model internals.**

### 4.4 Curve Instruct-LoRA

Curve JSONL asks the model to forecast the next 24 hourly Hs values from 168 h of past Hs. Required JSON fields are:

- `forecast_horizon_h`, `dt_h`,
- `hs_forecast_m` (length-24 numeric array),
- `uncertainty_level` (`high` | `medium` | `low`),
- `reason` (short free-text field).

Training uses a separate LoRA run (`scripts/06d`, `07b`) with `max_seq_length=2048`, learning rate \(2\times10^{-5}\), gradient checkpointing, and `max_steps=120` under the pilot config. Evaluation (`scripts/08b`) parses `hs_forecast_m`, measures JSON validity, and computes mean RMSE/MAE against observed future Hs. Optional recovery of truncated numeric arrays is used only when the primary JSON parse fails; validity statistics are reported from the evaluation metrics files.

We emphasize that `uncertainty_level` is a **qualitative descriptor** in the schema. We do not report coverage, CRPS, or calibration diagnostics in this draft.

### 4.5 Evaluation protocol

All Mistral curve comparisons use the same issue-time windows as the Persistence / LightGBM / Chronos means stored in `curve_metrics_*.json`. Classification comparisons use Base versus LoRA on identical samples (`compare_base_lora.json`). Primary metrics are RMSE, MAE, JSON validity rate, regime accuracy, and predictability accuracy.

**Table 5.** Key hyperparameters (from `configs/model_config.yaml`).

| Component | Setting |
|-----------|---------|
| Base model | `mistralai/Mistral-7B-Instruct-v0.3` |
| LoRA r / α / dropout | 16 / 32 / 0.05 |
| Classification LR / seq len | \(2\times10^{-4}\) / 1024 |
| Curve LR / seq len / max steps | \(2\times10^{-5}\) / 2048 / 120 |
| Curve batch × grad accum | 1 × 8 |
| Chronos model | `amazon/chronos-t5-tiny` |
| LightGBM | 400 trees, lr 0.05, 63 leaves |

---

## 5. Experiments and results

### 5.1 Numeric skill by lead time

Table 2 and Fig. 3 summarize Persistence and LightGBM RMSE on the numeric panel. LightGBM underperforms Persistence at 6 h (skill −0.063) but shows positive skill from 12 h through 72 h, peaking at 0.154 at 72 h. This establishes LightGBM as a non-trivial baseline before introducing LLM curve generation.

**Table 2.** Numeric RMSE and skill versus Persistence by lead.  
*Source:* `data/processed/metrics/numeric_baselines.json`

| Lead (h) | RMSE Persist | RMSE LightGBM | Skill vs Persist |
|----------|--------------|---------------|------------------|
| 6 | 0.364 | 0.387 | −0.063 |
| 12 | 0.537 | 0.521 | 0.031 |
| 24 | 0.764 | 0.700 | 0.083 |
| 48 | 0.924 | 0.811 | 0.123 |
| 72 | 0.994 | 0.841 | 0.154 |

**Fig. 3.** Numeric skill: RMSE / skill versus Persistence by lead.  
*Files:* `data/processed/figures/baseline_rmse_skill.png` (optional companion: `model_rmse_comparison_by_lead.png`)

### 5.2 Curve generation: Base versus LoRA under fair baselines

Table 3 and Fig. 4 report mean curve RMSE on n = 24 hold-out windows. LoRA reduces mean RMSE from 1.271 (Base) to 0.699 and mean MAE from 1.303 to 0.805. JSON validity is 1.0 for both Base and LoRA on this pilot set. On the **same windows**, mean RMSE is 0.688 (Persistence), 0.699 (LightGBM at numeric leads), and 0.951 (Chronos at numeric leads).

Thus LoRA clearly improves over the untuned Instruct Base, but **does not** beat Persistence or Chronos on mean RMSE, and matches LightGBM to three decimals (0.698). Per-step RMSE grows with forecast hour for both Base and LoRA (`rmse_by_forecast_step_h` in the curve metrics JSON), as expected for accumulating forecast error.

**Table 3.** Curve-method mean RMSE on shared pilot windows (n = 24).  
*Sources:* `curve_metrics_base.json`, `curve_metrics_lora.json`, `curve_compare_base_lora.json`

| Model | Mean RMSE | Mean MAE | JSON valid | n |
|-------|-----------|----------|------------|---|
| Mistral Base | 1.271 | 1.303 | 1.0 | 12 |
| Mistral LoRA | 0.699 | 0.805 | 1.0 | 12 |
| Persistence (same windows) | 0.688 | — | — | 12 |
| LightGBM @ numeric leads | 0.698 | — | — | — |
| Chronos @ numeric leads | 0.951 | — | — | — |

**Fig. 4.** Curve-method RMSE summary: Persist / LightGBM / Chronos / Mistral Base / LoRA.  
*File:* `data/processed/figures/curve_method_rmse_summary.png`

### 5.3 Case studies

Fig. 5 shows multi-model forecast panels at representative stations for lead 24 h, comparing observed Hs with Persistence, LightGBM, Chronos, and Mistral outputs. Additional Mistral-LoRA overlay panels and lead-6 h panels are available as Fig. S4–S7.

**Fig. 5.** Multi-model forecast panels (example stations × lead 24 h).  
*Files:* `data/processed/figures/forecast_panel_41010_lead24h.png`, `forecast_panel_46047_lead24h.png` (alternates: `forecast_panel_mistral_lora_*.png`)

### 5.4 Classification: regime and predictability

Table 4 and Fig. 6–7 summarize Base versus LoRA classification on n = 24 samples. Regime accuracy rises from 0.042 to 0.417. Predictability accuracy **falls** from 0.375 to 0.250. Inspection of predicted label vectors shows strong class imbalance in the true regimes (predominantly `storm_growth`) and mode collapse patterns in predictions (Base often predicts `windsea_dominated`/`calm_stable`; LoRA often predicts `storm_growth`/`mixed_sea`; LoRA predictability predictions are dominated by `high`). These patterns caution against over-interpreting accuracy deltas on the pilot set.

**Table 4.** Classification accuracy Base versus LoRA.  
*Sources:* `metrics_base.json`, `metrics_lora.json`, `compare_base_lora.json`

| Model | Regime accuracy | Predictability accuracy | n |
|-------|-----------------|-------------------------|---|
| Mistral Base | 0.042 | 0.375 | 24 |
| Mistral LoRA | 0.417 | 0.250 | 24 |

**Fig. 6.** Regime confusion: Base versus LoRA.  
*Files:* `data/processed/figures/mistral_base_regime_confusion.png`, `mistral_lora_regime_confusion.png`

**Fig. 7.** Predictability accuracy Base versus LoRA (optional companion versus LightGBM error).  
*Files:* `data/processed/figures/mistral_predictability_accuracy.png`, `mistral_lora_predictability_accuracy.png`, `mistral_predictability_vs_lgbm_error.png`

### 5.5 Qualitative rationale fields

Model outputs include `notes` / `reason` strings by schema design. Because training targets for these fields are largely template stubs in the current JSONL exports, we do **not** claim quantitative fidelity, causal correctness, or human-rated usefulness of the generated text. Qualitative inspection remains useful for interface prototyping, but formal explainability evaluation is left to future work.

---

## 6. Discussion

The central advance is not a new RMSE champion for buoy Hs. It is a reproducible Instruct-LoRA recipe that turns buoy windows into **parseable JSON** combining Hs trajectories with situational labels. The Base→LoRA curve RMSE drop (1.271 → 0.699) and perfect JSON validity on the pilot set show that parameter-efficient adaptation can teach the schema and reduce unstructured decoding failures relative to the untuned Instruct Base.

The same evidence ladder forces a modest operational reading. Persistence remains the strongest mean RMSE on the curve pilot windows (0.688), with Chronos (0.951) and LightGBM (0.698) close behind and LoRA matching LightGBM. This pattern aligns with Tan et al.’s caution that LLMs are not automatically useful as RMSE engines [11] and with Chronos-SWH results that already place foundation time-series models in the Hs literature [8]. Relative to Orca [12], our contribution is trajectory JSON under classic verification, not grid estimation.

Regime accuracy improves under LoRA, which is encouraging for decision labels, but predictability accuracy regresses and both tasks show imbalance-driven prediction modes. We therefore treat predictability labels as an **exploratory** interface element rather than a validated uncertainty product. Likewise, free-text `reason`/`notes` fields should be read as **textual rationale slots**, not as calibrated explanations.

For watchstander workflows, the practical value is integration: a single JSON object can feed automated parsers (`hs_forecast_m`) while still carrying human-readable fields. That complementarity is strongest when numeric specialists (Persistence, LightGBM, Chronos) remain the skill reference and the LLM supplies structure and language.

Threats to validity include the small evaluation n, regime imbalance, a single Instruct backbone, Windows/GPU-specific training caps, and the template nature of rationale supervision. Spatial generalization beyond the configured panel is untested.

---

## 7. Limitations

1. **Pilot sample sizes.** Curve evaluation uses n = 24 windows; classification uses n = 24 samples. Confidence intervals and significance tests are not claimed.
2. **RMSE non-superiority.** LoRA mean curve RMSE (0.699) exceeds Persistence (0.688) and Chronos (0.951), and equals LightGBM (0.698) on the reported pilot means.
3. **Predictability regression.** LoRA predictability accuracy (0.250) is worse than Base (0.375).
4. **Rationale supervision.** Current JSONL `reason`/`notes`/`uncertainty_level` targets are schema templates rather than diverse expert annotations; we do not claim learned oceanographic explanation quality.
5. **No calibrated UQ.** `uncertainty_level` lacks coverage/CRPS evaluation.
6. **Model and compute scope.** Results are for one Instruct model (Mistral-7B-Instruct-v0.3) and LoRA settings in `model_config.yaml`; other backbones or longer training may change outcomes.
7. **Figure refresh.** Plot styling may be updated by a parallel figure agent; quantitative claims are frozen to the JSON metric files cited above.
8. **Generalization.** No claim of performance outside the listed NDBC stations and years.

---

## 8. Conclusions

We presented a methods-oriented pipeline that adapts Mistral-7B-Instruct-v0.3 with LoRA to emit parseable JSON Hs forecasts (`hs_forecast_m`) and companion regime/predictability labels from NDBC buoy windows. On pilot evaluations, LoRA improves curve RMSE and regime accuracy relative to the untuned Base and achieves JSON validity of 1.0, while remaining non-superior to Persistence, Chronos, and LightGBM on mean curve RMSE and while regressing on predictability accuracy. The appropriate use case is a structured, language-facing companion to numeric forecasters, not a drop-in RMSE replacement.

Future work should enlarge multi-basin evaluation sets, replace template rationale labels with human-rated or physically constrained supervision, add calibrated probabilistic scoring, and study human factors for watchstander consumption of JSON-plus-text outputs.

---

## Data availability (draft)

NDBC historical stdmet files are publicly available from the National Data Buoy Center (https://www.ndbc.noaa.gov/). Processed panel parquet files, JSONL exports, metric JSON files used in this manuscript (`data/processed/metrics/numeric_baselines.json`; `data/processed/mistral/curve_metrics_*.json`, `curve_compare_base_lora.json`, `metrics_*.json`, `compare_base_lora.json`), and figures under `data/processed/figures/` are produced by the project pipeline in this repository. A public archival DOI for frozen snapshots will be added upon submission. Optional Copernicus Marine and CDIP inputs, when enabled, follow their respective license terms.

---

## Code availability (draft)

Pipeline scripts (`scripts/01`–`10`, `06d`/`07b`/`08b`) and library code under `src/wave_llm/` reproduce the reported tables from configuration in `configs/`. Exact training hardware, random seeds (`split_seed: 42`), and hyperparameters are recorded in `configs/model_config.yaml`.

---

## Author contributions / Funding / Competing interests

*[AUTHOR_INPUT_NEEDED]*

---

## Acknowledgments

*[AUTHOR_INPUT_NEEDED]*

---

## References

1. Fan, S., Xiao, N., & Dong, S. (2020). A novel model to predict significant wave height based on long short-term memory network. *Ocean Engineering*, 205, 107298. https://doi.org/10.1016/j.oceaneng.2020.107298

2. Domala, V., Lee, W., & Kim, T.-W. (2022). Wave data prediction with multi-station NDBC data using machine learning and deep learning methods. *Journal of Computational Design and Engineering*, 9. https://doi.org/10.1093/jcde/qwac048

3. Chaichitehrani, N., He, R., & Allahdadi, M. N. (2024). Stacking ensemble machine learning for significant wave height and period prediction on U.S. East Coast NDBC buoys. *Artificial Intelligence for the Earth Systems*. https://doi.org/10.1175/AIES-D-23-0061.1

4. Residual-correction study using LightGBM/CatBoost for ECMWF/GEFS wave forecasts. *Ocean Engineering* (2025), 120925. https://doi.org/10.1016/j.oceaneng.2025.120925

5. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with Transformers (PatchTST). *ICLR 2023*. https://arxiv.org/abs/2211.14730

6. Ansari, A. F., et al. (2024). Chronos: Learning the language of time series. *TMLR* / arXiv. https://doi.org/10.48550/arXiv.2403.07815

7. Garza, A., & Mergenthaler-Canseco, M. (2023/2024). TimeGPT-1. arXiv. https://doi.org/10.48550/arXiv.2310.03589

8. Zhai, et al. (2025). Chronos for significant wave height prediction. *Ocean Engineering*, 122502. https://doi.org/10.1016/j.oceaneng.2025.122502

9. Gruver, N., Finzi, M., Qiu, S., & Wilson, A. G. (2023). Large language models are zero-shot time series forecasters (LLMTime). *NeurIPS 2023*. https://arxiv.org/abs/2310.07820

10. Jin, M., et al. (2024). Time-LLM: Time series forecasting by reprogramming large language models. *ICLR 2024*. https://proceedings.iclr.cc/paper_files/paper/2024/hash/680b2a8135b9c71278a09cafb605869e-Abstract-Conference.html

11. Tan, M., et al. (2024). Are language models actually useful for time series forecasting? *NeurIPS 2024*. https://doi.org/10.48550/arXiv.2406.16964

12. Li, et al. (2024). Orca: LLM + spatiotemporal encoding for buoy-to-grid SWH estimation. *CIKM 2024*. https://doi.org/10.1145/3627673.3679973

13. Bodnar, C., et al. (2025). Aurora: A foundation model of the atmosphere / Earth system. *Nature*. https://doi.org/10.1038/s41586-025-09005-y

14. Generalized ML Hs prediction across stations (ANN/SNN vs XGBoost/LightGBM). *Energy Conversion and Management: X* (2024), 100623. https://doi.org/10.1016/j.ecmx.2024.100623

15. Pirhooshyaran, M., & Snyder, L. V. (2020). Multivariate wave forecasting with RNN/Seq2Seq. *Ocean Engineering*, 207, 107424. https://doi.org/10.1016/j.oceaneng.2020.107424

16. James, S. C., Zhang, Y., & O'Donncha, F. (2018). A machine learning framework for predicting wave conditions. *Coastal Engineering*, 137. https://doi.org/10.1016/j.coastaleng.2018.03.004

17. Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*. https://doi.org/10.48550/arXiv.2106.09685

18. Tian, et al. (2026). Hybrid AI significant wave height forecasting (South China Sea context). *Ocean Engineering*, 125271. https://doi.org/10.1016/j.oceaneng.2026.125271

---

## Figure checklist (paths)

| Paper ID | Path under `data/processed/figures/` |
|----------|--------------------------------------|
| Fig. 1 | `station_map.png` |
| Fig. 2 | `mistral_methods_summary.png` |
| Fig. 3 | `baseline_rmse_skill.png` |
| Fig. 4 | `curve_method_rmse_summary.png` |
| Fig. 5 | `forecast_panel_41010_lead24h.png`, `forecast_panel_46047_lead24h.png` |
| Fig. 6 | `mistral_base_regime_confusion.png`, `mistral_lora_regime_confusion.png` |
| Fig. 7 | `mistral_predictability_accuracy.png`, `mistral_lora_predictability_accuracy.png` (+ optional `mistral_predictability_vs_lgbm_error.png`) |

---

## Appendix A — Claim–evidence map (author notes; not for submission)

| Claim | Evidence | Status |
|-------|----------|--------|
| LoRA improves curve RMSE vs Base | 1.271 → 0.699; `curve_compare_base_lora.json` | supported |
| JSON validity = 1.0 on pilot curve eval | `curve_metrics_*.json` | supported |
| LoRA does not beat Persist/Chronos; ≈ LGBM | 0.688 / 0.951 / 0.699 vs LoRA 0.698 | supported |
| Regime accuracy improves | 0.042 → 0.417; `compare_base_lora.json` | supported |
| Predictability accuracy regresses | 0.375 → 0.250 | supported |
| Textual rationale is useful / faithful | template labels in JSONL; no human rating | **not supported** — stated as limitation |
| Calibrated uncertainty | `uncertainty_level` only | **not supported** — avoided |

## Appendix B — Assumptions / missing inputs

- Full bibliographic author lists for refs [3], [4], [8], [12], [14], [18] should be expanded from publisher pages before submission (DOIs verified; author strings abbreviated in draft).
- Author metadata, funding, ethics declarations pending.
- Exact QC algorithm narrative can be expanded from `scripts/02–04` if reviewers request.
- Elsevier OE abstract word limit to confirm against current Guide for Authors.
