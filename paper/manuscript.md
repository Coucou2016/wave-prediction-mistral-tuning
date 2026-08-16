# Schema-constrained significant-wave-height forecasting with instruction-tuned language models: 24-h JSON trajectories and sea-state labels from NDBC buoys

**Manuscript draft (v2.1)** — target: *Ocean Engineering* (methods + ocean forecasting)  
**Status:** Author draft from local metrics under `paper/metrics/` (2026-08-16 curve LoRA v2 eval). Numbers are not invented. `paper/metrics/*.json` is the single source of truth (supersedes any older n≈12 notes in `docs/`).  
**nature-writing axes:** `task=manuscript` · `paper_type=methods` · `journal=generic` (*Ocean Engineering*) · `language=en`  
**Code/data pointers:** https://github.com/Coucou2016/wave-prediction-mistral-tuning · figures in `paper/figures/` (SciencePlots + Times New Roman)  
**ChatGPT Round-1 ADOPT:** schema-constrained product framing; companion (not joint multitask) adapters; demote predictability from title; keep negative predictability result.

**One-sentence argument.** In NDBC buoy significant-wave-height forecasting, we show that instruction-tuned Mistral-7B-Instruct-v0.3 with LoRA can emit parseable JSON `hs_forecast_m` curves together with companion sea-state labels, supported by Base→LoRA gains and fair Persistence / LightGBM / Chronos comparisons, with the boundary that pilot curve RMSE does **not** claim supremacy over specialized numeric forecasters.

---

## Title

**Schema-constrained significant-wave-height forecasting with instruction-tuned language models: 24-h JSON trajectories and sea-state labels from NDBC buoys**

*Alternate (OE-leaning):* Structured buoy wave forecasting with Mistral-LoRA: Evaluation of JSON significant-wave-height trajectories and sea-state labels

---

## Abstract

Operational marine forecasting needs not only continuous significant wave height (Hs) trajectories, but also machine-readable situational labels that operators can inspect. Numeric machine-learning models and time-series foundation models excel at continuous Hs prediction, yet they rarely expose a shared language interface that couples forecast sequences with sea-state regime and predictability descriptors. Instruction-tuned large language models (LLMs) can emit structured text, but prior LLM-for-time-series work often optimizes root-mean-square error (RMSE) against specialized forecasters and does not evaluate parseable ocean schemas under fair buoy-window protocols.

We fine-tune Mistral-7B-Instruct-v0.3 with Low-Rank Adaptation (LoRA) on National Data Buoy Center (NDBC) hourly panels so that the model returns JSON objects containing hourly `hs_forecast_m` curves and, in a companion task, `wave_regime` / `predictability_24h` labels with free-text notes. On the curve evaluation set (**n = 24** windows spanning **10** stations), LoRA reduces mean RMSE from **1.271** (Base) to **0.699** while maintaining a JSON validity rate of **1.0**. On the same windows, Persistence, LightGBM (at configured numeric leads), and Chronos-T5 yield mean RMSE of **0.688**, **0.698**, and **0.951**, respectively. Classification LoRA raises regime accuracy from **0.042** to **0.417** (**n = 24**) but lowers predictability accuracy from **0.375** to **0.250**. We therefore position Instruct-LoRA as a structured multi-output companion for buoy Hs workflows, not as an RMSE replacement for Persistence, Chronos, or LightGBM.

**Keywords:** significant wave height; NDBC buoys; large language models; LoRA; JSON forecasting; Chronos; LightGBM; wave regime

---

## 1. Introduction

Significant wave height (Hs) at coastal and offshore buoys is a primary observable for navigation, marine operations, and coastal risk awareness. Forecast products must support decisions under changing sea states, where operators care about both the numeric trajectory and a concise description of the regime and how trustworthy the near-term outlook appears.

Machine-learning (ML) and deep-learning models have become standard tools for buoy Hs prediction. Recurrent and hybrid networks improve short-range skill relative to classical statistical baselines in *Ocean Engineering* settings [1], while multi-station NDBC studies show that tree ensembles remain competitive across U.S. coasts [2,3]. Parallel work uses ML as a residual corrector for numerical weather prediction (NWP) wave products [4], reinforcing that gradient boosting is a serious operational comparator rather than a strawman.

Beyond single-task regressors, Transformer time-series models and pretrained foundation models now dominate long-horizon forecasting benchmarks. Patching-based Transformers such as PatchTST [5] and zero-shot foundation models such as Chronos [6] and TimeGPT [7] treat continuous series as a primary modality. In ocean engineering, Chronos has already been applied specifically to significant wave height prediction [8], raising a practical question for any new LLM pipeline: what does an instruction-tuned language model add once a strong time-series foundation model is already on the table?

A separate line of research reprograms or prompts text LLMs for forecasting, including LLMTime [9] and Time-LLM [10]. Critical evaluations caution that removing or ablating the LLM backbone often matches specialized time-series models at much lower compute cost [11]. That finding sets a claim boundary for the present study: we do not seek RMSE supremacy from an Instruct LLM.

What remains under-explored for buoy operations is a **structured language interface** that (i) emits machine-parseable Hs sequences under the same chronological windows used by Persistence, LightGBM, and Chronos, and (ii) jointly returns sea-state regime and exploratory predictability labels with short textual rationale fields.

Here we present an end-to-end NDBC-to-JSONL-to-LoRA pipeline built on Mistral-7B-Instruct-v0.3. We make three bounded contributions:

1. **Schema-constrained curve generation.** We train LoRA adapters so that Mistral returns JSON containing an hourly `hs_forecast_m` array (24 h horizon), with JSON validity of 1.0 on the reported eval set.
2. **Companion sea-state labels (separate adapter).** A companion Instruct-LoRA task predicts `wave_regime` and `predictability_24h` together with free-text notes. These are **companion tasks** sharing the Instruct family, not a single jointly trained multitask adapter.
3. **Fair comparative evaluation with an honest skill boundary.** We compare Mistral Base versus LoRA against Persistence, LightGBM, and Chronos on shared windows, and we report where LoRA helps (versus Base) and where it remains comparable to—or trails—numeric baselines. LightGBM/Chronos aggregates on the curve subset use configured leads (†) and are not claimed to be dense 24-step apples-to-apples matches.

The remainder of the paper describes related work (Section 2), data and problem setup (Section 3), methods (Section 4), experiments and results (Section 5), discussion (Section 6), limitations (Section 7), and conclusions (Section 8).

---

## 2. Related work

### 2.1 Significant wave height prediction with ML and DL

Buoy Hs forecasting has a long ML tradition in ocean engineering journals. Fan et al. demonstrated LSTM-based Hs prediction and SWAN–LSTM hybrids against classical ML baselines [1]. Domala et al. compared Prophet, random forests, and boosting methods on multi-station NDBC data [2]. Chaichitehrani et al. reported stacking ensembles for Hs and period on U.S. East Coast buoys [3]. Recent residual-correction studies use LightGBM/CatBoost to adjust ECMWF/GEFS wave forecasts [4].

**Limitation relative to this work.** These approaches optimize continuous skill metrics. They do not provide a shared Instruct-style schema that couples Hs sequences with parseable regime/predictability labels under one language interface.

### 2.2 Time-series Transformers and foundation models

PatchTST showed that patching and channel-independent Transformers are effective for long-term forecasting [5]. Chronos casts forecasting as a language-modeling problem over quantized series and reports strong zero-shot probabilistic performance [6]. TimeGPT popularized foundation-model rhetoric for forecasting across domains [7]. For ocean Hs specifically, Chronos-based wave prediction studies in *Ocean Engineering* [8] motivate treating Chronos as a first-class numeric comparator rather than an optional baseline.

### 2.3 LLMs for time series and parameter-efficient adaptation

LLMTime demonstrated that pretrained LLMs can forecast by next-token prediction over digit strings [9]. Time-LLM reprograms frozen LLMs with text prototypes and prompt prefixes [10]. Subsequent critiques argue that LLM backbones are often unnecessary for RMSE [11]. LoRA provides a practical PEFT route for domain adaptation [12]. Our work sits beside—not above—these lines: we adapt an Instruct LLM to emit **ocean JSON schemas**, and we keep Persistence / LightGBM / Chronos in the comparison table.

### 2.4 Positioning

Closest adjacent systems either (i) optimize continuous Hs with TS foundation models [6,8], or (ii) reprogram LLMs for generic TS RMSE [9,10]. We instead evaluate **sequence-instruction → JSON curve + regime/predictability labels** on NDBC windows, with explicit honesty about pilot sample size and numeric skill ceilings.

---

## 3. Data and problem setup

### 3.1 Buoy panel

We assemble an hourly multi-station NDBC panel after QC and resampling. The working panel used for the reported experiments contains **10 stations** (`41010`, `42040`, `44013`, `46025`, `46026`, `46042`, `46047`, `46246`, `51000`, `51002`) and on the order of \(1.6\times10^5\) hourly rows. Features include Hs, peak period \(T_p\), and wind speed when available.

### 3.2 Tasks

1. **Curve forecast (primary numeric LLM task).** Given history at issue time \(t_0\), emit JSON with `hs_forecast_m` of length 24 (hourly leads \(+1\ldots+24\) h).
2. **Classification companion.** Emit `wave_regime` and `predictability_24h` labels with free-text notes (separate LoRA run).

### 3.3 Baselines on shared windows

- **Persistence:** last observed Hs held constant over the horizon.
- **LightGBM:** tabular regressor at configured lead hours (standard numeric pipeline).
- **Chronos-T5:** pretrained time-series foundation model zero-shot / panel-configured forecasts at numeric leads.
- **Mistral Base vs LoRA:** same Instruct prompt/JSON schema; LoRA adapters trained on curve JSONL.

---

## 4. Methods

### 4.1 Instruction schema and compression

Early curve LoRA pilots stalled when packing full 168-hour \(T_p\)/wind sequences into the prompt. The production recipe keeps full `history_hs_m` but compresses auxiliary channels to Hs statistics plus \(T_p\)/wind **mean / trend / last-6** summaries, with `max_seq_length = 2048` and gradient checkpointing.

### 4.2 LoRA fine-tuning

Curve LoRA v2 (`curve_lora_run_v2`) uses Mistral-7B-Instruct-v0.3 with **1024** training samples and a **24 h** horizon (see `paper/metrics/curve_lora_meta_v2.json`). Classification LoRA remains a separate adapter trained on regime/predictability JSONL.

### 4.3 Metrics

- Curve: mean RMSE / MAE over valid JSON samples; RMSE-by-lead; JSON validity rate.
- Classification: regime accuracy; predictability accuracy.
- Fairness note: LightGBM/Chronos RMSE on the curve subset is aggregated at configured numeric leads (†), not a full 24-step dense curve unless the numeric model emits one.

---

## 5. Experiments and results

### 5.1 Curve forecast (n = 24, 10 stations)

| Method | Mean RMSE (m) | Notes |
|--------|---------------|-------|
| Persistence | 0.688 | Same eval windows |
| LightGBM† | 0.698 | At numeric leads |
| Chronos† | 0.951 | At numeric leads |
| Mistral Base | 1.271 | JSON valid = 1.0 |
| Mistral LoRA | 0.699 | JSON valid = 1.0 |

Source: `paper/metrics/curve_metrics_*.json`, `curve_compare_base_lora.json`.

**Reading.** LoRA substantially improves over Base (1.271 → 0.699) and lands near Persistence / LightGBM on this pilot subset, while Chronos† is weaker on the same aggregated metric. We do **not** claim LoRA beats Persistence in a statistically powered multi-year benchmark; the takeaway is Base→LoRA recovery of parseable curves under a compressed instruction schema.

RMSE grows with lead hour for both Base and LoRA (Fig. `curve_rmse_by_lead_lines.png` / `model_rmse_comparison_by_lead.png`), consistent with predictability decay.

### 5.2 Classification companion (n = 24)

| Model | Regime acc. | Predictability acc. |
|-------|-------------|---------------------|
| Base | 0.042 | 0.375 |
| LoRA | 0.417 | 0.250 |

Source: `paper/metrics/compare_base_lora.json`. Regime accuracy rises; predictability accuracy falls—evidence that multi-label Instruct objectives can trade off and must be reported honestly.

### 5.3 Figures (SciencePlots)

All paper figures regenerated with SciencePlots + Times New Roman (`scripts/11_make_science_figures.py`):

- `paper/figures/curve_method_rmse_summary.png`
- `paper/figures/model_rmse_comparison_by_lead.png`
- `paper/figures/curve_rmse_by_lead_lines.png`
- `paper/figures/forecast_panel_mistral_lora_41010.png`
- `paper/figures/classification_base_vs_lora.png`
- regime/predictability confusion and accuracy panels

---

## 6. Discussion

The primary scientific value is **interface + evaluation discipline**, not a claim of RMSE dominance. Sequence-instruction fine-tuning yields machine-parseable Hs arrays with perfect JSON validity on the reported set, while regime labels improve under LoRA. Predictability labels do not; free-text `reason`/`notes` fields in training targets remain partially template-like and should not be marketed as calibrated explanations.

Relative to Chronos-for-Hs studies [8] and Time-LLM-style reprogramming [10], our setting asks a different question: can an Instruct LLM serve as a **structured ocean output layer** beside strong numeric models? On the pilot numbers, the answer is conditionally yes for JSON curves and regime tags, with numeric skill close to Persistence/LightGBM rather than clearly above them.

---

## 7. Limitations

1. **Pilot scale.** Curve and classification reported evals use **n = 24** windows; this is insufficient for operational certification or strong statistical ranking among near-tied methods (Persist 0.688 vs LoRA 0.699).
2. **Station imbalance.** True regime labels in the classification eval are heavily skewed toward `storm_growth`, inflating naive interpretation of accuracy.
3. **Lead aggregation asymmetry.** LightGBM/Chronos comparisons on the curve subset use configured leads (†), not identical dense 24-step curves.
4. **Textual rationales.** Free-text fields are not validated for factual correctness or calibrated uncertainty.
5. **Compute.** 7B Instruct LoRA requires GPU resources; Persistence/LightGBM remain far cheaper.
6. **Earlier pilot vs v2.** An earlier smaller curve pilot (n≈12) is superseded by the v2 metrics above; do not mix the two tables.

---

## 8. Conclusions

We present a reproducible NDBC→JSONL→Mistral-LoRA recipe for structured Hs curve generation and companion regime/predictability labeling. On the v2 curve eval (**n = 24**, **10** stations), LoRA improves mean RMSE from 1.271 to 0.699 with JSON validity 1.0, approaching Persistence/LightGBM while remaining an honest non-SOTA claim relative to specialized numeric forecasting. Future work should enlarge temporally blocked test sets, balance regimes, and either drop or rigorously evaluate free-text rationale fields.

---

## References (selected; verify before submission)

1. Fan et al., *Ocean Engineering*, 2020. https://doi.org/10.1016/j.oceaneng.2020.107298  
2. Domala et al., *JCDE*, 2022. https://doi.org/10.1093/jcde/qwac048  
3. Chaichitehrani et al., *AIES*, 2024. https://doi.org/10.1175/AIES-D-23-0061.1  
4. Residual-correction LightGBM/CatBoost study, *Ocean Engineering*, 2025. https://doi.org/10.1016/j.oceaneng.2025.120925  
5. Nie et al., PatchTST, ICLR 2023. https://arxiv.org/abs/2211.14730  
6. Ansari et al., Chronos, 2024. https://doi.org/10.48550/arXiv.2403.07815  
7. Garza & Mergenthaler-Canseco, TimeGPT-1. https://doi.org/10.48550/arXiv.2310.03589  
8. Chronos for significant wave height, *Ocean Engineering*, 2025. https://doi.org/10.1016/j.oceaneng.2025.122502  
9. Gruver et al., LLMTime, NeurIPS 2023. https://arxiv.org/abs/2310.07820  
10. Jin et al., Time-LLM, ICLR 2024.  
11. Tan et al., Are LMs useful for TS forecasting?, NeurIPS 2024. https://doi.org/10.48550/arXiv.2406.16964  
12. Hu et al., LoRA, ICLR 2022. https://doi.org/10.48550/arXiv.2106.09685  

---

## Author notes / assumptions

- Metrics frozen from local JSON copied to `paper/metrics/` on 2026-08-16.  
- Venue framing assumes *Ocean Engineering*; Nature-family claim discipline only.  
- “Explainability” wording intentionally avoided for free-text fields; use “textual rationale / notes”.
