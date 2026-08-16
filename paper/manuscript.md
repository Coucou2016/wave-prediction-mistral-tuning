# Schema-constrained significant-wave-height forecasting with instruction-tuned language models: 24-h JSON trajectories and sea-state labels from NDBC buoys

**Target journal:** *Ocean Engineering* (methods / ocean forecasting)  
**Manuscript version:** 3.1 (2026-08-17) — metrics frozen to public SSOT under the companion repository.

**One-sentence argument.** On a pilot NDBC buoy evaluation, instruction-tuned Mistral-7B-Instruct-v0.3 with LoRA recovers parseable JSON `hs_forecast_m` curves and companion sea-state labels relative to the pretrained Base model, while remaining numerically close to—but not superior to—Persistence under transparent lead-alignment caveats.

---

## Title

**Schema-constrained significant-wave-height forecasting with instruction-tuned language models: 24-h JSON trajectories and sea-state labels from NDBC buoys**

*Alternate (OE-leaning):* Structured buoy wave forecasting with Mistral-LoRA: Evaluation of JSON significant-wave-height trajectories and sea-state labels

---

## Abstract

Instruction-tuned large language models (LLMs) offer machine-readable output interfaces, but their role in ocean forecasting should be separated from claims of numerical superiority. We evaluate Mistral-7B-Instruct-v0.3 adapted with Low-Rank Adaptation (LoRA) for National Data Buoy Center significant-wave-height (Hs) forecasting. A curve adapter emits 24-h hourly Hs trajectories as JSON, while a separate companion adapter produces wave-regime and 24-h predictability labels. On **n = 24** forecast windows from **10** stations, LoRA reduced mean RMSE from **1.271** for the pretrained Base model to **0.699** with JSON validity **1.0**. Persistence, LightGBM†, and Chronos-T5† yielded RMSEs of **0.688**, **0.698**, and **0.951**, respectively; † denotes aggregation at configured numeric leads rather than dense 24-step scoring. For the companion task, regime accuracy increased from **0.042** to **0.417**, whereas predictability accuracy decreased from **0.375** to **0.250**. Because the classification evaluation is small and regime-imbalanced, these results are exploratory. The results support Instruct-LoRA as a structured companion to numeric buoy-forecasting workflows, but do not establish RMSE superiority or robust classification skill.

**Keywords:** significant wave height; NDBC buoys; large language models; LoRA; JSON forecasting; Chronos; LightGBM; wave regime

**Highlights (Ocean Engineering):**
1. Instruction tuning lowers mean root-mean-square error from 1.271 to 0.699
2. All reported wave-height curves satisfy the required machine-readable schema
3. Curve error is numerically close to persistence without superiority claims
4. Regime accuracy rises, while predictability accuracy declines

---

## 1. Introduction

Significant wave height (Hs) is a central variable for marine operations, navigation, and coastal risk assessment. Short-term Hs forecasts are therefore commonly formulated as continuous numerical trajectories. For automated downstream workflows, however, a forecast may also need to be delivered in a machine-readable form together with compact descriptors of the evolving sea state. This creates a problem distinct from numerical accuracy alone: whether wave forecasts can be exposed through a structured interface without obscuring the skill limits of the underlying predictor.

Data-driven Hs forecasting now spans recurrent and hybrid neural networks, tree-based ensembles, residual correction of numerical forecasts, Transformers, and pretrained time-series foundation models [1–8]. Recurrent and hybrid models have demonstrated short-range Hs forecasting capability [1], while multi-station studies have shown competitive performance from tree ensembles [2,3]. Machine learning has also been used to correct numerical wave forecasts [4]. More recently, Transformer architectures and time-series foundation models such as PatchTST and Chronos have extended the range of available forecasting approaches [5,6], and Chronos has been evaluated specifically for significant-wave-height prediction [8]. These developments make Persistence and established numerical or time-series models necessary comparators for any instruction-based forecasting pipeline.

Text-oriented large language models have been adapted to time-series forecasting through numerical tokenization, prompting, and reprogramming strategies, including LLMTime and Time-LLM [9,10]. At the same time, critical evaluations have shown that an LLM backbone does not necessarily confer superior forecasting accuracy and may add substantial computational cost [11]. Low-Rank Adaptation (LoRA) provides a parameter-efficient route for adapting large pretrained models [12], but efficient adaptation does not itself imply improved predictive skill. Accordingly, the present study treats root-mean-square error as an empirical benchmark rather than as a basis for claiming that an instruction-tuned LLM should replace specialized numerical forecasters.

Among the systems reviewed here, continuous Hs and general time-series forecasters primarily return numerical sequences, whereas LLM-for-time-series studies have largely evaluated forecasting accuracy. We consider a different evaluation target: whether an instruction-tuned model can emit a machine-parseable 24-h Hs trajectory while supporting companion sea-state outputs through a separate adapter. The companion task predicts wave-regime and 24-h predictability labels with textual notes; it is not a jointly trained multitask version of the curve model, and the textual fields are not interpreted as calibrated explanations. Structured output is therefore evaluated as an interface capability alongside, rather than in place of, numerical forecast skill.

Here we evaluate Mistral-7B-Instruct-v0.3 with LoRA on National Data Buoy Center observations. The study makes three bounded contributions. First, it evaluates schema-constrained generation of hourly 24-h Hs curves in a machine-readable JSON representation. Second, it examines a separate companion adapter for wave-regime and predictability labels while reporting positive and negative classification outcomes symmetrically. Third, it provides a protocol-aware comparison of Mistral Base and LoRA with Persistence, LightGBM†, and Chronos† on shared forecast windows, explicitly identifying the configured-lead aggregation used for † baselines rather than treating those scores as dense 24-step equivalents. The pilot scale and label imbalance are retained as limits on interpretation; the study tests a structured ocean-output interface and does not claim RMSE superiority or statistical equivalence to established numerical forecasters.

The remainder of the paper describes related work (Section 2), data and problem setup (Section 3), methods (Section 4), experiments and results (Section 5), discussion (Section 6), limitations (Section 7), and conclusions (Section 8).

---

## 2. Related work

### 2.1 Significant wave height prediction with ML and DL

Buoy Hs forecasting has a long ML tradition in ocean engineering journals. Fan et al. demonstrated LSTM-based Hs prediction and SWAN–LSTM hybrids against classical ML baselines [1]. Domala et al. compared Prophet, random forests, and boosting methods on multi-station NDBC data [2]. Chaichitehrani et al. reported stacking ensembles for Hs and period on U.S. East Coast buoys [3]. Recent residual-correction studies use LightGBM/CatBoost to adjust ECMWF/GEFS wave forecasts [4].

**Limitation relative to this work.** These approaches optimize continuous skill metrics. Among the systems reviewed here, they do not evaluate an Instruct-style schema that couples Hs sequences with parseable regime/predictability labels under one language interface.

### 2.2 Time-series Transformers and foundation models

PatchTST showed that patching and channel-independent Transformers are effective for long-term forecasting [5]. Chronos casts forecasting as a language-modeling problem over quantized series and reports strong zero-shot probabilistic performance [6]. TimeGPT popularized foundation-model rhetoric for forecasting across domains [7]. For ocean Hs specifically, Chronos-based wave prediction studies in *Ocean Engineering* [8] motivate treating Chronos as a first-class numeric comparator rather than an optional baseline.

### 2.3 LLMs for time series and parameter-efficient adaptation

LLMTime demonstrated that pretrained LLMs can forecast by next-token prediction over digit strings [9]. Time-LLM reprograms frozen LLMs with text prototypes and prompt prefixes [10]. Subsequent critiques argue that LLM backbones are often unnecessary for RMSE [11]. LoRA provides a practical PEFT route for domain adaptation [12]. Our work sits beside—not above—these lines: we adapt an Instruct LLM to emit **ocean JSON schemas**, and we keep Persistence / LightGBM / Chronos in the comparison table.

### 2.4 Positioning

Adjacent systems either (i) optimize continuous Hs with time-series foundation models [6,8], or (ii) reprogram LLMs for generic time-series RMSE [9,10]. We instead evaluate **sequence-instruction → JSON curve + companion regime/predictability labels** on NDBC windows, with explicit honesty about pilot sample size and numeric skill ceilings. Relative to Zhai et al. [8], Chronos remains a first-class numeric comparator rather than a competing Instruct-JSON interface. Relative to Tan et al. [11], we treat Instruct-LLM RMSE gains over Base as adaptation evidence, not as grounds for replacing specialized forecasters.

---

## 3. Data and problem setup

### 3.1 Buoy panel

We assemble an hourly multi-station NDBC panel after quality control and resampling. The working panel used for the reported experiments contains **10 stations** (`41010`, `42040`, `44013`, `46025`, `46026`, `46042`, `46047`, `46246`, `51000`, `51002`). Features include Hs, peak period \(T_p\), and wind speed when available. Exact panel cardinality and temporal coverage are documented with the public code and data-preparation release accompanying this manuscript.

### 3.2 Tasks

1. **Curve forecast (primary numeric LLM task).** Given history at issue time \(t_0\), emit JSON with `hs_forecast_m` of length 24 (hourly leads \(+1\ldots+24\) h).
2. **Classification companion.** Emit `wave_regime` and `predictability_24h` labels with free-text notes (separate LoRA adapter).

### 3.3 Baselines on shared windows

- **Persistence:** last observed Hs held constant over the horizon.
- **LightGBM†:** tabular regressor scored at configured numeric lead hours.
- **Chronos-T5†:** pretrained time-series foundation model forecasts aggregated at the same configured numeric leads.
- **Mistral Base vs LoRA:** same Instruct prompt/JSON schema; LoRA adapters trained on curve JSON samples.

---

## 4. Methods

### 4.1 Instruction schema and compression

Early curve fine-tuning attempts that packed full 168-hour \(T_p\)/wind sequences into the prompt were truncated by sequence-length limits. The configuration used for the reported adapters retains the full hourly `history_hs_m` series and compresses auxiliary channels to Hs summary statistics plus \(T_p\)/wind **mean / trend / last-6** summaries (verified against the released JSONL export implementation), with a maximum sequence length of 2048 tokens and gradient checkpointing.

### 4.2 LoRA fine-tuning

Curve LoRA uses Mistral-7B-Instruct-v0.3 with **1024** training samples and a **24 h** horizon. Classification LoRA is a separate adapter trained on regime/predictability JSON targets.

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
| LightGBM† | 0.698 | Configured numeric leads |
| Chronos† | 0.951 | Configured numeric leads |
| Mistral Base | 1.271 | JSON valid = 1.0 |
| Mistral LoRA | 0.699 | JSON valid = 1.0 |

**Reading.** On this pilot evaluation, LoRA substantially improves over Base (1.271 → 0.699) and achieves an RMSE numerically close to Persistence (0.688) and LightGBM† (0.698). Chronos† shows a higher aggregated-lead RMSE under this protocol; because † scoring is not dense 24-step, we do not interpret that gap as evidence that Chronos is an inferior model class. We do **not** claim LoRA superiority or statistical equivalence to Persistence.

Lead-wise RMSE profiles for Base and LoRA are shown in the accompanying figures. Errors are generally larger at later forecast leads, particularly in the latter part of the 24 h horizon, although the lead-wise RMSE profiles are not strictly monotonic.

### 5.2 Classification companion (n = 24)

| Model | Regime acc. | Predictability acc. |
|-------|-------------|---------------------|
| Base | 0.042 | 0.375 |
| LoRA | 0.417 | 0.250 |

Regime accuracy increased from 0.042 to 0.417, whereas predictability accuracy decreased from 0.375 to 0.250. Given n = 24 and severe regime-label imbalance on the evaluation set (true labels dominated by a single regime class), these results are exploratory and do not establish robust classification skill.

### 5.3 Figures

Main figures include: method-wise mean RMSE summary; lead-dependent Base versus LoRA RMSE; example multi-method Hs trajectories; and Base versus LoRA classification accuracy panels (SciencePlots rendering).

---

## 6. Discussion

The primary scientific value is **interface + evaluation discipline**, not a claim of RMSE dominance. Sequence-instruction fine-tuning yields machine-parseable Hs arrays with JSON validity of 1.0 on the reported set, while regime raw accuracy rises relative to Base under severe class imbalance. Predictability labels do not improve; free-text `reason`/`notes` fields in training targets remain partially template-like and should not be marketed as calibrated explanations.

Relative to Chronos-for-Hs studies [8] and Time-LLM-style reprogramming [10], our setting asks a different question: can an Instruct LLM serve as a **structured ocean output layer** beside strong numeric models? On the pilot numbers, LoRA recovers structured curve generation relative to Base and lands numerically close to Persistence on mean RMSE, while companion classification remains exploratory: raw regime accuracy rises relative to Base, but robust regime skill is not established, and predictability accuracy declines.

---

## 7. Limitations

1. **Pilot scale.** Curve and classification reported evaluations use **n = 24** windows; this is insufficient for operational certification or strong statistical ranking among near-tied methods (Persist 0.688 vs LoRA 0.699).
2. **Station / label imbalance.** True regime labels in the classification evaluation are heavily skewed toward a single class (`storm_growth`), limiting interpretation of raw accuracy.
3. **Lead aggregation asymmetry.** LightGBM/Chronos comparisons on the curve subset use configured leads (†), not identical dense 24-step curves.
4. **Textual rationales.** Free-text fields are not validated for factual correctness or calibrated uncertainty.
5. **Compute.** 7B Instruct LoRA requires GPU resources; Persistence/LightGBM remain far cheaper.
6. **Metric provenance.** Panel-wide numeric baseline tables at fixed leads (e.g., 6/12/24/48/72 h) must not be mixed with the curve-subset means reported above.

---

## 8. Conclusions

We present an NDBC→JSON→Mistral-LoRA study for structured Hs curve generation and companion regime/predictability labeling. On the curve evaluation (**n = 24**, **10** stations), LoRA improves mean RMSE from 1.271 to 0.699 with JSON validity 1.0, remaining numerically close to Persistence/LightGBM† while making no RMSE-superiority claim. Future work should enlarge temporally blocked test sets, balance regimes, and either drop or rigorously evaluate free-text rationale fields.

---

## Data availability

NDBC observations are publicly available from the U.S. National Data Buoy Center. Code, frozen evaluation metrics, and figure assets accompanying this manuscript are released at https://github.com/Coucou2016/wave-prediction-mistral-tuning. Large intermediate panels and model weights are not redistributed with the manuscript package; regeneration instructions are provided in the repository documentation.

---

## References (selected; verify DOIs before submission)

1. Fan et al., *Ocean Engineering*, 2020. https://doi.org/10.1016/j.oceaneng.2020.107298  
2. Domala et al., *JCDE*, 2022. https://doi.org/10.1093/jcde/qwac048  
3. Chaichitehrani et al., *AIES*, 2024. https://doi.org/10.1175/AIES-D-23-0061.1  
4. Henriques, M.R., Silva, D., Yanchin, I., Latas, M., Guedes Soares, C., Improving the forecast of wind speed and significant wave height using neural networks and gradient boosting trees. *Ocean Engineering* 327, 120925 (2025). https://doi.org/10.1016/j.oceaneng.2025.120925  
5. Nie, Y., Nguyen, N.H., Sinthong, P., Kalagnanam, J., A time series is worth 64 words: Long-term forecasting with Transformers (PatchTST). ICLR 2023. https://arxiv.org/abs/2211.14730  
6. Ansari, A.F., et al., Chronos: Learning the language of time series. 2024. https://doi.org/10.48550/arXiv.2403.07815  
7. Garza, A., Mergenthaler-Canseco, M., TimeGPT-1. 2023. https://doi.org/10.48550/arXiv.2310.03589  
8. Zhai, Y., Shi, H., Zhan, C., Wang, Q., You, Z., Wang, N., Improving significant wave height prediction using Chronos models. *Ocean Engineering* (2025). https://doi.org/10.1016/j.oceaneng.2025.122502  
9. Gruver, N., Finzi, M., Qiu, S., Wilson, A.G., Large language models are zero-shot time series forecasters (LLMTime). NeurIPS 2023. https://arxiv.org/abs/2310.07820  
10. Jin, M., et al., Time-LLM: Time series forecasting by reprogramming large language models. ICLR 2024.  
11. Tan, M., Merrill, M.A., Gupta, V., Althoff, T., Hartvigsen, T., Are language models actually useful for time series forecasting? NeurIPS 2024. https://doi.org/10.48550/arXiv.2406.16964  
12. Hu, E.J., et al., LoRA: Low-rank adaptation of large language models. ICLR 2022. https://doi.org/10.48550/arXiv.2106.09685  
