# Schema-constrained significant-wave-height forecasting with instruction-tuned language models: 24-h JSON trajectories and sea-state labels from NDBC buoys

**Target journal:** *Ocean Engineering* (methods / ocean forecasting)

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

### 2.1 Data-driven and hybrid significant-wave-height forecasting

Data-driven significant-wave-height (Hs) forecasting has progressed from site-specific statistical and neural predictors toward recurrent, ensemble, and hybrid numerical–machine-learning systems. Recurrent models provide an established reference point: Fan et al. [1] evaluated LSTM-based Hs forecasting and a SWAN–LSTM hybrid, illustrating how learned temporal dynamics can complement conventional wave modelling. Subsequent studies broadened the model class and spatial setting. NDBC-based evaluations have compared tree ensembles and other machine-learning approaches for wave variables [2], while stacking methods have combined heterogeneous learners for Hs and wave-period prediction along the U.S. East Coast [3]. Collectively, this literature establishes recurrent networks and tree-based ensembles as credible numerical comparators rather than weak baselines.

A complementary line uses machine learning as a post-processor for numerical wave forecasts. Henriques et al. [4], for example, learned residuals between ECMWF or GEFSWAVES forecasts and coastal-buoy observations using recurrent networks and gradient-boosting methods; LightGBM and CatBoost provided particularly strong corrections. This formulation is operationally important because machine learning augments rather than replaces an existing numerical forecast. Across these approaches, however, the principal evaluation target remains numerical forecast skill or numerical-model correction. Among the systems reviewed here, they do not evaluate the specific problem considered in this study: schema-constrained generation of an Hs trajectory together with machine-readable companion sea-state outputs through an instruction-oriented interface.

### 2.2 Transformers and time-series foundation models

Transformer forecasting has developed along a distinct trajectory from text-oriented large language models. PatchTST [5], for example, represents time series through patches and channel-independent processing. Chronos [6] goes further by pretraining probabilistic time-series models on large collections of real and synthetic series: scaled values are quantized into a discrete vocabulary and modelled using T5-family architectures. Chronos should therefore be described here as a pretrained time-series foundation model, not as an instruction-tuned text LLM merely because its architecture and token-prediction objective are language-model inspired. Other foundation-model efforts, including TimeGPT [7], similarly target transferable numerical forecasting.

This distinction is especially important for ocean forecasting because Chronos has already been evaluated directly for Hs prediction. Zhai et al. [8] adapted Chronos to significant-wave-height forecasting and examined forecast horizons extending from short-range to multi-day prediction. Their study makes Chronos a domain-relevant reference for the present work rather than a generic off-domain foundation-model baseline. The scientific comparison is therefore not “LLM versus no LLM” in the abstract; it is between an instruction-oriented structured-output formulation and established numerical forecasting approaches, including a time-series foundation model already applied to the same ocean variable.

### 2.3 Text-oriented LLMs for time-series forecasting

A separate literature repurposes pretrained text language models for numerical time series. LLMTime [9] encodes numerical observations as digit strings and formulates forecasting as next-token prediction by pretrained language models. Time-LLM [10] instead reprograms time-series inputs into representations compatible with a frozen LLM and augments them with prompt information before projecting the resulting representations back to numerical forecasts. These studies demonstrate mechanisms by which text-pretrained models can be applied to time-series prediction, but their central evaluation remains forecasting performance rather than schema-constrained ocean-product generation.

The numerical value of the LLM backbone itself remains contested. Tan et al. [11] systematically ablated several LLM-based time-series forecasting methods and found that removing the language-model component or replacing it with simpler attention or Transformer blocks generally did not degrade forecasting performance and often improved it, despite substantially lower computational cost. This evidence is directly relevant to the claim boundary adopted here. LoRA [12] provides a parameter-efficient mechanism for adapting a pretrained model, but neither parameter efficiency nor an improvement over the corresponding pretrained Base model demonstrates that an LLM backbone is necessary for, or superior at, numerical Hs forecasting.

### 2.4 Positioning

Among the systems reviewed here, Zhai et al. [8] address numerical Hs prediction with the Chronos time-series foundation model, Henriques et al. [4] address machine-learning correction of operational numerical forecasts, and LLMTime and Time-LLM [9,10] primarily address generic time-series forecasting with text-pretrained language models. The present study evaluates a different target: whether an instruction-tuned model can produce a schema-constrained, machine-parseable 24-h Hs trajectory while a separate companion adapter produces wave-regime and predictability labels, with numerical skill reported alongside the structured-output capability. Relative to Zhai et al. [8], we do not claim that Mistral-LoRA is a better Hs forecaster; Chronos remains a first-class numerical comparator, subject to the configured-lead aggregation caveat in our pilot evaluation. Relative to the critique of Tan et al. [11], the Base-to-LoRA improvement observed here is interpreted as evidence that adaptation improves performance under the present instruction-and-schema protocol, not as evidence that an LLM backbone is necessary or numerically superior to specialized forecasters. The differentiator is therefore the structured ocean-output interface and its protocol-aware evaluation, rather than an RMSE-winning claim.

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

### 4.1 Schema-constrained instruction outputs

The primary task maps an issue-time history to a machine-readable JSON object containing the significant-wave-height forecast field `hs_forecast_m`. This field is defined as an ordered array of 24 numeric values corresponding to hourly forecast leads \(+1,\ldots,+24\) h. The same output schema is used when evaluating the pretrained Mistral Base model and the curve LoRA adapter so that adaptation is assessed under a common interface.

The companion classification task is handled by a separate adapter rather than by a joint multitask model. Its target schema contains `wave_regime` and `predictability_24h` labels together with textual note fields. Consequently, curve forecasting and companion classification constitute separate instruction-tuning tasks and their performance is evaluated separately.

For the curve task, JSON validity is a syntactic/schema-level criterion. An output is counted as valid when it can be parsed as JSON and satisfies the required curve schema, including a numeric `hs_forecast_m` array of length 24. JSON validity does not measure forecast accuracy, physical plausibility, probabilistic calibration, or the factual quality of textual fields. The reported validity rate therefore refers only to this parse-and-schema layer on the n = 24 curve-evaluation windows.

### 4.2 Issue-time windowing

Each forecast sample is anchored at an issue time \(t_0\). Historical input terminates at \(t_0\), and the prediction target comprises observed Hs at hourly leads \(t_0+1,\ldots,t_0+24\) h. Thus, the value at \(t_0\) belongs to the available history rather than to the forecast target.

All historical sequences and compressed summary features supplied to the instruction are constructed from information available no later than \(t_0\). Future observations within the \(+1\) to \(+24\) h verification interval are used only as targets for evaluation and are not included in the model input. This issue-time convention is applied consistently when constructing the reported curve windows.

### 4.3 History representation and prompt compression

The curve instruction retains the complete `history_hs_m` sequence because the recent evolution of Hs is the principal temporal signal for the forecasting task. Hs summary statistics are additionally supplied as compact contextual features. In contrast, the auxiliary peak-period (\(T_p\)) and wind channels are compressed rather than inserted as full historical sequences. For each auxiliary channel, the input contains its mean, trend, and most recent six values.

This representation was adopted to reduce instruction length while preserving the full Hs trajectory and selected recent and aggregate information from the auxiliary variables. The reported training configuration uses a maximum sequence length of 2048 tokens. Gradient checkpointing is enabled during training to reduce memory demand; it is a training-memory mechanism and does not alter the issue-time information available to the model.

### 4.4 LoRA adaptation

The base language model is Mistral-7B-Instruct-v0.3. The curve adapter is trained with Low-Rank Adaptation (LoRA) using 1024 training samples, with each target representing a 24-h hourly Hs trajectory under the schema defined in Section 4.1. The pretrained Base model and the adapted model are evaluated with the same curve-task instruction format and output schema.

The companion regime/predictability task uses a separate LoRA adapter trained against its own classification JSON targets. No joint curve-and-classification multitask objective is used. Accordingly, an improvement or degradation in a companion classification metric should not be interpreted as an effect of a jointly optimized curve-forecasting objective.

Only training settings verified for the reported experiment are specified here. Unverified LoRA or optimizer hyperparameters are not inferred from model defaults.

### 4.5 Evaluation metrics and lead alignment

For each valid 24-step curve output, predicted Hs values are aligned with observations at the corresponding hourly leads. Root-mean-square error (RMSE) and mean absolute error (MAE) are computed on the aligned curve, and the reported curve-level summaries aggregate these errors across the evaluation windows. Lead-wise RMSE is additionally calculated at each of the 24 forecast leads to characterize the dependence of error on forecast horizon. JSON validity is reported separately from these numerical error metrics. The companion adapter is evaluated using raw accuracy for `wave_regime` and `predictability_24h`.

Persistence is naturally defined across the complete 24-h horizon and can therefore be evaluated on the same dense hourly targets as the Mistral curves. LightGBM† and Chronos† require a different interpretation. On the curve-evaluation windows, their reported RMSE values are aggregated only over their configured numeric forecast leads. The dagger (†) explicitly denotes this lead-selection protocol. These values therefore share the reported evaluation-window context but are not dense 24-step RMSE estimates directly equivalent to the Mistral or Persistence curves.

The † results are retained as protocol-aware numerical references rather than as a basis for model-class ranking. In particular, differences involving LightGBM† or Chronos† should not be interpreted as evidence of superiority or inferiority under a fully lead-matched 24-step benchmark.

---

## 5. Experiments and results

### 5.1 Curve forecast (n = 24, 10 stations)

| Method | Mean RMSE (m) | Notes |
|--------|---------------|-------|
| Persistence | 0.688 | Dense 24-h windows |
| LightGBM† | 0.698 | Configured numeric leads (§4.5) |
| Chronos† | 0.951 | Configured numeric leads (§4.5) |
| Mistral Base | 1.271 | JSON valid = 1.0 |
| Mistral LoRA | 0.699 | JSON valid = 1.0 |

On this pilot evaluation, LoRA substantially improves over Base (1.271 → 0.699) and achieves an RMSE numerically close to Persistence (0.688) and LightGBM† (0.698). Chronos† shows a higher aggregated-lead RMSE under the configured-lead protocol; because † scoring is not dense 24-step, that gap is not interpreted as evidence that Chronos is an inferior model class. No RMSE superiority or statistical equivalence to Persistence is claimed.

Lead-wise RMSE profiles for Base and LoRA (Fig. 2) show that errors are generally larger at later forecast leads, particularly in the latter part of the 24 h horizon, although the lead-wise profiles are not strictly monotonic.

### 5.2 Classification companion (n = 24)

| Model | Regime acc. | Predictability acc. |
|-------|-------------|---------------------|
| Base | 0.042 | 0.375 |
| LoRA | 0.417 | 0.250 |

Regime accuracy increased from 0.042 to 0.417, whereas predictability accuracy decreased from 0.375 to 0.250. Given n = 24 and severe regime-label imbalance on the evaluation set (true labels dominated by a single regime class), these results are exploratory and do not establish robust classification skill. Because the companion adapter is trained separately from the curve adapter, the opposite directions of the two classification metrics are reported as observed outcomes rather than as evidence of a joint multitask trade-off.

### 5.3 Figures

Main figures summarize method-wise mean RMSE (Fig. 1), lead-dependent Base versus LoRA RMSE (Fig. 2), an example multi-method Hs trajectory panel (Fig. 3), and Base versus LoRA classification accuracy (Fig. 4).

---

## 6. Discussion

The contribution is best read as three layered forms of evidence rather than a single accuracy claim. First, schema-constrained generation: on the reported evaluation set, LoRA returns JSON curves with validity 1.0 under the shared interface. Second, numeric adaptation: Base→LoRA reduces mean RMSE from 1.271 to 0.699, recovering performance to a level numerically close to Persistence (0.688) without establishing superiority or statistical equivalence. Third, companion classification: raw regime accuracy rises under severe class imbalance, while predictability accuracy declines from 0.375 to 0.250 and must be retained as a negative result.

Relative to Chronos-for-Hs studies [8] and text-LLM time-series reprogramming [10], the present question is whether an Instruct LLM can serve as a structured ocean-output layer beside strong numeric models. The pilot evidence supports that interface role for JSON curves, not an operational decision-support benefit that has not been measured here. Relative to Tan et al. [11], the Base→LoRA improvement is evidence that adaptation helps under the present instruction-and-schema protocol; it is not evidence that an LLM backbone is necessary or numerically superior to specialized forecasters. Free-text note fields remain partially template-like and are not treated as calibrated explanations.

---

## 7. Limitations

1. **Pilot scale.** Curve and classification evaluations use **n = 24** windows, which is insufficient for operational certification or strong ranking among near-tied methods (Persist 0.688 vs LoRA 0.699).
2. **Label imbalance.** True regime labels are heavily skewed toward a single class, so raw regime accuracy is exploratory rather than evidence of robust classification skill.
3. **Lead aggregation asymmetry.** LightGBM†/Chronos† comparisons use configured leads rather than dense 24-step curves (§4.5); model-class ranking under a fully lead-matched protocol is therefore unsupported.
4. **Textual rationales.** Free-text fields are not validated for factual correctness or calibrated uncertainty.
5. **Compute / deployment.** Controlled deployment cost benchmarks against Persistence and LightGBM were not performed; 7B Instruct LoRA remains a heavier model class than tabular or persistence baselines.
6. **Incomplete hyperparameter disclosure.** Only verified training settings are stated in Section 4; remaining adapter hyperparameters should accompany the final reproducibility package.

---

## 8. Conclusions

We present an NDBC→JSON→Mistral-LoRA study for structured Hs curve generation and companion regime/predictability labeling. On the curve evaluation (**n = 24**, **10** stations), LoRA improves mean RMSE from 1.271 to 0.699 with JSON validity 1.0, remaining numerically close to Persistence/LightGBM† while making no RMSE-superiority claim. Future work should enlarge temporally blocked test sets, balance regimes, and either drop or rigorously evaluate free-text rationale fields.

---

## Data availability

The NDBC observations used in this study are publicly accessible through the U.S. National Oceanic and Atmospheric Administration National Data Buoy Center (NOAA/NDBC). The code, frozen evaluation metrics, and figure assets supporting the reported analyses are publicly available in the companion GitHub repository: https://github.com/Coucou2016/wave-prediction-mistral-tuning. Derived intermediate data panels and model weights are not redistributed with the manuscript package; the repository documents the data-preparation and model-generation procedures required to reconstruct the reported workflow.

---

## References

1. Fan et al., *Ocean Engineering*, 2020. https://doi.org/10.1016/j.oceaneng.2020.107298  
2. Domala et al., *JCDE*, 2022. https://doi.org/10.1093/jcde/qwac048  
3. Chaichitehrani et al., *AIES*, 2024. https://doi.org/10.1175/AIES-D-23-0061.1  
4. Henriques, M.R., Silva, D., Yanchin, I., Latas, M., Guedes Soares, C., Improving the forecast of wind speed and significant wave height using neural networks and gradient boosting trees. *Ocean Engineering* 327, 120925 (2025). https://doi.org/10.1016/j.oceaneng.2025.120925  
5. Nie, Y., Nguyen, N.H., Sinthong, P., Kalagnanam, J., A time series is worth 64 words: Long-term forecasting with Transformers (PatchTST). ICLR 2023. https://arxiv.org/abs/2211.14730  
6. Ansari, A.F., et al., Chronos: Learning the language of time series. *Transactions on Machine Learning Research*, 2024. https://doi.org/10.48550/arXiv.2403.07815  
7. Garza, A., Mergenthaler-Canseco, M., TimeGPT-1. 2023. https://doi.org/10.48550/arXiv.2310.03589  
8. Zhai, Y., Shi, H., Zhan, C., Wang, Q., You, Z., Wang, N., Improving significant wave height prediction using Chronos model. *Ocean Engineering* 341 (Part 2), 122502 (2025). https://doi.org/10.1016/j.oceaneng.2025.122502  
9. Gruver, N., Finzi, M., Qiu, S., Wilson, A.G., Large language models are zero-shot time series forecasters (LLMTime). NeurIPS 2023. https://arxiv.org/abs/2310.07820  
10. Jin, M., et al., Time-LLM: Time series forecasting by reprogramming large language models. ICLR 2024.  
11. Tan, M., Merrill, M.A., Gupta, V., Althoff, T., Hartvigsen, T., Are language models actually useful for time series forecasting? NeurIPS 2024. https://doi.org/10.52202/079017-1922  
12. Hu, E.J., et al., LoRA: Low-rank adaptation of large language models. ICLR 2022. https://doi.org/10.48550/arXiv.2106.09685  
