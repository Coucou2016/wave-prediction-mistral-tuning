# Literature review notes (verified)

**Date:** 2026-08-16  
**ChatGPT conversation (web search ON):** https://chatgpt.com/c/6a812828-a690-83ea-a218-25721d148a25  
**nature-skills:** `nature-academic-search` workflow = multi-source-search + citation-verification (MCP academic-search unavailable in this session → WebSearch/WebFetch + CrossRef/arXiv/DOI URLs).  
**nature-writing axes for framing:** `task=manuscript`, `paper_type=methods`, `journal=generic` (primary venue Ocean Engineering / Applied Ocean Research), `language=zh-to-en` notes → EN outline.

**Verification policy:** Only entries with reachable DOI / conference proceedings / publisher page are marked **已核验**. ChatGPT suggestions not independently confirmed are listed under **未核验 / 剔除**.

---

## A. Time-series foundation models & LLM-for-TS (已核验)

| # | Citation | Venue / year | DOI / URL | One-line contribution | Relation to this work | Status |
|---|----------|--------------|-----------|----------------------|----------------------|--------|
| 1 | Ansari et al., *Chronos: Learning the Language of Time Series* | TMLR / arXiv 2024 | https://doi.org/10.48550/arXiv.2403.07815 ; OpenReview | Pretrained T5-style TS foundation model via quantization + CE loss; strong zero-shot probabilistic forecasts | Direct numeric baseline in our pipeline (`amazon/chronos-t5-*`) | 已核验 |
| 2 | Garza & Mergenthaler-Canseco, *TimeGPT-1* | arXiv 2023/2024 | https://doi.org/10.48550/arXiv.2310.03589 | Foundation TS transformer claiming broad zero-shot forecasting | Related-work for foundation-model framing; not used as our runtime baseline | 已核验 (preprint) |
| 3 | Nie et al., *A Time Series is Worth 64 Words* (PatchTST) | ICLR 2023 | https://arxiv.org/abs/2211.14730 ; OpenReview ICLR | Patching + channel-independent Transformer for long-term forecasting | Config mentions PatchTST; architecture contrast to text-LLM curve generation | 已核验 |
| 4 | Jin et al., *Time-LLM* | ICLR 2024 | https://proceedings.iclr.cc/paper_files/paper/2024/hash/680b2a8135b9c71278a09cafb605869e-Abstract-Conference.html | Reprogram frozen LLM with text prototypes + Prompt-as-Prefix for TS forecasting | Closest LLM-reprogramming line; we instead LoRA-instruct Mistral to emit **JSON Hs arrays + text reason** | 已核验 |
| 5 | Gruver et al., *LLMTime* | NeurIPS 2023 | https://arxiv.org/abs/2310.07820 ; NeurIPS proceedings | Zero-shot LLM forecasting by digit-string next-token prediction | Motivates text encoding of numbers; our Instruct+LoRA path is supervised domain adaptation | 已核验 |
| 6 | Tan / BennyTMT et al., *Are Language Models Actually Useful for Time Series Forecasting?* | NeurIPS 2024 | https://doi.org/10.48550/arXiv.2406.16964 | Ablations: removing LLM often matches/beats LLM-TS methods; compute much higher | **Critical for honest novelty:** do not claim RMSE SOTA from LLM; claim structured multi-task outputs | 已核验 |
| 7 | Hu et al., *LoRA* | ICLR 2022 | https://doi.org/10.48550/arXiv.2106.09685 | Low-rank adapters for PEFT of LLMs | Training method for our Mistral adapters | 已核验 |

## B. Significant wave height / ocean ML (已核验)

| # | Citation | Venue / year | DOI / URL | One-line contribution | Relation to this work | Status |
|---|----------|--------------|-----------|----------------------|----------------------|--------|
| 8 | Fan, Xiao & Dong | *Ocean Engineering* 2020, 205:107298 | https://doi.org/10.1016/j.oceaneng.2020.107298 | LSTM Hs prediction vs classical ML; SWAN-LSTM hybrid | Classic Hs ML baseline narrative for Ocean Engineering architecture | 已核验 |
| 9 | Domala, Lee & Kim | *J. Computational Design and Engineering* 2022 | https://doi.org/10.1093/jcde/qwac048 | NDBC multi-station wave ML/DL (Prophet, RF, GB, XGB) | NDBC panel + ensemble ML context for our LightGBM baseline | 已核验 |
| 10 | Chaichitehrani, He & Allahdadi | *Artificial Intelligence for the Earth Systems* 2024 | https://doi.org/10.1175/AIES-D-23-0061.1 | Stacking ensemble for Hs & period on US East Coast NDBC | Strong “fair numeric baselines matter” comparator; AMS journal structure | 已核验 |
| 11 | Energy Conversion & Management: X 2024 (generalized ML Hs) | *Energy Conversion and Management: X* | https://doi.org/10.1016/j.ecmx.2024.100623 | ANN/SNN vs XGBoost/LightGBM for Hs from wind/atmosphere; generalization across stations | Supports LightGBM as serious competitor, not a strawman | 已核验 |
| 12 | Ocean Engineering 2025 residual-correction study | *Ocean Engineering* | https://doi.org/10.1016/j.oceaneng.2025.120925 | LightGBM/CatBoost correct ECMWF/GEFS wave forecast residuals | Positions ML as post-processor to NWP—useful Discussion contrast | 已核验 |

## C. Writing-architecture patterns worth imitating

### C1. Methods / algorithmic TS papers (Time-LLM, PatchTST, Chronos, LLMTime)

Typical spine:

1. **Abstract** — task → gap (modality / specialization) → method name → 1–2 quantitative claims → code link.  
2. **Introduction** — funnel: importance → prior specialized models → LLM/TS opportunity → **explicit contributions as 3 bullets**.  
3. **Related Work** — topic clusters (not chronological dump): classical TS; Transformer TS; LLM-for-TS.  
4. **Method** — figures of pipeline modules; equations for tokenization / reprogramming / loss.  
5. **Experiments** — datasets table → metrics → main table → ablations → efficiency.  
6. **Discussion / Conclusion** — when it fails; compute cost; future work.

**Contribution sentence pattern (imitate):**  
“We propose X that [does Y] while [keeping Z], and show [metric] under [protocol], without claiming [overclaim].”

**Figure budget:** 1 architecture schematic + 1–2 main result figures + ablation panels. Tables for multi-dataset RMSE/MAE.

### C2. Ocean Engineering–style Hs ML papers (Fan 2020; residual-correction OE 2025)

Typical spine (IMRaD-leaning):

1. Introduction (engineering motivation: navigation, coastal safety).  
2. Methodology (model + features + train/test split).  
3. Study sites / data (buoy IDs, QC).  
4. Results (lead-time RMSE/MAE/R² tables + time-series panels).  
5. Discussion (physical interpretability, storm cases).  
6. Conclusions.

**Figure budget:** station map; observed vs predicted series; error-by-lead; confusion/skill if classification exists.

### C3. What to copy for *our* paper

| Element | Copy from | Adapt for us |
|---------|-----------|--------------|
| Contribution bullets | Time-LLM / Chronos | Emphasize **JSON schema + regime/predictability + reason**, not SOTA RMSE |
| Fair baselines table | PatchTST / OE Hs papers | Persistence, LightGBM, Chronos, Mistral Base vs LoRA on **same windows** |
| Critical self-position | NeurIPS 2024 “Are LLMs useful…” | Explicitly agree: numeric RMSE may lag; value is **structured + textual rationale** (not unproven “explainability”) |
| Data transparency | Domala / Chaichitehrani | List NDBC IDs, resample rule, split seed |
| Pilot honesty | — | Report small `n` (curve **n=24**, 10 stations; class n=24). Older n=24 (v2; older pilot superseded) pilot superseded by `paper/metrics/` v2. |

---

## D. Target journal structure differences (for venue choice)

| Venue | Structure bias | Fit for this work | Risk |
|-------|----------------|-------------------|------|
| **Ocean Engineering** (Elsevier) | Classical IMRaD; engineering motivation; buoy case studies; RMSE tables | **Primary recommendation** for Hs + operational framing | Must not oversell LLM vs LGBM |
| **Applied Ocean Research** | Similar to OE; slightly more applied coastal | Good alternate if OE desk-reject | Same |
| **Coastal Engineering** | Process/morphodynamics heavy | Weaker unless coastal impact emphasized | Scope mismatch |
| **JGR: Oceans** | Geophysical process + statistics | Needs stronger physical diagnostics | Hard for pure ML methods |
| **Nature Communications / Nat Mach Intell** | Broad significance; high novelty bar; short main text + SI | Only if multi-ocean scale + clear “why LLM language interface matters” for geophysics | Pilot sample too small; RMSE story weak |

**Recommendation:** Draft for **Ocean Engineering** (methods + applications). Keep a Nat-family **argument skeleton** via nature-writing (claim–evidence–boundary) but do not submit to Nat Commun until evaluation scale grows.

---

## E. Innovation positioning (cross-checked with ChatGPT web-search themes)

ChatGPT (search ON) correctly stressed: literature **does not** support “LLM beats specialized TS models on RMSE.” Independent verification of Tan et al. (NeurIPS 2024) confirms this.

**Safe novelty claims (supported by local metrics):**

1. **Structured generation:** Instruct LoRA emits parseable `hs_forecast_m` JSON with 100% JSON validity on pilot eval (`curve_metrics_*.json`).  
2. **Multi-output oceanology language:** same LLM family covers **regime classification** + **predictability labels** + free-text **reason** (`metrics_base/lora.json`).  
3. **Fair triad comparison:** persistence / LightGBM / Chronos / Mistral Base→LoRA on shared holdout windows—LoRA curve RMSE improves vs Base (1.271→0.699) but remains **near** Chronos/LGBM and **above** persistence mean RMSE (0.688).

**Unsafe claims (do not write):** “first LLM for waves”; “outperforms Chronos/LGBM”; “production-ready storm forecasting.”

---

## F. ChatGPT-sourced items — verification outcome

| ChatGPT claim | Our check | Outcome |
|---------------|-----------|---------|
| Chronos / TimeGPT / PatchTST / Time-LLM / LLMTime | DOI/proceedings confirmed | **Retain (已核验)** |
| Domala et al. 2022 NDBC ML | DOI 10.1093/jcde/qwac048 | **Retain** |
| Chaichitehrani et al. 2024 | DOI 10.1175/AIES-D-23-0061.1 | **Retain** |
| Follow-up papers “questioning LLM numerical robustness” | Matched to Tan et al. NeurIPS 2024 | **Retain** |
| OE Abstract ≤250 words / highlights advice | Plausible for Elsevier OE; not re-fetched from author guide in this pass | **部分核验** — confirm on Elsevier OE Guide for Authors before submission |

### B2. Additional ChatGPT-suggested items (post-hoc verified)

| # | Citation | Venue / year | DOI / URL | One-line + relation | Status |
|---|----------|--------------|-----------|---------------------|--------|
| 13 | Li et al., *Orca* | CIKM 2024 | https://doi.org/10.1145/3627673.3679973 | LLM + spatiotemporal encoding for buoy→grid SWH estimation | 已核验 — **blocks “first LLM+SWH” claim**; differentiate: we do **point-forecast trajectory JSON**, not grid estimation |
| 14 | Zhai et al., Chronos for SWH | *Ocean Engineering* 2025, 122502 | https://doi.org/10.1016/j.oceaneng.2025.122502 | Chronos applied to SWH vs PatchTST etc. | 已核验 — **must cite in OE submission**; answer what Mistral structured generation adds beyond Chronos-SWH |
| 15 | Bodnar et al., *Aurora* | *Nature* 2025 | https://doi.org/10.1038/s41586-025-09005-y | Earth-system FM incl. ocean waves at global scale | 已核验 — raises bar for Nat-family; context only for OE |
| 16 | Pirhooshyaran & Snyder | *Ocean Engineering* 2020, 207:107424 | https://doi.org/10.1016/j.oceaneng.2020.107424 | RNN/Seq2Seq multivariate wave forecast | 已核验 |
| 17 | James, Zhang & O'Donncha | *Coastal Engineering* 2018, 137 | https://doi.org/10.1016/j.coastaleng.2018.03.004 | Early ML surrogate for wave conditions | 已核验 |
| 18 | Tian et al., hybrid AI Hs (SCS) | *Ocean Engineering* 2026, 125271 | https://doi.org/10.1016/j.oceaneng.2026.125271 | Recent OE Hs-AI bar (extended-range SCS) | 已核验 (context) |

**Counts:** 已核验 **18** listed above.  
**未核验 / 剔除:** ChatGPT-mentioned Kang et al. Frontiers Geoformer (not fully cross-read here); Acta Oceanologica Sinica Wang 2023 (DOI seen in search but not double-checked in this pass)—exclude from claim-critical cites until re-verified. Any ChatGPT DOI not listed → default reject.

### ChatGPT architecture advice retained (after our verification)

- Prefer OE IMRaD with Data before Methods; Results by **research question**, not by model.  
- Must answer reviewers: *beyond Chronos-SWH (Zhai 2025) and Orca (Li 2024), what does Instruct+JSON trajectory add?*  
- Wording guardrails: use **textual rationale** / **uncertainty descriptor**, not “explainable” / “calibrated UQ” without metrics.  
- Demote predictability accuracy from headline (pilot LoRA worse than Base).

---

## G. Local experiment numbers (authoritative; not from ChatGPT)

Source files only:

- `data/processed/metrics/numeric_baselines.json`
- `data/processed/mistral/curve_metrics_base.json`, `curve_metrics_lora.json`, `curve_compare_base_lora.json`
- `data/processed/mistral/metrics_base.json`, `metrics_lora.json`, `compare_base_lora.json`
