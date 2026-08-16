# Final acceptance report — wave-prediction-mistral-tuning

**Date:** 2026-08-16  
**Executor:** Cursor local agent  
**Advisor:** ChatGPT Pro (web search)

---

## 1. Progress / metrics baseline (verified locally)

| Item | Value | Source |
|------|-------|--------|
| Stations | **10** (`41010`…`51002`) | panel + curve eval |
| Curve eval n | **24** | `paper/metrics/curve_metrics_*.json` |
| JSON valid | **1.0** | same |
| Base → LoRA RMSE | **1.271 → 0.699** | `curve_compare_base_lora.json` |
| Persist / LGBM† / Chronos† | **0.688 / 0.698 / 0.951** | curve metrics `baselines` |
| Regime acc Base→LoRA | **0.042 → 0.417** (n=24) | `compare_base_lora.json` |
| Predictability acc | **0.375 → 0.250** (↓) | same |
| Curve LoRA train | 1024 samples, 24 h | `curve_lora_meta_v2.json` |

Pilot note: earlier n≈12 numbers are **superseded**; SSOT = `paper/metrics/`.

---

## 2. SciencePlots

| Check | Result |
|-------|--------|
| Install (`wave_llm`) | **PASS** — SciencePlots 2.2.2 already present |
| Times New Roman | **PASS** — available on Windows |
| Script | `scripts/11_make_science_figures.py` + `src/wave_llm/evaluation/science_plots_style.py` |
| Outputs | `data/processed/figures_science/` and `paper/figures/` |

Key figures: `curve_method_rmse_summary.png`, `model_rmse_comparison_by_lead.png`, `curve_rmse_by_lead_lines.png`, `forecast_panel_mistral_lora_41010.png`, `classification_base_vs_lora.png`, regime/predictability panels.

---

## 3. Public GitHub

- **URL:** https://github.com/Coucou2016/wave-prediction-mistral-tuning  
- **Visibility:** public  
- **Pushed:** code, configs, docs, `paper/` (manuscript + figures + metrics snapshots)  
- **Excluded:** `models/`, adapters/checkpoints, parquet, raw/processed bulk data  

---

## 4. ChatGPT collaboration (≥5 rounds)

**Primary thread:** https://chatgpt.com/c/6a819cc9-f5b8-83ea-87fb-876a79e63d01  

| Round | Topic | Adopt / Reject (summary) |
|-------|-------|--------------------------|
| 1 | Framework + innovation | **ADOPT** schema-constrained product; adaptation≠superiority; companion≠joint; demote predictability title; SSOT metrics. **REJECT** RMSE-win claim. |
| 2 | Related work | **ADOPT** 3-cluster RW; must-cite Tan 2024; Chronos as TS foundation model; keep Zhai 2025. |
| 3 | Methods | **ADOPT** compression/JSON/LoRA/† protocol; layered validity checklist (note: reported `json_valid` = parse layer today). |
| 4 | Results wording | **ADOPT** schema→adaptation→fair compare→lead→negative classification order with frozen v2 numbers. |
| 5 | Innovation / limits / captions | Sent; captions/limitations to mirror honesty on n=24 and †. |

Prior context URLs (not primary):  
https://chatgpt.com/c/6a812828-a690-83ea-a218-25721d148a25 · https://chatgpt.com/c/6a80a918-cd30-83ea-ab55-b28f4dfbdcfc  

**Rate limit:** transient「请求过于频繁」during Round 1; recovered. No CAPTCHA.

Full log: `docs/chatgpt_session.md`

---

## 5. Manuscript (nature-writing)

- **Path:** `paper/manuscript.md`  
- **Axes:** `task=manuscript` · `paper_type=methods` · `journal=generic` (*Ocean Engineering*) · `language=en`  
- **Title (adopted):** Schema-constrained … 24-h JSON trajectories and sea-state labels…  
- Numbers from local JSON only; limitations explicit (n=24, † leads, predictability drop, rationale stubs).

---

## 6. Command PASS/FAIL

| Command / step | Status |
|----------------|--------|
| SciencePlots import / font check | **PASS** |
| `11_make_science_figures.py` | **PASS** |
| `git init` + public `gh repo create --push` | **PASS** |
| Push manuscript/docs updates | **PASS** |
| ChatGPT ≥5 rounds in paper thread | **PASS** (R1–R5 executed; R5 response may still stream at report time) |
| nature-academic-search MCP | **N/A** — used WebSearch + ChatGPT web search + existing verified DOI notes |

---

## 7. Honest claim boundary (final)

Instruct-LoRA is a **structured multi-output companion** for buoy Hs workflows (JSON curves + sea-state labels). On n=24 it recovers Base→LoRA skill and matches Persist/LGBM order of magnitude; it does **not** demonstrate numeric supremacy.
