# Round 1 — Authenticity audit brief (SSOT vs manuscript)

**Role for ChatGPT:** Paper-refinement advisor. Read this brief (and optionally the linked manuscript raw URL). Do **not** invent numbers. Flag exaggeration, inconsistency, or fabrication risk. Reply with: (1) CRITICAL issues, (2) MAJOR issues, (3) MINOR polish, (4) concrete ADOPT/REJECT recommendations for the local executor.

**Public repo:** https://github.com/Coucou2016/wave-prediction-mistral-tuning  
**Manuscript (blob):** https://github.com/Coucou2016/wave-prediction-mistral-tuning/blob/main/paper/manuscript.md  
**Manuscript (raw):** https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/manuscript.md  
**SSOT directory:** https://github.com/Coucou2016/wave-prediction-mistral-tuning/tree/main/paper/metrics  

**Constraint:** Paper must remain free of local paths, conda, script filenames, Cursor/ChatGPT process notes. Those belong only in `docs/`.

---

## Frozen SSOT (verified locally 2026-08-17; report to 3 decimals as in draft)

### Curve eval (`curve_compare_base_lora.json` / `curve_metrics_*.json`)

| Quantity | Exact JSON | Draft rounding |
|----------|------------|----------------|
| n | 24 | 24 |
| Base mean RMSE | 1.2710926958757351 | **1.271** |
| LoRA mean RMSE | 0.6990227680261466 | **0.699** |
| JSON valid rate (Base/LoRA) | 1.0 / 1.0 | 1.0 |
| Persistence mean RMSE | 0.6881267642914985 | **0.688** |
| LightGBM† mean RMSE | 0.6980680977053485 | **0.698** |
| Chronos† mean RMSE | 0.9507183400221131 | **0.951** |

† = aggregated at configured numeric leads on the same curve windows — **not** dense 24-step RMSE.

### Classification companion (`compare_base_lora.json`)

| Quantity | Exact | Draft |
|----------|-------|-------|
| n | 24 | 24 |
| Base regime acc. | 0.041666… | **0.042** |
| LoRA regime acc. | 0.416666… | **0.417** |
| Base predictability acc. | 0.375 | **0.375** |
| LoRA predictability acc. | 0.25 | **0.250** |

### Training meta (`curve_lora_meta_v2.json`)

- train_samples: **1024**
- horizon_hours: **24**
- base_model id: Mistral-7B-Instruct-v0.3 (local absolute path present in meta JSON — **must not appear in paper**)

---

## Pre-audit findings by local executor (please confirm / extend)

### CRITICAL (claim / SSOT risk)

1. **No RMSE supremacy claim allowed.** LoRA (0.699) ≈ LightGBM† (0.698) and **worse than** Persistence (0.688). Any wording implying “beats specialized forecasters” is false.
2. **† asymmetry must stay explicit** wherever LightGBM/Chronos appear beside dense 24-h LLM curves.
3. **Predictability accuracy declines** (0.375 → 0.250). Must remain a reported negative result; never framed as success.
4. **Pilot n = 24** is too small for operational ranking among near-tied methods; “pilot / exploratory” language required.
5. **Do not mix** `numeric_baselines.json` (panel leads 6/12/24/48/72) with curve-subset means in the same table without a clear protocol note.

### MAJOR (paper hygiene / consistency)

6. Manuscript header still contains **engineering / process notes** (ChatGPT ADOPT line, `paper/metrics/` paths, script names). These must move to `docs/` before submission tone.
7. Station count **10** and ID list appear in §3.1 — keep only if reproducible from released data docs; otherwise soften to “ten QC’d NDBC stations (IDs in Data Availability / repo)”.
8. “On the order of \(1.6\times10^5\) hourly rows” — verify against released panel stats or mark as approximate with source.
9. Classification eval regime labels are reportedly **skewed toward `storm_growth`** — accuracy inflation risk; keep in Limitations.
10. Free-text `reason`/`notes` must not be called “explainability” or calibrated uncertainty.

### MINOR

11. Rounding of Chronos 0.9507→0.951 and Persist 0.6881→0.688 is acceptable if stated as three decimals consistently.
12. References still incomplete / DOI placeholders for some entries — flag for R3 citation pass.
13. Meta JSON leaking Windows path is a **repo hygiene** issue (sanitize for public SSOT).

---

## Questions for ChatGPT (answer explicitly)

1. Which manuscript sentences currently overclaim relative to the SSOT table above?
2. What authenticity checklist items should we add that we missed?
3. For *Ocean Engineering* methods-style papers, what claim boundary language is standard when LLM RMSE ≈ Persistence?
4. Should we demote Chronos† “weaker” phrasing given † aggregation (avoid implying Chronos is inferior as a model class)?
5. Produce a short **ADOPT / REJECT** list for the executor’s next manuscript edit.

---

## Desired ChatGPT output format

```
CRITICAL:
- ...
MAJOR:
- ...
MINOR:
- ...
ADOPT:
- ...
REJECT:
- ...
```
