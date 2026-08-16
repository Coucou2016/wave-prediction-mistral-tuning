# Writing notes — manuscript draft decisions

**Date:** 2026-08-16  
**Axes (nature-writing):** `task=manuscript` · `paper_type=methods` · `journal=generic` (target *Ocean Engineering*) · `language=en` · sections: full IMRaD scaffold  
**One-sentence argument:** In NDBC buoy significant-wave-height forecasting, instruction-tuned Mistral-7B-Instruct-v0.3 with LoRA can emit parseable JSON `hs_forecast_m` curves together with sea-state regime / predictability labels and textual rationale fields, with Base→LoRA gains and fair Persistence / LightGBM / Chronos baselines, without claiming RMSE superiority over specialized numeric forecasters.

## Terminology ledger (locked)

| Canonical term | First-use | Notes |
|----------------|-----------|-------|
| significant wave height (Hs) | spell out once | metres |
| NDBC | National Data Buoy Center | |
| Mistral-7B-Instruct-v0.3 | full id once | then Mistral Instruct / Base |
| LoRA | Low-Rank Adaptation | Hu et al. |
| `hs_forecast_m` | JSON array field | hourly Hs forecast |
| `wave_regime` | six-class label | calm_stable … storm_decay |
| `predictability_24h` | high \| medium \| low | from LGBM vs persistence skill |
| `uncertainty_level` | high \| medium \| low | **qualitative descriptor only** |
| `reason` / `notes` | free-text field | **not** formal explainability |
| Persistence | last-observation forecast | |
| LightGBM | gradient boosting baseline | |
| Chronos-T5 | Amazon Chronos tiny | zero-shot TS FM |
| RMSE / MAE | root/mean absolute error | |
| skill vs persistence | 1 − RMSE_model/RMSE_persist | |

## Critical honesty gates (do not dilute)

1. **RMSE boundary:** LoRA mean curve RMSE **0.977** improves on Base **1.510** but does **not** beat Persistence **0.884** or Chronos **0.965**; ≈ LightGBM **0.977** on the same pilot windows (n=12).
2. **Predictability regression:** LoRA predictability accuracy **0.250** < Base **0.375** (n=24). Report in Results; demote from Abstract headline.
3. **Rationale wording:** Prefer *textual rationale* / *uncertainty descriptor*. Avoid *explainable AI* and *calibrated UQ* without coverage/CRPS.
4. **Training-label audit (2026-08-16):** Curve JSONL `reason` strings and classification `notes` appear **template stubs** (e.g. single repeated English/Chinese sentence; `uncertainty_level` often constant `medium`). Manuscript must **not** claim diverse supervised rationale learning. Schema + parseability are the supported claims; qualitative reason fidelity is future work.
5. **Differentiation:** vs Orca (buoy→grid estimation); vs Zhai Chronos-SWH (numeric Chronos optimization). We study Instruct LLM as structured companion *beside* Chronos.
6. **Sample size:** Curve eval n=12; classification n=24. Pilot language throughout.

## Intro variant chosen

**application-first** (OE engineering stake: navigation / coastal ops) → numeric ML → TS foundation models → LLM-for-TS + Tan et al. skepticism → gap (structured ocean JSON + multi-task language interface under fair baselines) → contributions.

## ChatGPT use this pass

None. Framing and wording risks already settled in `literature_review_notes.md` / existing thread. No new browser consult required for draft v1.

## Open placeholders for author fill-in

- Author list, affiliations, corresponding author, funding.
- Exact QC step prose if journal asks for more than pipeline-level description.
- Code/data DOI upon archival.
- Confirm Elsevier *Ocean Engineering* abstract word limit before submission (currently drafted ~230 words).
