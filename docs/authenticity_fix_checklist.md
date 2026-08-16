# Authenticity / claim-fix checklist (paper refine)

Last updated: 2026-08-17 (R1–R6 landed). SSOT = `paper/metrics/`.

## Fixed or enforced

| ID | Issue | Status |
|----|-------|--------|
| A1 | No claim that LoRA RMSE beats Persistence / LightGBM | **Fixed** |
| A2 | † lead aggregation for LightGBM/Chronos | **Fixed** |
| A3 | Predictability drop 0.375→0.250 reported as negative | **Fixed** |
| A4 | n=24 pilot scale in Limitations | **Fixed** |
| A5 | Sanitize `curve_lora_meta_v2.json` local Windows path | **Fixed** |
| A6 | Remove ChatGPT/Cursor process notes from manuscript | **Fixed** |
| A7 | Remove script filenames / local paths from manuscript | **Fixed** |
| A8 | Do not mix `numeric_baselines.json` with curve-subset means | **Fixed** |
| A9 | Chronos† class-inferior claim | **Fixed** |
| A10 | Monotonic “RMSE grows with lead” | **Fixed** |
| A11 | Multi-label “trade-off” causal claim | **Fixed** |
| A12 | Regime skill overclaim under imbalance | **Fixed** |
| A13 | Abstract length (project ≤200; OE official ≤250) | **Fixed** (~161 words) |
| A14 | Caption “decaying predictability” conflation | **Fixed** (R6) |
| A15 | Refs [4]/[8] placeholder titles | **Fixed** |

## Open (submission polish)

| ID | Issue | Action |
|----|-------|--------|
| B1 | Exact hourly row count | Softened; optional panel stats later |
| B2 | Full author lists for older refs [1]–[3] | Pre-submission bibliography pass |
| B3 | Remaining LoRA hyperparameters | Release with reproducibility package |
| B4 | Archival DOI | Optional Zenodo/etc. |

## Numbers freeze (do not invent)

- Curve: Base 1.271 → LoRA 0.699; Persist 0.688; LGBM† 0.698; Chronos† 0.951; n=24; 10 stations; JSON valid 1.0
- Class: Regime 0.042→0.417; Predictability 0.375→0.250; n=24
- Train: 1024 samples; horizon 24 h
