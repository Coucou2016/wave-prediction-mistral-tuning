# Authenticity / claim-fix checklist (paper refine)

Last updated: 2026-08-17 (Round-1 prep). SSOT = `paper/metrics/`.

## Fixed or enforced

| ID | Issue | Status |
|----|-------|--------|
| A1 | No claim that LoRA RMSE beats Persistence / LightGBM | Enforced in draft (LoRA 0.699 > Persist 0.688) |
| A2 | † lead aggregation for LightGBM/Chronos | Enforced in tables + captions |
| A3 | Predictability drop 0.375→0.250 reported as negative | Enforced |
| A4 | n=24 pilot scale in Limitations | Enforced |
| A5 | Sanitize `curve_lora_meta_v2.json` local Windows path | **Fixed 2026-08-17** → HuggingFace model id |
| A6 | Remove ChatGPT/Cursor process notes from manuscript body | **In progress** (R1–R2) |
| A7 | Remove script filenames / local paths from manuscript | **In progress** |
| A8 | Do not mix `numeric_baselines.json` panel leads with curve-subset means | Enforced (numeric_baselines not cited as primary table) |

## Open / verify next

| ID | Issue | Action |
|----|-------|--------|
| B1 | Exact hourly row count \(1.6\times10^5\) | Soften or cite panel stats file if present |
| B2 | Station ID list reproducibility | Keep + Data Availability pointer to repo |
| B3 | Regime label skew magnitude | Keep qualitative Limitation until counts exported |
| B4 | Incomplete reference DOIs | R3 citation pass |
| B5 | Chronos “weaker” wording | Soften pending ChatGPT R1 advice |

## Numbers freeze (do not invent)

- Curve: Base 1.271 → LoRA 0.699; Persist 0.688; LGBM† 0.698; Chronos† 0.951; n=24; 10 stations; JSON valid 1.0
- Class: Regime 0.042→0.417; Predictability 0.375→0.250; n=24
- Train: 1024 samples; horizon 24 h
