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
| A6 | Remove ChatGPT/Cursor process notes from manuscript body | **Fixed** (v3.0) |
| A7 | Remove script filenames / local paths from manuscript | **Fixed** (v3.0) |
| A8 | Do not mix `numeric_baselines.json` panel leads with curve-subset means | Enforced |
| A9 | Chronos† “weaker” / class-inferior claim | **Fixed** (protocol-only wording) |
| A10 | Monotonic “RMSE grows with lead” | **Fixed** (non-monotonic caveat) |
| A11 | Multi-label “trade-off” causal claim | **Fixed** (observation-only) |
| A12 | Regime skill overclaim under imbalance | **Fixed** (exploratory + skew caveat) |

## Open / verify next

| ID | Issue | Action |
|----|-------|--------|
| B1 | Exact hourly row count | Softened in v3.0; optional panel stats later |
| B2 | Station ID list reproducibility | Keep + Data Availability |
| B3 | Regime label skew counts (23/24) | Stated qualitatively; export counts optional |
| B4 | Incomplete reference DOIs / author lists | R3 citation pass |
| B5 | Abstract ≤200 words (OE) | R2 polish |
| B6 | Highlights 3–5 × ≤85 chars | R6 / submission package |

## Numbers freeze (do not invent)

- Curve: Base 1.271 → LoRA 0.699; Persist 0.688; LGBM† 0.698; Chronos† 0.951; n=24; 10 stations; JSON valid 1.0
- Class: Regime 0.042→0.417; Predictability 0.375→0.250; n=24
- Train: 1024 samples; horizon 24 h
