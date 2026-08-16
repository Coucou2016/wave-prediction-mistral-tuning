# Round 5 — Results + Discussion + Limitations

**Role:** Academic adviser. Read manuscript §§5–7 + this brief. No uploads. Numbers immutable.

**URLs**
- Brief: https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/chatgpt_rounds/R05_results_discussion.md  
- Manuscript: https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/manuscript.md  
- Captions: https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/figure_captions.md  

## Frozen numbers

Curve n=24 / 10 stations: Base 1.271 → LoRA 0.699; Persist 0.688; LGBM† 0.698; Chronos† 0.951; JSON valid 1.0.  
Class n=24: Regime 0.042→0.417; Predictability **0.375→0.250** (negative; must keep).  
Lead RMSE: generally larger later; **not monotonic**.  
Regime labels heavily skewed (`storm_growth` dominant).

## Deliverables

1. Drop-in Results, Discussion, Limitations EN.  
2. Explicit: no RMSE supremacy; numerically close ≠ equivalent; Chronos† protocol-only; predictability negative result; imbalance.  
3. ADOPT/REJECT.

## Reject

Any “beats / SOTA / trade-off proves / Chronos weaker / fair apples-to-apples” language.
