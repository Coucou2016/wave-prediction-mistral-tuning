# Round 2 — Abstract + Introduction academic rewrite

**Role for ChatGPT:** Paper-refinement advisor. Read the updated manuscript (raw URL) and this brief. Propose a submission-grade Abstract (≤200 words for *Ocean Engineering*) and Introduction rewrite. No attachments. Do not invent numbers.

**ChatGPT thread:** https://chatgpt.com/c/6a81efce-fc5c-83ea-a615-54eae70ac13e  

**URLs**
- This brief: https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/chatgpt_rounds/R02_abstract_intro.md  
- Manuscript v3.0 (post-R1 landing): https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/manuscript.md  

**nature-writing axes:** `task=manuscript` · `paper_type=methods` · `section=abstract,intro` · `journal=generic` (*Ocean Engineering*) · `language=en`

## Frozen numbers (immutable)

n=24; 10 stations; Base 1.271 → LoRA 0.699; Persist 0.688; LGBM† 0.698; Chronos† 0.951; Regime 0.042→0.417; Predictability 0.375→0.250; JSON valid 1.0. † = configured numeric-lead aggregation.

## R1 verdicts already landed (do not reopen)

- REJECT: Chronos weaker; RMSE monotonic growth; multi-label trade-off causal claim; fair→protocol-aware; engineering headers removed.
- ADOPT: numerically close (not equivalent); opposite-direction classification observation; imbalance caveat; Abstract marks both LGBM† and Chronos†.

## Your tasks

1. Web-search *Ocean Engineering* Guide for Authors (abstract ≤200 words; highlights 3–5 × ≤85 chars) and mirror claim-boundary tone from recent OE Hs / Chronos papers if useful.
2. Deliver a **drop-in Abstract** (≤200 words) and **Introduction** (4–6 paragraphs) in English.
3. Soften remaining novelty overclaims (`rarely`, `under-explored`) if still present.
4. Keep companion ≠ joint multitask; keep negative predictability result.
5. End with ADOPT/REJECT list for the local executor (sentence-level if needed).

## Output format

```
ABSTRACT (<=200 words):
...

INTRODUCTION:
...

HIGHLIGHTS (optional, 3-5 lines, <=85 chars each):
...

ADOPT:
...
REJECT:
...
```
