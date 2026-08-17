# Round 11 — Data Presentation, References, and Journal Fit

**Role:** Data editor and journal-match reviewer.

**URLs**
- Manuscript (raw): https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/manuscript.md
- Metrics SSOT: https://github.com/Coucou2016/wave-prediction-mistral-tuning/tree/main/paper/metrics
- Prior rounds: https://github.com/Coucou2016/wave-prediction-mistral-tuning/tree/main/paper/chatgpt_rounds

## Instruction

### 1. Data Presentation Audit
- **† notation:** Is every instance of LightGBM† and Chronos† consistent? Is the dagger defined everywhere it appears?
- **Negative results:** Is the predictability accuracy decline (0.375→0.250) presented honestly throughout?
- **n values:** Is n=24 consistently stated? Is it clear this is n=24 forecast windows, not n=24 stations or n=24 hours?
- **Decimal consistency:** Are all RMSE/accuracy values reported to 3 decimal places consistently?
- **Station count:** Is "10 stations" consistent with the station list in §3.1?

### 2. References
- Complete reference [10] (Time-LLM) — currently missing DOI
- Check all DOI links are valid
- Check reference format consistency (some have full author lists, others "et al.")
- Check that all citations in text appear in References and vice versa
- Ocean Engineering typically uses numbered references [1] style — is this followed?

### 3. Data Availability
- Does the Data Availability statement meet Ocean Engineering requirements?
- Is the GitHub repo link correct?
- Does it clearly state what IS and IS NOT available?

### 4. Journal Fit (Ocean Engineering)
- Does the paper's framing match OE's scope (marine structures, ocean environment, coastal engineering)?
- Re-read the Abstract and Introduction through an OE reviewer's eyes
- Are "LLM" and "JSON" terms sufficiently explained for an ocean engineering audience?
- Is the "structured ocean-output interface" contribution framed as engineering-relevant?

### 5. Highlights
- Check Highlights ≤ 85 characters each (OE requirement)
- Check that all 4 highlights are substantive and not repetitive

### Desired Output Format
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