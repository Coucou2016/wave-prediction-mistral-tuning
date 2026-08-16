# ChatGPT session — paper refine (本回合)

**Protocol:** ChatGPT = advisor; local Cursor agent = executor. No file uploads — only public GitHub Markdown URLs.  
**Repo:** https://github.com/Coucou2016/wave-prediction-mistral-tuning  
**Thread:** https://chatgpt.com/c/6a81efce-fc5c-83ea-a615-54eae70ac13e  
**Started:** 2026-08-17  
**Model UI:** 极高 + web search  

Prior paper-adviser thread (context only): https://chatgpt.com/c/6a819cc9-f5b8-83ea-87fb-876a79e63d01  

## Round ledger

| Round | Topic | Brief URL | ChatGPT reply | Local verdict | Landed |
|-------|-------|-----------|---------------|---------------|--------|
| R1 | Authenticity audit vs SSOT | [R01](https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/chatgpt_rounds/R01_authenticity_audit.md) | Full CRITICAL/MAJOR/MINOR/ADOPT/REJECT | **ADOPT** claim-boundary fixes; Chronos† demotion; non-monotonic lead wording; no trade-off causal claim; strip engineering headers; Soften row count | manuscript v3.0 |
| R2 | Abstract + Introduction | [R02](https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/chatgpt_rounds/R02_abstract_intro.md) | pending | pending | pending |
| R3 | Related Work + Positioning | pending | pending | pending | pending |
| R4 | Methods (JSON / issue-time / †) | pending | pending | pending | pending |
| R5 | Results + Discussion + Limitations | pending | pending | pending | pending |
| R6 | Captions / Data Availability / OE fit | optional | — | — | — |

## R1 independent verdict notes

- ADOPT ChatGPT’s Chronos† / monotonic RMSE / trade-off / regime-skill / fair→protocol-aware / OE abstract ≤200 / hygiene points.
- REJECT inventing new numbers or dropping predictability negative result.
- Compression recipe verified in `src/wave_llm/nxt/curve_jsonl_export.py` (mean/trend/last_6).

## Rules

- Independent ADOPT/REJECT after each ChatGPT reply.
- Paper must not contain research-log / tooling traces.
- Process detail → `docs/research_log.md` / this file.
