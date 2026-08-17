# ChatGPT session — paper refine (本回合)

**Protocol:** ChatGPT = advisor; local Cursor agent = executor. No file uploads — only public GitHub Markdown URLs.  
**Repo:** https://github.com/Coucou2016/wave-prediction-mistral-tuning  
**Thread:** https://chatgpt.com/c/6a81efce-fc5c-83ea-a615-54eae70ac13e  
**UI title:** 论文真实性审计 / 论文精修  
**Started:** 2026-08-17  
**Model UI:** 极高 + web search  

Prior paper-adviser thread (context only): https://chatgpt.com/c/6a819cc9-f5b8-83ea-87fb-876a79e63d01  

## Round ledger (≥5 substantive rounds)

| Round | Topic | Brief (raw) | Verdict | Landed |
|-------|-------|-------------|---------|--------|
| **R1** | Authenticity audit vs SSOT | [R01](https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/chatgpt_rounds/R01_authenticity_audit.md) | **ADOPT** Chronos† demotion; non-monotonic lead wording; no trade-off causal claim; strip engineering headers; OE abstract limit; soft row count. **REJECT** RMSE supremacy / Chronos-weaker / equivalent-to-Persist | manuscript hygiene + claim fixes |
| **R2** | Abstract + Introduction | [R02](https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/chatgpt_rounds/R02_abstract_intro.md) | **ADOPT** 161-word Abstract; 5-para Intro; Highlights 4×≤85; “Among the systems reviewed here”. **REJECT** rarely/under-explored | Abstract/Intro/Highlights |
| **R3** | Related Work + Positioning | [R03](https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/chatgpt_rounds/R03_related_work.md) | **ADOPT** synthesis §2; Chronos=TSFM; Zhai/Henriques/Tan positioning; fix [8] singular title. **REJECT** Chronos-is-LLM / exhaustive novelty | §2 + refs [4][6][8][11] |
| **R4** | Methods (JSON / \(t_0\) / †) | [R04](https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/chatgpt_rounds/R04_methods.md) | **ADOPT** five-subsection Methods; issue-time leakage boundary; JSON=schema-only; † ≠ dense 24-step. **REJECT** script paths / guessed hyperparameters | §4 rewrite |
| **R5** | Results + Discussion + Limitations | [R05](https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/chatgpt_rounds/R05_results_discussion.md) | **ADOPT** three evidence layers; keep predictability negative; soften compute claim. **REJECT** trade-off / SOTA / Chronos weaker | §§5–7 |
| **R6** | Captions / Data Availability / OE fit | [R06](https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/chatgpt_rounds/R06_captions_data_oe.md) | **ADOPT** journal captions; Data Availability; OE framing = short-term wave env for marine ops. **REJECT** all-data-public / representative-panel overclaim | captions + Data Availability |
| **R7** | Innovation review | [R07](https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/chatgpt_rounds/R07_innovation_review.md) | **ADOPT** two-level contribution hierarchy; add Orca [13] + marine LLM [14]; "JSON syntax not novel"; forecast-product reframing; operational problem statement; remove sea-state labels from title; soften Discussion support→demonstrate. **REJECT** first-LLM claims; restructuring Methods/Data; moving companion to SI; adding many baselines; deleting predictability negative | Title, Abstract, §1, §2.3, §2.4, §6, §8, refs [13][14] |
| **R8** | Text expression polish | [R08](https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/chatgpt_rounds/R08_text_expression_polish.md) | **ADOPT** tighten §1 prose (remove "therefore"/"also" fillers); fix duplicate title; remove "Among the systems reviewed here" redundancy in §2.1; consistent "Hs prediction" (not "significant-wave-height prediction"); trim verbosity. **REJECT** any claim-level changes | §1, §2.1, title |

## Independent executor notes

- SSOT numbers never altered.
- Compression mean/trend/last-6 verified in `src/wave_llm/nxt/curve_jsonl_export.py`.
- Meta local Windows path sanitized in `paper/metrics/curve_lora_meta_v2.json`.
- Paper vs research-log separation: process → `docs/research_log.md` + this file.

## Key public URLs

- Manuscript: https://github.com/Coucou2016/wave-prediction-mistral-tuning/blob/main/paper/manuscript.md  
- Manuscript raw: https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/manuscript.md  
- Rounds folder: https://github.com/Coucou2016/wave-prediction-mistral-tuning/tree/main/paper/chatgpt_rounds  
- Metrics SSOT: https://github.com/Coucou2016/wave-prediction-mistral-tuning/tree/main/paper/metrics  
