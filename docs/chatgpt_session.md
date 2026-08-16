# ChatGPT research session log

## Paper / literature thread (THIS TURN — primary)

- **URL:** https://chatgpt.com/c/6a819cc9-f5b8-83ea-87fb-876a79e63d01  
- **Title (UI):** 论文顾问 海浪预测  
- **Date:** 2026-08-16  
- **Mode:** Web search used (UI citations to GitHub / ScienceDirect / arXiv / NeurIPS / ICLR). Model「极高」.  
- **Attachments:** none (text-only + public GitHub URL).  
- **GitHub:** https://github.com/Coucou2016/wave-prediction-mistral-tuning  

### Rounds

| Round | Deliverable | Status | Adopt / Reject |
|-------|-------------|--------|----------------|
| 1 | Framework + literature + honest innovation | Done (web search + repo read) | **ADOPT** schema-constrained product framing; split adaptation vs superiority; companion≠joint multitask; demote predictability from title; SSOT=`paper/metrics`; matched-lead caveat. **REJECT** RMSE-win narrative. |
| 2 | Related-work 3 clusters + EN paragraphs | Done | **ADOPT** Hs-ML / TSFM / LLM+skepticism spine; keep Zhai 2025 Chronos-Hs; **must-cite** Tan et al. 2024; Chronos = TS foundation model (not “LLM-TSFM”). |
| 3 | Methods EN paragraphs | Done | **ADOPT** layered json_valid checklist; compression+LoRA recipe; † lead-alignment wording. |
| 4 | Results wording with frozen v2 metrics | Done | **ADOPT** order: schema → adaptation → fair compare → lead → negative classification. |
| 5 | Innovation / limitations / figure captions | Done (streaming captions verified in UI) | **ADOPT** n=24 / † honesty in captions; mirror in `paper/figure_captions.md`. |

### Rate limit note
Transient modal「请求过于频繁」appeared mid-Round-1; recovered after wait. No CAPTCHA/login failure.

---

## Prior related threads (context)

- https://chatgpt.com/c/6a812828-a690-83ea-a218-25721d148a25 — literature shortlist (web search ON)  
- https://chatgpt.com/c/6a80a918-cd30-83ea-ab55-b28f4dfbdcfc — curve LoRA compression / honesty vs Persist  

---

## Local landing

- Manuscript: `paper/manuscript.md`  
- Figures: `paper/figures/` (SciencePlots + Times New Roman)  
- Metrics SSOT: `paper/metrics/`
