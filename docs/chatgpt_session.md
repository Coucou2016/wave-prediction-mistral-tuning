# ChatGPT research session log

- **URL:** https://chatgpt.com/c/6a812828-a690-83ea-a218-25721d148a25  
- **Title (UI):** LLM海洋波浪预报创新 / LLM 海洋预报文献  
- **Date:** 2026-08-16  
- **Mode:** Web search explicitly enabled（菜单「网页搜索」）; model UI showed「极高」.  
- **Attachments:** none (text-only, sanitized metrics).  
- **Problems resolved in-thread:** literature shortlist; writing architecture; journal ranking; innovation positioning; wording risks (reason/uncertainty); Orca & Chronos-SWH differentiation.  
- **Independent verification:** see `literature_review_notes.md` (WebSearch/DOI; nature-academic-search MCP unavailable → T1-style DOI/publisher checks).

## Manuscript drafting pass (2026-08-16, later)

- **ChatGPT consult:** none (framing already settled; no new browser session).  
- **Deliverable:** `docs/manuscript_draft.md` + `docs/writing_notes.md`.  
- **Extra honesty gate discovered locally:** JSONL `reason`/`notes`/`uncertainty_level` training targets are template stubs — reflected in Limitations.


## Curve LoRA v2 strategy consult (2026-08-16)

- **URL:** https://chatgpt.com/c/6a80a918-cd30-83ea-ab55-b28f4dfbdcfc
- **Title (UI):** 模型训练与评估建议
- **Attachments:** none (text CONTEXT only).
- **Ask:** compressed Tp/wind features; expand to 1024/200 at 24h; station round-robin; honesty vs Persist.
- **Advice (independent review applied):**
  - A) ADOPT compressed mean/trend/last_N (no 168× Tp/wind).
  - B) ADOPT 1024 samples / 200 steps + station round-robin as stage-2.
  - C) REJECT mandatory 6h-first; keep 24h + RMSE-by-lead.
  - D) MODIFY claim: LoRA beats Base and is on-par with numeric baselines; do **not** claim beat Persistence yet.
