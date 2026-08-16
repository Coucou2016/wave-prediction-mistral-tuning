# ChatGPT research session log

## Paper / literature thread (THIS TURN — primary)

- **URL:** https://chatgpt.com/c/6a819cc9-f5b8-83ea-87fb-876a79e63d01  
- **Title (UI):** 论文顾问任务  
- **Date:** 2026-08-16  
- **Mode:** Web search requested explicitly; UI showed「正在搜索 github.com」during Round 1. Model「极高」.  
- **Attachments:** none (text-only + public GitHub URL).  
- **GitHub given to ChatGPT:** https://github.com/Coucou2016/wave-prediction-mistral-tuning  

### Rounds (≥5 planned)

| Round | Deliverable | Status | Adopt / Reject |
|-------|-------------|--------|----------------|
| 1 | Framework + literature shortlist + honest innovation | Started; rate-limited mid-reply after acknowledging structured-JSON claim boundary | **ADOPT** keep claim boundary (JSON interface + Base→LoRA + fair baselines; not RMSE supremacy). Continue after cooldown. |
| 2 | Related-work cluster synthesis | Pending cooldown | — |
| 3 | Methods wording (compression + LoRA recipe) | Pending | — |
| 4 | Results wording with frozen v2 metrics | Pending | — |
| 5 | Innovation / limitations / figure captions | Pending | — |

### Rate limit note
During Round 1 generation, ChatGPT showed modal「请求过于频繁」and temporarily blocked conversation history access. Operator waits and retries; no CAPTCHA/login failure.

---

## Prior related threads (context only; not this-turn primary)

- **URL:** https://chatgpt.com/c/6a812828-a690-83ea-a218-25721d148a25  
- **Title (UI):** LLM海洋波浪预报创新 / LLM 海洋预报文献  
- **Mode:** Web search ON. Literature shortlist + writing architecture.  
- **Independent verification:** `docs/literature_review_notes.md`

- **URL:** https://chatgpt.com/c/6a80a918-cd30-83ea-ab55-b28f4dfbdcfc  
- **Title (UI):** 模型训练与评估建议  
- **Ask:** compressed Tp/wind; expand samples; honesty vs Persist.  
- **Advice applied:** ADOPT compression + larger v2 train; MODIFY claim (Base↑, on-par numeric; do not claim beat Persistence).

---

## Local manuscript landing

- `paper/manuscript.md` — nature-writing methods draft with **v2 metrics** (n=24).  
- Figures: `paper/figures/` (SciencePlots). Metrics: `paper/metrics/`.
