# Research log (engineering / process — not for manuscript)

Separated from `paper/manuscript.md` so the paper stays free of tooling traces.

## ChatGPT paper-refine thread (2026-08-17)

- URL: https://chatgpt.com/c/6a81efce-fc5c-83ea-a615-54eae70ac13e  
- Title: 学术论文真实性审计 / 论文精修  
- Protocol: public GitHub Markdown URLs only; no uploads.

## R1 authenticity landing (2026-08-17)

- Softened Chronos† wording; fixed non-monotonic lead RMSE language.
- Classification: observation-only; severe imbalance caveat.
- Removed ChatGPT/nature-writing/script path headers from manuscript.
- Softened unverified \(1.6\times10^5\) row count; kept 10 station IDs.
- Sanitized `curve_lora_meta_v2.json` absolute Windows path.
- Compression mean/trend/last-6 verified against `src/wave_llm/nxt/curve_jsonl_export.py`.

## Local SSOT

- `paper/metrics/*.json`  
- Figures: `paper/figures/` via SciencePlots scripts under `scripts/`.

## Prior adviser thread

- https://chatgpt.com/c/6a819cc9-f5b8-83ea-87fb-876a79e63d01
