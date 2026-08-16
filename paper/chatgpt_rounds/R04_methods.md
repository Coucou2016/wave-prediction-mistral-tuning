# Round 4 — Methods (schema JSON, issue-time, compression, † alignment)

**Role:** Academic adviser. Read manuscript Methods + this brief. No uploads. Do not invent hyperparameters beyond what the brief/manuscript state.

**URLs**
- Brief: https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/chatgpt_rounds/R04_methods.md  
- Manuscript: https://raw.githubusercontent.com/Coucou2016/wave-prediction-mistral-tuning/main/paper/manuscript.md  

## Verified local facts (use these)

- Base model: Mistral-7B-Instruct-v0.3  
- Curve LoRA train samples: **1024**; horizon: **24 h**  
- Prompt compression: full `history_hs_m` + Hs stats + \(T_p\)/wind mean/trend/last-6 (export implementation verified)  
- max_seq_length = 2048; gradient checkpointing used in training recipe  
- Issue time \(t_0\): history ends at \(t_0\); forecast leads \(+1\ldots+24\) h  
- Companion classification = **separate adapter** (not joint multitask)  
- † protocol: LightGBM/Chronos RMSE on curve windows aggregated at configured numeric leads only  

## Deliverables

1. Drop-in §4 Methods EN (subsections: schema; issue-time windowing; compression; LoRA; metrics & † alignment).  
2. Short checklist language for JSON validity (parse/schema layer only on n=24).  
3. ADOPT/REJECT.  

## Reject

Script filenames, conda, local paths, Cursor/ChatGPT process notes.
