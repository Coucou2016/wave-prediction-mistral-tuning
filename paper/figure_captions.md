# Figure captions (SciencePlots assets)

**Source of numbers:** `paper/metrics/` (n=24 curve eval; 10 stations). † = LightGBM/Chronos at configured numeric leads only.

**Fig. 1 — `curve_method_rmse_summary.png`.** Mean RMSE on the held-out curve subset (n = 24). Persistence (0.688), LightGBM† (0.698), Chronos† (0.951), Mistral Base (1.271), and Mistral LoRA (0.699). LoRA improves over Base but does not beat Persistence on this pilot set.

**Fig. 2 — `model_rmse_comparison_by_lead.png` / `curve_rmse_by_lead_lines.png`.** Hourly lead-dependent RMSE for Mistral Base versus LoRA over the 24 h horizon. Errors generally increase with lead, consistent with decaying predictability.

**Fig. 3 — `forecast_panel_mistral_lora_41010.png`.** Example multi-method Hs trajectories at station 41010: history, truth, Persistence, Mistral Base/LoRA curves, and optional LightGBM†/Chronos† markers at available leads.

**Fig. 4 — `classification_base_vs_lora.png`.** Companion classification accuracy (n = 24): regime improves (0.042 → 0.417) while predictability decreases (0.375 → 0.250), showing task-dependent LoRA effects.
