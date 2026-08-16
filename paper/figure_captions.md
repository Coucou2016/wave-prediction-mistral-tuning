# Figure captions (submission-ready)

**Fig. 1.** Mean significant-wave-height (Hs) RMSE on the curve evaluation set (n = 24 forecast windows from 10 NDBC stations). Persistence, LightGBM†, Chronos†, Mistral Base, and Mistral LoRA yield RMSEs of 0.688, 0.698, 0.951, 1.271, and 0.699 m, respectively. LoRA substantially reduces error relative to Base but does not outperform Persistence. †LightGBM and Chronos values are aggregated at configured numeric forecast leads and are not dense 24-step RMSE estimates.

**Fig. 2.** Lead-wise RMSE of Mistral Base and Mistral LoRA over the 24-h Hs forecast horizon on the curve evaluation set (n = 24). Errors are generally larger at later forecast leads, particularly in the latter part of the horizon, although neither lead-wise profile is strictly monotonic.

**Fig. 3.** Example 24-h Hs forecast trajectories at NDBC station 41010, showing the observed history, verification observations, Persistence forecast, and Mistral Base and LoRA predictions. LightGBM† and Chronos† markers, where present, correspond only to their configured numeric forecast leads. The panel illustrates one forecast window and is not intended to represent performance across the full evaluation set.

**Fig. 4.** Raw companion-task classification accuracy for Mistral Base and LoRA (n = 24). Regime accuracy increases from 0.042 to 0.417 after adaptation, whereas predictability accuracy decreases from 0.375 to 0.250. Because the regime labels are strongly imbalanced, these results are exploratory and do not establish robust classification skill.
