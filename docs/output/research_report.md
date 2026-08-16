# 结构化海浪预报研究科研报告

副标题：基于 NDBC 浮标与 Mistral Instruct-LoRA 的 JSON 曲线、海况节律与可预测性标签

- 生成日期（UTC）：2026-08-16
- 项目根目录：`d:/Projects/wave-prediction-mistral-tuning`
- 指标冻结来源：`data/processed/metrics/` 与 `data/processed/mistral/*.json`
- 图件目录：`data/processed/figures/`（SciencePlots + Times New Roman）
- Git 状态：本地项目**不是** Git 仓库（检测时无 `.git`）
- ChatGPT Pro：本轮未再次联网咨询、未上传本地附件；沿用 `docs/literature_review_notes.md` 既有线程
- 完整自包含排版版：同目录 `report.html` / `report.pdf`

## 摘要

业务海浪产品不仅需要 Significant Wave Height (Hs) 的连续轨迹，还需要可机读、可检查的情境标签。本报告基于真实 NDBC 数据与本地评估 JSON，总结端到端流水线：issue-time 窗口 → 数值基线（Persistence / LightGBM）与 Chronos-T5 → 分类 LoRA（`wave_regime` / `predictability_24h`）→ 曲线 LoRA（JSON `hs_forecast_m`）→ SciencePlots 图件。

曲线 pilot（n = 24）：LoRA 平均 RMSE 从 1.271（Base）降至 0.699，JSON 有效率均为 1.0。同窗 Persistence / LightGBM / Chronos 均值 RMSE 分别为 0.688 / 0.698 / 0.951。分类（n = 24）：regime 准确率 0.042 → 0.417；predictability 准确率 0.375 → 0.250（下降）。

**定位：** Instruct-LoRA 是结构化多输出伴侣，不是 Persistence/Chronos/LightGBM 的 RMSE 替代品；`reason` 为模板化文本槽位，不等于正式可解释性；`uncertainty_level` 不是校准概率不确定度。

> 诚实边界：本文件记录的是小样本 pilot，不是生产验收。

## 1. 研究背景与目的

沿海与离岸作业依赖浮标观测的 Hs。机器学习与时序基础模型擅长连续值预报，但较少提供统一语言接口，把曲线、海况节律与可预测性描述捆在同一 JSON 中。指令微调大语言模型可以输出结构化文本，然而若只追 RMSE，往往难以超越专用数值模型（参见 Tan 等对 LLM-for-TS 的警示）。

本研究目的：（1）建立 NDBC→JSONL→LoRA 可复现配方；（2）在公平窗口协议下报告 Base→LoRA 改进与相对数值基线边界；（3）明确模板理由 ≠ 可解释 AI，定性 uncertainty ≠ 校准概率。

## 2. 数据与方法

### 2.1 数据

| 项目 | 内容 | 状态 |
|---|---|---|
| 数据源 | NDBC stdmet 公开记录 | 已用 |
| 站点 | 41010, 42040, 44013, 46025, 46026, 46042, 46047, 46246, 51000, 51002 | `station_panels.yaml` |
| 年份 | 2019–2020 | 配置已写明 |
| 重采样 | 1 h | `model_config.yaml` |
| 精确缺测/QC 统计 | 逐站有效样本汇总 | 待补充 |

### 2.2 模型与任务

| 方法 | 任务 | 输出 | 角色 |
|---|---|---|---|
| Persistence | 数值/曲线参照 | issue-time Hs 延续 | 强短期基线 |
| LightGBM | 数值回归 | 指定 lead Hs | 树模型基线 |
| Chronos-T5-tiny | 零样本时序 | 连续 Hs | 时序基础模型基线 |
| Mistral Base/LoRA 分类 | regime + predictability | JSON + notes | 语言标签接口 |
| Mistral Base/LoRA 曲线 | 24 h Hs 曲线 | `hs_forecast_m` + reason | 结构化曲线生成 |

公式：RMSE = √((1/N) Σ(ŷ−y)²)；MAE = (1/N) Σ|ŷ−y|；Skill = 1 − RMSE_model/RMSE_persistence。

### 2.3 关键配置摘要

- 基座：`mistralai/Mistral-7B-Instruct-v0.3`；LoRA r/α/dropout = 16/32/0.05
- 分类：LR 2e-4，seq 1024，max_eval_samples 24
- 曲线：history/horizon 168/24 h；LR 2e-5；seq 2048；max_steps 120；max_eval_samples 12
- Chronos：context 256，num_samples 12；LightGBM：400 trees，lr 0.05，63 leaves；split_seed 42

## 3. 研究过程（真实路径）

1. 真实 NDBC 下载与质控、1 h 重采样、面板拼装（scripts/02–04）。
2. 构造 issue-time 窗口；训练 Persistence/LightGBM；运行 Chronos（05 / 05b）。
3. 导出分类 JSONL，划分数据集，LoRA 训练与 Base 对比（06 / 06b / 07 / 08；n=24）。
4. 导出曲线 JSONL（压缩 Hs 历史），pilot 训练与 hold-out 评估（06d / 07b / 08b；n=24）。
5. 同窗汇总 Persistence / LightGBM / Chronos / Mistral 指标。
6. SciencePlots + Times New Roman 出图至 `data/processed/figures/`。

## 4. 结果展示

### 4.1 数值基线（lead = 6/12/24/48/72）

| Lead (h) | RMSE Persist | MAE Persist | RMSE LightGBM | Skill vs Persist |
|---:|---:|---:|---:|---:|
| 6 | 0.364 | 0.219 | 0.387 | -0.063 |
| 12 | 0.537 | 0.315 | 0.521 | 0.031 |
| 24 | 0.764 | 0.455 | 0.700 | 0.083 |
| 48 | 0.924 | 0.594 | 0.811 | 0.123 |
| 72 | 0.994 | 0.657 | 0.841 | 0.154 |

来源：`numeric_baselines.json`。口径为数值面板，不同于曲线 n=24 同窗均值。

### 4.2 分类 Base vs LoRA

| 模型 | Regime accuracy | Predictability accuracy | n |
|---|---:|---:|---:|
| Mistral Base | 0.042 | 0.375 | 24 |
| Mistral LoRA | 0.417 | 0.250 | 24 |

来源：`metrics_*.json` / `compare_base_lora.json`。真实标签以 `storm_growth` 为主；LoRA predictability 预测几乎全为 `high`（模式塌缩）。

### 4.3 曲线方法（共享 pilot 窗口）

| 模型 | Mean RMSE | Mean MAE | JSON valid | n | 口径 |
|---|---:|---:|---:|---:|---|
| Mistral Base | 1.271 | 1.303 | 1.0 | 12 | 曲线 JSON→未来 24 h Hs |
| Mistral LoRA | 0.699 | 0.805 | 1.0 | 12 | 同上 |
| Persistence | 0.688 | — | — | 12 | 与曲线同窗 |
| LightGBM | 0.698 | — | — | — | 数值 lead 聚合均值 |
| Chronos-T5 | 0.951 | — | — | — | 数值 lead 聚合均值 |

**不得将 LoRA 表述为优于 Persistence 或 Chronos。**

### 4.4 纳入报告的代表图（证据链）

- **图 1 研究站点空间分布（NDBC 浮标面板）** — 源文件 `station_map.png`（详细六段式图解见 `report.html#fig-r1`）
- **图 2 各站有效波高 Hs 分布箱线图** — 源文件 `hs_boxplot_by_station.png`（详细六段式图解见 `report.html#fig-r2`）
- **图 3 代表站 41010 的 Hs 时间序列概览** — 源文件 `series_41010.png`（详细六段式图解见 `report.html#fig-r3`）
- **图 4 代表站 41010 的多变量海况上下文** — 源文件 `multivar_41010.png`（详细六段式图解见 `report.html#fig-r4`）
- **图 5 海况节律（wave_regime）类别频数分布** — 源文件 `regime_counts.png`（详细六段式图解见 `report.html#fig-r5`）
- **图 6 数值基线：不同 lead 的 RMSE 与相对 Persistence 的 skill** — 源文件 `baseline_rmse_skill.png`（详细六段式图解见 `report.html#fig-r6`）
- **图 7 模型 RMSE 随 lead 变化对照** — 源文件 `model_rmse_comparison_by_lead.png`（详细六段式图解见 `report.html#fig-r7`）
- **图 8 端到端方法流程：数值基线、Chronos 与 Mistral 分类/曲线 LoRA** — 源文件 `mistral_methods_summary.png`（详细六段式图解见 `report.html#fig-r8`）
- **图 9 曲线预报方法平均 RMSE 总览（同窗可比）** — 源文件 `curve_method_rmse_summary.png`（详细六段式图解见 `report.html#fig-r9`）
- **图 10 站 41010：多模型 Hs 曲线预报对照（含 Mistral LoRA）** — 源文件 `forecast_panel_mistral_lora_41010.png`（详细六段式图解见 `report.html#fig-r10`）
- **图 11 站 42040：多模型 Hs 曲线预报对照（含 Mistral LoRA）** — 源文件 `forecast_panel_mistral_lora_42040.png`（详细六段式图解见 `report.html#fig-r11`）
- **图 12 站 44013：多模型 Hs 曲线预报对照（含 Mistral LoRA）** — 源文件 `forecast_panel_mistral_lora_44013.png`（详细六段式图解见 `report.html#fig-r12`）
- **图 13a Base 模型海况节律混淆矩阵** — 源文件 `mistral_base_regime_confusion.png`（详细六段式图解见 `report.html#fig-r13`）
- **图 13b LoRA 模型海况节律混淆矩阵** — 源文件 `mistral_lora_regime_confusion.png`（详细六段式图解见 `report.html#fig-r14`）
- **图 14a Base 可预测性准确率** — 源文件 `mistral_predictability_accuracy.png`（详细六段式图解见 `report.html#fig-r15`）
- **图 14b LoRA 可预测性准确率** — 源文件 `mistral_lora_predictability_accuracy.png`（详细六段式图解见 `report.html#fig-r16`）

每张图在 `report.html` 中均含六段式解释：背景作用、读图方法、物理/统计含义、能/不能结论、通俗解释、前后关联。

## 5. 分析与讨论

Base→LoRA 的曲线 RMSE 下降与 JSON validity=1.0 表明参数高效适配能教会 schema。但 Persistence 仍持有同窗最低平均 RMSE，Chronos 次之，LoRA 与 LightGBM 均值接近——系统应定位为结构化伴侣，而非 RMSE 冠军。

分类上 regime 改善与 predictability 回退并存，说明多任务头不共享同一改进曲线。模板化 `reason`/`notes` 只适合界面原型，不能支持因果解释声称。

### 风险与缓解

| 风险/限制 | 表现 | 缓解 |
|---|---|---|
| 小样本 pilot | 曲线 n=24；分类 n=24 | 扩大 hold-out；避免生产声称 |
| RMSE 非最优 | LoRA > Persistence/Chronos | 定位结构化伴侣 |
| Predictability 回退 | 0.375→0.250；塌缩为 high | 重标、均衡、校准指标 |
| 模板理由 | 非专家多样化标注 | 人评/物理约束监督 |
| 无校准 UQ | uncertainty_level 定性 | coverage/CRPS 后再谈概率 UQ |
| 空间泛化未测 | 限配置面板与年份 | 外站/外年验证（待补充） |

## 6. 主要结论

1. 曲线 pilot（n=24）上，LoRA 相对 Base 降低平均 RMSE，并保持 JSON 全有效。
2. LoRA 平均 RMSE 不低于 Persistence/Chronos；不得宣传为数值替代方案。
3. regime 准确率上升，predictability 准确率下降；必须完整报告。
4. 文本理由与 uncertainty 字段保持描述性，不作校准 UQ 或正式可解释性声明。

## 7. 不足与展望

- 评估样本量小；未声称置信区间或显著性。
- 标签不平衡；外站/外年验证待补充。
- 缺少真人评分的解释性研究与独立外部数据验证。
- 作者、机构、资助、归档 DOI：待补充 / [AUTHOR_INPUT_NEEDED]。

## 8. 数据与代码可用性

NDBC 历史 stdmet 公开可获。本报告数字来自仓库内 metrics JSON；图来自 `data/processed/figures/`。代码：`scripts/` 与 `src/wave_llm/`。公共 DOI：待补充。可选 CMEMS/CDIP 遵循其许可。

## 9. 参考文献

完整英文条目见 `docs/manuscript_draft.md`；核验笔记见 `docs/literature_review_notes.md`（Chronos、PatchTST、Time-LLM、LLMTime、Tan et al. 2024、LoRA、Fan et al. 2020、Domala et al. 2022、Chaichitehrani et al. 2024 等）。

## 10. 术语与符号表

| 术语 | 含义 | 本报告角色 |
|---|---|---|
| Hs | Significant Wave Height（m） | 主预报量 |
| Tp | Peak Wave Period | 上下文变量（若可得） |
| lead / issue time | 预报时效 / 起报时刻 | 窗口对齐 |
| Persistence | 持续法 | 强基线 |
| LoRA | Low-Rank Adaptation | 微调方法 |
| RMSE / MAE / Skill | 误差与技巧 | 主定量指标 |
| wave_regime | 海况节律六类 | 分类标签 |
| predictability_24h | high/medium/low | 探索性标签 |
| reason / notes | 自由文本字段 | 模板槽，非 XAI |
| uncertainty_level | 高/中/低 | 非校准概率 |
| JSON validity | 可解析比例 | 结构成功率 |

## 附录

- **A. Git/发布：** 非 Git 仓库；仅本地生成，未 commit/push/PR/部署。
- **B. 复现：** `python scripts/build_research_report.py`（建议 `wave_llm` 环境）；校验 `python scripts/validate_report_html.py`。
- **C. 图说明：** Markdown 为便于版本阅读的源文件；逐图完整六段式解释与 Base64 内嵌图以 `report.html` 为准。
