#!/usr/bin/env python3
"""
Build self-contained research report + paper HTML/MD/PDF from local metrics and figures.

Usage (from project root, wave_llm env recommended):
  python scripts/build_research_report.py
  python scripts/build_research_report.py --skip-pdf
  python scripts/validate_report_html.py

Outputs under docs/output/:
  paper.html, paper.pdf, research_report.md, report.html, report.pdf
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "data" / "processed" / "figures"
OUT_DIR = ROOT / "docs" / "output"
METRICS_DIR = ROOT / "data" / "processed" / "metrics"
MISTRAL_DIR = ROOT / "data" / "processed" / "mistral"

# Representative figures for the Chinese research report (evidence chain; ~14 panels).
REPORT_FIGURES: list[dict[str, str]] = [
    {
        "id": "fig-r1",
        "file": "station_map.png",
        "title": "图 1 研究站点空间分布（NDBC 浮标面板）",
        "role": "data",
    },
    {
        "id": "fig-r2",
        "file": "hs_boxplot_by_station.png",
        "title": "图 2 各站有效波高 Hs 分布箱线图",
        "role": "data",
    },
    {
        "id": "fig-r3",
        "file": "series_41010.png",
        "title": "图 3 代表站 41010 的 Hs 时间序列概览",
        "role": "data",
    },
    {
        "id": "fig-r4",
        "file": "multivar_41010.png",
        "title": "图 4 代表站 41010 的多变量海况上下文",
        "role": "data",
    },
    {
        "id": "fig-r5",
        "file": "regime_counts.png",
        "title": "图 5 海况节律（wave_regime）类别频数分布",
        "role": "labels",
    },
    {
        "id": "fig-r6",
        "file": "baseline_rmse_skill.png",
        "title": "图 6 数值基线：不同 lead 的 RMSE 与相对 Persistence 的 skill",
        "role": "baseline",
    },
    {
        "id": "fig-r7",
        "file": "model_rmse_comparison_by_lead.png",
        "title": "图 7 模型 RMSE 随 lead 变化对照",
        "role": "baseline",
    },
    {
        "id": "fig-r8",
        "file": "mistral_methods_summary.png",
        "title": "图 8 端到端方法流程：数值基线、Chronos 与 Mistral 分类/曲线 LoRA",
        "role": "methods",
    },
    {
        "id": "fig-r9",
        "file": "curve_method_rmse_summary.png",
        "title": "图 9 曲线预报方法平均 RMSE 总览（同窗可比）",
        "role": "curve",
    },
    {
        "id": "fig-r10",
        "file": "forecast_panel_mistral_lora_41010.png",
        "title": "图 10 站 41010：多模型 Hs 曲线预报对照（含 Mistral LoRA）",
        "role": "cases",
    },
    {
        "id": "fig-r11",
        "file": "forecast_panel_mistral_lora_42040.png",
        "title": "图 11 站 42040：多模型 Hs 曲线预报对照（含 Mistral LoRA）",
        "role": "cases",
    },
    {
        "id": "fig-r12",
        "file": "forecast_panel_mistral_lora_44013.png",
        "title": "图 12 站 44013：多模型 Hs 曲线预报对照（含 Mistral LoRA）",
        "role": "cases",
    },
    {
        "id": "fig-r13",
        "file": "mistral_base_regime_confusion.png",
        "title": "图 13a Base 模型海况节律混淆矩阵",
        "role": "class",
    },
    {
        "id": "fig-r14",
        "file": "mistral_lora_regime_confusion.png",
        "title": "图 13b LoRA 模型海况节律混淆矩阵",
        "role": "class",
    },
    {
        "id": "fig-r15",
        "file": "mistral_predictability_accuracy.png",
        "title": "图 14a Base 可预测性准确率",
        "role": "class",
    },
    {
        "id": "fig-r16",
        "file": "mistral_lora_predictability_accuracy.png",
        "title": "图 14b LoRA 可预测性准确率",
        "role": "class",
    },
]

PAPER_FIGURES: list[dict[str, str]] = [
    {"id": "fig1", "file": "station_map.png", "caption": "Fig. 1. Study stations on coastlines (NDBC panel)."},
    {"id": "fig2", "file": "mistral_methods_summary.png", "caption": "Fig. 2. End-to-end methods: numeric baselines, Chronos, and Mistral classification/curve LoRA."},
    {"id": "fig3", "file": "baseline_rmse_skill.png", "caption": "Fig. 3. Numeric skill: RMSE and skill versus Persistence by lead."},
    {"id": "fig4", "file": "curve_method_rmse_summary.png", "caption": "Fig. 4. Curve-method mean RMSE: Persistence / LightGBM / Chronos / Mistral Base / LoRA."},
    {"id": "fig5a", "file": "forecast_panel_41010_lead24h.png", "caption": "Fig. 5a. Multi-model forecast panel at station 41010 (lead 24 h)."},
    {"id": "fig5b", "file": "forecast_panel_46047_lead24h.png", "caption": "Fig. 5b. Multi-model forecast panel at station 46047 (lead 24 h)."},
    {"id": "fig6a", "file": "mistral_base_regime_confusion.png", "caption": "Fig. 6a. Regime confusion matrix — Mistral Base (n = 24)."},
    {"id": "fig6b", "file": "mistral_lora_regime_confusion.png", "caption": "Fig. 6b. Regime confusion matrix — Mistral LoRA (n = 24)."},
    {"id": "fig7a", "file": "mistral_predictability_accuracy.png", "caption": "Fig. 7a. Predictability accuracy — Base."},
    {"id": "fig7b", "file": "mistral_lora_predictability_accuracy.png", "caption": "Fig. 7b. Predictability accuracy — LoRA."},
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def img_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    suffix = path.suffix.lower().lstrip(".")
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    return f"data:{mime};base64,{b64}"


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def load_metrics() -> dict[str, Any]:
    numeric = load_json(METRICS_DIR / "numeric_baselines.json")
    curve_base = load_json(MISTRAL_DIR / "curve_metrics_base.json")
    curve_lora = load_json(MISTRAL_DIR / "curve_metrics_lora.json")
    curve_cmp = load_json(MISTRAL_DIR / "curve_compare_base_lora.json")
    class_base = load_json(MISTRAL_DIR / "metrics_base.json")
    class_lora = load_json(MISTRAL_DIR / "metrics_lora.json")
    class_cmp = load_json(MISTRAL_DIR / "compare_base_lora.json")
    return {
        "numeric": numeric,
        "curve_base": curve_base,
        "curve_lora": curve_lora,
        "curve_cmp": curve_cmp,
        "class_base": class_base,
        "class_lora": class_lora,
        "class_cmp": class_cmp,
    }


def shared_css(zh: bool = True) -> str:
    fonts = (
        '"Microsoft YaHei", "SimSun", "Noto Serif CJK SC", "Source Han Sans SC", sans-serif'
        if zh
        else '"Times New Roman", Times, serif'
    )
    body_font = fonts if zh else '"Times New Roman", Times, serif'
    return f"""
:root {{
  --ink: #1a1a1a;
  --muted: #444;
  --line: #c8c8c8;
  --bg: #fafafa;
  --card: #f3f6f9;
  --warn: #fff8e6;
  --warn-border: #e0c36a;
  --info: #eef5fb;
  --info-border: #7aa2c4;
  --accent: #1f4e79;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  margin: 0 auto;
  max-width: 920px;
  padding: 24px 28px 64px;
  color: var(--ink);
  background: #fff;
  font-family: {body_font};
  font-size: 11.5pt;
  line-height: 1.65;
}}
h1, h2, h3, h4 {{
  font-family: {fonts};
  color: var(--accent);
  line-height: 1.35;
  page-break-after: avoid;
}}
h1 {{ font-size: 1.7rem; text-align: center; margin: 0.4em 0 0.6em; }}
h2 {{ font-size: 1.28rem; border-bottom: 1.5px solid var(--line); padding-bottom: 0.25em; margin-top: 1.6em; }}
h3 {{ font-size: 1.08rem; margin-top: 1.2em; }}
p {{ margin: 0.55em 0; text-align: justify; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.cover {{
  text-align: center;
  padding: 36px 12px 28px;
  border-bottom: 2px solid var(--accent);
  margin-bottom: 1.2em;
}}
.cover .meta {{ color: var(--muted); font-size: 0.95rem; margin-top: 0.6em; }}
.toc {{ background: var(--bg); border: 1px solid var(--line); padding: 14px 18px; margin: 1em 0 1.5em; }}
.toc ol {{ margin: 0.3em 0 0 1.2em; padding: 0; }}
.toc li {{ margin: 0.25em 0; }}
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 0.8em 0 1.1em;
  font-size: 0.92rem;
  page-break-inside: avoid;
}}
th, td {{
  border: 1px solid #999;
  padding: 6px 8px;
  vertical-align: top;
}}
th {{ background: #e8eef5; font-weight: 600; }}
caption {{
  caption-side: top;
  text-align: left;
  font-weight: 600;
  margin-bottom: 0.4em;
  color: var(--ink);
}}
.table-note {{
  font-size: 0.85rem;
  color: var(--muted);
  margin: -0.6em 0 1em;
}}
figure {{
  margin: 1.1em 0 1.4em;
  page-break-inside: avoid;
}}
figure img {{
  display: block;
  width: 100%;
  max-width: 100%;
  height: auto;
  border: 1px solid #ddd;
}}
figcaption {{
  font-size: 0.9rem;
  color: var(--muted);
  margin-top: 0.45em;
  text-align: left;
}}
.callout {{
  border-left: 4px solid var(--info-border);
  background: var(--info);
  padding: 10px 14px;
  margin: 0.9em 0;
}}
.callout.warn {{
  border-left-color: var(--warn-border);
  background: var(--warn);
}}
.callout strong {{ color: var(--accent); }}
.fig-explain {{
  background: var(--card);
  border: 1px solid var(--line);
  padding: 12px 14px;
  margin: 0.4em 0 1.2em;
  font-size: 0.95rem;
}}
.fig-explain h4 {{ margin: 0 0 0.4em; color: var(--ink); font-size: 1rem; }}
.fig-explain ol {{ margin: 0.3em 0 0.3em 1.2em; padding: 0; }}
.fig-explain li {{ margin: 0.25em 0; }}
.kbd {{ font-family: Consolas, "Courier New", monospace; font-size: 0.9em; }}
.footer-note {{
  margin-top: 2.5em;
  padding-top: 0.8em;
  border-top: 1px solid var(--line);
  font-size: 0.85rem;
  color: var(--muted);
}}
@media print {{
  body {{ max-width: none; padding: 12mm 14mm; font-size: 10.5pt; }}
  h2, h3, figure, table {{ break-inside: avoid; page-break-inside: avoid; }}
  a {{ color: inherit; text-decoration: none; }}
  .toc a::after {{ content: ""; }}
  @page {{ size: A4; margin: 16mm 14mm; }}
}}
"""


def fig_block(fid: str, title: str, src: str, source_file: str, explain_html: str) -> str:
    return f"""
<figure id="{html.escape(fid)}">
  <img src="{src}" alt="{html.escape(title)}" />
  <figcaption><strong>{html.escape(title)}</strong><br/>来源文件：<span class="kbd">{html.escape(source_file)}</span></figcaption>
</figure>
<div class="fig-explain">
{explain_html}
</div>
"""


def figure_explanations(m: dict[str, Any]) -> dict[str, str]:
    """Detailed provenance-style explanations for each report figure."""
    cb = m["curve_base"]
    cl = m["curve_lora"]
    persist = cb["baselines"]["mean_rmse_persist"]
    lgbm = cb["baselines"]["mean_rmse_lgbm_at_numeric_leads"]
    chronos = cb["baselines"]["mean_rmse_chronos_at_numeric_leads"]
    return {
        "fig-r1": f"""
<h4>图解：站点地图在证据链中的位置</h4>
<ol>
<li><strong>背景与作用</strong>：在比较任何预报误差之前，必须先明确“在哪些地点、哪类海域做研究”。本图给出配置面板中 NDBC 浮标的大致空间分布，支撑后续多站可比性声明。</li>
<li><strong>如何读图</strong>：底图为海岸线/陆海轮廓；点位为浮标站号位置。颜色仅为制图区分，不代表误差大小。</li>
<li><strong>物理/统计含义</strong>：站点跨越大西洋、墨西哥湾、太平洋与夏威夷附近，对应不同风浪/涌浪气候背景，有利于检验方法是否依赖单一海区。</li>
<li><strong>能/不能得出的结论</strong>：能说明研究覆盖多盆地；不能从地图本身推断模型精度，也不能声称全球代表性。</li>
<li><strong>通俗解释</strong>：就像在地图上标出“我们听了哪些浮标的海浪录音”，后面所有分数都只对这批浮标负责。</li>
<li><strong>前后关联</strong>：与表 1（站点清单）、图 2–4（分布与时间序列）共同构成数据章；后文案例站 41010/42040/44013 均来自此面板。</li>
</ol>
""",
        "fig-r2": f"""
<h4>图解：Hs 箱线分布</h4>
<ol>
<li><strong>背景与作用</strong>：有效波高 Hs 是核心预报量。箱线图用于展示各站 Hs 的中心趋势与极端尾部，防止读者把“平均误差”误读为所有海况同等困难。</li>
<li><strong>如何读图</strong>：横轴为站号；纵轴为 Hs（米）。箱体通常表示四分位区间，须线与离群点显示尾部；站间箱体高度差异反映气候差。</li>
<li><strong>物理含义</strong>：开阔洋站常有更高浪高尾部；近岸/遮蔽站更窄。分布偏态意味着 RMSE 会被少数风暴窗口强烈拉动。</li>
<li><strong>能/不能得出的结论</strong>：能比较站间浪高气候；不能直接读出模型好坏。若某站样本稀疏，箱线可能不稳定——本项目未在本图中给出精确有效样本数（待补充）。</li>
<li><strong>通俗解释</strong>：每个箱子像“这座浮标常见浪有多高、偶尔会飙到多高”。</li>
<li><strong>前后关联</strong>：解释为何曲线评估对风暴窗敏感（后文 per-sample RMSE 也显示个别大误差样本）。</li>
</ol>
""",
        "fig-r3": f"""
<h4>图解：代表站时间序列</h4>
<ol>
<li><strong>背景与作用</strong>：时间序列展示 Hs 随时间的起伏与风暴过程，是理解“为什么需要不同 lead / 为什么 Persistence 有时很强”的直观入口。</li>
<li><strong>如何读图</strong>：横轴为时间，纵轴为 Hs。连续起伏对应天气尺度过程；尖峰对应高浪事件。</li>
<li><strong>物理含义</strong>：浪高变化由局地风生浪与远程涌浪叠加决定；短 lead 常更接近“惯性延续”，长 lead 更依赖气象强迫变化。</li>
<li><strong>能/不能得出的结论</strong>：能看到非平稳性与事件结构；单站图不能代表全面板，也不能替代 hold-out 定量指标。</li>
<li><strong>通俗解释</strong>：像看海浪高度的“心电图”，平静时平缓，风暴时突然抬升。</li>
<li><strong>前后关联</strong>：与图 4 多变量上下文、图 10 预报面板衔接；案例站优先选 41010。</li>
</ol>
""",
        "fig-r4": f"""
<h4>图解：多变量上下文</h4>
<ol>
<li><strong>背景与作用</strong>：仅看 Hs 不够理解海况。多变量图把风速、波向、周期等（视可用字段）与 Hs 并置，说明分类标签与可预测性为何需要窗口统计特征。</li>
<li><strong>如何读图</strong>：多个子图共享时间轴；每个子图纵轴对应不同物理量。对齐时间可识别“风增强—浪增高”等过程。</li>
<li><strong>物理含义</strong>：风海主导、涌浪主导、混合海等 regime，依赖 Hs 统计量与风浪关系；本项目 regime 标签由规则基于窗口统计生成。</li>
<li><strong>能/不能得出的结论</strong>：能展示特征丰富度；不能把相关当成因果，也不能把多变量图当作模型输入可视化（曲线 LoRA 试点主要用压缩 Hs 历史）。</li>
<li><strong>通俗解释</strong>：不仅看浪有多高，还顺便看风有多猛、浪从哪边来，帮助理解“这是什么类型的海”。</li>
<li><strong>前后关联</strong>：为图 5 regime 分布与分类任务提供物理动机。</li>
</ol>
""",
        "fig-r5": f"""
<h4>图解：regime 频数</h4>
<ol>
<li><strong>背景与作用</strong>：分类任务的难度高度依赖类别是否均衡。本图展示 wave_regime 标签的总体频数，解释后文混淆矩阵中的模式塌缩风险。</li>
<li><strong>如何读图</strong>：横轴类别名（如 calm_stable、storm_growth 等），纵轴计数。柱高悬殊即表示不平衡。</li>
<li><strong>统计含义</strong>：若某类占绝大多数，准确率可能被“总是猜多数类”抬高或扭曲；本 pilot 评估集真实标签以 storm_growth 为主（见 metrics JSON）。</li>
<li><strong>能/不能得出的结论</strong>：能提示不平衡；不能单独证明模型学会了物理分型。空类别/极少样本类别若出现，应视为稀疏而非“完美分类”。</li>
<li><strong>通俗解释</strong>：如果试卷里 90% 都是同一题型，得分高不一定代表样样都会。</li>
<li><strong>前后关联</strong>：直接预警图 13 混淆矩阵与表 4 准确率解读边界。</li>
</ol>
""",
        "fig-r6": f"""
<h4>图解：数值基线 skill</h4>
<ol>
<li><strong>背景与作用</strong>：在引入 LLM 之前，先建立 Persistence 与 LightGBM 的 lead-wise 基准。这是“公平比较”的第一级台阶。</li>
<li><strong>如何读图</strong>：横轴 lead（6/12/24/48/72 h）；纵轴为 RMSE 或 skill。skill = 1 − RMSE<sub>LGBM</sub>/RMSE<sub>persist</sub>；skill&gt;0 表示优于 Persistence。</li>
<li><strong>统计含义</strong>：本地 metrics 显示 lead=6 h 时 skill 为负（约 −0.063），12–72 h 为正并在 72 h 达约 0.154。短时效 Persistence 很强是海洋预报常见现象。</li>
<li><strong>能/不能得出的结论</strong>：能说明 LightGBM 是认真的数值竞争者；不能把该表直接等同于曲线任务 n=12 窗口上的方法排名（口径不同，见表注）。</li>
<li><strong>通俗解释</strong>：先问“如果我傻傻地认为明天浪高=现在浪高，错多少？”再看机器学习能不能做得更好。</li>
<li><strong>前后关联</strong>：为图 9 曲线方法总览提供数值参照；也定义 predictability 标签所用的 24 h skill 思想。</li>
</ol>
""",
        "fig-r7": f"""
<h4>图解：RMSE 随 lead</h4>
<ol>
<li><strong>背景与作用</strong>：补充展示不同方法/设置下 RMSE 随预报时效增长的趋势，强调误差随 lead 累积。</li>
<li><strong>如何读图</strong>：横轴 lead，纵轴 RMSE；不同颜色/线型对应不同模型。曲线上升表示更远的未来更难预报。</li>
<li><strong>物理含义</strong>：大气与海浪系统混沌增长使长时效确定性下降；任何模型都应预期更长 lead 的更大误差（除非评估窗或气候不同）。</li>
<li><strong>能/不能得出的结论</strong>：能比较相对排序趋势；若图中某线缺失或样本不足，不得脑补为“完美表现”。</li>
<li><strong>通俗解释</strong>：预报越远，越像猜更远的天气，通常错得更多。</li>
<li><strong>前后关联</strong>：与曲线逐步 RMSE（metrics 中 rmse_by_forecast_step_h）相互印证：逐步误差总体随小时增加。</li>
</ol>
""",
        "fig-r8": f"""
<h4>图解：方法总览</h4>
<ol>
<li><strong>背景与作用</strong>：把整条流水线画成一张图：NDBC→窗口→数值基线/Chronos→分类 LoRA→曲线 LoRA→评估。避免读者把“两个 LoRA 任务”混成一个模型。</li>
<li><strong>如何读图</strong>：按箭头从数据到输出阅读；注意分类 JSON（regime/predictability/notes）与曲线 JSON（hs_forecast_m/uncertainty_level/reason）是并列任务。</li>
<li><strong>方法含义</strong>：LoRA 只训练低秩适配器，不重训全部 7B 权重；评估强调同 issue-time 窗口与 JSON 可解析性。</li>
<li><strong>能/不能得出的结论</strong>：能理解系统结构；流程图本身不含精度证据，精度只看后文表与图 9–14。</li>
<li><strong>通俗解释</strong>：像工厂流水线说明书：原料是浮标数据，成品是“可解析的 JSON 预报+标签”。</li>
<li><strong>前后关联</strong>：对应第 4 章方法与脚本 05/05b/06/07/08 系列。</li>
</ol>
""",
        "fig-r9": f"""
<h4>图解：曲线方法 RMSE 总览（核心结果图）</h4>
<ol>
<li><strong>背景与作用</strong>：回答“在同一批 hold-out 曲线窗口上，谁的平均 RMSE 更低”。这是 RMSE 边界声明的主证据。</li>
<li><strong>如何读图</strong>：横轴方法，纵轴平均 RMSE。比较对象包含 Persistence、LightGBM（数值 lead 聚合口径）、Chronos（同上）、Mistral Base、Mistral LoRA。</li>
<li><strong>具体数字（本地 JSON）</strong>：Base {fmt3(cb['mean_rmse'])} → LoRA {fmt3(cl['mean_rmse'])}；Persistence {fmt3(persist)}；Chronos {fmt3(chronos)}；LightGBM {fmt3(lgbm)}；n={cb['n_samples']}；JSON valid=1.0。</li>
<li><strong>能/不能得出的结论</strong>：能得出 LoRA 相对 Base 明显改进；不能声称 LoRA 优于 Persistence/Chronos；与 LightGBM 均值到三位小数相近不等于全面等价。小样本 pilot，不作显著性检验。</li>
<li><strong>通俗解释</strong>：微调后语言模型更会“按格式报浪高曲线”，但平均仍未必比“认为未来=现在”的笨办法更准。</li>
<li><strong>前后关联</strong>：数字锁定自 curve_metrics_*.json / curve_compare_base_lora.json；案例图 10–12 展示个体窗口差异。</li>
</ol>
""",
        "fig-r10": f"""
<h4>图解：站 41010 预报面板</h4>
<ol>
<li><strong>背景与作用</strong>：平均 RMSE 会掩盖“有的窗口很好、有的很差”。面板把观测真值与多模型曲线叠在一起，展示误差形态。</li>
<li><strong>如何读图</strong>：黑线/标记通常为观测 Hs；灰/蓝等为 Persistence、LightGBM；另有 Chronos 与 Mistral Base/LoRA（以图例为准）。横轴为预报时段内小时，纵轴 Hs（m）。</li>
<li><strong>物理/误差含义</strong>：若模型整体平移、相位滞后或风暴峰值低估，会在视觉上直接显现；逐步误差常在后期放大。</li>
<li><strong>能/不能得出的结论</strong>：能做定性诊断；单站单窗不能外推到全样本。若某条模型曲线缺失，表示该窗未产出或未叠加，不得当作“误差为 0”。</li>
<li><strong>通俗解释</strong>：像把几位预报员的“未来 24 小时浪高草图”叠在真实海浪上对比。</li>
<li><strong>前后关联</strong>：与图 9 均值表互补；站点选自大西洋侧代表站 41010。</li>
</ol>
""",
        "fig-r11": f"""
<h4>图解：站 42040 预报面板</h4>
<ol>
<li><strong>背景与作用</strong>：墨西哥湾站 42040 提供不同海区情景，检查方法是否只在单一站点“看起来好看”。</li>
<li><strong>如何读图</strong>：同图 10。注意不同站的浪高量级与过程形状可能不同。</li>
<li><strong>含义</strong>：跨站对照用于展示可迁移的可视化协议，而非宣称跨站泛化已验证（正式空间泛化实验待补充）。</li>
<li><strong>能/不能得出的结论</strong>：能观察跨站行为差异；不能因一两个窗口断言盆地级稳健性。</li>
<li><strong>通俗解释</strong>：换一个海湾的浮标再看一眼，避免“只挑好例子”。</li>
<li><strong>前后关联</strong>：与 41010、44013 组成三站案例链。</li>
</ol>
""",
        "fig-r12": f"""
<h4>图解：站 44013 预报面板</h4>
<ol>
<li><strong>背景与作用</strong>：波士顿附近进港航道站 44013，强调近岸/航道场景对业务读者的相关性。</li>
<li><strong>如何读图</strong>：同图 10–11。若图面较空或曲线段较短，可能与该窗有效点或导出叠加有关，应如实视为样本/出图限制，而非优异性能。</li>
<li><strong>含义</strong>：近岸站可能有不同非平稳性与局部效应；本 pilot 未单独报告该站的专属 n 与显著性。</li>
<li><strong>能/不能得出的结论</strong>：能作为第三站定性对照；不能替代表 3 的平均指标。</li>
<li><strong>通俗解释</strong>：再看一个靠近港口的浮标，想想船员会不会关心这种曲线产品。</li>
<li><strong>前后关联</strong>：收束案例章，转入分类混淆与可预测性结果。</li>
</ol>
""",
        "fig-r13": f"""
<h4>图解：Base 节律混淆矩阵</h4>
<ol>
<li><strong>背景与作用</strong>：分类 LoRA 的第一目标是 wave_regime。混淆矩阵展示“真标签 vs 预测标签”的计数结构，比单一准确率更诚实。</li>
<li><strong>如何读图</strong>：行/列对应类别；对角线为正确；颜色深浅表示计数。注意真实标签高度集中于 storm_growth（n=24 的 JSON 列表）。</li>
<li><strong>观察</strong>：Base 预测常塌缩到 windsea_dominated / calm_stable 等模式，导致 regime accuracy 仅约 {fmt3(m['class_base']['regime_accuracy'])}。</li>
<li><strong>能/不能得出的结论</strong>：能说明 Base 几乎未学会面板标签分布；不能解释为物理意义的“系统性偏见机制”已证实。</li>
<li><strong>通俗解释</strong>：没微调时，模型像几乎总用少数几个口头禅回答“现在是什么海况”。</li>
<li><strong>前后关联</strong>：对照图 13b LoRA；准确率见表 4。</li>
</ol>
""",
        "fig-r14": f"""
<h4>图解：LoRA 节律混淆矩阵</h4>
<ol>
<li><strong>背景与作用</strong>：展示微调后标签分布是否更接近真值。regime accuracy 升至约 {fmt3(m['class_lora']['regime_accuracy'])}（n=24）。</li>
<li><strong>如何读图</strong>：同 13a。LoRA 预测更多落在 storm_growth / mixed_sea，与真值中 storm_growth 占优一致，但也提示可能存在“多数类迎合”。</li>
<li><strong>能/不能得出的结论</strong>：能支持“LoRA 改善了相对 Base 的 regime 匹配”；不能在类别极不均衡、样本很小的情况下宣称生产级分类器。</li>
<li><strong>通俗解释</strong>：微调后更常说出“风暴增强”这类在数据里常见的标签，分数上升，但仍需小心是否投机取巧。</li>
<li><strong>前后关联</strong>：与可预测性准确率下降形成对比，防止单一叙事。</li>
</ol>
""",
        "fig-r15": f"""
<h4>图解：Base 可预测性准确率</h4>
<ol>
<li><strong>背景与作用</strong>：predictability_24h ∈ {{high, medium, low}} 来自 LightGBM 相对 Persistence 在 24 h lead 的 skill 思想，是探索性情境标签，不是概率预报校准产物。</li>
<li><strong>如何读图</strong>：柱状/条形显示准确率水平。Base 约 {fmt3(m['class_base']['predictability_accuracy'])}（n=24）。</li>
<li><strong>能/不能得出的结论</strong>：能给出基线；不能把准确率当成可靠不确定度。标签定义依赖数值模型 skill，有循环解释风险，需在讨论中保持克制。</li>
<li><strong>通俗解释</strong>：这是在问模型“未来一天好不好报”，用的是简化分级，不是给出 80% 置信区间那种正式不确定度。</li>
<li><strong>前后关联</strong>：必须与图 14b 一起读：LoRA 在该指标上变差。</li>
</ol>
""",
        "fig-r16": f"""
<h4>图解：LoRA 可预测性准确率（诚实边界）</h4>
<ol>
<li><strong>背景与作用</strong>：报告负面结果。LoRA 可预测性准确率约 {fmt3(m['class_lora']['predictability_accuracy'])}，低于 Base 的 {fmt3(m['class_base']['predictability_accuracy'])}。</li>
<li><strong>如何读图</strong>：同 14a。结合 metrics 中预测向量可见 LoRA 几乎总是预测 high，属于模式塌缩。</li>
<li><strong>能/不能得出的结论</strong>：能明确“微调并不自动改善所有头”；不能把该标签当作已验证的业务不确定度产品；不能因 regime 变好而掩盖此项变差。</li>
<li><strong>通俗解释</strong>：模型微调后更会说海况类型，但对“好不好报”变得过于乐观、口径单一。</li>
<li><strong>前后关联</strong>：支撑结论定位——Instruct-LoRA 是结构化多输出伴侣，而非全面优于 Base 的全能分类器，更非 RMSE 冠军。</li>
</ol>
""",
    }


def build_tables_html(m: dict[str, Any]) -> dict[str, str]:
    rows_num = []
    for r in m["numeric"]:
        rows_num.append(
            f"<tr><td>{r['lead_h']}</td><td>{fmt3(r['rmse_persist'])}</td>"
            f"<td>{fmt3(r['mae_persist'])}</td><td>{fmt3(r['rmse_lgbm'])}</td>"
            f"<td>{fmt3(r['skill_vs_persist'])}</td></tr>"
        )
    t_numeric = f"""
<table id="tab-numeric">
<caption>表 3 数值基线指标（lead = 6/12/24/48/72 h）</caption>
<thead><tr><th>Lead (h)</th><th>RMSE Persist</th><th>MAE Persist</th><th>RMSE LightGBM</th><th>Skill vs Persist</th></tr></thead>
<tbody>{''.join(rows_num)}</tbody>
</table>
<p class="table-note">来源：<span class="kbd">data/processed/metrics/numeric_baselines.json</span>。Skill = 1 − RMSE_LGBM/RMSE_persist。此表为数值面板评估口径，不同于曲线任务 n=12 同窗均值。</p>
"""
    cb, cl = m["curve_base"], m["curve_lora"]
    t_curve = f"""
<table id="tab-curve">
<caption>表 5 曲线方法平均误差（共享 pilot 窗口）</caption>
<thead><tr><th>模型</th><th>Mean RMSE</th><th>Mean MAE</th><th>JSON valid</th><th>n</th><th>口径说明</th></tr></thead>
<tbody>
<tr><td>Mistral Base</td><td>{fmt3(cb['mean_rmse'])}</td><td>{fmt3(cb['mean_mae'])}</td><td>{cb['json_valid_rate']}</td><td>{cb['n_samples']}</td><td>曲线 JSON 解析后对未来 24 h Hs</td></tr>
<tr><td>Mistral LoRA</td><td>{fmt3(cl['mean_rmse'])}</td><td>{fmt3(cl['mean_mae'])}</td><td>{cl['json_valid_rate']}</td><td>{cl['n_samples']}</td><td>同上</td></tr>
<tr><td>Persistence</td><td>{fmt3(cb['baselines']['mean_rmse_persist'])}</td><td>—</td><td>—</td><td>{cb['n_samples']}</td><td>与曲线评估同窗口</td></tr>
<tr><td>LightGBM</td><td>{fmt3(cb['baselines']['mean_rmse_lgbm_at_numeric_leads'])}</td><td>—</td><td>—</td><td>—</td><td>数值 lead 聚合均值（见 JSON 字段名）</td></tr>
<tr><td>Chronos-T5</td><td>{fmt3(cb['baselines']['mean_rmse_chronos_at_numeric_leads'])}</td><td>—</td><td>—</td><td>—</td><td>数值 lead 聚合均值（见 JSON 字段名）</td></tr>
</tbody>
</table>
<p class="table-note">来源：curve_metrics_base.json / curve_metrics_lora.json / curve_compare_base_lora.json。不得将 LoRA 表述为优于 Persistence 或 Chronos。</p>
"""
    t_class = f"""
<table id="tab-class">
<caption>表 4 分类任务：Base vs LoRA</caption>
<thead><tr><th>模型</th><th>Regime accuracy</th><th>Predictability accuracy</th><th>n</th></tr></thead>
<tbody>
<tr><td>Mistral Base</td><td>{fmt3(m['class_base']['regime_accuracy'])}</td><td>{fmt3(m['class_base']['predictability_accuracy'])}</td><td>{m['class_base']['n_samples']}</td></tr>
<tr><td>Mistral LoRA</td><td>{fmt3(m['class_lora']['regime_accuracy'])}</td><td>{fmt3(m['class_lora']['predictability_accuracy'])}</td><td>{m['class_lora']['n_samples']}</td></tr>
</tbody>
</table>
<p class="table-note">来源：metrics_base.json / metrics_lora.json / compare_base_lora.json。Predictability 准确率下降必须保留在结论边界中。</p>
"""
    t_data = """
<table id="tab-data">
<caption>表 1 数据源 / 站点 / 变量说明</caption>
<thead><tr><th>项目</th><th>内容</th><th>状态</th></tr></thead>
<tbody>
<tr><td>数据源</td><td>NDBC standard meteorological (stdmet) 公开记录</td><td>已用</td></tr>
<tr><td>站点 ID</td><td>41010, 42040, 44013, 46025, 46026, 46042, 46047, 46246, 51000, 51002</td><td>来自 configs/station_panels.yaml</td></tr>
<tr><td>年份</td><td>2019–2020</td><td>配置已写明</td></tr>
<tr><td>核心变量</td><td>Significant Wave Height (Hs)；另含风速、波向、周期等（视站时可用）</td><td>Hs 为曲线目标</td></tr>
<tr><td>重采样</td><td>resample_rule = 1h</td><td>model_config.yaml</td></tr>
<tr><td>可选数据</td><td>Copernicus Marine / CDIP</td><td>可选；许可条款待补充说明</td></tr>
<tr><td>精确有效样本时段统计</td><td>逐站缺测率、QC 剔除量等</td><td>待补充 / To be completed</td></tr>
</tbody>
</table>
"""
    t_models = """
<table id="tab-models">
<caption>表 2 模型与任务对照</caption>
<thead><tr><th>模型/方法</th><th>任务</th><th>输出</th><th>角色</th></tr></thead>
<tbody>
<tr><td>Persistence</td><td>数值/曲线参照</td><td>issue-time Hs 延续</td><td>强短期基线</td></tr>
<tr><td>LightGBM</td><td>数值回归</td><td>指定 lead 的 Hs</td><td>树模型基线；亦参与 predictability 定义</td></tr>
<tr><td>Chronos-T5-tiny</td><td>零样本时序</td><td>连续 Hs 轨迹</td><td>时序基础模型基线</td></tr>
<tr><td>Mistral Base</td><td>分类 / 曲线</td><td>JSON 标签或 hs_forecast_m</td><td>未适配 Instruct 对照</td></tr>
<tr><td>Mistral LoRA（分类）</td><td>regime + predictability</td><td>JSON + notes</td><td>语言标签接口</td></tr>
<tr><td>Mistral LoRA（曲线）</td><td>24 h Hs 曲线</td><td>JSON hs_forecast_m + reason 等</td><td>结构化曲线生成</td></tr>
</tbody>
</table>
"""
    t_cfg = """
<table id="tab-cfg">
<caption>表 6 实验配置摘要（configs/model_config.yaml）</caption>
<thead><tr><th>组件</th><th>关键设置</th></tr></thead>
<tbody>
<tr><td>基座模型</td><td>mistralai/Mistral-7B-Instruct-v0.3</td></tr>
<tr><td>LoRA r / α / dropout</td><td>16 / 32 / 0.05</td></tr>
<tr><td>分类 LR / max_seq_length / max_eval_samples</td><td>2e-4 / 1024 / 24</td></tr>
<tr><td>曲线 history/horizon</td><td>168 h / 24 h</td></tr>
<tr><td>曲线 LR / seq len / max_steps / max_eval_samples</td><td>2e-5 / 2048 / 120 / 12</td></tr>
<tr><td>曲线 batch × grad accum</td><td>1 × 8</td></tr>
<tr><td>Chronos</td><td>amazon/chronos-t5-tiny；context 256；num_samples 12</td></tr>
<tr><td>LightGBM</td><td>400 trees；lr 0.05；63 leaves；random_state 42</td></tr>
<tr><td>split_seed</td><td>42</td></tr>
</tbody>
</table>
"""
    t_risk = """
<table id="tab-risk">
<caption>表 7 风险 / 限制与缓解计划</caption>
<thead><tr><th>风险/限制</th><th>表现</th><th>缓解计划</th></tr></thead>
<tbody>
<tr><td>小样本 pilot</td><td>曲线 n=12；分类 n=24</td><td>扩大 hold-out；报告置信区间；避免生产声称</td></tr>
<tr><td>RMSE 非最优</td><td>LoRA &gt; Persistence/Chronos（均值）</td><td>定位为结构化伴侣；数值专家模型继续作为技能参照</td></tr>
<tr><td>Predictability 回退</td><td>0.375 → 0.250；预测塌缩为 high</td><td>重标标签；类别均衡；独立校准指标</td></tr>
<tr><td>模板化 reason/notes</td><td>非专家多样化标注</td><td>人工评分/物理约束监督；禁止因果解释声称</td></tr>
<tr><td>无校准 UQ</td><td>uncertainty_level 定性</td><td>引入 coverage/CRPS 等后再谈概率不确定度</td></tr>
<tr><td>单骨干/算力上限</td><td>仅 Mistral-7B-Instruct-v0.3 + 既定 LoRA</td><td>多骨干对照；更长训练与完整数据</td></tr>
<tr><td>空间泛化未测</td><td>限配置面板与年份</td><td>外站/外年独立验证（待补充）</td></tr>
</tbody>
</table>
"""
    return {
        "data": t_data,
        "models": t_models,
        "numeric": t_numeric,
        "class": t_class,
        "curve": t_curve,
        "cfg": t_cfg,
        "risk": t_risk,
    }


def build_research_report_md(m: dict[str, Any]) -> str:
    cb, cl = m["curve_base"], m["curve_lora"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    num_rows = "\n".join(
        f"| {r['lead_h']} | {fmt3(r['rmse_persist'])} | {fmt3(r['mae_persist'])} | "
        f"{fmt3(r['rmse_lgbm'])} | {fmt3(r['skill_vs_persist'])} |"
        for r in m["numeric"]
    )
    fig_list = "\n".join(
        f"- **{spec['title']}** — 源文件 `{spec['file']}`（详细六段式图解见 `report.html#{spec['id']}`）"
        for spec in REPORT_FIGURES
    )
    lines = [
        "# 结构化海浪预报研究科研报告",
        "",
        "副标题：基于 NDBC 浮标与 Mistral Instruct-LoRA 的 JSON 曲线、海况节律与可预测性标签",
        "",
        f"- 生成日期（UTC）：{now}",
        "- 项目根目录：`d:/Projects/wave-prediction-mistral-tuning`",
        "- 指标冻结来源：`data/processed/metrics/` 与 `data/processed/mistral/*.json`",
        "- 图件目录：`data/processed/figures/`（SciencePlots + Times New Roman）",
        "- Git 状态：本地项目**不是** Git 仓库（检测时无 `.git`）",
        "- ChatGPT Pro：本轮未再次联网咨询、未上传本地附件；沿用 `docs/literature_review_notes.md` 既有线程",
        "- 完整自包含排版版：同目录 `report.html` / `report.pdf`",
        "",
        "## 摘要",
        "",
        "业务海浪产品不仅需要 Significant Wave Height (Hs) 的连续轨迹，还需要可机读、可检查的情境标签。"
        "本报告基于真实 NDBC 数据与本地评估 JSON，总结端到端流水线：issue-time 窗口 → "
        "数值基线（Persistence / LightGBM）与 Chronos-T5 → 分类 LoRA（`wave_regime` / `predictability_24h`）→ "
        "曲线 LoRA（JSON `hs_forecast_m`）→ SciencePlots 图件。",
        "",
        f"曲线 pilot（n = {cb['n_samples']}）：LoRA 平均 RMSE 从 {fmt3(cb['mean_rmse'])}（Base）降至 {fmt3(cl['mean_rmse'])}，"
        f"JSON 有效率均为 {cb['json_valid_rate']}。同窗 Persistence / LightGBM / Chronos 均值 RMSE 分别为 "
        f"{fmt3(cb['baselines']['mean_rmse_persist'])} / {fmt3(cb['baselines']['mean_rmse_lgbm_at_numeric_leads'])} / "
        f"{fmt3(cb['baselines']['mean_rmse_chronos_at_numeric_leads'])}。"
        f"分类（n = {m['class_base']['n_samples']}）：regime 准确率 "
        f"{fmt3(m['class_base']['regime_accuracy'])} → {fmt3(m['class_lora']['regime_accuracy'])}；"
        f"predictability 准确率 {fmt3(m['class_base']['predictability_accuracy'])} → "
        f"{fmt3(m['class_lora']['predictability_accuracy'])}（下降）。",
        "",
        "**定位：** Instruct-LoRA 是结构化多输出伴侣，不是 Persistence/Chronos/LightGBM 的 RMSE 替代品；"
        "`reason` 为模板化文本槽位，不等于正式可解释性；`uncertainty_level` 不是校准概率不确定度。",
        "",
        "> 诚实边界：本文件记录的是小样本 pilot，不是生产验收。",
        "",
        "## 1. 研究背景与目的",
        "",
        "沿海与离岸作业依赖浮标观测的 Hs。机器学习与时序基础模型擅长连续值预报，但较少提供统一语言接口，"
        "把曲线、海况节律与可预测性描述捆在同一 JSON 中。指令微调大语言模型可以输出结构化文本，"
        "然而若只追 RMSE，往往难以超越专用数值模型（参见 Tan 等对 LLM-for-TS 的警示）。",
        "",
        "本研究目的：（1）建立 NDBC→JSONL→LoRA 可复现配方；（2）在公平窗口协议下报告 Base→LoRA 改进与相对数值基线边界；"
        "（3）明确模板理由 ≠ 可解释 AI，定性 uncertainty ≠ 校准概率。",
        "",
        "## 2. 数据与方法",
        "",
        "### 2.1 数据",
        "",
        "| 项目 | 内容 | 状态 |",
        "|---|---|---|",
        "| 数据源 | NDBC stdmet 公开记录 | 已用 |",
        "| 站点 | 41010, 42040, 44013, 46025, 46026, 46042, 46047, 46246, 51000, 51002 | `station_panels.yaml` |",
        "| 年份 | 2019–2020 | 配置已写明 |",
        "| 重采样 | 1 h | `model_config.yaml` |",
        "| 精确缺测/QC 统计 | 逐站有效样本汇总 | 待补充 |",
        "",
        "### 2.2 模型与任务",
        "",
        "| 方法 | 任务 | 输出 | 角色 |",
        "|---|---|---|---|",
        "| Persistence | 数值/曲线参照 | issue-time Hs 延续 | 强短期基线 |",
        "| LightGBM | 数值回归 | 指定 lead Hs | 树模型基线 |",
        "| Chronos-T5-tiny | 零样本时序 | 连续 Hs | 时序基础模型基线 |",
        "| Mistral Base/LoRA 分类 | regime + predictability | JSON + notes | 语言标签接口 |",
        "| Mistral Base/LoRA 曲线 | 24 h Hs 曲线 | `hs_forecast_m` + reason | 结构化曲线生成 |",
        "",
        "公式：RMSE = √((1/N) Σ(ŷ−y)²)；MAE = (1/N) Σ|ŷ−y|；Skill = 1 − RMSE_model/RMSE_persistence。",
        "",
        "### 2.3 关键配置摘要",
        "",
        "- 基座：`mistralai/Mistral-7B-Instruct-v0.3`；LoRA r/α/dropout = 16/32/0.05",
        "- 分类：LR 2e-4，seq 1024，max_eval_samples 24",
        "- 曲线：history/horizon 168/24 h；LR 2e-5；seq 2048；max_steps 120；max_eval_samples 12",
        "- Chronos：context 256，num_samples 12；LightGBM：400 trees，lr 0.05，63 leaves；split_seed 42",
        "",
        "## 3. 研究过程（真实路径）",
        "",
        "1. 真实 NDBC 下载与质控、1 h 重采样、面板拼装（scripts/02–04）。",
        "2. 构造 issue-time 窗口；训练 Persistence/LightGBM；运行 Chronos（05 / 05b）。",
        "3. 导出分类 JSONL，划分数据集，LoRA 训练与 Base 对比（06 / 06b / 07 / 08；n=24）。",
        "4. 导出曲线 JSONL（压缩 Hs 历史），pilot 训练与 hold-out 评估（06d / 07b / 08b；n=12）。",
        "5. 同窗汇总 Persistence / LightGBM / Chronos / Mistral 指标。",
        "6. SciencePlots + Times New Roman 出图至 `data/processed/figures/`。",
        "",
        "## 4. 结果展示",
        "",
        "### 4.1 数值基线（lead = 6/12/24/48/72）",
        "",
        "| Lead (h) | RMSE Persist | MAE Persist | RMSE LightGBM | Skill vs Persist |",
        "|---:|---:|---:|---:|---:|",
        num_rows,
        "",
        "来源：`numeric_baselines.json`。口径为数值面板，不同于曲线 n=12 同窗均值。",
        "",
        "### 4.2 分类 Base vs LoRA",
        "",
        "| 模型 | Regime accuracy | Predictability accuracy | n |",
        "|---|---:|---:|---:|",
        f"| Mistral Base | {fmt3(m['class_base']['regime_accuracy'])} | {fmt3(m['class_base']['predictability_accuracy'])} | {m['class_base']['n_samples']} |",
        f"| Mistral LoRA | {fmt3(m['class_lora']['regime_accuracy'])} | {fmt3(m['class_lora']['predictability_accuracy'])} | {m['class_lora']['n_samples']} |",
        "",
        "来源：`metrics_*.json` / `compare_base_lora.json`。真实标签以 `storm_growth` 为主；LoRA predictability 预测几乎全为 `high`（模式塌缩）。",
        "",
        "### 4.3 曲线方法（共享 pilot 窗口）",
        "",
        "| 模型 | Mean RMSE | Mean MAE | JSON valid | n | 口径 |",
        "|---|---:|---:|---:|---:|---|",
        f"| Mistral Base | {fmt3(cb['mean_rmse'])} | {fmt3(cb['mean_mae'])} | {cb['json_valid_rate']} | {cb['n_samples']} | 曲线 JSON→未来 24 h Hs |",
        f"| Mistral LoRA | {fmt3(cl['mean_rmse'])} | {fmt3(cl['mean_mae'])} | {cl['json_valid_rate']} | {cl['n_samples']} | 同上 |",
        f"| Persistence | {fmt3(cb['baselines']['mean_rmse_persist'])} | — | — | {cb['n_samples']} | 与曲线同窗 |",
        f"| LightGBM | {fmt3(cb['baselines']['mean_rmse_lgbm_at_numeric_leads'])} | — | — | — | 数值 lead 聚合均值 |",
        f"| Chronos-T5 | {fmt3(cb['baselines']['mean_rmse_chronos_at_numeric_leads'])} | — | — | — | 数值 lead 聚合均值 |",
        "",
        "**不得将 LoRA 表述为优于 Persistence 或 Chronos。**",
        "",
        "### 4.4 纳入报告的代表图（证据链）",
        "",
        fig_list,
        "",
        "每张图在 `report.html` 中均含六段式解释：背景作用、读图方法、物理/统计含义、能/不能结论、通俗解释、前后关联。",
        "",
        "## 5. 分析与讨论",
        "",
        "Base→LoRA 的曲线 RMSE 下降与 JSON validity=1.0 表明参数高效适配能教会 schema。"
        "但 Persistence 仍持有同窗最低平均 RMSE，Chronos 次之，LoRA 与 LightGBM 均值接近——"
        "系统应定位为结构化伴侣，而非 RMSE 冠军。",
        "",
        "分类上 regime 改善与 predictability 回退并存，说明多任务头不共享同一改进曲线。"
        "模板化 `reason`/`notes` 只适合界面原型，不能支持因果解释声称。",
        "",
        "### 风险与缓解",
        "",
        "| 风险/限制 | 表现 | 缓解 |",
        "|---|---|---|",
        "| 小样本 pilot | 曲线 n=12；分类 n=24 | 扩大 hold-out；避免生产声称 |",
        "| RMSE 非最优 | LoRA > Persistence/Chronos | 定位结构化伴侣 |",
        "| Predictability 回退 | 0.375→0.250；塌缩为 high | 重标、均衡、校准指标 |",
        "| 模板理由 | 非专家多样化标注 | 人评/物理约束监督 |",
        "| 无校准 UQ | uncertainty_level 定性 | coverage/CRPS 后再谈概率 UQ |",
        "| 空间泛化未测 | 限配置面板与年份 | 外站/外年验证（待补充） |",
        "",
        "## 6. 主要结论",
        "",
        f"1. 曲线 pilot（n={cb['n_samples']}）上，LoRA 相对 Base 降低平均 RMSE，并保持 JSON 全有效。",
        "2. LoRA 平均 RMSE 不低于 Persistence/Chronos；不得宣传为数值替代方案。",
        "3. regime 准确率上升，predictability 准确率下降；必须完整报告。",
        "4. 文本理由与 uncertainty 字段保持描述性，不作校准 UQ 或正式可解释性声明。",
        "",
        "## 7. 不足与展望",
        "",
        "- 评估样本量小；未声称置信区间或显著性。",
        "- 标签不平衡；外站/外年验证待补充。",
        "- 缺少真人评分的解释性研究与独立外部数据验证。",
        "- 作者、机构、资助、归档 DOI：待补充 / [AUTHOR_INPUT_NEEDED]。",
        "",
        "## 8. 数据与代码可用性",
        "",
        "NDBC 历史 stdmet 公开可获。本报告数字来自仓库内 metrics JSON；图来自 `data/processed/figures/`。"
        "代码：`scripts/` 与 `src/wave_llm/`。公共 DOI：待补充。可选 CMEMS/CDIP 遵循其许可。",
        "",
        "## 9. 参考文献",
        "",
        "完整英文条目见 `docs/manuscript_draft.md`；核验笔记见 `docs/literature_review_notes.md`"
        "（Chronos、PatchTST、Time-LLM、LLMTime、Tan et al. 2024、LoRA、Fan et al. 2020、Domala et al. 2022、Chaichitehrani et al. 2024 等）。",
        "",
        "## 10. 术语与符号表",
        "",
        "| 术语 | 含义 | 本报告角色 |",
        "|---|---|---|",
        "| Hs | Significant Wave Height（m） | 主预报量 |",
        "| Tp | Peak Wave Period | 上下文变量（若可得） |",
        "| lead / issue time | 预报时效 / 起报时刻 | 窗口对齐 |",
        "| Persistence | 持续法 | 强基线 |",
        "| LoRA | Low-Rank Adaptation | 微调方法 |",
        "| RMSE / MAE / Skill | 误差与技巧 | 主定量指标 |",
        "| wave_regime | 海况节律六类 | 分类标签 |",
        "| predictability_24h | high/medium/low | 探索性标签 |",
        "| reason / notes | 自由文本字段 | 模板槽，非 XAI |",
        "| uncertainty_level | 高/中/低 | 非校准概率 |",
        "| JSON validity | 可解析比例 | 结构成功率 |",
        "",
        "## 附录",
        "",
        "- **A. Git/发布：** 非 Git 仓库；仅本地生成，未 commit/push/PR/部署。",
        "- **B. 复现：** `python scripts/build_research_report.py`（建议 `wave_llm` 环境）；校验 `python scripts/validate_report_html.py`。",
        "- **C. 图说明：** Markdown 为便于版本阅读的源文件；逐图完整六段式解释与 Base64 内嵌图以 `report.html` 为准。",
        "",
    ]
    return "\n".join(lines)


def build_report_html(m: dict[str, Any], img_map: dict[str, str]) -> str:
    tables = build_tables_html(m)
    explains = figure_explanations(m)
    cb, cl = m["curve_base"], m["curve_lora"]
    toc_items = [
        ("sec-abstract", "摘要"),
        ("sec-bg", "1. 研究背景与目的"),
        ("sec-data", "2. 数据与方法"),
        ("sec-process", "3. 研究过程"),
        ("sec-results", "4. 结果展示"),
        ("sec-discuss", "5. 分析与讨论"),
        ("sec-concl", "6. 主要结论"),
        ("sec-limit", "7. 不足与展望"),
        ("sec-avail", "8. 数据与代码可用性"),
        ("sec-refs", "9. 参考文献"),
        ("sec-glossary", "10. 术语与符号表"),
        ("sec-appendix", "附录"),
    ]
    toc_html = "<ul>" + "".join(f'<li><a href="#{a}">{html.escape(t)}</a></li>' for a, t in toc_items) + "</ul>"

    fig_html_parts = []
    for spec in REPORT_FIGURES:
        path = FIG_DIR / spec["file"]
        if not path.exists():
            fig_html_parts.append(
                f'<div class="callout warn"><strong>缺图：</strong>{html.escape(spec["file"])} 待补充</div>'
            )
            continue
        uri = img_map[spec["file"]]
        fig_html_parts.append(
            fig_block(spec["id"], spec["title"], uri, spec["file"], explains.get(spec["id"], ""))
        )

    body = f"""
<div class="cover">
  <p style="letter-spacing:0.12em;color:#666;margin:0;">科研报告 · RESEARCH REPORT</p>
  <h1>结构化海浪预报：NDBC 浮标面板上的 Mistral Instruct-LoRA<br/>JSON 曲线、海况节律与可预测性标签</h1>
  <p class="meta">作者 / 机构 / 资助：[AUTHOR_INPUT_NEEDED] / 待补充</p>
  <p class="meta">基于本地真实 metrics 与 figures 自动生成 · 生成时间（UTC）{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}</p>
  <p class="meta">目标定位：方法型研究记录（非生产验证报告）</p>
</div>

<nav class="toc" id="toc">
  <strong>目录</strong>
  {toc_html}
</nav>

<section id="sec-abstract">
<h2>摘要</h2>
<p>本报告汇总项目 <span class="kbd">wave-prediction-mistral-tuning</span> 的真实实验结果：在 NDBC 多站小时面板上，比较 Persistence、LightGBM、Chronos-T5 与 Mistral-7B-Instruct-v0.3（Base / LoRA）在<strong>数值 lead 预报</strong>与<strong>结构化 JSON 曲线/分类</strong>两类任务中的表现。</p>
<p>曲线 pilot（n = {cb['n_samples']}）：LoRA 平均 RMSE {fmt3(cl['mean_rmse'])}，优于 Base {fmt3(cb['mean_rmse'])}，JSON validity = {cb['json_valid_rate']}；
同窗 Persistence {fmt3(cb['baselines']['mean_rmse_persist'])}、Chronos {fmt3(cb['baselines']['mean_rmse_chronos_at_numeric_leads'])}、LightGBM {fmt3(cb['baselines']['mean_rmse_lgbm_at_numeric_leads'])}。
<strong>LoRA 并未在数值上优于 Persistence 或 Chronos。</strong></p>
<p>分类 pilot（n = {m['class_base']['n_samples']}）：regime 准确率 {fmt3(m['class_base']['regime_accuracy'])} → {fmt3(m['class_lora']['regime_accuracy'])}；
predictability 准确率 {fmt3(m['class_base']['predictability_accuracy'])} → {fmt3(m['class_lora']['predictability_accuracy'])}（下降）。
文本字段 <span class="kbd">reason</span>/<span class="kbd">notes</span> 为模板化监督，不作因果解释声称。</p>
<div class="callout warn"><strong>诚实边界：</strong>本报告是小样本 pilot 的方法记录，不是业务上线验收。</div>
</section>

<section id="sec-bg">
<h2>1. 研究背景与目的</h2>
<p>Significant Wave Height (Hs，有效波高) 是航海与海洋工程的核心观测。数值 ML 与时序基础模型提供连续预报技巧，但业务值守往往还需要可解析的情境标签与简短文字槽位。指令微调 LLM 能输出 JSON，可是若把目标写成“击败 LightGBM/Chronos 的 RMSE”，既与文献警示（Tan 等）不符，也被本地数字否定。</p>
<p><strong>目的：</strong>（1）建立 NDBC→JSONL→LoRA 可复现配方；（2）在相同窗口协议下报告 Base→LoRA 改进与相对数值基线的边界；（3）明确模板理由 ≠ 可解释 AI，定性 uncertainty ≠ 校准概率。</p>
</section>

<section id="sec-data">
<h2>2. 数据与方法</h2>
{tables['data']}
{tables['models']}
<p>数值基线：Persistence 取起报时刻最近观测；LightGBM 使用窗口特征回归；Chronos-T5-tiny 零样本。分类 LoRA 输出 <span class="kbd">wave_regime</span>、<span class="kbd">predictability_24h</span>、<span class="kbd">notes</span>。
曲线 LoRA 输出 24 个小时的 <span class="kbd">hs_forecast_m</span>，以及 <span class="kbd">uncertainty_level</span> 与 <span class="kbd">reason</span>。</p>
<p>评价指标：RMSE = √((1/N) Σ(ŷ−y)²)；MAE = (1/N) Σ|ŷ−y|；Skill = 1 − RMSE<sub>model</sub>/RMSE<sub>persistence</sub>；另报 JSON validity 与分类准确率。</p>
{tables['cfg']}
</section>

<section id="sec-process">
<h2>3. 研究过程</h2>
<ol>
<li>真实 NDBC 下载与质控、1 h 重采样、面板拼装（scripts/02–04）。</li>
<li>构造 issue-time 窗口；训练数值基线并运行 Chronos（05 / 05b）。</li>
<li>导出分类 JSONL，划分 train/val/test，LoRA 训练与 Base 对比（06 / 06b / 07 / 08）。</li>
<li>导出曲线 JSONL（压缩 Hs 历史），pilot 训练与 hold-out 评估（06d / 07b / 08b）。</li>
<li>在共享窗口上汇总 Persistence / LightGBM / Chronos / Mistral 指标。</li>
<li>SciencePlots + Times New Roman 出图至 <span class="kbd">data/processed/figures/</span>。</li>
</ol>
<div class="callout"><strong>说明：</strong>本轮文档生成未再次调用 ChatGPT 附件或上传本地文件；写作边界沿用本地 manuscript / writing_notes。</div>
</section>

<section id="sec-results">
<h2>4. 结果展示</h2>
{tables['numeric']}
{tables['class']}
{tables['curve']}
<p>下列图件按证据链选取（约 {len(REPORT_FIGURES)} 张），避免把全部重复面板机械堆入。每张图均含来龙去脉式解释。</p>
{''.join(fig_html_parts)}
</section>

<section id="sec-discuss">
<h2>5. 分析与讨论</h2>
<p>Base→LoRA 的曲线 RMSE 下降与 JSON validity=1.0 说明参数高效适配能够教会 schema 并减少无结构乱码。与此同时，Persistence 仍持有同窗最低平均 RMSE，Chronos 次之，LoRA 与 LightGBM 均值接近——这强制我们把系统定位为<strong>结构化伴侣</strong>。</p>
<p>分类方面，regime 改善与 predictability 回退并存，且预测分布呈模式塌缩，说明多任务头并不共享同一改进曲线。模板化 <span class="kbd">reason</span> 只适合界面原型，不能支持“模型给出了海洋学因果解释”的表述。</p>
{tables['risk']}
</section>

<section id="sec-concl">
<h2>6. 主要结论</h2>
<ol>
<li>在 n={cb['n_samples']} 曲线 pilot 上，LoRA 相对 Base 降低平均 RMSE，并保持 JSON 全有效。</li>
<li>LoRA 平均 RMSE 不低于 Persistence/Chronos；不得宣传为数值替代方案。</li>
<li>regime 准确率上升，predictability 准确率下降；必须完整报告。</li>
<li>文本理由与 uncertainty 字段保持描述性，不作校准 UQ 或正式可解释性声明。</li>
</ol>
</section>

<section id="sec-limit">
<h2>7. 不足与展望</h2>
<ul>
<li>评估样本量小；未声称置信区间或显著性。</li>
<li>标签不平衡；外站/外年验证待补充。</li>
<li>缺少真人评分的解释性研究与独立外部数据验证。</li>
<li>作者、机构、资助、归档 DOI：待补充。</li>
</ul>
</section>

<section id="sec-avail">
<h2>8. 数据与代码可用性</h2>
<p>NDBC 历史 stdmet 公开可获。本报告数字来自仓库内 metrics JSON；图来自 <span class="kbd">data/processed/figures/</span>。
代码：<span class="kbd">scripts/</span> 与 <span class="kbd">src/wave_llm/</span>。公共 DOI：待补充。可选 CMEMS/CDIP 遵循其许可。</p>
</section>

<section id="sec-refs">
<h2>9. 参考文献</h2>
<p>完整英文条目见 <span class="kbd">docs/manuscript_draft.md</span>。关键已核验方向包括：Chronos、PatchTST、Time-LLM、LLMTime、Tan et al. 2024、LoRA、Fan et al. 2020、Domala et al. 2022、Chaichitehrani et al. 2024 等（详见 <span class="kbd">docs/literature_review_notes.md</span>）。</p>
</section>

<section id="sec-glossary">
<h2>10. 术语与符号表</h2>
<table>
<caption>表 8 术语与符号</caption>
<thead><tr><th>符号/术语</th><th>全称与含义</th><th>本报告中的角色</th></tr></thead>
<tbody>
<tr><td>Hs</td><td>Significant Wave Height，有效波高（m）</td><td>主预报量</td></tr>
<tr><td>Tp</td><td>Peak Wave Period，峰值周期</td><td>上下文变量（若可得）</td></tr>
<tr><td>lead time</td><td>预报时效</td><td>6–72 h 数值评估</td></tr>
<tr><td>issue time</td><td>起报时刻</td><td>窗口对齐的关键</td></tr>
<tr><td>Persistence</td><td>持续法预报</td><td>强基线</td></tr>
<tr><td>residual / regime</td><td>残差思维 / 海况节律</td><td>标签与讨论</td></tr>
<tr><td>predictability</td><td>可预报性等级</td><td>探索性标签</td></tr>
<tr><td>LoRA</td><td>Low-Rank Adaptation</td><td>微调方法</td></tr>
<tr><td>JSON validity</td><td>输出可解析比例</td><td>结构成功率</td></tr>
<tr><td>reason / notes</td><td>自由文本字段</td><td>模板槽，非 XAI</td></tr>
<tr><td>uncertainty_level</td><td>高/中/低描述</td><td>非校准概率</td></tr>
<tr><td>RMSE / MAE / Skill</td><td>误差与技巧</td><td>主定量指标</td></tr>
</tbody>
</table>
</section>

<section id="sec-appendix">
<h2>附录</h2>
<p><strong>A. Git / 发布状态：</strong>项目根目录未检测到 Git 仓库；本输出仅为本地文件生成，未 commit / push / PR / 部署。</p>
<p><strong>B. 复现命令：</strong><span class="kbd">python scripts/build_research_report.py</span>（建议 <span class="kbd">wave_llm</span> 环境）。</p>
<p><strong>C. 验证：</strong><span class="kbd">python scripts/validate_report_html.py</span>。</p>
</section>

<div class="footer-note">
页眉/页脚说明：本 HTML 为自包含学术报告；打印请使用浏览器“打印→另存 PDF”，或由构建脚本经 Chromium/Chrome 生成的 report.pdf。
中文字体优先 Microsoft YaHei / SimSun；英文 Times New Roman。
</div>
"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>结构化海浪预报科研报告 — NDBC × Mistral LoRA</title>
<style>
{shared_css(zh=True)}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def md_to_simple_html_paragraphs(text: str) -> str:
    """Very small Markdown subset → HTML for the English paper body."""
    lines = text.splitlines()
    out: list[str] = []
    in_table = False
    table_buf: list[str] = []
    in_code = False
    code_buf: list[str] = []

    def flush_table() -> None:
        nonlocal table_buf, in_table
        if not table_buf:
            return
        rows = [r for r in table_buf if r.strip()]
        table_buf = []
        in_table = False
        if len(rows) < 2:
            return
        def split_row(r: str) -> list[str]:
            r = r.strip().strip("|")
            return [c.strip() for c in r.split("|")]
        header = split_row(rows[0])
        body_rows = []
        for r in rows[2:] if re.match(r"^\s*\|?\s*:?-+:?\s*\|", rows[1]) else rows[1:]:
            if re.match(r"^\s*\|?\s*:?-+:?\s*\|", r):
                continue
            body_rows.append(split_row(r))
        html_rows = ["<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in header) + "</tr>"]
        for br in body_rows:
            html_rows.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in br) + "</tr>")
        out.append("<table>\n" + "\n".join(html_rows) + "\n</table>")

    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                out.append("<pre class='kbd'>" + html.escape("\n".join(code_buf)) + "</pre>")
                code_buf = []
                in_code = False
            else:
                if in_table:
                    flush_table()
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue
        if "|" in line and re.match(r"^\s*\|", line):
            in_table = True
            table_buf.append(line)
            continue
        elif in_table:
            flush_table()

        if not line.strip():
            continue
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:].strip())}</h1>")
        elif line.startswith("## "):
            title = line[3:].strip()
            aid = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()
            out.append(f'<h2 id="{aid}">{html.escape(title)}</h2>')
        elif line.startswith("### "):
            out.append(f"<h3>{html.escape(line[4:].strip())}</h3>")
        elif line.startswith("**") and line.endswith("**"):
            out.append(f"<p><strong>{html.escape(line.strip('*').strip())}</strong></p>")
        elif line.startswith("- ") or line.startswith("* "):
            out.append(f"<li>{html.escape(line[2:].strip())}</li>")
        else:
            # light inline escapes
            t = html.escape(line)
            t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
            t = re.sub(r"`([^`]+)`", r"<span class='kbd'>\1</span>", t)
            out.append(f"<p>{t}</p>")
    if in_table:
        flush_table()
    # wrap consecutive li
    merged: list[str] = []
    i = 0
    while i < len(out):
        if out[i].startswith("<li>"):
            merged.append("<ul>")
            while i < len(out) and out[i].startswith("<li>"):
                merged.append(out[i])
                i += 1
            merged.append("</ul>")
        else:
            merged.append(out[i])
            i += 1
    return "\n".join(merged)


def build_paper_html(m: dict[str, Any], img_map: dict[str, str]) -> str:
    draft_path = ROOT / "docs" / "manuscript_draft.md"
    draft = draft_path.read_text(encoding="utf-8")
    # Drop figure checklist appendix-heavy tails for cleaner HTML body but keep refs
    body_md = draft
    body_html = md_to_simple_html_paragraphs(body_md)

    fig_sections = []
    for spec in PAPER_FIGURES:
        path = FIG_DIR / spec["file"]
        if not path.exists():
            continue
        uri = img_map[spec["file"]]
        fig_sections.append(
            f"""<figure id="{spec['id']}">
<img src="{uri}" alt="{html.escape(spec['caption'])}" />
<figcaption>{html.escape(spec['caption'])}<br/>Source file: {html.escape(spec['file'])}</figcaption>
</figure>"""
        )

    metrics_box = f"""
<div class="callout">
<strong>Frozen pilot metrics (local JSON):</strong>
Curve n={m['curve_base']['n_samples']}: Base RMSE {fmt3(m['curve_base']['mean_rmse'])},
LoRA RMSE {fmt3(m['curve_lora']['mean_rmse'])}, Persistence {fmt3(m['curve_base']['baselines']['mean_rmse_persist'])},
Chronos {fmt3(m['curve_base']['baselines']['mean_rmse_chronos_at_numeric_leads'])},
LightGBM {fmt3(m['curve_base']['baselines']['mean_rmse_lgbm_at_numeric_leads'])}; JSON valid=1.0.
Classification n={m['class_base']['n_samples']}: regime {fmt3(m['class_base']['regime_accuracy'])}→{fmt3(m['class_lora']['regime_accuracy'])};
predictability {fmt3(m['class_base']['predictability_accuracy'])}→{fmt3(m['class_lora']['predictability_accuracy'])}.
Author/funding: [AUTHOR_INPUT_NEEDED].
</div>
"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Structured wave forecasting with instruction-tuned LLMs</title>
<style>
{shared_css(zh=False)}
body {{ font-family: "Times New Roman", Times, serif; }}
h1,h2,h3 {{ font-family: "Times New Roman", Times, serif; }}
</style>
</head>
<body>
<div class="cover">
<h1>Structured wave forecasting with instruction-tuned LLMs: JSON Hs curves, regime labels, and predictability rationales from NDBC buoys</h1>
<p class="meta">Manuscript HTML export · Ocean Engineering–style draft · [AUTHOR_INPUT_NEEDED]</p>
<p class="meta">Numbers frozen from local metrics JSON · Figures embedded as Base64</p>
</div>
{metrics_box}
{body_html}
<h2 id="embedded-figures">Embedded figures</h2>
{''.join(fig_sections)}
<div class="footer-note">Generated by scripts/build_research_report.py · Offline self-contained HTML</div>
</body>
</html>
"""


def html_to_pdf(html_path: Path, pdf_path: Path) -> str:
    """Render PDF via Playwright using system Chrome/Edge if possible."""
    from playwright.sync_api import sync_playwright

    uri = html_path.resolve().as_uri()
    last_err = None
    with sync_playwright() as p:
        for channel in ("chrome", "msedge", None):
            try:
                if channel:
                    browser = p.chromium.launch(channel=channel)
                else:
                    browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(uri, wait_until="networkidle", timeout=120000)
                page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    print_background=True,
                    margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"},
                )
                browser.close()
                return f"playwright:{channel or 'chromium'}"
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
    raise RuntimeError(f"PDF generation failed: {last_err}")


def verify_pdf_text(pdf_path: Path) -> dict[str, Any]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages[: min(5, len(reader.pages))]:
        texts.append(page.extract_text() or "")
    blob = "\n".join(texts)
    return {
        "pages": len(reader.pages),
        "sample_chars": len(blob),
        "has_chinese": bool(re.search(r"[\u4e00-\u9fff]", blob)),
        "has_english_title_hint": ("Structured" in blob) or ("wave" in blob.lower()) or ("海浪" in blob) or ("科研" in blob),
        "snippet": blob[:400].replace("\n", " "),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pdf", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m = load_metrics()

    needed = {f["file"] for f in REPORT_FIGURES} | {f["file"] for f in PAPER_FIGURES}
    img_map: dict[str, str] = {}
    missing = []
    for name in sorted(needed):
        path = FIG_DIR / name
        if not path.exists():
            missing.append(name)
            continue
        img_map[name] = img_data_uri(path)
    if missing:
        print("WARNING missing figures:", ", ".join(missing))

    md = build_research_report_md(m)
    (OUT_DIR / "research_report.md").write_text(md, encoding="utf-8")

    report_html = build_report_html(m, img_map)
    paper_html = build_paper_html(m, img_map)
    (OUT_DIR / "report.html").write_text(report_html, encoding="utf-8")
    (OUT_DIR / "paper.html").write_text(paper_html, encoding="utf-8")

    print(f"Wrote {OUT_DIR / 'research_report.md'} ({len(md)} chars)")
    print(f"Wrote {OUT_DIR / 'report.html'} ({(OUT_DIR / 'report.html').stat().st_size} bytes)")
    print(f"Wrote {OUT_DIR / 'paper.html'} ({(OUT_DIR / 'paper.html').stat().st_size} bytes)")

    if not args.skip_pdf:
        for html_name, pdf_name in (("report.html", "report.pdf"), ("paper.html", "paper.pdf")):
            html_path = OUT_DIR / html_name
            pdf_path = OUT_DIR / pdf_name
            method = html_to_pdf(html_path, pdf_path)
            info = verify_pdf_text(pdf_path)
            print(f"PDF {pdf_name}: method={method}, pages={info['pages']}, "
                  f"chars_sample={info['sample_chars']}, chinese={info['has_chinese']}, "
                  f"title_hint={info['has_english_title_hint']}")
            print(f"  snippet: {info['snippet'][:180]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
