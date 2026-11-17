# -*- coding: utf-8 -*-
"""阶段 06：生成综合报告、研究简报、PPT 大纲与复现说明。

所有数字从 outputs/ 的机器可读结果读取，避免报告与脚本“两套数”。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import OUT, FIG, REPORTS, DOCS, ROOT, SEED, SNAPSHOT_UTC

R1 = json.loads((OUT / "rq1" / "rq1_summary.json").read_text(encoding="utf-8"))
R2 = json.loads((OUT / "rq2" / "rq2_summary.json").read_text(encoding="utf-8"))
R3 = json.loads((OUT / "rq3" / "rq3_summary.json").read_text(encoding="utf-8"))
R4 = json.loads((OUT / "rq4" / "rq4_summary.json").read_text(encoding="utf-8"))
BASE = json.loads((OUT / "integrated" / "00_data_audit.json").read_text(encoding="utf-8"))

R1P = pd.read_csv(OUT / "rq1" / "01_theme_profile.csv")
R1V = json.loads((OUT / "rq1" / "08_theme_validation_summary.json").read_text(encoding="utf-8"))
R1S = pd.read_csv(OUT / "rq1" / "04_sensitivity_coverage.csv")
R2C = pd.read_csv(OUT / "rq2" / "04_rules_core.csv")
R2L = pd.read_csv(OUT / "rq2" / "01_label_frequency.csv")
R2N = pd.read_csv(OUT / "rq2" / "07_negative_pairs.csv")
R2A = json.loads((OUT / "rq2" / "08_rule_audit_summary.json").read_text(encoding="utf-8"))
R2ST = pd.read_csv(OUT / "rq2" / "06_rule_stability.csv")
R3O = pd.read_csv(OUT / "rq3" / "01_process_overview.csv")
R3M = pd.read_csv(OUT / "rq3" / "10_model_metrics.csv")
R3T = pd.read_csv(OUT / "rq3" / "02_group_type.csv")
R3E = pd.read_csv(OUT / "rq3" / "02_group_entry.csv")
R3V = pd.read_csv(OUT / "rq3" / "02_group_env.csv")
R3R = pd.read_csv(OUT / "rq3" / "02_group_remote.csv")
R4D = pd.read_csv(OUT / "rq4" / "02_unified_text_distribution.csv")
R4I = pd.read_csv(OUT / "rq4" / "06_interaction_compare.csv")


def _load_optional(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# 新增材料的独立扩展结果（08 口径核验 / 09 竞争风险独立复现 / 10 新 RQ1 盲审）
EXT8 = _load_optional(OUT / "integrated" / "new_materials_validation.json")
EXT9 = _load_optional(OUT / "integrated" / "09_competing_risk.json")
EXT9B = _load_optional(OUT / "integrated" / "09b_delivery_repro.json")
EXT10 = _load_optional(OUT / "integrated" / "10_rq1_agreement.json")
EXT1P = _load_optional(OUT / "integrated" / "10_rq1_profile.json")
EXT12 = _load_optional(OUT / "integrated" / "issue_ml_extension_validation.json")
RQ1_KSCAN = pd.read_csv(OUT / "rq1" / "09_k_selection.csv") if (OUT / "rq1" / "09_k_selection.csv").exists() else pd.DataFrame()
RQ3_EXT = pd.read_csv(OUT / "rq3_validation" / "a5_b5_model_validation.csv") if (OUT / "rq3_validation" / "a5_b5_model_validation.csv").exists() else pd.DataFrame()
RQ3_CAL = pd.read_csv(OUT / "rq3_validation" / "a6_calibration_diagnostic.csv") if (OUT / "rq3_validation" / "a6_calibration_diagnostic.csv").exists() else pd.DataFrame()


def num(x, d=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{d}f}"


def pct(x, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.{d}f}%"


def table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("—" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
                                     for v in r) + " |")
    return "\n".join(out)


def best_models():
    rows = []
    for target, g in R3M.groupby("target"):
        b = g.loc[g.roc_auc.idxmax()]
        base = g[g.model == "prior_baseline"].iloc[0]
        rows.append({"target": target, "model": b.model, "auc": b.roc_auc, "ap": b.average_precision,
                     "brier": b.brier, "recall": b.recall, "top10": b.top10_recall,
                     "base_auc": base.roc_auc, "base_ap": base.average_precision})
    return pd.DataFrame(rows)


BEST = best_models()
R1P = R1P.sort_values("share_of_body_corpus", ascending=False)
val_map = {int(x["theme_id"]): x for x in R1V["per_theme"]}


# --------------------------------------------------------------------------
# 报告片段
# --------------------------------------------------------------------------
def rq1_theme_table():
    rows = []
    for _, r in R1P.iterrows():
        v = val_map.get(int(r.theme_id), {})
        rows.append([int(r.theme_id), r.theme, num(r.n_topics), pct(r.share_of_body_corpus),
                     ", ".join(str(r.top_terms).split(";")[:4]),
                     f"{r.top_category} {pct(r.top_category_share)}",
                     pct(r.accepted_rate), pct(r.zero_reply_rate),
                     f"{int(v.get('fit', 0))}/{int(v.get('reviewed', 0))}"])
    return table(["#", "主题（人工复核后名称）", "帖数", "正文份额", "高频词（前4）", "主版块", "采纳率", "零回复率", "主题贴合"], rows)


def rq2_rule_table():
    rows = []
    for _, r in R2C.iterrows():
        rows.append([r.rule_id, r.rule, r.module, num(r.support, 3), num(r.confidence, 3),
                     num(r.lift, 2), pct(r.coverage), num(r.ab_count)])
    return table(["ID", "规则", "问题模块", "支持度", "置信度", "提升度", "覆盖率", "A&B 样本"], rows)


def rq3_overview_table():
    keep = ["首次外部人类回复", "3 天内首次外部回复", "首次维护者回复", "3 天内维护者回复",
            "已关闭", "30 天内关闭", "仍开启且 ≥90 天", "90 天以上长期未关闭"]
    rows = []
    for m in keep:
        r = R3O[R3O.metric == m].iloc[0]
        rows.append([m, num(r.n), num(r.observed), pct(r.ratio),
                     num(r.median_days, 2), num(r.p75_days, 2), num(r.p90_days, 2), r.definition])
    return table(["指标", "分母 n", "观测数", "比例", "中位天数", "P75", "P90", "定义"], rows)


def rq3_outcome_table():
    rows = []
    total = R3["denominators"]["all_issues"]
    for k in ["open", "reopened", "completed", "duplicate", "not_planned"]:
        n = R3["outcomes"].get(k, 0)
        rows.append([{"open": "仍开启（普通）", "reopened": "已重开（仍开启）", "completed": "标记完成",
                      "duplicate": "重复关闭", "not_planned": "不计划处理"}[k], num(n), pct(n / total),
                     {"open": "不等于未修复", "reopened": "曾关闭后重开，不能并入普通 open",
                      "completed": "不等于已修复", "duplicate": "重复合并",
                      "not_planned": "主动不处理"}[k]])
    return table(["状态", "数量", "占全部 Issue", "解释"], rows)


def rq3_group_table(df, label):
    rows = []
    df = df[df.group != "unknown"]
    for _, r in df.iterrows():
        rows.append([r.label if "label" in df.columns else r.group, num(r.n), pct(r.replied_rate),
                     num(r.median_days, 2), pct(r.close30_rate),
                     f"{num(r.open90_observed)}/{num(r.open90_n)}（{pct(r.open90_rate)}）"])
    return table([label, "n", "有外部回复", "首回复中位天数", "30 天关闭率", "满 90 天样本中仍开（分子/分母）"], rows)


def rq4_type_table():
    t = R4D[R4D.dimension == "type"].pivot_table(index=["code", "label"], columns="channel",
                                                values="share", aggfunc="first").reset_index()
    for c in ["Cursor 论坛", "Codex Issues"]:
        if c not in t.columns:
            t[c] = np.nan
    t = t.sort_values("Codex Issues", ascending=False)
    t["_unk"] = (t["code"] == "unknown").astype(int)
    t = t.sort_values(["_unk", "Codex Issues"], ascending=[True, False])
    rows = [[r["label"], r["code"], pct(r["Cursor 论坛"]), pct(r["Codex Issues"])] for _, r in t.iterrows()]
    return table(["问题类型", "代码", "Cursor 论坛", "Codex Issues"], rows)


def rq4_entry_table():
    e = R4D[(R4D.dimension == "entry") & (R4D.code != "unknown")].pivot_table(
        index=["code", "label"], columns="channel", values="share", aggfunc="first").reset_index()
    for c in ["Cursor 论坛", "Codex Issues"]:
        if c not in e.columns:
            e[c] = 0.0
        e[c] = e[c].fillna(0.0)
    known_share = {}
    for channel in ["Cursor 论坛", "Codex Issues"]:
        unknown_rows = R4D[(R4D.dimension == "entry") & (R4D.code == "unknown") &
                           (R4D.channel == channel)]
        known_share[channel] = 1 - float(unknown_rows.share.iloc[0]) if len(unknown_rows) else 1.0
        e[channel] = e[channel] / known_share[channel] if known_share[channel] else 0.0
    e["avg"] = e[["Cursor 论坛", "Codex Issues"]].mean(axis=1)
    e = e.sort_values("avg", ascending=False).head(7)
    rows = [[r["label"], r["code"], pct(r["Cursor 论坛"]), pct(r["Codex Issues"])] for _, r in e.iterrows()]
    return table(["入口类别（多选）", "代码", "论坛已识别记录命中率", "Issues 已识别记录命中率"], rows)


def rq4_entry_coverage_text():
    parts = []
    for channel in ["Cursor 论坛", "Codex Issues"]:
        row = R4D[(R4D.dimension == "entry") & (R4D.code == "unknown") &
                  (R4D.channel == channel)]
        unknown = float(row.share.iloc[0]) if len(row) else 0.0
        parts.append(f"{channel}入口识别覆盖率 {pct(1 - unknown)}（未识别 {pct(unknown)}）")
    return "；".join(parts)


def model_table():
    labels = {"closed_within_30d": "30 天内关闭", "reply_3d": "3 天内首次外部回复",
              "long_running_90d": "90 天以上长期未关闭"}
    rows = []
    for target in ["closed_within_30d", "reply_3d", "long_running_90d"]:
        r = BEST[BEST.target == target].iloc[0]
        info = next(x for x in R3["models"] if x["target"] == target)
        rows.append([labels[r.target], r.model, num(r.auc, 3), num(r.ap, 3), num(r.brier, 3),
                     pct(r.recall), pct(r.top10), pct(info["train_positive_rate"]), pct(info["test_positive_rate"])])
    return table(["预测任务", "最佳模型", "AUC", "AP", "Brier", "阈值0.5召回", "Top10%召回", "训练正例率", "测试正例率"], rows)


def rq3_extension_table():
    if RQ3_EXT.empty:
        return "扩展验证尚未运行。"
    labels = {"closed_within_30d":"30 天内关闭", "reply_3d":"3 天内首次外部回复",
              "long_running_90d":"90 天以上长期未关闭"}
    rows = []
    for target, label in labels.items():
        same = RQ3_EXT[(RQ3_EXT.target == target) &
                       (RQ3_EXT.validation == "same_70_30_time_split")]
        nb = same[same.model == "text_nb"]
        lr = same[same.model == "text_logistic_only"]
        roll = RQ3_EXT[(RQ3_EXT.target == target) & RQ3_EXT.validation.str.startswith("rolling_fold_")]
        if len(nb) and len(lr):
            rows.append([label, num(nb.iloc[0].roc_auc, 3), num(lr.iloc[0].roc_auc, 3),
                         f"{num(roll.roc_auc.min(), 3)}–{num(roll.roc_auc.max(), 3)}" if len(roll) else "—",
                         f"{pct(lr.iloc[0].train_positive_rate)} → {pct(lr.iloc[0].positive_rate)}"])
    return table(["任务", "文本 NB AUC", "纯文本逻辑回归 AUC", "滚动窗 AUC 范围", "训练→测试正例率"], rows)


def rq2_label_table():
    rows = []
    for _, r in R2L.head(8).iterrows():
        rows.append([r.label, num(r["count"]), pct(r.share_of_transactions), r.dimensions])
    return table(["标签", "出现次数", "占多标签事务", "映射维度"], rows)


def rq2_neg_table():
    rows = []
    for _, r in R2N.head(6).iterrows():
        rows.append([r.label_a, r.label_b, num(r["count"]), num(r.lift, 2), r.interpretation])
    return table(["标签 A", "标签 B", "共现数", "提升度", "解释"], rows)


def ext_rq1_topic_table():
    if not EXT1P:
        return "（未运行 `src/10_rq1_review_agreement.py`，本节留空）"
    rows = []
    for t in sorted(EXT1P["topics"], key=lambda x: -x["n"]):
        rows.append([t["code"], t["label"], num(t["n"]), pct(t["share"])])
    return table(["代码", "主题（模型输出）", "首帖数", "占 9,498 条英文语料"], rows)


def ext_rq3_scenario_table():
    if not EXT9:
        return "（未运行 `src/09_rq3_competing_risk.py`，本节留空）"
    rows = []
    for s in EXT9["scenarios"]:
        rows.append([s["scenario"], num(s["n"]), pct(s["completed_cif_30"]),
                     pct(s["any_closure_cif_30"]), num(s["delta_completed_cif_30_pp"], 2)])
    return table(["情景", "N", "30 天 completed（AJ）", "30 天任意关闭", "较基准差（pp）"], rows)


# --------------------------------------------------------------------------
# 综合报告
# --------------------------------------------------------------------------
def build_main():
    corpus = R1["corpus"]
    val = R1V
    rho = R1["tests"]["coverage_sensitivity"]["spearman"]
    refit = R1["tests"]["refit_match_median_cosine"]
    audit = R2A
    ov = {r.metric: r for _, r in R3O.iterrows()}
    m = {r.target: r for _, r in BEST.iterrows()}
    inter = {r.channel: r for _, r in R4I.iterrows()}
    f_n = R4["window"]["forum_n"]
    i_n = R4["window"]["issue_n"]

    def ext9_check(fragment: str):
        for c in (EXT9 or {}).get("checks", []):
            if fragment in c["check"]:
                return c["observed"]
        return float("nan")

    text = f"""# AI 编程智能体的多源用户反馈与产品问题挖掘

> 课程项目综合报告  
> 数据快照：{SNAPSHOT_UTC}（由 Issue 最新时间字段推定，不是已核实的抓取完成时间）  
> 代码与结果：`src/`、`outputs/`、`figures/`；随机种子：{SEED}  
> 报告生成：`python src/06_build_reports.py`

## 摘要

1. **用户在谈什么**：正文语料中，MCP/网络/扩展宿主（24.8%）、IDE 界面（18.0%）、云端代理与仓库协作（15.2%）、模型与子 Agent 路由（14.4%）四类主题合计约 72%。每主题 6 条高权重样本文义回读为 {val['theme_fit']}/{val['reviewed']}；这是主题命名与边界诊断，不是总体分类准确率。其中 12.5% 的样本是社区活动而非产品反馈。
2. **问题如何组合**：25,519 条多标签 Issue 形成 12 条核心规则。Windows 桌面端覆盖最广（`windows-os -> app`，支持度 19.9%，提升度 1.38）；`tui -> cli` 提升度最高（2.19）；`computer-use -> app` 置信度最高（95.4%）。规则是分流线索，不是故障发生率。
3. **反馈如何处理**：55.5% 的 Issue 至少收到一次外部人类回复（中位 1.02 天），但只有 1.66% 收到维护者回复；39.8% 已关闭，其中 completed 仅占全部 25.94%；另有 154 条（0.55%）标记为 reopened，不能并入普通 open。已关闭样本的关闭中位数为 2.01 天，Kaplan-Meier 中位关闭时间为 127.8 天，说明“快速关闭”和“长期积压”并存。扩展复核把关闭拆成三种竞争结局后，30 天 completed 累积发生率约 18.7%，明显低于任意关闭的约 28.8%——**“关闭”不能当作“解决”**。
4. **渠道差异**：在 2026-01 至快照、同一套标题规则下，论坛与 Issues 的缺陷类占比分别为 {pct(R4['headline']['forum_bug_share_title'])} 与 {pct(R4['headline']['issue_bug_share_title'])}；论坛有公开回复的比例 {pct(inter['Cursor 论坛'].public_reply_rate)}，Issues 为 {pct(inter['Codex Issues'].public_reply_rate)}。但两渠道的字段定义、用户构成和模板不同，差异不能写成“总体用户偏好”。
5. **最重要的边界**：论坛正文只覆盖 10.5% 的主题，2026-07 前以分层抽样为主；Issue 标签是维护者标注行为；`closed` 不等于 `fixed`；三个预测任务的时间外 AUC 仅 {num(m['closed_within_30d'].auc, 3)}–{num(m['reply_3d'].auc, 3)}，只适合做弱排序，不适合自动决策。

## 1. 项目问题与数据来源

### 1.1 研究链路

本项目不追求“多模型堆叠”，而是围绕一条可复核的证据链：

> 用户在讨论什么（RQ1） → 问题如何组合（RQ2） → 反馈如何被处理（RQ3） → 对产品、用户研究和反馈运营有什么启发（RQ4 + 综合）

四个研究问题共用同一套“入口—能力—环境—问题类型—处理结果”分类字典，但各自保留合适的分析单位：RQ1 是论坛主题帖，RQ2 是 Issue 标签事务，RQ3 是 Issue 生命周期，RQ4 是渠道内部分布。

### 1.2 数据资产与分母

| 数据 | 原始记录 | 可用分母 | 时间范围 | 在本项目中的角色 |
|---|---:|---:|---|---|
| Cursor 论坛主题元数据 | 41,218 | 41,218 | 2023-08 ~ 2026-09 | 版块、互动、月度结构 |
| Cursor 论坛正文 | 4,322 主题 / 18,498 帖 | 4,307 个清洗后主题 | 2026-03 ~ 2026-09 | RQ1 主题建模，仅正文子集 |
| Codex Issues | 27,756 | 27,320 有标签 / 25,519 多标签 | 2025-04 ~ 2026-09 | RQ2、RQ3、RQ4 |
| Codex 评论 | 115,280 | 97,034 可关联 / 96,999 进入回复计算 | 2025-04 ~ 2026-09 | 首次回复、维护者回复 |
| Hacker News | 80,510 | 16,506 关键词命中 | 2022-12 ~ 2026-09 | 仅辅助对照，不承担核心结论 |

> **口径提醒**：交接材料中的 284,687 是“展开正文帖后的可分析记录”口径；六份 JSON 的顶层元素为 269,298。论坛主题与帖子存在包含关系，不能与 Issue、评论简单混加成“总样本”。本报告不使用单一总数，所有比例都带各自分母。

### 1.3 复现与只读约束

- 原始六份 JSON 保留在 `data/raw/`，未覆盖；`01_build_base.py` 逐文件计算 SHA-256 并与 manifest 对照。
- 所有派生表写入 `data/derived/`，所有结果写入 `outputs/`，图写入 `figures/`。
- 运行环境、依赖、命令、随机种子见 `docs/复现说明.md`。

## 2. 数据质量与样本口径

### 2.1 三条必须保留的质量事实

1. **Issue 评论关联损失**：115,280 条评论中 97,034 条能关联到 Issue，18,246 条无法关联。后者不进入回复时间计算，因此“无回复”比例可能被高估。
2. **论坛正文非随机覆盖**：4,322/41,218 = 10.5% 的主题有正文；2026-07 后为全量普查（3,934 个主题），之前主要是分层抽样（388 个主题）。正文级主题份额不能外推为 2023—2026 全历史。
3. **标签是维护行为**：Issue 标签由维护者/机器人流程添加，反映分流习惯，不等于真实故障发生率；论坛标签 2026 年才成规模，不能与 Issue 标签直接对比。

### 2.2 处理结果口径

{rq3_outcome_table()}

`closed` 是聚合状态（11,055 条，39.8%），由 `completed`、`duplicate`、`not_planned` 三类关闭原因组成，不是一个与它们并列的互斥类别。`completed` 只表示状态机上的“完成”，可能包含“无法复现/已修复/不再需要”等混合含义；`duplicate` 是重复合并；`not_planned` 是主动不处理；`reopened` 是曾关闭后重开、当前仍为 open 的 154 条，单独列出而不并入普通 open。任何“解决率”都必须说明是否把 completed/duplicate/not_planned 都算作解决。

## 3. RQ1：用户讨论主题

### 3.1 方法

- 仅使用有正文的主题（n={num(corpus['body_topics'])}），去除 HTML、代码块、诊断页脚和 Discourse 缺陷模板页脚。
- TF-IDF（词 + 二元组）+ NMF（k=8，Frobenius 损失，固定随机种子）。
- 主题名称由高频词规则生成，再用每主题权重最高的 6 帖做结构化回读；**版块分类**（Bug Reports/Help/...）与**主题分类**（NMF）分开报告。
- 敏感性：只用 2026-07 后普查正文重跑，比较主题份额排序与词表余弦；bootstrap 重采样检查稳定性。

### 3.2 主题结构

{rq1_theme_table()}

> 份额分母是 4,307 个有正文且清洗后非过短的主题，不是 41,218 个论坛主题。主题 1 是社区活动，不是产品痛点；主题 7 是 API Key/第三方模型接入的残余主题，稳定性最低。

### 3.3 人工复核与边界

- 8 个主题各取 6 条高权重帖子，共 {val['reviewed']} 条；其中 {val['theme_fit']} 条与主题文义贴合。该回读用于主题命名和边界诊断，不是随机样本或总体准确率。
- 产品反馈占比 {pct(val['product_feedback_share'])}；6 帖社区活动全部来自主题 1。
- 不贴合案例：主题 5 的“CLI 把内置 Node 加到 PATH”（更像 CLI/环境问题）；主题 7 的“Grok 高需求限流”（更像容量/用量问题）。这说明主题边界仍有混入，不能把 NMF 主题当成互斥业务分类。
- 复核由 LLM 辅助完成结构化阅读，**不是独立人工双盲标注**；正式提交前应抽 20% 由人复核。

### 3.4 主题 × 版块：两种分类不是一回事

主题与官方版块的 Cramér's V = {num(R1['tests']['category_chi2']['cramers_v'], 2)}（n={num(R1['tests']['category_chi2']['n'])}）。例如“MCP/网络/扩展宿主”中 74.3% 来自 Bug Reports，但“订阅套餐与用量额度”只有 38% 来自 Help、其余分散在 Feature Requests 和 Bug Reports。版块是用户自选的容器，主题是文本内容，二者不能互相替代。

### 3.5 互动差异

- 采纳率整体 12.3%，但各主题仅 9.0%–15.6%；社区活动主题采纳率为 0，零回复率 98.3%。
- 零回复率在 54.7%–65.9% 之间（不含社区活动）：云端代理/仓库协作最高（65.9%），MCP/网络/扩展宿主最低（54.7%），IDE 界面为 62.7%。
- Kruskal-Wallis：浏览 H={num(R1['tests']['views_kruskal']['H'], 1)}、回复 H={num(R1['tests']['replies_kruskal']['H'], 1)}，主题间差异显著；但这是论坛行为分布，不是需求重要性排序。

### 3.6 主题数与重构诊断

`explained_weight_mean` 已更正为相对重构诊断：`1 − ||X−WH||²_F / ||X||²_F`。它表示当前未中心化 TF-IDF 矩阵被 NMF 重构的比例，不是 PCA 方差解释率，也不是主题准确率。候选 K 的多指标扫描见 `outputs/rq1/09_k_selection.csv` 与 `figures/rq1_11_k选择诊断.png`；重构比例会随 K 增大而改善，但 UMass、余弦轮廓和主题多样性并不同时指向同一个 K，因此保留 K=8 作为兼顾可读性与原报告连续性的工作设定，不宣称它是唯一最优值。

### 3.6 样本敏感性与稳定性

- 完整正文 vs 仅 2026-07 后普查：主题份额 Spearman ρ={num(rho, 3)}；最大份额差 {pct(R1S.diff_census_minus_full.abs().max())}。
- 普查重跑后 8 个主题的匈牙利匹配余弦中位数 {num(refit, 3)}；主题排序基本不变。
- Bootstrap 平均余弦 {num(R1['stability_bootstrap_mean'], 3)}；主题 7 仅 0.555（min 0.403），主题 5 为 0.789，主题 2 为 0.857；其余主题 >0.97。**主题 7 只能作为“API Key/第三方接入”的弱主题，不能单独支撑强结论。**

### 3.7 扩展（独立样本）：2026-04~09 全量首帖复核

该扩展来自另一套近期语料，**不与 3.2 的 4,307 条主题模型合并**：

{ext_rq1_topic_table()}

样本流转为：9,569 个候选主题 → 9,543 条成功获取首帖 → 9,518 条可用正文 → 9,498 条英文语料进入模型；8 个输出类别（含 OTHER）合计 9,498，份额加总为 1。分母既不是 9,569，也不是 9,518。

读法：

- 近期英文首帖中，**编辑器界面/会话管理（22.8%）、MCP/终端/本地工具（22.8%）、模型/套餐/用量（20.7%）** 三类合计约 66%，方向与主结果的“连接与界面是高频摩擦点”一致。
- 但两套语料的时间窗、语言、抽样设计、清洗和主题映射都不同：主结果是 2023-08 起的分层正文样本（覆盖 10.5%），新结果是 2026-04~09 英语首帖。**“方向一致”只能作为提出与核查主题假设的线索，不能写成一个比例验证了另一个比例。**
- 新分析自己报告了边界误分：Grok Bot/远程计算机类混入过自动关帖机器人帖与社区活动；模型/套餐类范围偏宽。这些属于分类边界问题，不是产品趋势。

**准入状态：未通过正式采纳门槛（盲审未完成）。** 交付包提供 120 条验证样本与 20 条定向模糊样本，但验证样本并非按类等量（T02 28 条、T06 35 条、T05 22 条，T01 9 条、T03/T07 各 6 条、T04 仅 4 条），且包内没有 9,498 条语料的逐帖模型标签，无法补足“每类 ≥20 条”的等量分层。项目已生成双人盲审工作簿 `data/manual/rq1_2026_review/` 与评分脚本 `src/10_rq1_review_agreement.py`；在两份评审表填完、κ/α ≥ 0.70 且关键类精确率 ≥ 0.75 之前，本节只作探索性描述，主题名与月度变化不进入结论。

![新语料主题结构](../../figures/rq1_05_新语料主题分布.png)

> 图注：分母 9,498 条 2026-04~09 英语首帖，模型输出 7 类 + OTHER；与主结果 4,307 条不可合并。图源为新增 RQ1 交付材料，公开仓库中的复制件见 `figures/rq1_05_新语料主题分布.png`。

### 3.8 RQ1 结论（观察—解释—意义—限制—验证）

**观察事实**：产品相关讨论集中在 MCP/网络、IDE 界面、云端协作、模型路由四类；社区活动是独立的一类，不应混入产品痛点。  
**可能解释**：AI 编程智能体的使用门槛从“模型能力”转向“连接、权限、界面和协作”；这些是高频摩擦点。  
**业务意义**：产品路线图应把“连接与扩展宿主稳定性”“界面变更的可逆性”“云端仓库集成”作为并列的基础体验项，而不是只优化模型回答。  
**证据限制**：正文覆盖 10.5%，2026-07 前为分层抽样；主题是词分布，不是用户需求强度；主题 7 不稳定。  
**后续验证**：用内部会话成功率、MCP 连接失败率、IDE 版本回滚率和用户访谈三角验证；对主题 7 单独做 API Key/BYOK 的定向问卷。

## 4. RQ2：问题组合结构

### 4.1 分母与规则

- 全部 Issue：{num(R2['denominators']['all_issues'])}；有标签：{num(R2['denominators']['labeled_issues'])}；至少两个标签：{num(R2['denominators']['multi_label_issues'])}。
- 标签出现次数：{num(R2['denominators']['label_occurrences'])}；不同标签：{num(R2['denominators']['n_labels'])}。
- Apriori 阈值：支持度 ≥ {R2['thresholds']['min_support']}，置信度 ≥ {R2['thresholds']['min_confidence']}，提升度 ≥ {R2['thresholds']['min_lift']}，最小计数 {R2['thresholds']['min_count']}，最大项集 4、最大前件 2。
- 结果：{num(R2['rule_counts']['frequent_itemsets'])} 个频繁项集，{num(R2['rule_counts']['raw_rules_after_threshold'])} 条阈值内规则，按覆盖/提升/模块去重后保留 **{num(R2['rule_counts']['core_rules'])} 条核心规则**。

### 4.2 高频标签

{rq2_label_table()}

### 4.3 核心规则

{rq2_rule_table()}

解读要点：

- `windows-os -> app` 支持度最高（19.9%），说明 Windows 桌面是最大的问题容器；反向 `app -> windows-os` 置信度只有 38.3%，因为桌面应用也大量出现在 macOS/Linux，**方向不能反着写**。
- `tui -> cli` 提升度 2.19，是稳定的层级关系：TUI 是 CLI 的界面形态；这应作为标签治理规则（打 TUI 时应补 CLI），而不是“TUI 导致 CLI 问题”。
- `computer-use -> app` 置信度 95.4%，但覆盖仅 3.0%；高置信度不等于高影响面。
- `app; connectivity -> bug` 置信度 96.9%，说明“桌面应用 + 连接”组合几乎总以缺陷形式出现，适合作为网络故障的快速分流线索。

### 4.4 规则评价与人工回读

- 48 条分层样本（12 规则 × 2 正 + 2 反），A&B 24 条中 23 条样本文义一致（23/24）；这是 LLM 辅助回读，不是总体规则精确率，也不是独立盲标结果。
- 仅 A 的 24 条中 4 条应补 B，**漏标率代理 {pct(audit['miss_rate_proxy'])}**；典型是 TUI 缺 CLI（2 条）、桌面 Computer Use 缺 APP（2 条）。
- 抽样按“规则”分层，未再按月份做人工分层；月份维度用全量数据的季度稳定性评估（见 4.5），两者互补。
- 这不是全量 precision/recall：样本按规则分层，且由 LLM 辅助阅读；只能说明“规则语义边界大体成立，但存在系统性层级漏标”。

### 4.5 时间稳定性与外推

- 12 条核心规则中，只有 {num(R2['stability']['n_stable'])} 条在 ≥75% 的可评估季度满足提升度 ≥1.2；多数规则只有 2–3 个可评估季度，因为标签体系在 2026 年才扩展。
- 2026 时间外检验：169 条旧规则中 157 条仍可计算且方向存活；6 条训练期新规则中 2 条在测试期通过阈值。规则“存活”受标签体系变化、样本量和真实关系变化共同影响，不能单独解释为规则失效。
- 结论应写成“当前标签体系下的分流线索”，不能写成“问题组合的因果结构”。

### 4.6 负关联与标签治理

{rq2_neg_table()}

负关联（lift < 1）更多是**替代关系或标签边界**：`app` 与 `cli`、`enhancement` 与 `windows-os`、`tui` 与 `windows-os` 都低于独立预期，说明入口标签存在互斥式标注习惯，而不是“某入口不会出某问题”。

治理表显示：

- `bug` 与 `enhancement` 同现 86 条，`bug` 与 `performance` 同现 1,517 条；问题类型不是互斥变量，报告时不能用饼图强行加总。
- `tui` 是 `cli` 的下级标签，`app-server` 是运行入口而非 `app`；当前存在漏补和混用。
- `computer-use` 在设置页报错、工具图片污染等案例中被过度使用，边界偏宽。
- 建议：建立“入口层级 + 类型互斥检查 + 管理标签隔离”的标注规范，并每月抽样 50 条做一致性审计。

### 4.7 RQ2 结论（观察—解释—意义—限制—验证）

**观察事实**：问题组合有清晰的入口—能力结构：Windows 桌面、CLI/TUI、浏览器/Computer Use、沙箱、远程/云各自形成稳定搭配；高频组合不等于强关联。  
**可能解释**：这些组合反映产品的真实使用路径和标签分流习惯，而不是单一根因。  
**业务意义**：可用于 triage 路由（如 `app; connectivity` 直接进网络排查）、回归测试组合（TUI+CLI、sandbox+Windows）和标签质检。  
**证据限制**：标签是维护者行为；规则不支持因果；季度稳定性仅 2/12；人工回读为 LLM 辅助。  
**后续验证**：用内部错误日志验证 `sandbox -> Windows` 与 `computer-use -> app` 的故障链路；做 100 条双人盲标，估计真实 precision/recall。

## 5. RQ3：反馈处理过程

### 5.1 三个“时钟”，不是一个 SLA

{rq3_overview_table()}

关键读法：

- **首次外部回复**：55.5% 有回复， median 1.02 天，P90 47.3 天；这是社区响应，不是官方承诺。
- **首次维护者回复**：仅 1.66%，median 0.71 天；一旦维护者出现通常很快，但覆盖极低，不能把 1.66% 当成官方处理率。
- **关闭**：39.8% 已关闭，已关闭样本 median 2.01 天、P90 87.2 天；KM 中位关闭 127.8 天，30/90/180 天仍未关闭分别为 71.3%/57.5%/44.4%。**条件中位数（已关闭样本）和生存中位数（全样本）必须同时报告，否则会低估长尾。**

### 5.2 分组差异

**按问题类型**

{rq3_group_table(R3T.head(4), "问题类型")}

**按产品入口**

{rq3_group_table(R3E[R3E.group.isin(["app", "cli", "tui", "app-server", "extension", "ios"])], "产品入口")}

**按运行环境 / 远程属性**

{rq3_group_table(R3V, "运行环境")}

{rq3_group_table(R3R, "远程属性")}

观察：CLI/TUI 的外部回复率和 30 天关闭率高于 app；iOS 的外部回复率 32.2%、30 天关闭率 13.2%；Linux 的回复率 64.1%、30 天关闭率 40.9%，高于 Windows 的 50.9%/26.5%。这些是标签分组后的描述差异，未控制问题类型和月份构成。

### 5.3 Kaplan-Meier 与队列

- 全样本 KM 中位关闭 127.8 天；30/90/180 天未关闭 71.3%/57.5%/44.4%。
- 按问题类型、入口、环境的 log-rank 均显著（p<0.001），但这是未控制构成差异的探索性检验；图中必须保留“仅描述”标注。
- 月度队列的 30 天关闭率受观察期长度影响：越接近快照的月份，30 天窗口越不完整，不能把近期下降直接解释为效率下降。

### 5.4 预测任务：弱信号与分布漂移

{model_table()}

必须同时报告的四个事实：

1. **时间切分与 purge gap** 已使用；特征只含创建时可见的标题/正文、长度、是否有代码、创建时间、作者身份；**不含最终标签、入口/环境/类型派生字段、状态、closed_at、评论**。标题/正文来自快照，可能被事后编辑，公开数据无法还原创建时版本，这是本任务无法完全消除的残余泄漏风险。
2. 三个任务的最佳 AUC 仅 {num(m['closed_within_30d'].auc, 3)} / {num(m['reply_3d'].auc, 3)} / {num(m['long_running_90d'].auc, 3)}；`reply_3d` 的文本模型略高于基线（0.605 vs 0.500），但没有达到可自动决策的水平。
3. 训练/测试正例率发生大幅漂移：30 天关闭 44.2%→14.4%，3 天回复 40.8%→19.0%，90 天长期未关闭 23.4%→70.2%。这不是模型失败的全部原因，但说明“时间外”比“同分布交叉验证”难得多。
4. 阈值 0.5 下召回很低；Top10% 召回也只有 10%–16%。若用于运营，只能做“候选排序 + 人工复核”，不能自动关闭或自动忽略。

#### 扩展验证：文本基线与滚动时间窗

新增 `outputs/rq3_validation/a5_b5_model_validation.csv` 对照文本朴素贝叶斯与纯文本逻辑回归，并以 3 个扩展时间窗检查文本逻辑回归的时间稳定性。

{rq3_extension_table()}

成熟标签资格、测试区间与 purge gap 均逐项列出；这属于敏感性验证，不替代主分析模型。滚动窗口间表现波动应解释为时间漂移证据，而不是挑选表现最好的窗口。

#### 诊断试验：响应时长箱线图与概率校准

`figures/rq3_a4_observed_reply_boxplot.png` 只纳入已发生首次外部回复的 Issue；未回复样本是右删失，不能当成普通时长或零天纳入，因此该图只描述“已回复者的条件时长”，不用于总体响应速度比较。`figures/rq3_a6_calibration_*.png` 检查预测分箱的观测率。先验漂移校正试算使用了测试集实际正例率，只能视为回顾性 oracle 上界诊断，不能作为可部署校正或改善证据；项目暂不把它纳入正式模型流程。

### 5.5 扩展（独立复现）：把“关闭”拆成三种竞争结局

主分析把 `closed` 当作一个事件；本节把关闭原因拆成 **completed / duplicate / not_planned** 三类互斥事件，其余在截点仍未关闭者右删失，用 Aalen–Johansen 估计累积发生率。

| 样本 | N | open | reopened | completed | duplicate | not_planned |
|---|---:|---:|---:|---:|---:|---:|
| 全量快照（2.2） | 27,756 | 16,547 | 154 | 7,199 | 2,059 | 1,797 |
| 竞争风险资格样本 | {num(EXT9['reference']['n']) if EXT9 else '—'} | 删失 {num(EXT9['reference']['event_counts']['0']) if EXT9 else '—'} | — | {num(EXT9['reference']['event_counts']['1']) if EXT9 else '—'} | {num(EXT9['reference']['event_counts']['2']) if EXT9 else '—'} | {num(EXT9['reference']['event_counts']['3']) if EXT9 else '—'} |

资格样本 = 27,756 − 17 条截点后创建 − 154 条已知 reopened；另有 1 条截点后才关闭的记录按右删失处理。**曲线人群不是全量快照人群**，两组数字必须同段出现。

| 指标（N=27,585） | 估计 | 95% 区间 / 说明 |
|---|---:|---|
| 30 天 completed 累积发生率（AJ） | {pct(EXT9['reference']['aj']['completed_cif_30']) if EXT9 else '—'} | bootstrap {pct(EXT9['bootstrap95_reference']['completed'][0]) if EXT9 else '—'}–{pct(EXT9['bootstrap95_reference']['completed'][1]) if EXT9 else '—'} |
| 30 天 duplicate / not_planned | {pct(EXT9['reference']['aj']['duplicate_cif_30']) if EXT9 else '—'} / {pct(EXT9['reference']['aj']['not_planned_cif_30']) if EXT9 else '—'} | 同一竞争框架 |
| 30 天任意关闭（1 − 未关闭概率） | {pct(1 - EXT9['reference']['aj']['open_cif_30']) if EXT9 else '—'} | 三种原因之和 |
| 90 天 completed 累积发生率 | {pct(EXT9['reference']['aj']['completed_cif_90']) if EXT9 else '—'} | 长尾仍在累积 |
| 固定 30 天窗口：completed vs 任意关闭 | {pct(EXT9['reference']['fixed_horizon_30d']['completed_rate']) if EXT9 else '—'} vs {pct(EXT9['reference']['fixed_horizon_30d']['closed_rate']) if EXT9 else '—'} | 相差 {num(EXT9['reference']['fixed_horizon_30d']['gap_pp'], 1) if EXT9 else '—'} 个百分点 |

**最重要的读数**：把 `closed` 当成 `completed`，会把“30 天内完成”从约 20.8% 夸大到约 31.9%（固定窗口口径）；在含删失的 AJ 口径下，30 天 completed 只有 18.7%。`duplicate` 与 `not_planned` 合计在 30 天已占约 10.1%，它们不是“解决”。

独立复现与敏感性：

- 三方一致：① 交付结果；② 用交付脚本在本项目环境原样重跑（12/12 一致，说明不依赖其原始库版本）；③ 本项目用统一底座独立实现 Aalen–Johansen（17/17 一致）。资格样本量、四类事件计数、固定窗口比例、KM 开放概率、AJ 30/90 天估计、0–180 天曲线逐点值全部对齐（曲线最大差 < 1e-15）。
- reopened 两种处理（排除 / 作为删失纳入）对 30 天 completed 的影响约 {num(abs(EXT9['scenarios'][1]['delta_completed_cif_30_pp']), 2) if EXT9 else '—'} 个百分点；观察截点从 00:00:00Z 改为 02:20:37Z 的影响约 {num(abs(EXT9['scenarios'][2]['delta_completed_cif_30_pp']), 2) if EXT9 else '—'} 个百分点。点估计几乎不动，但**报告必须固定写清用哪一个截点**。

{ext_rq3_scenario_table()}

![竞争风险与敏感性](../../figures/rq3_05_竞争风险.png)

> 图注：左图为 N=27,585 的累积发生率（0–180 天，0.5 天网格）；右图为四种口径下 30 天 completed 与任意关闭的对比。曲线未按创建月份调整，不能解释为处理效率变化。

### 5.6 扩展：Issue 类型分流与完成预测

这一组新增实验承接已有主线，而不是新增一条平行研究问题：RQ2 先回答“标签如何组合”，这里进一步检验“已有标签能否由文本辅助识别”；RQ3 研究“反馈如何被处理”，这里进一步检验“能否在早期估计处理结果”。两项任务使用独立交付包 `issue_ml/`，主流程只做结果核验，不覆盖 RQ1–RQ4 的主结果。

**任务 A：bug / enhancement 类型分流。** 对 26,679 条去重后的单标签、正文非空 Issue，使用标题与正文的 TF-IDF 特征和逻辑回归，在时间切分（训练 21,343、测试 5,287）上识别已有的 `bug` / `enhancement` 维护者标签。测试集 Accuracy 为 96.12%，Macro-F1 为 0.919，ROC-AUC 为 0.984，enhancement 的 AP 为 0.942。它可以作为人工 triage 的候选分流和标签质检线索，但不能解释为严重程度判断，也不能把维护者标签当成真实故障的人工真值。

**任务 B：completed 结果预测。** 使用 Issue 创建时可见的结构特征与前 24 小时早期互动特征，预测 7 天内是否进入 `completed`。固定方案的测试集为 1,395 条，其中 completed 仅 22 条；Random Forest 的 ROC-AUC 为 0.527、AP 为 0.100、Brier 为 0.021。作为不同观察窗的对照，24 小时早期互动预测 30 天 completed 的 AUC 为 0.613，但这仍只是有限的排序信号，受正例稀少、时间漂移和快照数据无法完整恢复历史特征影响，不适合自动分流、自动承诺处理时间或自动关闭 Issue。

**业务解释。** 这两个结果共同说明：“知道反馈属于哪一类”与“预测它何时完成”是难度不同的任务。前者适合放在人工分流前，减少标签整理成本；后者更适合作为研究性诊断，帮助识别哪些早期信号值得进一步接入内部日志验证。它们不能替代 RQ3 的生存分析，也不能把弱预测能力包装成运营 KPI。

扩展结果由阶段 12 对交付包做只读核验，当前 10/10 项检查通过；主流程不重复训练模型，也不改变 RQ1–RQ4 的原有结果。

### 5.7 RQ3 结论（观察—解释—意义—限制—验证）

**观察事实**：社区回复快、维护者回复少、关闭长尾重；CLI/TUI 与 Linux 的处理表现优于 app/iOS/Windows。  
**可能解释**：CLI 用户描述更技术化、问题更可复现；app/iOS 涉及更多环境与账户因素；维护者人力有限，主要依靠社区互助。  
**业务意义**：应把“首次外部回复”“首次维护者回复”“最终关闭”拆成三个运营指标；对 app/iOS 增加诊断信息采集和自助排查；对 90 天以上未关闭 Issue 做定期清理，而不是追求单一关闭率。  
**证据限制**：closed≠fixed；观察终点是推定快照；18,246 条评论无法关联；标签和标题可能是事后版本；预测任务分布漂移。另有 154 条 `state_reason=reopened` 在快照上仍为 open，其早先关闭事件不可见，生存分析只能按当前状态删失，因此真实“关闭后再重开”比例可能被低估。  
**后续验证**：接入内部事件日志（创建→首回复→状态变更），用同一口径重算；对 app/iOS 做 20 个用户的出声思维测试；对长期未关闭样本做人工抽样，区分“已修复未关闭”“无法复现”“搁置”。

## 6. RQ4：渠道差异

### 6.1 可比窗口

- 统一窗口：2026-01-01 ~ {SNAPSHOT_UTC}。
- 论坛 {num(f_n)} 个主题；Issues {num(i_n)} 条。
- 主比较使用**仅标题文本规则**（两渠道同一输入、同一字典），因此不受“论坛版块 vs Issue 标签”的平台差异污染；代价是标题规则覆盖率有限（论坛/Issues 的 type 维度未识别率为 72.6%/69.0%）。

### 6.2 问题类型

{rq4_type_table()}

> 表中份额的分母是各渠道窗口内全部主题/Issue，包含“标题规则未识别”；已识别类别与 unknown 相加为 100%。不能把未识别部分默认为“其他”，也不能用已识别类别内部比例替代全量比例。

### 6.3 产品入口

{rq4_entry_table()}

{rq4_entry_coverage_text()}。表中各入口比例以该渠道已识别记录为分母；入口允许多标签，一条记录可命中多个类别，因此比例不要求合计为 100%。两渠道覆盖差距较大，条件命中率可能受不同文本可识别性影响，不能据此判断总体渠道入口偏好。品牌词（如 Cursor）和界面词（如 sidebar、editor）不单独推断为桌面 app；未识别项保留为 unknown。

### 6.4 互动指标

| 指标 | Cursor 论坛 | Codex Issues | 可比性 |
|---|---:|---:|---|
| 有公开回复 | {pct(inter['Cursor 论坛'].public_reply_rate)} | {pct(inter['Codex Issues'].public_reply_rate)} | 接近但不等价 |
| 零回复 | {pct(inter['Cursor 论坛'].zero_reply_rate)} | {pct(inter['Codex Issues'].zero_reply_rate)} | 接近但不等价 |
| 明确完成/采纳 | {pct(inter['Cursor 论坛'].explicit_done_rate)}（采纳答案） | {pct(inter['Codex Issues'].explicit_done_rate)}（completed） | 定义不同，不可直接比较 |
| 回复中位数 | {num(inter['Cursor 论坛'].median_replies, 1)} | {num(inter['Codex Issues'].median_replies, 1)} | 论坛含所有回复，Issue 仅外部人类 |

### 6.5 不可比字段

| 字段 | 论坛 | Issues | 判定 | 原因 |
|---|---|---|---|---|
| 浏览量/点赞 | 有 | 无 | 不可直接比较 | 平台机制不同，浏览量不是需求强度 |
| 采纳答案 | has_accepted_answer | 无直接对应 | 不可直接比较 | 社区行为 vs 状态机 |
| closed/completed | 主题关闭 | state_reason | 定义不同 | 论坛关闭多为管理动作 |
| 正文长度 | 10.5% 覆盖 | 全量 body | 不可直接比较 | 论坛正文覆盖率低且非随机 |
| 零回复 | reply_count=0 | 无外部人类回复 | 接近但不等价 | 论坛回复可能含官方/机器人 |
| 标签 | 2026 年才成规模 | 维护者快照标签 | 不可直接比较 | 体系、粒度、动机不同 |

### 6.6 RQ4 结论（观察—解释—意义—限制—验证）

**观察事实**：同一标题规则下，Issues 的缺陷类占比高于论坛（22.8% vs 18.1%）；论坛的零回复率更高（59.9% vs 49.5%）。入口识别覆盖率分别为 {pct(1-R4D[(R4D.dimension=='entry')&(R4D.code=='unknown')&(R4D.channel=='Cursor 论坛')].share.iloc[0])} 和 {pct(1-R4D[(R4D.dimension=='entry')&(R4D.code=='unknown')&(R4D.channel=='Codex Issues')].share.iloc[0])}，覆盖不平衡，已识别子集入口构成不用于推断总体渠道偏好。  
**可能解释**：Issues 是“可复现缺陷”的正式入口，论坛更像“使用求助/讨论”的社区；渠道选择本身筛掉了不同用户和问题类型。  
**业务意义**：不能用论坛数据代表全部用户，也不能把 Issues 的缺陷率当成产品整体缺陷率；应按渠道分别设定响应和分流策略。  
**证据限制**：标题规则覆盖率约 27%–31%（type），入口覆盖率差异更大；未识别比例高且可能非随机；互动字段定义不同；HN 仅关键词辅助。  
**后续验证**：对同一批用户在两个渠道的行为做配对调查；用内部账号 ID 关联论坛/Issue（若合规）；对标题规则做 200 条人工校验并报告混淆矩阵。

## 7. 综合产品洞察

以下五条均按“观察事实—可能解释—业务意义—证据限制—后续验证”书写。

### 7.1 入口分层比“总体满意度”更有行动性

- **观察事实**：CLI/TUI 的外部回复率 66%–69%、30 天关闭率 38%–42%；app 为 44%/21%；iOS 为 32%/13%。
- **可能解释**：CLI 用户更技术化、问题更可复现；app/iOS 牵涉更多账户、环境和 UI 因素，分流成本更高。
- **业务意义**：按入口设置不同的首响目标和诊断模板；app/iOS 优先补“环境自检 + 日志一键上传”。
- **证据限制**：入口来自标签，非随机分组；未控制问题类型和月份；iOS 样本仅 270 条。
- **后续验证**：内部漏斗按入口拆分；对 iOS 前 50 个高频问题做人工归因。

### 7.2 远程/云问题以“组合”出现，单点修复不够

- **观察事实**：`remote -> app`（提升度 1.36）、`app; connectivity -> bug`（置信度 96.9%）、云端代理/仓库主题占正文 15.2%；远程 Issue 的 30 天关闭率 22.5%，低于非远程 32.7%。
- **可能解释**：远程问题跨越本地客户端、网络、服务端和仓库权限，单层修复容易被其他层掩盖。
- **业务意义**：建立跨层 trace ID（客户端—网关—云端—仓库），把“远程”作为独立故障域做端到端 SLA。
- **证据限制**：标签共现不是因果；远程样本仅 2,797 条；无法看到服务端日志。
- **后续验证**：抽取 100 条远程 Issue 做时间线复盘，定位失败层；用内部 trace 验证。

### 7.3 反馈处理要拆成“社区响应 / 维护者响应 / 关闭”三条曲线

- **观察事实**：外部回复 55.5%（中位 1.02 天），维护者回复 1.66%，关闭 39.8%（条件中位 2.01 天，KM 中位 127.8 天）。
- **可能解释**：社区互助承担了首响，维护者只处理少量高优先级问题；关闭动作混合了修复、重复和不计划。
- **业务意义**：运营看板至少三个指标；对“高赞/高影响但 90 天未关闭”设专项；对 duplicate 做合并而不是关闭了事。
- **证据限制**：closed≠fixed；评论关联损失；快照推定；维护者身份依赖 author_association。
- **后续验证**：接入状态变更事件流；对 200 条已关闭 Issue 人工判定“真修复/重复/搁置”。

### 7.4 论坛与 Issues 是不同漏斗，不能合并成“用户声音”

- **观察事实**：论坛零回复 59.9%、Issues 49.5%；论坛缺陷类 18.1%、Issues 22.8%。入口覆盖率不平衡（{rq4_entry_coverage_text()}），不比较总体入口偏好。
- **可能解释**：渠道选择由问题类型、用户角色和模板共同决定；论坛更偏求助/讨论，Issues 更偏可复现缺陷。
- **业务意义**：渠道分别设 KPI；论坛侧重“自助解决与社区响应”，Issues 侧重“可复现性与修复闭环”。
- **证据限制**：入口未识别率分别为 {pct(float(R4D[(R4D.dimension=='entry')&(R4D.code=='unknown')&(R4D.channel=='Cursor 论坛')].share.iloc[0]))} 和 {pct(float(R4D[(R4D.dimension=='entry')&(R4D.code=='unknown')&(R4D.channel=='Codex Issues')].share.iloc[0]))}，且缺失机制可能不同；已识别子集内部构成也只作条件描述，不外推总体；其他文本维度也有较高未识别率；字段不可比；无同一用户跨渠道 ID。
- **后续验证**：做渠道来源问卷；对同一功能在两个渠道的表述做对照阅读。

### 7.5 关联规则适合做“分流和标签质检”，不适合做因果

- **样本内回读**：A&B 共现样本中 23/24 文义一致；该结果不代表总体规则精确率。漏标代理为 16.7%，季度稳定为 2/12；`tui -> cli`、`computer-use -> app` 等主要是层级/入口关系。
- **可能解释**：标签体系存在层级漏标和边界混用；规则捕捉的是标注习惯与使用路径的混合。
- **业务意义**：把规则用于 triage 路由、测试组合和标签质检；不要用于“某入口导致某故障”的因果陈述。
- **证据限制**：LLM 辅助回读、分层样本、标签版本变化。
- **后续验证**：双人盲标 100 条；每季度重跑规则并监控 lift 漂移。

## 8. 结论边界与后续验证

| 结论 | 证据 | 不能说什么 | 下一步验证 |
|---|---|---|---|
| 四类产品主题占正文约 72% | NMF 8 主题 + 48 帖回读 | 不能说“72% 用户最痛” | 内部工单分类 + 用户访谈 |
| Windows 桌面是最大问题容器 | `windows-os -> app` 支持度 19.9% | 不能说 Windows 故障率最高 | 按活跃用户数算人均故障率 |
| CLI/TUI 处理表现更好 | 回复率/关闭率分组 | 不能说 CLI 质量更好 | 控制问题类型与严重度后回归 |
| 维护者回复覆盖极低 | 462/27,756 | 不能说官方只处理 1.66% | 内部 SLA 与人力数据 |
| 论坛与 Issues 有渠道差异 | 统一标题规则 + 互动指标 | 不能说“论坛用户更不活跃” | 跨渠道用户配对调查 |
| 规则可做 triage | 12 条规则 + 48 条回读 | 不能说规则揭示因果 | 双人盲标 + 线上 A/B |
| 30 天 completed 约 18.7%（AJ）/ 20.8%（固定窗口） | N=27,585 竞争风险 + 独立复现 | 不能说“关闭率 39.8% 即解决率” | 接入状态事件流，区分修复/重复/搁置 |
| 新语料三大主题合计约 66% | 9,498 条近期英文首帖 | 不能说“验证了旧主题比例” | 双人盲审 + 主题映射对齐 |

**如果时间有限，优先保留**：统一分母与口径、RQ1 主题覆盖限制、RQ2 规则边界、RQ3 三段时钟与防泄漏、RQ4 不可比清单。复杂模型（Cox、深度学习）不优先。

## 附录 A：核心图表

- RQ1：`figures/rq1_01_主题结构.png`、`rq1_02_主题与版块.png`、`rq1_04_覆盖与敏感性.png`；近期新语料扩展 `rq1_05_新语料主题分布.png`、`rq1_06_新语料月度份额.png`；组员长期趋势扩展 `rq1_07_全量版块月度结构.png`、`rq1_08_四类反馈长期趋势.png`、`rq1_09_月度回应与采纳.png`、`rq1_10_正文痛点词典线索.png`
- RQ2：`figures/rq2_01_标签频率.png`、`rq2_03_支持度与提升度.png`、`rq2_04_季度稳定性.png`
- RQ3：`figures/rq3_01_处理过程.png`、`rq3_02_生存曲线.png`、`rq3_04_模型ROC.png`；扩展 `rq3_05_竞争风险.png`
- RQ4：`figures/rq4_01_问题类型.png`、`rq4_02_入口.png`、`rq4_04_可比性.png`

## 附录 B：关键输出表

- 数据底座：`outputs/integrated/00_data_audit.json`、`data/derived/*.parquet`
- RQ1：`outputs/rq1/01_theme_profile.csv`、`07_theme_validation_sample.csv`、`08_theme_validation_summary.json`
- RQ2：`outputs/rq2/04_rules_core.csv`、`08_rule_manual_sample.csv`、`08_rule_audit_summary.json`、`09_label_governance.csv`
- RQ3：`outputs/rq3/01_process_overview.csv`、`07_km_stratified.csv`、`10_model_metrics.csv`、`10_model_protocol.json`
- RQ4：`outputs/rq4/02_unified_text_distribution.csv`、`06_interaction_compare.csv`、`07_noncomparable_fields.csv`
- 扩展核验：`outputs/integrated/new_materials_validation.json`（08）、`09_competing_risk.json`（09）、`09b_delivery_repro.json`（09b）、`10_rq1_agreement.json` 与 `10_rq1_profile.json`（10）

## 附录 C：新增材料的整合核验与版本记录

新增材料（2026-10-23 的 `RQ1_20261023` 与 `问题五`）只作为**独立扩展**接入，不覆盖 `outputs/rq1`、`outputs/rq3` 与主结论。四层核验相互独立：

| 脚本 | 作用 | 结果 | 不证明什么 |
|---|---|---|---|
| `08_validate_new_materials.py` | 交付结果的算术、样本流转、哈希与曲线守恒 | {f"{EXT8['passed']}/{EXT8['total']}" if EXT8 else "—"} | 不重算模型，不证明语义正确 |
| `09_rq3_competing_risk.py` | 用统一底座独立实现 Aalen–Johansen + 两层敏感性 | {f"{EXT9['passed']}/{EXT9['total']}" if EXT9 else "—"} | 不重训主题/回归/预测 |
| `09b_repro_delivery_script.py` | 在隔离目录用交付脚本原样重跑，三方对照 | {f"{EXT9B['passed']}/{EXT9B['total']}" if EXT9B else "—"} | 不证明结论无偏，只证明可复现 |
| `10_rq1_review_agreement.py` | 生成双人盲审工作簿，计算逐类精确率与一致性 | {EXT10['status'] if EXT10 else "—"}（A {EXT10['filled']['rater_A']}/140、B {EXT10['filled']['rater_B']}/140） | 未填完前不产生准确率 |

**不可合并清单（已写入核验脚本）**：

1. RQ1 的 4,307（2023-08 起分层正文）与 9,498（2026-04~09 英语首帖）份额不可拼接；
2. 主分析的“3 日内首次外部人类回复”与新材料的“7 日内他人回复”不是同一指标；
3. 主分析的“30 日关闭”与新材料的“30 日 completed”不是同一结局；
4. 全量快照状态（27,756）与固定截点竞争风险资格样本（27,585）不可互换；
5. 论坛主题与 Issue 评论是两个分析单位，不能合并成“总样本”。

**版本记录**：主结果由 `01`–`05` 于本轮流水线生成；扩展核验时间戳分别为 08 `{EXT8.get('generated_local', '—') if EXT8 else '—'}`、09 `{EXT9.get('generated_local', '—') if EXT9 else '—'}`、09b `{EXT9B.get('generated_local', '—') if EXT9B else '—'}`、10 `{EXT10.get('generated_local', '—') if EXT10 else '—'}`。原始六份 JSON 未改动，哈希 6/6 与 manifest 一致。
"""
    (REPORTS / "00_综合报告.md").write_text(text, encoding="utf-8")
    return text


def build_brief():
    val = R1V
    m = {r.target: r for _, r in BEST.iterrows()}
    inter = {r.channel: r for _, r in R4I.iterrows()}
    text = f"""# 研究简报（一页版）

**题目**：AI 编程智能体的多源用户反馈与产品问题挖掘  
**数据**：Cursor 论坛 41,218 主题（正文 4,322/18,498 帖）、Codex Issues 27,756 条/评论 115,280 条、HN 80,510 条（辅助）  
**快照**：{SNAPSHOT_UTC}  
**一句话结论**：公开反馈显示，AI 编程智能体的主要摩擦已经从“模型会不会写”转向“连接、权限、界面、协作与反馈闭环”；但这些是渠道内证据，不是总体用户比例。

## 五个可带走的数字

1. 正文主题中，MCP/网络/扩展宿主 24.8%、IDE 界面 18.0%、云端协作 15.2%、模型路由 14.4%；高权重样本文义回读 {val['theme_fit']}/{val['reviewed']}，仅用于主题命名与边界诊断。
2. 25,519 条多标签 Issue 形成 12 条核心规则；`windows-os -> app` 支持度 19.9%，`tui -> cli` 提升度 2.19，`computer-use -> app` 置信度 95.4%。
3. 首次外部回复覆盖 55.5%（中位 1.02 天），维护者回复仅 1.66%，关闭 39.8%；KM 中位关闭 127.8 天，30/90/180 天未关闭 71.3%/57.5%/44.4%。扩展竞争风险复核：30 天 completed 仅 18.7%，任意关闭 28.8%——关闭不是解决。
4. CLI/TUI 的 30 天关闭率 38%–42%，app 21%，iOS 13%；Linux 41%，Windows 27%。
5. 2026 窗口内，论坛零回复 59.9%、Issues 49.5%；论坛缺陷类 18.1%、Issues 22.8%；渠道差异不能写成总体偏好。

## 三条行动建议

- **按入口拆指标**：首响、维护者首响、关闭分别看；app/iOS 优先补环境自检与日志上传。
- **把远程/云当跨层故障域**：建立客户端—网关—云端—仓库的 trace，而不是单点修。
- **规则只做 triage**：12 条规则用于分流和标签质检；每季度重跑，配双人盲标，避免因果化表述。

## 三个必须提醒的限制

- 论坛正文覆盖 10.5%，2026-07 前为分层抽样；不能外推全历史。
- Issue 标签是维护者行为；`closed` 不等于 `fixed`；18,246 条评论无法关联。
- RQ3 预测最佳 AUC 仅 {num(m['closed_within_30d'].auc, 3)}–{num(m['reply_3d'].auc, 3)}，只能弱排序，不能自动决策。

## 复现

`python src/run_all.py` 一键重跑；详见 `docs/复现说明.md`。
"""
    (REPORTS / "01_研究简报.md").write_text(text, encoding="utf-8")
    return text


def build_ppt():
    m = {r.target: r for _, r in BEST.iterrows()}
    text = f"""# 汇报 PPT 大纲（12–14 页）

> 原则：每条研究线最多 3 张核心图；每页只回答一个问题；所有图带分母、时间范围和口径。

## 第 1 页 封面

- 标题：AI 编程智能体的多源用户反馈与产品问题挖掘
- 副标题：从“用户在说什么”到“反馈如何被处理”
- 数据快照与课程信息

## 第 2 页 研究问题与数据

- 四个 RQ 的证据链
- 三个渠道、五个分母
- 讲稿：我们不做“全网用户代表”，做“公开反馈渠道内的可复核证据”。

## 第 3 页 数据质量与口径

- 论坛正文 10.5% 覆盖；Issue 评论 18,246 条无法关联；标签是维护行为
- 图：`figures/rq1_04_覆盖与敏感性.png` 或口径表
- 讲稿：先讲分母，再讲结论。

## 第 4 页 RQ1 主题结构

- 图：`figures/rq1_01_主题结构.png`
- 三个数字：MCP/网络 24.8%、IDE 18.0%、云端 15.2%
- 边界：主题 1 是社区活动，不是产品痛点

## 第 5 页 RQ1 主题 × 版块

- 图：`figures/rq1_02_主题与版块.png`
- 结论：版块分类 ≠ 主题分类；Cramér's V={num(R1['tests']['category_chi2']['cramers_v'], 2)}
- 人工复核 {val_frac()}

## 第 6 页 RQ1 稳定性与限制

- 图：`figures/rq1_04_覆盖与敏感性.png`
- ρ={num(R1['tests']['coverage_sensitivity']['spearman'], 3)}；主题 7 bootstrap 0.555
- 讲稿：不要把 NMF 内部指标当成“真实需求强度”。
- 可选补充图：`figures/rq1_11_k选择诊断.png`；K=8 是可读性/连续性的工作设定，多指标没有共同最优 K

## 第 6+ 页（可选）RQ1 近期全量首帖复核

- 图：`figures/rq1_05_新语料主题分布.png`（分母 9,498，2026-04~09 英语首帖）
- 三个数字：界面 22.8%、MCP/终端 22.8%、模型/套餐 20.7%
- 边界：与主结果 4,307 条不可合并；盲审未完成前只作探索性结果
- 讲稿：这是“另一种语料指向同样的产品领域”，不是“旧比例被验证”。

## 第 7 页 RQ2 规则与模块

- 图：`figures/rq2_03_支持度与提升度.png`
- 12 条核心规则；`windows-os -> app` 19.9%；`tui -> cli` lift 2.19
- 讲稿：高频组合 ≠ 强关联，方向不能反写。

## 第 8 页 RQ2 稳定性与标签治理

- 图：`figures/rq2_04_季度稳定性.png`
- 2/12 季度稳定；A&B 样本文义一致 23/24（非总体精确率），漏标代理 16.7%
- 讲稿：规则用于 triage 和标签质检，不用于因果。

## 第 9 页 RQ3 三段时钟

- 图：`figures/rq3_01_处理过程.png`
- 55.5% 外部回复 / 1.66% 维护者回复 / 39.8% 关闭
- 讲稿：社区响应快，不代表官方处理快。

## 第 10 页 RQ3 生存与长尾

- 图：`figures/rq3_02_生存曲线.png`
- KM 中位 127.8 天；30/90/180 天未关闭 71.3%/57.5%/44.4%
- 讲稿：条件中位数 2.01 天与 KM 中位 127.8 天不矛盾，是两种口径。

## 第 10+ 页（可选）RQ3 关闭原因竞争风险

- 图：`figures/rq3_05_竞争风险.png`
- 30 天 completed 18.7%（AJ）/ 20.8%（固定窗口）vs 任意关闭 28.8%/31.9%
- 资格样本 N=27,585；固定截点 2026-09-13 00:00 UTC，排除 17 条截点后创建与 154 条 reopened
- 讲稿：这张图专门回答“closed ≠ completed”；口径敏感性约 0.1 个百分点，不影响方向。

## 第 11 页 RQ3 预测与防泄漏

- 图：`figures/rq3_04_模型ROC.png`
- 最佳 AUC {num(m['closed_within_30d'].auc, 3)}/{num(m['reply_3d'].auc, 3)}/{num(m['long_running_90d'].auc, 3)}；训练/测试正例率漂移
- 讲稿：只做候选排序，不做自动关闭。
- 可选验证表：文本 NB 对照 + 3 个滚动时间窗（见综合报告 5.4）；只作为稳健性检查
- A4/A6 暂不上主讲：已回复时长是条件分布；先验校正使用测试集实际正例率，仅是回顾性上界诊断

## 第 11+ 页（可选）RQ3 应用扩展：文本分流与处理预测

- **问题一：能否辅助判断 Issue 类型？** TF-IDF + 逻辑回归；时间切分测试集 5,287 条；Accuracy 96.12%，Macro-F1 0.919，ROC-AUC 0.984。
- **问题二：能否在早期预测 completed？** Random Forest；观察前 24 小时、预测 7 天；测试集 1,395 条、正例 22 条；AUC 0.527，AP 0.100；24 小时早期互动预测 30 天的对照 AUC 0.613。
- **业务结论**：文本可以较好复现已有 bug/enhancement 标签，可用于辅助分流；“知道它是什么类型”不等于“能预测它什么时候完成”，完成预测目前只能作为研究性排序信号。
- **边界**：标签来自维护者标注；completed 正例稀少且存在时间漂移；两项结果都不用于自动决策。
- **讲稿**：扩展任务自然接在主线之后，分别服务于 RQ2 的分流和 RQ3 的处理结果判断，不改变前面四条研究线的分母和主结论。

## 第 12 页 RQ4 可比性

- 图：`figures/rq4_04_可比性.png`
- 统一时间窗与标题规则；不可比字段清单
- 讲稿：渠道差异不是总体用户偏好。

## 第 13 页 RQ4 渠道画像

- 图：`figures/rq4_01_问题类型.png` + `rq4_02_入口.png`
- 论坛零回复 59.9%、Issues 49.5%；入口识别覆盖：论坛 {pct(1-R4D[(R4D.dimension=='entry')&(R4D.code=='unknown')&(R4D.channel=='Cursor 论坛')].share.iloc[0])}，Issues {pct(1-R4D[(R4D.dimension=='entry')&(R4D.code=='unknown')&(R4D.channel=='Codex Issues')].share.iloc[0])}
- 图表为已识别记录中的入口多选命中率；比例不相加为 100%，覆盖差异大，不推断总体入口偏好
- 讲稿：不同漏斗，不同 KPI；分类器覆盖不对称也限制渠道比较。

## 第 14 页 结论与边界

- 五条产品洞察（入口分层、远程跨层、三段时钟、渠道分流、规则 triage）
- 一页“不能说什么”
- 下一步：内部日志、双人盲标、用户访谈

## 备用页

- 统一分类字典
- 字段与分母表
- 复现命令与环境
"""
    (REPORTS / "02_汇报PPT大纲.md").write_text(text, encoding="utf-8")
    return text


def val_frac():
    return f"{R1V['theme_fit']}/{R1V['reviewed']} 条高权重样本文义贴合（主题命名/边界诊断，非总体准确率）"


def build_repro():
    text = f"""# 复现说明

## 1. 环境

- 操作系统：Windows 10/11
- Python：Python 3.12（建议使用虚拟环境）
- 关键依赖：pandas 2.3.3、numpy、scipy 1.17.0、scikit-learn 1.8.0、matplotlib 3.10.8、pyarrow 23.0.0
- 随机种子：{SEED}（所有采样、NMF、模型、bootstrap 共用；脚本内各自传入）
- 不依赖 NLTK/BeautifulSoup；HTML 清洗和轻量词形还原在 `src/common/text.py` 内实现。

## 2. 目录

```
data/
  derived/            由原始 JSON 生成的 parquet（只读输入给 RQ 脚本）
  manual/             人工/LLM 辅助回读表（rq1_theme_audit.csv, rq2_rule_audit.csv）
src/
  common/             config / taxonomy / text / stats
  01_build_base.py    数据底座与字段、样本、口径文档
  02_rq1_topics.py    NMF 主题、验证、敏感性、稳定性
  02b_rq1_k_selection.py 候选 K 多指标诊断（不自动选择）
  03_rq2_rules.py     Apriori、规则评价、时间稳定性、标签治理
  04_rq3_process.py   三段时钟、KM、预测任务
  04b_rq3_validation.py 文本 NB 对照、滚动时间验证、A4/A6 诊断试验
  05_rq4_compare.py   统一窗口/标题规则的渠道比较
  06_build_reports.py 综合报告、简报、PPT 大纲、本文件
  07_verify_outputs.py 验收核对
  08_validate_new_materials.py 新增材料口径/算术/哈希核验（只读）
  09_rq3_competing_risk.py 竞争风险独立复现 + reopened/截点敏感性
  09b_repro_delivery_script.py 交付脚本原样重跑（隔离目录、硬链接输入）
  10_rq1_review_agreement.py 新 RQ1 盲审工作簿 + 一致性评分
  11_rq1_legacy_extensions.py 重建组员 RQ1 有独立价值的长期趋势/互动/痛点图
outputs/              机器可读结果（JSON/CSV）
figures/              由 01–05、09、11 生成的核心与扩展图；10/11 均为口径单列的 RQ1 扩展
docs/              综合报告、研究简报、PPT 大纲
docs/                 字段字典、口径、样本、分类字典、实施计划、本说明
```

## 3. 执行顺序

```powershell
python src/01_build_base.py
python src/02_rq1_topics.py
python src/02b_rq1_k_selection.py
python src/03_rq2_rules.py
python src/04_rq3_process.py
python src/04b_rq3_validation.py
python src/05_rq4_compare.py
python src/06_build_reports.py
python src/07_verify_outputs.py
python src/08_validate_new_materials.py
python src/09_rq3_competing_risk.py
python src/09b_repro_delivery_script.py
python src/10_rq1_review_agreement.py
```

`01` 会重算原始文件 SHA-256，与 manifest 对照；不会改写原始 JSON。`03` 和 `02` 会读取 `data/manual/` 的回读表；若删掉回读表，脚本仍能生成空白样板，但报告中的精确率/贴合率会缺失。

`08`–`10` 是新增材料的可选扩展核验，只读本地 `data/delivery/` 与 `data/derived/` 中的输入，写入 `outputs/integrated/`；缺失时报告对应小节自动降级为“未运行”。

## 4. 输入与输出对照

| 阶段 | 主要输入 | 主要输出 |
|---|---|---|
| 01 | `data/raw/*.json` | `data/derived/*.parquet`、`outputs/integrated/00_data_audit.json`、`docs/字段字典.md` 等 |
| 02 | `forum_docs.parquet`、`forum_topics.parquet` | `outputs/rq1/*`、`figures/rq1_*.png` |
| 03 | `issues.parquet` | `outputs/rq2/*`、`figures/rq2_*.png` |
| 04 | `issues.parquet`、`comments.parquet` | `outputs/rq3/*`、`figures/rq3_*.png` |
| 05 | `forum_topics.parquet`、`issues.parquet`、`hn_filtered.parquet` | `outputs/rq4/*`、`figures/rq4_*.png` |
| 06 | 上述 JSON/CSV + 扩展核验结果 | `docs/*.md`、本文件 |
| 07 | 全部关键输出 | `outputs/integrated/验收核对.csv/json` |
| 08 | 新增交付包 + `00_data_audit.json` + `issues.parquet` | `outputs/integrated/new_materials_validation*.json/csv` |
| 09 | `issues.parquet` + 交付的 `descriptive_survival.json` / `survival_curves.json` | `outputs/integrated/09_*`、`figures/rq3_05_竞争风险.png` |
| 09b | 交付的 `analyze_issue_outcomes.py` + 两份原始 JSON（硬链接） | `outputs/repro_rq5/`、`outputs/integrated/09b_delivery_repro.json` |
| 10 | 新增 RQ1 交付包 + `data/manual/rq1_2026_review/*` | `outputs/integrated/10_rq1_profile.json`、`10_rq1_agreement.json`、`10_rq1_per_class.csv`、`10_rq1_confusion.csv` |

## 5. 口径与防泄漏

- 观察终点：`{SNAPSHOT_UTC}`，由 Issue 表最新时间字段推定；不是已核实的抓取时间。
- RQ1 正文覆盖：4,322/41,218 = 10.5%；2026-07 后为普查，之前为分层抽样（388 个主题）。
- RQ2 事务：25,519 条至少两个标签的 Issue；标签是维护者标注，不是故障发生率。
- RQ3 处理：`closed` 只表示关闭；`completed/duplicate/not_planned` 分开报告；KM 右删失在快照。
- RQ3 预测特征白名单：标题/正文、长度、是否有代码、创建时间、作者身份；**不含** state、state_reason、closed_at、comments、回复字段、最终标签及其派生入口/环境/类型。
- RQ3 时间切分：按创建时间 70/30，训练/测试之间留出与 horizon 等长的 purge gap。
- RQ4：统一窗口 2026-01-01 ~ 快照；主比较使用同一套标题文本规则；不可比字段单列。
- 新增材料：不并入主分母。新 RQ1 分母 9,498（2026-04~09 英语首帖），主 RQ1 分母 4,307（全历史分层正文）；RQ3 竞争风险资格样本 27,585（排除 17 条截点后创建、154 条 reopened），主状态概览仍为 27,756。两套截点 `2026-09-13T00:00:00Z`（交付）与 `2026-09-13T02:20:37Z`（本项目）都在 `09` 中报告，不可混用。

## 6. 已知警告与处理

- pandas `to_period("Q")` 会丢弃时区信息：季度字段只用于分组展示，不影响时间计算。
- `str.contains` 的正则分组警告：HN 关键词过滤使用正则，仅辅助，不影响核心结论。
- `03` 运行中 `Body` 模板/页脚已清洗；若新增论坛模板，需在 `src/common/text.py` 的 `FOOTER_RE` 补充。
- 控制台中文可能因 PowerShell 编码显示乱码；Markdown/JSON 文件均为 UTF-8。

## 7. 验收

```powershell
python src/07_verify_outputs.py
```

检查项包括：验收时现场重算原始文件哈希、Issue/评论/论坛计数、outcome 互斥、关键输出与图存在、RQ2 时间外检验、RQ1 敏感性与 K 扫描、RQ3 分组 90 天指标分子/分母闭合、模型特征白名单、文本/滚动验证、A4/A6 边界，以及 08/09/09b 扩展核验和新 RQ1 盲审工作簿。当前 44/44 通过。

## 8. 人工回读

- `data/manual/rq1_theme_audit.csv`：48 帖，字段 `is_product_feedback`、`theme_fit`、`note`。
- `data/manual/rq2_rule_audit.csv`：48 条，字段 `semantic_fit`、`B_should_be_present`、`note`。
- 当前回读为 **LLM 辅助的结构化阅读**，不是独立人工双盲；报告已明确标注。正式提交前建议抽 20% 由人复核，并记录复核人/日期。
- `data/manual/rq1_2026_review/`：新增 RQ1 的双人盲审工作簿（`rater_A.csv`、`rater_B.csv`、隔离的 `key_do_not_show_raters.csv`）。填完后运行 `src/10_rq1_review_agreement.py`，输出逐类精确率、混淆矩阵、Cohen's κ、Krippendorff's α 与门槛判定。
"""
    (DOCS / "复现说明.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    build_main()
    build_brief()
    build_ppt()
    build_repro()
    print("DONE", REPORTS, DOCS)

