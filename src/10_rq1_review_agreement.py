# -*- coding: utf-8 -*-
"""新增 RQ1（2026-04~09 全量首帖）的独立盲审工作簿与一致性评分。

职责分三层：
1) build_scaffold(): 从交付模板生成两份互不相同的盲审表（rater_A / rater_B）
   与一份隔离的 key（含模型类，标注员不可见）。
2) evaluate(): 读入已填写的评审表，计算逐类精确率（Wilson 95% CI）、混淆矩阵、
   双人 Cohen's κ 与 Krippendorff's α（nominal），并按预设门槛给出结论。
3) 若评审表未填写，只输出 pending 状态，不伪造任何准确率。

门槛（项目自定，非学术通用标准）：
- 双人一致性 κ/α ≥ 0.70；
- 模型份额 ≥5% 的主题逐类精确率 ≥ 0.75；
- 每类盲审样本 ≥20 条（不满足时该类标注为“样本不足，仅方向性”）。

注意：交付模板的 120 条验证样本并非按类等量（T03/T04 仅 4–6 条），
因此“每类 ≥20 条”只能由新的分层抽样补足；当前包未提供逐帖模型标签，
脚本会在报告中把该缺口写成数据请求，而不是用现有样本冒充。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.config import SEED  # noqa: E402
from common.stats import wilson_ci  # noqa: E402

NEW = ROOT / "data" / "delivery" / "rq1_20261023"
OUT = ROOT / "outputs" / "integrated"
MANUAL = ROOT / "data" / "manual" / "rq1_2026_review"
CODES = ["T01", "T02", "T03", "T04", "T05", "T06", "T07", "OTHER"]
LABELS = {
    "T01": "社区活动与线下聚会", "T02": "编辑器界面、窗口与会话管理",
    "T03": "Grok Bot 与远程计算机", "T04": "连接、模型服务与执行超时",
    "T05": "模型选择、套餐与用量反馈", "T06": "MCP、终端与本地工具工作流",
    "T07": "云端代理与代码仓库集成", "OTHER": "其他／待复核",
}
KEY_SHARE = 0.05  # 模型份额 ≥5% 的主题：门槛要求逐类精确率 ≥0.75


def build_scaffold() -> dict:
    """生成盲审表与 key；已存在则不覆盖。"""
    MANUAL.mkdir(parents=True, exist_ok=True)
    tpl = pd.read_csv(NEW / "04_复核材料" / "independent_human_review_template.csv")
    ref = pd.read_csv(NEW / "04_复核材料" / "助手参考" / "manual_review.csv")
    ref = ref[["topic_id", "model_topic_id", "split", "review_topic_id"]].rename(
        columns={"split": "sample_layer"})
    df = tpl.drop(columns=["split"]).merge(ref, on="topic_id", how="left")
    df["model_topic_id"] = df["model_topic_id"].fillna("OTHER")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(df))
    df = df.iloc[order].reset_index(drop=True)
    df["review_id"] = [f"R{i:03d}" for i in range(1, len(df) + 1)]

    key = df[["review_id", "topic_id", "sample_layer", "model_topic_id"]].copy()
    key.to_csv(MANUAL / "key_do_not_show_raters.csv", index=False, encoding="utf-8-sig")

    wrote = []
    for rater in ["rater_A", "rater_B"]:
        path = MANUAL / f"{rater}.csv"
        if path.exists():
            continue
        sheet = pd.DataFrame({
            "review_id": df["review_id"],
            "title": df["title"].str.slice(0, 220),
            "text_excerpt": df["text_excerpt"].fillna("").str.slice(0, 1200),
            "label": "",
            "notes": "",
        })
        sheet.to_csv(path, index=False, encoding="utf-8-sig")
        wrote.append(path.name)

    readme = MANUAL / "填写说明.md"
    if not readme.exists():
        lines = [
            "# 新增 RQ1 独立盲审填写说明", "",
            "本目录用于 2026-04~09 全量首帖主题模型（9,498 条英文语料）的独立人工复核。",
            "**先独立判断，再看 `key_do_not_show_raters.csv` 或助手参考目录。**", "",
            "## 文件", "",
            "- `rater_A.csv` / `rater_B.csv`：两位组员各自的盲审表，互不参照；",
            "- `key_do_not_show_raters.csv`：review_id 与原 topic_id、抽样层、模型类的对应表，仅评分人使用；",
            "- 两表行序一致，但都不含模型标签与抽样层。", "",
            "## 填写规则", "",
            "1. 只填 `label` 列，取值为 T01–T07 或 OTHER；无法归类时填 OTHER，理由写在 `notes`。",
            "2. 正文片段不足时，可访问 `https://forum.cursor.com/t/<topic_id>` 查看原帖；原帖可能已更新。",
            "3. 不要修改行序、不删行、不改 review_id；不确定时保留具体理由，不强行归类。",
            "4. 建议先各自独立完成，再讨论分歧；记录实际标注人与日期。", "",
            "## 主题编码", "",
        ]
        lines += [f"- `{c}` {LABELS[c]}" for c in CODES]
        lines += ["", "## 评分", "",
                  "```powershell",
                  "python src/10_rq1_review_agreement.py",
                  "```", "",
                  "两份表都填完后，脚本输出逐类精确率、混淆矩阵、Cohen's κ、Krippendorff's α 与门槛判定。", "",
                  "## 已知样本缺口", "",
                  "交付模板的 120 条验证样本不是按类等量抽取：T02 28 条、T06 35 条、T05 22 条，",
                  "但 T04 仅 4 条、T03/T07 各 6 条、T01 9 条。要满足“每类 ≥20 条”，",
                  "需要原分析方补充 9,498 条语料的逐帖模型标签，才能重新做等量分层抽样；",
                  "在此之前，小类的精确率只能作方向性描述，不能写进主结论。", ""]
        readme.write_text("\n".join(lines), encoding="utf-8")

    return {"rows": len(df), "created": wrote,
            "per_class_in_template": key.model_topic_id.value_counts().sort_index().to_dict()}


def krippendorff_alpha_nominal(a: list[str], b: list[str]) -> float:
    """两位编码者、nominal 变量的 Krippendorff's α。"""
    units = [(x, y) for x, y in zip(a, b) if x in CODES and y in CODES]
    n = len(units)
    if n < 2:
        return float("nan")
    cats = sorted({x for u in units for x in u})
    do_num = 0.0
    for x, y in units:
        if x != y:
            do_num += 1.0
    do = do_num / n
    counts = {c: sum((x == c) + (y == c) for x, y in units) for c in cats}
    total = 2 * n
    de = 1 - sum(v * (v - 1) for v in counts.values()) / (total * (total - 1))
    return 1 - do / de if de > 0 else float("nan")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x in CODES and y in CODES]
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in pairs) / n
    pe = sum((sum(x == c for x, _ in pairs) / n) * (sum(y == c for _, y in pairs) / n)
             for c in CODES)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def write_profile() -> dict:
    """把新 RQ1 的分母、样本流转与主题分布固化到 outputs/，供报告只读引用。"""
    dist = pd.read_csv(NEW / "03_统计与案例" / "topic_distribution.csv")
    flow = pd.read_csv(NEW / "03_统计与案例" / "sample_flow_cleaning.csv")
    lang = pd.read_csv(NEW / "03_统计与案例" / "language_distribution.csv")
    profile = {
        "corpus": {"candidate": int(flow.iloc[0]["input"]),
                   "first_post_fetched": int(flow.iloc[1]["retained"]),
                   "usable_body": int(flow.iloc[2]["retained"]),
                   "english_model": int(flow.iloc[3]["retained"]),
                   "english_share_of_usable": float(flow.iloc[3]["retained"] / flow.iloc[2]["retained"])},
        "window": "2026-04 ~ 2026-09（近期全量首帖；语言限英语）",
        "topics": dist.rename(columns={"content_topic_id": "code", "content_topic": "label",
                                       "topics": "n"})[["code", "label", "n", "share"]].to_dict("records"),
        "flow": flow.to_dict("records"),
        "language": lang.to_dict("records"),
        "difference_from_primary": (
            "主结果 4,307 条为 2023-08 起全历史的分层正文样本（覆盖 10.5%）；"
            "新语料 9,498 条为 2026-04~09 英语首帖。时间窗、语言、抽样设计、清洗与主题映射均不同，"
            "两套份额不可拼接成趋势线，也不可互相验证比例。"),
    }
    (OUT / "10_rq1_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return profile


def evaluate() -> dict:
    scaffold = build_scaffold()
    write_profile()
    key = pd.read_csv(MANUAL / "key_do_not_show_raters.csv")
    raters = {}
    for name in ["rater_A", "rater_B"]:
        path = MANUAL / f"{name}.csv"
        if path.exists():
            d = pd.read_csv(path)
            d["label"] = d["label"].astype(str).str.strip().str.upper().replace("NAN", "")
            raters[name] = d

    filled = {k: int(v["label"].isin(CODES).sum()) for k, v in raters.items()}
    total = len(key)
    status = "pending"
    if any(v > 0 for v in filled.values()):
        status = "single_rater_only" if filled.get("rater_A", 0) < total and filled.get("rater_B", 0) < total else "evaluated"

    dist_all = pd.read_csv(NEW / "03_统计与案例" / "topic_distribution.csv")
    key_classes = dist_all.loc[dist_all.share >= KEY_SHARE, "content_topic_id"].tolist()
    result = {"generated_local": pd.Timestamp.now().isoformat(timespec="seconds"),
              "status": status, "rows": total, "filled": filled,
              "scaffold": scaffold, "thresholds": {
                  "agreement": 0.70, "precision": 0.75, "min_per_class": 20,
                  "key_share": KEY_SHARE, "key_classes": key_classes}}

    # 只有至少一位评审填完才计算；否则只报 pending
    complete = [k for k, v in filled.items() if v == total]
    if not complete:
        result["message"] = ("评审表尚未填完：只生成/校验工作簿，不计算准确率。"
                             f"当前 A={filled.get('rater_A', 0)}/{total}，B={filled.get('rater_B', 0)}/{total}。")
        (OUT / "10_rq1_agreement.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return result

    merged = key.copy()
    for name in complete:
        merged = merged.merge(raters[name][["review_id", "label"]].rename(columns={"label": name}),
                              on="review_id", how="left")
    if len(complete) == 2:
        both = merged.dropna(subset=["rater_A", "rater_B"])
        result["agreement"] = {
            "cohen_kappa": float(cohen_kappa(both.rater_A.tolist(), both.rater_B.tolist())),
            "krippendorff_alpha": float(krippendorff_alpha_nominal(both.rater_A.tolist(), both.rater_B.tolist())),
            "n_overlap": int(len(both)),
            "disagreement_n": int((both.rater_A != both.rater_B).sum()),
        }

    # 逐类精确率：以已完成的评审（两人一致时取一致标签；否则取 A）对模型类
    if len(complete) == 2:
        merged["human"] = np.where(merged.rater_A == merged.rater_B, merged.rater_A, np.nan)
        basis = "两人一致样本（分歧样本不进入精确率）"
    else:
        merged["human"] = merged[complete[0]]
        basis = f"单人评审（{complete[0]}）"
    valid = merged.dropna(subset=["human"])
    valid = valid[valid.human.isin(CODES)]

    rows = []
    for c in CODES:
        sub = valid[valid.model_topic_id == c]
        k = int((sub.human == c).sum())
        n = int(len(sub))
        lo, hi = wilson_ci(k, n) if n else (np.nan, np.nan)
        rows.append({"model_class": c, "label": LABELS[c], "n_reviewed": n,
                     "correct_n": k, "precision": k / n if n else np.nan,
                     "wilson95_low": lo, "wilson95_high": hi,
                     "share_of_corpus": None,
                     "sample_sufficient": n >= 20})
    per_class = pd.DataFrame(rows)

    dist = pd.read_csv(NEW / "03_统计与案例" / "topic_distribution.csv")
    share = dict(zip(dist.content_topic_id, dist.share))
    per_class["share_of_corpus"] = per_class.model_class.map(share)
    per_class.to_csv(OUT / "10_rq1_per_class.csv", index=False, encoding="utf-8-sig")

    conf = pd.crosstab(valid.model_topic_id, valid.human).reindex(index=CODES, columns=CODES, fill_value=0)
    conf.to_csv(OUT / "10_rq1_confusion.csv", encoding="utf-8-sig")

    # 覆盖门槛：8 个类别每类 ≥20 条；精确率门槛：仅对份额 ≥5% 的主题
    insufficient = per_class[~per_class.sample_sufficient].model_class.tolist()
    low_precision = per_class[(per_class.share_of_corpus.fillna(0) >= KEY_SHARE)
                              & (per_class.precision.fillna(0) < 0.75)].model_class.tolist()
    kappa = result.get("agreement", {}).get("cohen_kappa", float("nan"))
    alpha = result.get("agreement", {}).get("krippendorff_alpha", float("nan"))
    gate_pass = (len(complete) == 2 and np.nanmin([kappa, alpha]) >= 0.70
                 and not insufficient and not low_precision)
    result.update({
        "basis": basis, "per_class": rows,
        "gate": {"passed": bool(gate_pass),
                 "insufficient_sample_classes": insufficient,
                 "low_precision_classes": low_precision,
                 "reason": ("通过：双人一致性达标、关键类精确率≥0.75 且每类样本≥20"
                            if gate_pass else
                            "未通过：小类样本不足或一致性/精确率未达门槛，主题名与趋势只能作探索性结果")},
    })
    (OUT / "10_rq1_agreement.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result


if __name__ == "__main__":
    res = evaluate()
    print("status:", res["status"], "| filled:", res["filled"])
    if res.get("agreement"):
        print("kappa", round(res["agreement"]["cohen_kappa"], 4),
              "alpha", round(res["agreement"]["krippendorff_alpha"], 4))
    for r in res.get("per_class", []):
        print(f"{r['model_class']:>5} n={r['n_reviewed']:>3} precision={r['precision']:.3f}"
              if r["n_reviewed"] else f"{r['model_class']:>5} n=0")
    if res.get("gate"):
        print("gate:", res["gate"]["passed"], "|", res["gate"]["reason"])
    if res.get("message"):
        print(res["message"])
