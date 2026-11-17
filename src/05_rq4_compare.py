# -*- coding: utf-8 -*-
"""阶段 05：RQ4 跨渠道比较（Cursor 论坛 vs Codex Issues）。

可比性设计
----------
1. 统一时间窗：2026-01-01 ~ 2026-09-13（快照）。
2. 主比较使用同一套“仅标题文本规则”分类，避免把论坛版块 vs Issue 标签的
   平台差异误写成用户差异。
3. 辅助比较使用各自平台原生信号（论坛版块/标签 vs Issue 标签），并明确标注
   “分类来源不同，只能看方向，不能看精确差值”。
4. 互动指标只比较定义接近的“是否有公开回复/零回复/明确完成或采纳”，
   浏览量 vs 评论数等不可比字段单列。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DERIVED, OUT, FIG, COLORS, COMPARE_START, SNAPSHOT_UTC, setup_matplotlib
from common import taxonomy as tx
from common.stats import save_csv, save_json, cramers_v, wilson_ci, md_table

OUTD = OUT / "rq4"
OUTD.mkdir(parents=True, exist_ok=True)
plt = setup_matplotlib()
START = pd.Timestamp(COMPARE_START, tz="UTC")
END = pd.Timestamp(SNAPSHOT_UTC)


def dist_table(df, dim, prefix, label, suffix="_multi_t"):
    order = {"type": tx.TYPE_ORDER, "entry": tx.ENTRY_ORDER,
             "env": tx.ENV_ORDER, "cap": tx.CAP_ORDER}[dim]
    covered = df[f"{prefix}{suffix}"].fillna("").str.strip().ne("")
    recognized_n = int(covered.sum())
    rows = []
    for code in order:
        n = int((df[f"{prefix}{suffix}"].fillna("").str.split(";").apply(lambda xs: code in xs)).sum())
        if n == 0:
            continue
        lo, hi = wilson_ci(n, len(df))
        rows.append({"dimension": dim, "code": code, "label": dict(
            {"type": tx.TYPE_DEF, "entry": tx.ENTRY_DEF, "env": tx.ENV_DEF, "cap": tx.CAP_DEF}[dim])[code][0],
            "channel": label, "n": n, "denominator": len(df), "share": n / len(df),
            "recognized_n": recognized_n,
            "recognition_coverage": recognized_n / len(df) if len(df) else np.nan,
            "ci_low": lo, "ci_high": hi})
    n_unknown = int((~covered).sum())
    if n_unknown:
        rows.append({"dimension": dim, "code": "unknown", "label": "标题规则未识别",
                     "channel": label, "n": n_unknown, "denominator": len(df),
                     "share": n_unknown / len(df), "recognized_n": recognized_n,
                     "recognition_coverage": recognized_n / len(df) if len(df) else np.nan,
                     "ci_low": np.nan, "ci_high": np.nan})
    return pd.DataFrame(rows)


def add_title_multi(df):
    recs = [tx.classify_text(t) for t in df["title"].fillna("")]
    for dim in ["entry", "cap", "env", "type"]:
        df[f"{dim}_multi_t"] = [";".join(r[dim]) for r in recs]
    return df


def primary_from_multi(s, order):
    def f(x):
        vals = [v for v in str(x).split(";") if v]
        for o in order:
            if o in vals:
                return o
        return "unknown"
    return s.apply(f)


def compare_dim(t_forum, t_issue, dim, prefix):
    rows = []
    for lab, d in [("Cursor 论坛", t_forum), ("Codex Issues", t_issue)]:
        r = dist_table(d, dim, prefix, lab)
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    if dim == "entry":
        coverage = out[out.code == "unknown"].set_index("channel").recognition_coverage.to_dict()
        recognized_n = out[out.code == "unknown"].set_index("channel").recognized_n.to_dict()
        out["share_within_recognized"] = out.apply(
            lambda r: (float(r.n) / float(recognized_n.get(r.channel, 0))
                       if r.code != "unknown" and float(recognized_n.get(r.channel, 0)) > 0
                       else np.nan), axis=1)
        save_csv(out, OUTD / "03_entry_distribution_with_coverage.csv")
        save_csv(out[out.code != "unknown"], OUTD / "04_entry_known_subset_rates.csv")
    # 主类卡方（只对 primary 且排除 unknown）
    fcol = f"{dim}_primary_t" if dim != "cap" else "cap_primary_t"
    a = t_forum[fcol].replace("unknown", np.nan).dropna()
    b = t_issue[fcol].replace("unknown", np.nan).dropna()
    cats = sorted(set(a.unique()) | set(b.unique()))
    table = pd.DataFrame({0: a.value_counts().reindex(cats, fill_value=0),
                          1: b.value_counts().reindex(cats, fill_value=0)})
    chi2, p, v = cramers_v(table) if table.shape[0] > 1 else (np.nan, np.nan, np.nan)
    meta = {"dimension": dim, "chi2": chi2, "p_value": p, "cramers_v": v,
            "n_forum": int(len(a)), "n_issue": int(len(b)),
            "note": "主类分布检验；多标签维度的占比不可相加为 100%"}
    return out, meta


def main():
    forum = pd.read_parquet(DERIVED / "forum_topics.parquet")
    issues = pd.read_parquet(DERIVED / "issues.parquet")
    forum["created_at"] = pd.to_datetime(forum["created_at"], utc=True)
    issues["created_at"] = pd.to_datetime(issues["created_at"], utc=True)
    f = forum[(forum.created_at >= START) & (forum.created_at <= END)].copy()
    i = issues[(issues.created_at >= START) & (issues.created_at <= END)].copy()
    for d in [f, i]:
        d["month"] = d["created_at"].dt.strftime("%Y-%m")

    # 统一文本规则：仅标题（两渠道完全相同的输入与规则）
    f = add_title_multi(f)
    i = add_title_multi(i)
    for d in [f, i]:
        for dim, order in [("type", tx.TYPE_ORDER), ("entry", tx.ENTRY_ORDER),
                           ("env", tx.ENV_ORDER), ("cap", tx.CAP_ORDER)]:
            d[f"{dim}_primary_t"] = primary_from_multi(d[f"{dim}_multi_t"], order)

    # 平台原生信号：论坛=版块+标签；Issue=标签
    f["native_type"] = f["type_primary_forum"].fillna("unknown")
    i["native_type"] = i["type_label_primary"].replace("unknown", np.nan)
    i["native_type"] = i["native_type"].fillna("unknown")
    for d in [f, i]:
        d["native_entry"] = d["entry_label_primary"]
        d["native_env"] = d["env_label_primary"]
        d["native_cap"] = d["cap_multi"]

    window = pd.DataFrame([
        {"channel": "Cursor 论坛", "n": len(f), "start": str(f.created_at.min()), "end": str(f.created_at.max()),
         "definition": "全部主题元数据（标题/版块/标签/互动）；正文仅覆盖 10.5%"},
        {"channel": "Codex Issues", "n": len(i), "start": str(i.created_at.min()), "end": str(i.created_at.max()),
         "definition": "全部 Issue（标题/正文/标签/状态/评论）"}])
    save_csv(window, OUTD / "01_window_counts.csv")

    all_dims, metas = [], []
    for dim in ["type", "entry", "env", "cap"]:
        d, m = compare_dim(f, i, dim, dim)
        all_dims.append(d); metas.append(m)
    dims = pd.concat(all_dims)
    save_csv(dims, OUTD / "02_unified_text_distribution.csv")
    save_csv(pd.DataFrame(metas), OUTD / "02_unified_text_tests.csv")

    # 主类并排表
    for dim in ["type", "entry"]:
        piv = dims[dims.dimension == dim].pivot_table(index=["code", "label"], columns="channel",
                                                     values="share", aggfunc="first").reset_index()
        piv["diff_issue_minus_forum"] = piv.get("Codex Issues", 0) - piv.get("Cursor 论坛", 0)
        save_csv(piv.sort_values("Codex Issues", ascending=False), OUTD / f"03_{dim}_compare.csv")

    # 平台原生信号
    native_rows = []
    for dim, col in [("type", "native_type"), ("entry", "native_entry"), ("env", "native_env")]:
        for lab, d in [("Cursor 论坛", f), ("Codex Issues", i)]:
            vc = d[col].fillna("unknown").value_counts(normalize=True)
            for code, share in vc.items():
                native_rows.append({"dimension": dim, "channel": lab, "code": code,
                                    "share": float(share), "n": int((d[col] == code).sum()),
                                    "denominator": len(d)})
    native = pd.DataFrame(native_rows)
    save_csv(native, OUTD / "04_native_signal_distribution.csv")

    # 月度趋势（统一文本规则）
    trend = []
    for lab, d in [("Cursor 论坛", f), ("Codex Issues", i)]:
        for m, g in d.groupby("month"):
            if len(g) < 30:
                continue
            vc = g["type_primary_t"].value_counts(normalize=True)
            trend.append({"channel": lab, "month": m, "n": len(g),
                          "bug": vc.get("bug", 0), "help": vc.get("help", 0),
                          "enhancement": vc.get("enhancement", 0),
                          "performance": vc.get("performance", 0),
                          "connectivity": vc.get("connectivity", 0),
                          "unknown": vc.get("unknown", 0)})
    trend = pd.DataFrame(trend)
    save_csv(trend, OUTD / "05_monthly_trend.csv")

    # 互动指标（只比较定义接近的字段）
    inter = pd.DataFrame([
        {"channel": "Cursor 论坛", "n": len(f),
         "public_reply_rate": float((f.reply_count > 0).mean()),
         "zero_reply_rate": float((f.reply_count == 0).mean()),
         "explicit_done_rate": float(f.has_accepted_answer.mean()),
         "explicit_done_label": "有采纳答案",
         "median_replies": float(f.reply_count.median()),
         "median_views": float(f.views.median()),
         "note": "采纳≠解决；浏览/点赞无 Issue 对应"},
        {"channel": "Codex Issues", "n": len(i),
         "public_reply_rate": float(i.has_external_reply.mean()),
         "zero_reply_rate": float(i.no_external_reply.mean()),
         "explicit_done_rate": float((i.outcome == "completed").mean()),
         "explicit_done_label": "state_reason=completed",
         "median_replies": float(i.external_reply_count.median()),
         "median_views": np.nan,
         "note": "完成≠修复；评论已排除机器人"},
    ])
    save_csv(inter, OUTD / "06_interaction_compare.csv")

    # 不可比字段清单
    noncomp = pd.DataFrame([
        {"field": "浏览量 / 点赞", "forum": "有", "issue": "无", "verdict": "不可直接比较",
         "reason": "平台机制不同；浏览量不能当需求强度"},
        {"field": "采纳答案", "forum": "has_accepted_answer", "issue": "无直接对应", "verdict": "不可直接比较",
         "reason": "论坛采纳是社区行为；Issue 没有采纳字段"},
        {"field": "closed / completed", "forum": "closed 主题", "issue": "state_reason", "verdict": "定义不同",
         "reason": "论坛关闭多为管理动作；Issue completed 也不等于修复"},
        {"field": "正文长度", "forum": "仅 10.5% 主题有正文", "issue": "全量 body", "verdict": "不可直接比较",
         "reason": "论坛正文覆盖率低且非随机"},
        {"field": "零回复", "forum": "reply_count=0", "issue": "无外部人类回复", "verdict": "接近但不等价",
         "reason": "论坛回复可能含官方/机器人；Issue 已排除机器人"},
        {"field": "标签", "forum": "2026 年才成规模", "issue": "维护者快照标签", "verdict": "不可直接比较",
         "reason": "标签体系、粒度、维护者动机不同"},
    ])
    save_csv(noncomp, OUTD / "07_noncomparable_fields.csv")

    # HN 辅助
    hn = pd.read_parquet(DERIVED / "hn_filtered.parquet")
    hn["created_at"] = pd.to_datetime(hn["created_at"], utc=True)
    h = hn[(hn.created_at >= START) & (hn.created_at <= END)]
    hn_rows = h.groupby(h["created_at"].dt.strftime("%Y-%m")).agg(
        n=("objectID", "size"), stories=("is_story", "sum"),
        median_points=("points", "median"), median_comments=("num_comments", "median")).reset_index()
    hn_rows.columns = ["month", "n", "stories", "median_points", "median_comments"]
    save_csv(hn_rows, OUTD / "08_hn_aux_monthly.csv")

    # 图 1：问题类型（统一文本规则）
    order = ["bug", "help", "enhancement", "feedback", "performance", "connectivity", "docs", "other", "unknown"]
    pv = dims[dims.dimension == "type"].pivot_table(index="code", columns="channel",
                                                   values="share", aggfunc="first").reindex(order)
    pv = pv.dropna(how="all").fillna(0)
    # 把 unknown 单独放在右侧面板，避免它的大占比压扁其他已识别类别。
    known = pv.drop(index="unknown", errors="ignore")
    fig, (ax, ax_unknown) = plt.subplots(1, 2, figsize=(11.2, 5.2),
                                         gridspec_kw={"width_ratios": [4.2, 1.25]})
    x = np.arange(len(known)); w = .38
    forum_vals = known.get("Cursor 论坛", pd.Series(0, index=known.index)).to_numpy() * 100
    issue_vals = known.get("Codex Issues", pd.Series(0, index=known.index)).to_numpy() * 100
    b1 = ax.bar(x - w/2, forum_vals, w, label=f"Cursor 论坛 (n={len(f):,})", color=COLORS[0])
    b2 = ax.bar(x + w/2, issue_vals, w, label=f"Codex Issues (n={len(i):,})", color=COLORS[1])
    ax.bar_label(b1, labels=[f"{v:.1f}" for v in forum_vals], padding=2, fontsize=7, rotation=90)
    ax.bar_label(b2, labels=[f"{v:.1f}" for v in issue_vals], padding=2, fontsize=7, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([dict(zip(tx.TYPE_ORDER, [v[0] for v in tx.TYPE_DEF.values()]))[c] for c in known.index],
                                         rotation=25, ha="right")
    ax.set_ylim(0, max(27, float(max(forum_vals.max(initial=0), issue_vals.max(initial=0)) * 1.18)))
    ax.set_ylabel("该渠道内占比（%）")
    ax.set_title("已识别问题类型")
    ax.grid(axis="y", alpha=.2)
    unknown_vals = pv.loc["unknown"] if "unknown" in pv.index else pd.Series(dtype=float)
    uv = [float(unknown_vals.get("Cursor 论坛", 0)) * 100,
          float(unknown_vals.get("Codex Issues", 0)) * 100]
    ub = ax_unknown.bar([0, 1], uv, width=.62, color=[COLORS[0], COLORS[1]])
    ax_unknown.bar_label(ub, labels=[f"{v:.1f}%" for v in uv], padding=3, fontsize=8)
    ax_unknown.set_xticks([0, 1]); ax_unknown.set_xticklabels(["论坛", "Issues"])
    ax_unknown.set_ylim(0, max(80, max(uv, default=0) * 1.15))
    ax_unknown.set_title("未识别")
    ax_unknown.grid(axis="y", alpha=.2)
    ax_unknown.tick_params(axis="y", labelleft=False)
    fig.suptitle("RQ4 问题类型：同一套标题规则下的渠道差异（2026-01 ~ 2026-09）")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "rq4_01_问题类型.png", bbox_inches="tight"); plt.close(fig)

    # 图 2：入口
    entry_dist = dims[dims.dimension == "entry"]
    pv2 = entry_dist.pivot_table(index="code", columns="channel", values="share",
                                aggfunc="first").fillna(0)
    coverage = {lab: float(1 - pv2.loc["unknown", lab]) if "unknown" in pv2.index else 1.0
                for lab in ["Cursor 论坛", "Codex Issues"]}
    known_pv2 = pv2.drop(index="unknown", errors="ignore").copy()
    for lab in coverage:
        known_pv2[lab] = known_pv2[lab] / coverage[lab] if coverage[lab] else 0.0
    known_pv2 = known_pv2.sort_values("Codex Issues")
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    y = np.arange(len(known_pv2))
    f_label = f"Cursor 论坛（识别覆盖 {coverage['Cursor 论坛']:.1%}）"
    i_label = f"Codex Issues（识别覆盖 {coverage['Codex Issues']:.1%}）"
    b1 = ax.barh(y - .19, known_pv2.get("Cursor 论坛", 0) * 100, .38, label=f_label, color=COLORS[0])
    b2 = ax.barh(y + .19, known_pv2.get("Codex Issues", 0) * 100, .38, label=i_label, color=COLORS[1])
    ax.bar_label(b1, labels=[f"{v:.1f}%" for v in known_pv2.get("Cursor 论坛", 0) * 100], padding=3, fontsize=7)
    ax.bar_label(b2, labels=[f"{v:.1f}%" for v in known_pv2.get("Codex Issues", 0) * 100], padding=3, fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([tx.ENTRY_DEF[c][0] for c in known_pv2.index])
    ax.set_xlabel("已识别记录中命中该入口的比例（多选，%）")
    ax.set_title("RQ4 产品入口：已识别子集内多选命中率；覆盖率差异较大")
    ax.legend(); ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); fig.savefig(FIG / "rq4_02_入口.png", bbox_inches="tight"); plt.close(fig)

    # 图 3：月度趋势
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, code in zip(axes, ["bug", "help", "enhancement"]):
        for lab, color in [("Cursor 论坛", COLORS[0]), ("Codex Issues", COLORS[1])]:
            d = trend[trend.channel == lab]
            ax.plot(d.month, d[code] * 100, marker="o", label=lab, color=color)
        ax.set_title(dict(zip(tx.TYPE_ORDER, [v[0] for v in tx.TYPE_DEF.values()]))[code])
        ax.tick_params(axis="x", rotation=60); ax.grid(alpha=.2)
    axes[0].set_ylabel("渠道内占比（%）"); axes[0].legend(fontsize=8)
    fig.suptitle("月度结构：两渠道都向缺陷/支持型反馈集中，但水平与波动不同", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "rq4_03_月度趋势.png", bbox_inches="tight"); plt.close(fig)

    # 图 4：可比性矩阵
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.axis("off")
    cell = [["字段", "论坛", "Issues", "判定"],
            ["问题类型", "版块/文本", "标签/文本", "可比（统一规则）"],
            ["产品入口", "标签/文本", "标签/文本", "可比（统一规则）"],
            ["是否有公开回复", "reply_count>0", "外部人类回复", "接近但不等价"],
            ["明确完成/采纳", "采纳答案", "completed", "定义不同"],
            ["浏览量/点赞", "有", "无", "不可比"],
            ["正文长度", "10.5% 覆盖", "全量", "不可比"]]
    table = ax.table(cellText=cell[1:], colLabels=cell[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1, 1.7)
    for j in range(4):
        table[(0, j)].set_facecolor("#dbe5f1")
    ax.set_title("RQ4 可比性登记：只有统一口径的字段才进入结论", pad=12)
    fig.tight_layout(); fig.savefig(FIG / "rq4_04_可比性.png", bbox_inches="tight"); plt.close(fig)

    summary = {
        "window": {"start": str(START), "end": str(END), "forum_n": len(f), "issue_n": len(i)},
        "tests": metas,
        "interaction": inter.to_dict("records"),
        "noncomparable": noncomp.to_dict("records"),
        "hn_aux": {"n_2026_window": int(len(h)), "note": "关键词过滤后的辅助样本，不承担核心结论"},
        "headline": {
            "forum_bug_share_title": float(pv.loc["bug", "Cursor 论坛"]) if "bug" in pv.index else np.nan,
            "issue_bug_share_title": float(pv.loc["bug", "Codex Issues"]) if "bug" in pv.index else np.nan,
            "forum_entry_top_known_subset": known_pv2["Cursor 论坛"].idxmax(),
            "issue_entry_top_known_subset": known_pv2["Codex Issues"].idxmax(),
        },
        "entry_recognition": {
            lab: {"n_total": int(entry_dist[entry_dist.channel == lab].denominator.iloc[0]),
                  "n_recognized": int(entry_dist[(entry_dist.channel == lab) & (entry_dist.code == "unknown")].recognized_n.iloc[0]),
                  "coverage": coverage[lab], "unknown_rate": 1 - coverage[lab],
                  "known_subset_share": {code: float(known_pv2.loc[code, lab])
                                          for code in known_pv2.index}}
            for lab in coverage
        },
        "entry_comparability": "入口规则覆盖在两渠道间显著不同；全样本份额和已识别子集构成都不能直接解释为总体渠道偏好。",
        "note": "渠道差异是渠道机制、用户构成和反馈模板共同作用的结果，不能写成总体用户偏好。",
    }
    save_json(summary, OUTD / "rq4_summary.json")
    print(md_table(dims[dims.dimension == "type"][["code", "channel", "n", "share"]]))
    print(md_table(inter[["channel", "n", "public_reply_rate", "explicit_done_rate"]]))
    print("DONE", OUTD)


if __name__ == "__main__":
    main()
