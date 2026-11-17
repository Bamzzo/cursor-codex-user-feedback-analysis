# -*- coding: utf-8 -*-
"""阶段 03：RQ2 标签组合、规则评价、时间稳定性与标签治理。

相对已有任务包的改进
--------------------
1. 从原始 Issue 重建 25,519 条双标签以上事务，复算 114 个标签；
2. 规则评价补齐 precision / recall / coverage，并做 2025→2026 的时间外检验；
3. 核心规则按季度检查稳定性；
4. 输出 lift<1 的负关联/边界关系；
5. 12 条规则做分层人工回读（正例+反例），区分“语义一致”与“漏标”；
6. 标签治理表：维度、层级、冲突、低频、漏标候选。
"""
from __future__ import annotations
import sys, json, itertools
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DERIVED, OUT, FIG, ROOT, SEED, COLORS, setup_matplotlib
from common import taxonomy as tx
from common.stats import save_csv, save_json, md_table
from common.text import audit_text

OUTD = OUT / "rq2"
OUTD.mkdir(parents=True, exist_ok=True)
plt = setup_matplotlib()

MIN_SUPPORT = 0.002
MIN_CONF = 0.30
MIN_LIFT = 1.20
MIN_COUNT = 50
MAX_ITEMS = 4
MAX_ANTECEDENT = 2
CORE_N = 12


def apriori(transactions: list[set], n: int, max_len: int = 4, min_count: int = 1):
    """返回 {k: {frozenset: count}}；用垂直 tidset 交集计数。"""
    freq, tids = {}, {}
    c1 = Counter()
    for t in transactions:
        c1.update(t)
    freq[1] = {frozenset([i]): c for i, c in c1.items() if c >= min_count}
    tids[1] = {}
    for i in freq[1]:
        lab = next(iter(i))
        tids[1][i] = {idx for idx, t in enumerate(transactions) if lab in t}
    for k in range(2, max_len + 1):
        prev = list(freq[k - 1].keys())
        candidates = set()
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                union = prev[i] | prev[j]
                if len(union) != k:
                    continue
                if all(frozenset(s) in freq[k - 1] for s in itertools.combinations(union, k - 1)):
                    candidates.add(frozenset(union))
        freq[k], tids[k] = {}, {}
        for cand in candidates:
            item = frozenset([next(iter(cand))])
            rest = cand - item
            ts = tids[k - 1].get(rest)
            if ts is None:
                continue
            inter = ts & tids[1][item]
            if len(inter) >= min_count:
                freq[k][cand] = len(inter)
                tids[k][cand] = inter
        if not freq[k]:
            break
    return freq


def rule_stats(a_count, b_count, ab_count, n):
    supp = ab_count / n
    conf = ab_count / a_count if a_count else np.nan
    lift = conf / (b_count / n) if b_count and n else np.nan
    recall = ab_count / b_count if b_count else np.nan
    coverage = a_count / n
    f1 = 2 * conf * recall / (conf + recall) if conf and recall and (conf + recall) > 0 else np.nan
    return dict(support=supp, confidence=conf, lift=lift, recall=recall, coverage=coverage, f1=f1)


def mine_rules(transactions, n, min_count=MIN_COUNT):
    freq = apriori(transactions, n, max_len=MAX_ITEMS, min_count=max(1, min_count))
    rules = []
    for k in range(2, MAX_ITEMS + 1):
        for itemset, ab in freq.get(k, {}).items():
            items = sorted(itemset)
            for r in range(1, min(MAX_ANTECEDENT, k - 1) + 1):
                for ant in itertools.combinations(items, r):
                    cons = tuple(sorted(set(items) - set(ant)))
                    if len(cons) != 1:   # 只保留单后件规则，便于业务解释
                        continue
                    A, B = frozenset(ant), cons[0]
                    a_count = freq[r].get(A, 0)
                    b_count = freq[1].get(frozenset([B]), 0)
                    st = rule_stats(a_count, b_count, ab, n)
                    rules.append({"antecedent": "; ".join(sorted(A)), "consequent": B,
                                  "k": k, "a_count": a_count, "b_count": b_count, "ab_count": ab, **st})
    r = pd.DataFrame(rules)
    if r.empty:
        return r, freq
    r = r[(r.confidence >= MIN_CONF) & (r.lift >= MIN_LIFT) & (r.ab_count >= min_count)]
    r = r.sort_values(["ab_count", "lift"], ascending=False).drop_duplicates(["antecedent", "consequent"])
    # 冗余规则：若存在更一般的子前件且置信度不低，则删除
    key = {(row.antecedent, row.consequent): row for _, row in r.iterrows()}
    keep = []
    for _, row in r.iterrows():
        ant = set(row.antecedent.split("; "))
        redundant = False
        if len(ant) >= 2:
            for sub in itertools.combinations(sorted(ant), len(ant) - 1):
                other = key.get(("; ".join(sub), row.consequent))
                if other is not None and other.confidence >= row.confidence - 0.01:
                    redundant = True
                    break
        if not redundant:
            keep.append(row)
    r = pd.DataFrame(keep).sort_values(["ab_count", "lift"], ascending=False).reset_index(drop=True)
    r.insert(0, "rule_id", [f"R{i+1:02d}" for i in range(len(r))])
    r["rule"] = r["antecedent"] + " -> " + r["consequent"]
    return r, freq


def rule_metrics_on(df, rule_row, label_col="labels"):
    """在给定子集上重算 P/R/coverage；返回 None 表示前件不出现。"""
    ant = set(rule_row.antecedent.split("; "))
    cons = rule_row.consequent
    n = len(df)
    if n == 0:
        return None
    a = df[label_col].apply(lambda xs: ant <= set(xs))
    b = df[label_col].apply(lambda xs: cons in xs)
    ab = (a & b).sum()
    ac, bc = int(a.sum()), int(b.sum())
    if ac == 0:
        return None
    st = rule_stats(ac, bc, int(ab), n)
    return {"n": n, "a_count": ac, "b_count": bc, "ab_count": int(ab), **st}


def module_of(rule_row):
    labs = set(rule_row.antecedent.split("; ")) | {rule_row.consequent}
    if {"iOS", "ios", "remote"} & labs:
        return "移动端与远程连接"
    if {"sandbox", "TCC / Permissions", "tcc / permissions"} & labs:
        return "沙箱与权限"
    if {"browser", "computer-use", "Browser Use"} & labs:
        return "浏览器与计算机使用"
    if {"CLI", "cli", "TUI", "tui", "terminal", "exec"} & labs:
        return "CLI 与终端交互"
    if {"windows-os", "Windows", "windows"} & labs:
        return "Windows 桌面端"
    if {"mcp", "mcp-server", "connectivity"} & labs:
        return "MCP 与网络"
    if {"app", "session", "performance", "tool-calls", "sandbox"} & labs:
        return "桌面应用与能力"
    return "其他"


def as_labels(x):
    if x is None:
        return []
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return [str(v).strip() for v in x if str(v).strip()]
    return [str(x).strip()] if str(x).strip() else []


def main():
    issues = pd.read_parquet(DERIVED / "issues.parquet")
    issues["created_at"] = pd.to_datetime(issues["created_at"], utc=True)
    issues["labels"] = issues["labels"].apply(as_labels)
    labeled = issues[issues.label_count > 0].copy()
    trans_df = labeled[labeled.label_count >= 2].copy()
    N_ALL, N_LAB, N = len(issues), len(labeled), len(trans_df)
    transactions = [set(x) for x in trans_df["labels"]]

    # ---- 标签频率与维度 ----
    freq = Counter()
    for t in transactions:
        freq.update(t)
    rows = []
    for lab, c in freq.most_common():
        d = tx.label_dimensions(lab)
        rows.append({"label": lab, "count": c, "share_of_transactions": c / N,
                     "share_of_labeled_issues": c / N_LAB, "label_occurrences": sum(1 for t in transactions if lab in t),
                     "dimensions": ",".join(k for k in ["entry", "cap", "env", "type"] if d.get(k)),
                     "dimension_detail": "; ".join(f"{k}={','.join(v)}" for k, v in d.items() if isinstance(v, list)),
                     "management": int(tx.is_management(lab))})
    lf = pd.DataFrame(rows)
    save_csv(lf, OUTD / "01_label_frequency.csv")

    # ---- 共现 ----
    co = Counter()
    for t in transactions:
        for a, b in itertools.combinations(sorted(t), 2):
            co[(a, b)] += 1
    pair_rows = []
    for (a, b), c in co.most_common():
        ca, cb = freq[a], freq[b]
        pair_rows.append({"label_a": a, "label_b": b, "count": c, "support": c / N,
                          "lift": (c / N) / ((ca / N) * (cb / N)) if ca and cb else np.nan,
                          "confidence_a_to_b": c / ca, "confidence_b_to_a": c / cb})
    pairs = pd.DataFrame(pair_rows)
    save_csv(pairs.head(300), OUTD / "02_cooccurrence_top300.csv")

    # ---- Apriori 规则 ----
    rules, freq_items = mine_rules(transactions, N)
    freq_rows = [{"itemset_size": k, "itemset": "; ".join(sorted(s)), "count": v,
                  "support": v / N} for k, d in freq_items.items() for s, v in d.items()]
    save_csv(pd.DataFrame(freq_rows), OUTD / "03_frequent_itemsets.csv")
    save_csv(rules, OUTD / "04_rules_all.csv")
    print("rules", len(rules), "itemsets", len(freq_rows), flush=True)

    # 核心规则：每个问题模块取覆盖最大的，再按 lift 补足
    rules["module"] = rules.apply(module_of, axis=1)
    core = (rules.sort_values(["module", "ab_count"], ascending=[True, False])
            .groupby("module").head(2).sort_values("ab_count", ascending=False).head(CORE_N))
    if len(core) < CORE_N:
        extra = rules[~rules.rule_id.isin(core.rule_id)].sort_values("lift", ascending=False).head(CORE_N - len(core))
        core = pd.concat([core, extra])
    core = core.sort_values(["ab_count", "lift"], ascending=False).reset_index(drop=True)
    save_csv(core, OUTD / "04_rules_core.csv")

    # ---- 全样本与时间外评价 ----
    all_rows = []
    for _, rr in rules.iterrows():
        st = rule_metrics_on(trans_df, rr)
        if st:
            all_rows.append({"rule_id": rr.rule_id, "rule": rr.rule, "module": rr.module, **st})
    met = pd.DataFrame(all_rows)
    save_csv(met, OUTD / "05_rule_metrics_all.csv")

    split_date = pd.Timestamp("2026-01-01", tz="UTC")
    train_df = trans_df[trans_df.created_at < split_date]
    test_df = trans_df[trans_df.created_at >= split_date]
    train_rules, _ = mine_rules([set(x) for x in train_df.labels], len(train_df))
    hold_rows = []
    # 以 2025 训练集挖出的规则，在 2026 上检验；同时报告原 85 条规则的存活
    test_eval = []
    for _, rr in train_rules.iterrows():
        st = rule_metrics_on(test_df, rr)
        if st:
            test_eval.append({"rule_id": rr.rule_id, "rule": rr.rule, **st,
                              "source": "mined_2025"})
    test_eval = pd.DataFrame(test_eval)
    if not test_eval.empty:
        test_eval["pass"] = (test_eval.confidence >= MIN_CONF) & (test_eval.lift >= MIN_LIFT) & (test_eval.ab_count >= 20)
    save_csv(test_eval, OUTD / "05_holdout_rules_2026.csv")

    # 已有 85 条规则在 2026 的存活
    surv = []
    for _, rr in rules.iterrows():
        st = rule_metrics_on(test_df, rr)
        base = met[met.rule_id == rr.rule_id].iloc[0] if (met.rule_id == rr.rule_id).any() else None
        if st:
            surv.append({"rule_id": rr.rule_id, "rule": rr.rule, "module": rr.module,
                         "full_confidence": base.confidence if base is not None else np.nan,
                         "full_lift": base.lift if base is not None else np.nan,
                         "test_n": st["n"], "test_a_count": st["a_count"],
                         "test_confidence": st["confidence"], "test_lift": st["lift"],
                         "test_recall": st["recall"], "test_coverage": st["coverage"],
                         "survives": bool(st["confidence"] >= MIN_CONF and st["lift"] >= MIN_LIFT and st["ab_count"] >= 20)})
    surv = pd.DataFrame(surv)
    save_csv(surv, OUTD / "05_rule_survival_2026.csv")
    holdout = {
        "train_n": int(len(train_df)), "test_n": int(len(test_df)),
        "train_rule_count": int(len(train_rules)),
        "test_rules_evaluable": int(len(test_eval)),
        "test_rules_pass": int(test_eval["pass"].sum()) if not test_eval.empty else 0,
        "pass_rate": float(test_eval["pass"].mean()) if not test_eval.empty else np.nan,
        "existing_rule_survival": int(surv["survives"].sum()) if len(surv) else 0,
        "existing_rule_n": int(len(surv)),
        "note": "时间外检验会同时受标签体系变化、样本量和真实关系变化影响，不能单独解释为规则失效。",
    }
    save_json(holdout, OUTD / "05_holdout_summary.json")

    # ---- 季度稳定性（核心规则）----
    trans_df["quarter"] = trans_df["created_at"].dt.to_period("Q").astype(str)
    q_rows = []
    for _, rr in core.iterrows():
        for q, g in trans_df.groupby("quarter"):
            st = rule_metrics_on(g, rr)
            if st:
                q_rows.append({"rule_id": rr.rule_id, "rule": rr.rule, "quarter": q, **st})
    qs = pd.DataFrame(q_rows)
    save_csv(qs, OUTD / "06_rule_quarterly.csv")
    stab = []
    for rid, g in qs.groupby("rule_id"):
        valid = g[g.a_count >= 20]
        stab.append({"rule_id": rid, "rule": g.rule.iloc[0], "quarters_total": len(g),
                     "quarters_evaluable": len(valid),
                     "quarters_lift_gt_1.2": int((valid.lift >= 1.2).sum()),
                     "stable_75pct": bool(len(valid) >= 3 and (valid.lift >= 1.2).mean() >= 0.75),
                     "min_confidence": float(valid.confidence.min()) if len(valid) else np.nan,
                     "max_confidence": float(valid.confidence.max()) if len(valid) else np.nan})
    stab = pd.DataFrame(stab)
    save_csv(stab, OUTD / "06_rule_stability.csv")

    # ---- 负关联 ----
    neg = pairs[(pairs.lift < 1.0) & (pairs["count"] >= max(20, int(N * 0.004)))].copy()
    neg = neg.sort_values("lift").head(20)
    neg["interpretation"] = "低于独立预期，可能是替代关系/标签边界，不等于互斥或因果"
    save_csv(neg, OUTD / "07_negative_pairs.csv")

    # ---- 人工回读样本（12 规则 × 2 正 + 2 反；删除/撤回帖不进入审计）----
    rng = np.random.default_rng(SEED)
    audit = []
    for _, rr in core.iterrows():
        ant = set(rr.antecedent.split("; ")); cons = rr.consequent
        a_mask = trans_df.labels.apply(lambda xs: ant <= set(xs))
        for kind, pool in [("A&B", trans_df[a_mask & trans_df.labels.apply(lambda xs: cons in xs)]),
                           ("A&¬B", trans_df[a_mask & trans_df.labels.apply(lambda xs: cons not in xs)])]:
            pool = pool.sample(frac=1, random_state=int(rng.integers(1e9)))
            taken = 0
            for _, r in pool.iterrows():
                body = audit_text(str(r.get("body_snippet", "")))
                title = str(r["title"] or "").strip()
                if title.lower() in {"[withdrawn]", "removed", "deleted"} or len(body) < 60:
                    continue
                audit.append({"rule_id": rr.rule_id, "rule": rr.rule, "module": rr.module,
                              "case_type": kind, "number": int(r["number"]), "title": title,
                              "labels": "; ".join(r["labels"]),
                              "snippet": body[:900],
                              "url": r["url"],
                              "semantic_fit_A&B": "", "B_should_be_present": "",
                              "judgement_note": ""})
                taken += 1
                if taken >= 2:
                    break
    audit = pd.DataFrame(audit)
    # 人工回读结果作为外部输入保存，脚本只负责抽样、合并与计算，避免把判断写成生成逻辑
    manual_path = ROOT / "data" / "manual" / "rq2_rule_audit.csv"
    audit_metrics = {"reviewed": 0, "note": "未找到人工回读表，仅输出空白样板"}
    if manual_path.exists() and len(audit):
        man = pd.read_csv(manual_path).rename(columns={
            "semantic_fit": "fit", "B_should_be_present": "b_missing", "note": "review_note"})
        audit = audit.drop(columns=["semantic_fit_A&B", "B_should_be_present", "judgement_note"])
        audit = audit.merge(man[["rule_id", "number", "case_type", "fit", "b_missing", "review_note"]],
                            on=["rule_id", "number", "case_type"], how="left")
        audit = audit.rename(columns={"fit": "semantic_fit_A&B", "b_missing": "B_should_be_present",
                                      "review_note": "judgement_note"})
        audited = audit[audit["semantic_fit_A&B"].notna()].copy()
        aud_pos = audited[audited.case_type == "A&B"]
        aud_neg = audited[audited.case_type == "A&¬B"]
        audit_metrics = {
            "reviewed": int(len(audited)),
            "A&B_reviewed": int(len(aud_pos)),
            "A&B_fit": int(aud_pos["semantic_fit_A&B"].sum()) if len(aud_pos) else 0,
            "precision_estimate": float(aud_pos["semantic_fit_A&B"].mean()) if len(aud_pos) else np.nan,
            "A_only_reviewed": int(len(aud_neg)),
            "A_only_B_missing": int(aud_neg["B_should_be_present"].sum()) if len(aud_neg) else 0,
            "miss_rate_proxy": float(aud_neg["B_should_be_present"].mean()) if len(aud_neg) else np.nan,
            "method": "LLM 辅助的结构化回读（依据标题+清洗后正文判断），不是独立人工双盲标注；用于估计规则语义边界，需人工抽检确认。",
        }
        save_json(audit_metrics, OUTD / "08_rule_audit_summary.json")
    save_csv(audit, OUTD / "08_rule_manual_sample.csv")

    # ---- 标签治理 ----
    gov = []
    for _, r in lf.iterrows():
        lab = r["label"]
        gov.append({"label": lab, "count": r["count"], "dimensions": r["dimensions"],
                    "management": r["management"],
                    "hierarchy": {"tui": "TUI 是 CLI 的下级界面标签",
                                  "app-server": "app-server 是运行入口，不等同 app",
                                  "ios": "iOS 与 remote 是入口—能力关系"}.get(lab, ""),
                    "conflict": "",
                    "boundary": ""})
    gov = pd.DataFrame(gov)
    # 冲突/边界
    conflicts = [("bug", "enhancement"), ("bug", "performance"), ("app", "CLI"), ("TUI", "CLI")]
    for a, b in conflicts:
        ca = trans_df.labels.apply(lambda xs: a in xs and b in xs).sum()
        if ca:
            gov.loc[gov.label == b, "conflict"] += f"与 {a} 同现 {ca} 条；"
    gov.to_csv(OUTD / "09_label_governance.csv", index=False, encoding="utf-8-sig")

    # 月度标签覆盖
    cov = issues.groupby("created_month").agg(
        n=("number", "size"),
        labeled=("label_count", lambda x: (x > 0).sum()),
        multi=("label_count", lambda x: (x >= 2).sum())).reset_index()
    cov["labeled_pct"] = cov.labeled / cov.n
    cov["multi_pct"] = cov.multi / cov.n
    save_csv(cov, OUTD / "10_label_coverage_month.csv")

    # 漏标候选
    miss = []
    for _, rr in core.iterrows():
        ant = set(rr.antecedent.split("; ")); cons = rr.consequent
        a_mask = trans_df.labels.apply(lambda xs: ant <= set(xs))
        a_n = int(a_mask.sum()); missing = int((a_mask & trans_df.labels.apply(lambda xs: cons not in xs)).sum())
        miss.append({"rule_id": rr.rule_id, "rule": rr.rule, "module": rr.module,
                     "a_count": a_n, "missing_consequent": missing,
                     "missing_rate": missing / a_n if a_n else np.nan,
                     "suggestion": f"出现 [{rr.antecedent}] 时检查是否漏标 [{cons}]"})
    miss = pd.DataFrame(miss).sort_values("missing_rate", ascending=False)
    save_csv(miss, OUTD / "11_missing_label_candidates.csv")

    # ---- 图 ----
    top = lf.head(18).sort_values("count")
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    colors = [COLORS[4] if m else COLORS[0] for m in top["management"]]
    ax.barh(top["label"], top["count"], color=colors)
    ax.set_xlabel(f"包含该标签的 Issue 数（分母 = {N:,} 条双标签以上 Issue）")
    ax.set_title("RQ2 标签频率：bug 覆盖最广，但不说明入口或故障域")
    fig.tight_layout(); fig.savefig(FIG / "rq2_01_标签频率.png", bbox_inches="tight"); plt.close(fig)

    # 共现热力图（前 14 标签）
    top14 = lf.head(14)["label"].tolist()
    mat = pd.DataFrame(0, index=top14, columns=top14)
    for t in transactions:
        for a in top14:
            if a in t:
                for b in top14:
                    if b in t:
                        mat.loc[a, b] += 1
    fig, ax = plt.subplots(figsize=(8, 6.4))
    im = ax.imshow(np.log1p(mat.to_numpy()), cmap="YlGnBu")
    ax.set_xticks(range(len(top14))); ax.set_xticklabels(top14, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(top14))); ax.set_yticklabels(top14, fontsize=8)
    ax.set_title("标签共现（log1p 计数）")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(FIG / "rq2_02_共现热力图.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    sc = ax.scatter(rules.support * 100, rules.lift, s=np.sqrt(rules.ab_count) * 3,
                    c=[COLORS[0] if m == "其他" else COLORS[2] for m in rules.module], alpha=.65)
    # 只标注关键规则；完整规则清单保留在输出表，避免密集区域文字互相遮挡。
    label_offsets = {"R01": (-12, 12), "R02": (-12, -14), "R04": (8, 8),
                     "R10": (8, 9), "R20": (-20, -13)}
    for _, r in core[core.rule_id.isin(label_offsets)].iterrows():
        ax.annotate(r.rule_id, (r.support * 100, r.lift), fontsize=8, fontweight="bold",
                    xytext=label_offsets[r.rule_id], textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=.18", fc="white", ec="#888", lw=.5),
                    arrowprops=dict(arrowstyle="-", color="#777", lw=.6))
    ax.set_xlabel("支持度（%）"); ax.set_ylabel("提升度")
    ax.set_title("规则支持度—提升度：高频不等于强关联（仅标注代表规则）")
    ax.axhline(1.2, ls="--", color="gray", lw=.8)
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIG / "rq2_03_支持度与提升度.png", bbox_inches="tight"); plt.close(fig)

    if len(qs):
        piv = qs.pivot_table(index="rule", columns="quarter", values="lift", aggfunc="mean")
        piv = piv.reindex([r for r in core.rule if r in piv.index])
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        im = ax.imshow(piv.to_numpy(), cmap="RdYlGn", vmin=0.8, vmax=max(6, np.nanmax(piv.to_numpy())))
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=30, ha="right")
        ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([r[:42] for r in piv.index], fontsize=7)
        ax.set_title("核心规则提升度的季度稳定性（空白=该季度无足够样本）")
        fig.colorbar(im, ax=ax, fraction=0.03, label="lift")
        fig.tight_layout(); fig.savefig(FIG / "rq2_04_季度稳定性.png", bbox_inches="tight"); plt.close(fig)

    summary = {
        "denominators": {"all_issues": N_ALL, "labeled_issues": N_LAB, "multi_label_issues": N,
                         "label_occurrences": int(freq.total()) if hasattr(freq, "total") else int(sum(freq.values())),
                         "n_labels": int(len(freq))},
        "thresholds": {"min_support": MIN_SUPPORT, "min_confidence": MIN_CONF, "min_lift": MIN_LIFT,
                       "min_count": MIN_COUNT, "max_items": MAX_ITEMS, "max_antecedent": MAX_ANTECEDENT},
        "rule_counts": {"frequent_itemsets": len(freq_rows), "raw_rules_after_threshold": int(len(rules)),
                        "core_rules": int(len(core))},
        "holdout": holdout,
        "stability": {"n_core": int(len(stab)), "n_stable": int(stab.stable_75pct.sum()),
                      "share_stable": float(stab.stable_75pct.mean()) if len(stab) else np.nan},
        "negative_pairs": neg[["label_a", "label_b", "count", "lift"]].to_dict("records"),
        "manual_audit_rows": int(len(audit)),
        "manual_audit": audit_metrics,
        "note": "规则是分流与漏标检查线索，不是因果或真实故障发生率。",
    }
    save_json(summary, OUTD / "rq2_summary.json")
    print(md_table(lf.head(12)[["label", "count", "share_of_transactions", "dimensions", "management"]]))
    print("holdout", holdout)
    print("stability", summary["stability"])
    print("DONE", OUTD)


if __name__ == "__main__":
    main()
