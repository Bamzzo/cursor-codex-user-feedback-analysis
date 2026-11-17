# -*- coding: utf-8 -*-
"""阶段 02：RQ1 论坛主题、主题验证、样本敏感性与互动关系。

核心改进（相对已有任务包）
1. 从原始 JSON 重建语料，独立复现 8 主题结构；
2. 每个主题抽取代表帖，生成人工验证样本并登记边界问题；
3. 区分“版块分类”和“NMF 主题分类”；
4. 只用 2026-07 后普查正文重跑，检查主题排序敏感性；
5. 主题与浏览/回复/采纳/零回复的关系；
6. 主题稳定性（bootstrap + 与旧版主题的词表映射）。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DERIVED, OUT, FIG, ROOT, SEED, COLORS, MAIN4, setup_matplotlib
from common.stats import save_csv, save_json, cramers_v, md_table

OUTD = OUT / "rq1"
OUTD.mkdir(parents=True, exist_ok=True)
K = 8
plt = setup_matplotlib()

NAMING_RULES = [
    ("社区活动与线下聚会", {"luma", "meetup", "community", "event", "join", "roadshow", "excite"}),
    ("订阅套餐与用量额度", {"usage", "plan", "billing", "subscription", "quota", "price", "limit", "spend", "token", "cost"}),
    ("模型、子 Agent 与任务路由", {"model", "subagent", "fast", "picker", "gpt", "sonnet", "opus", "composer"}),
    ("API Key / 第三方模型接入（残余主题）", {"openai", "api", "key", "byok", "moonshot", "custom", "base"}),
    ("MCP / 网络 / 扩展宿主", {"mcp", "server", "network", "proxy", "connect", "host", "oauth", "connection"}),
    ("云端代理与仓库协作", {"cloud", "github", "repo", "slack", "pr", "automation", "linear", "branch"}),
    ("Bot / Computer Use / 设备", {"bot", "computer", "screen", "device", "ios", "desktop", "recover"}),
    ("CLI / 终端 / Tab 补全", {"cli", "terminal", "tui", "tab", "completion", "shell", "command"}),
    ("IDE 界面 / 窗口 / 编辑器", {"window", "editor", "sidebar", "layout", "panel", "ui", "view", "ide"}),
    ("模板化缺陷报告与编辑器/扩展故障（残余主题）", {"template", "bug", "autocomplete", "bedrock", "wsl"}),
    ("Agent 工具 / 规则 / 上下文", {"agent", "tool", "rule", "context", "session", "prompt", "skill", "hook", "subagent"}),
]


def name_themes(H: np.ndarray, terms: list[str]) -> tuple[list[str], list[list[tuple[str, float]]]]:
    names, tops = [], []
    for k in range(H.shape[0]):
        idx = np.argsort(H[k])[::-1][:15]
        top = [(terms[i], float(H[k, i])) for i in idx]
        tops.append(top)
        vocab = {t for t, _ in top}
        best, score = None, 0
        for name, keys in NAMING_RULES:
            s = len(vocab & keys)
            if s > score:
                best, score = name, s
        if best is None or score < 2:
            best = " / ".join([t for t, _ in top[:3]])
        names.append(best)
    # 去重：同名主题追加序号
    seen = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n} ({seen[n]})")
        else:
            seen[n] = 0
            out.append(n)
    return out, tops


def cosine_match(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A/B 行向量；返回 Hungarian 匹配索引与余弦相似度。"""
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    S = An @ Bn.T
    r, c = linear_sum_assignment(-S)
    return c, S[r, c]


def umass_coherence(H, terms, token_sets, topn=10):
    vals = []
    for k in range(H.shape[0]):
        top = [terms[i] for i in np.argsort(H[k])[::-1][:topn]]
        df = {w: sum(w in s for s in token_sets) for w in top}
        for i in range(1, len(top)):
            for j in range(i):
                wi, wj = top[i], top[j]
                pair = sum((wi in s) and (wj in s) for s in token_sets)
                vals.append(np.log((pair + 1) / (df[wj] + 1e-9)))
    return float(np.mean(vals)) if vals else np.nan


def main():
    docs = pd.read_parquet(DERIVED / "forum_docs.parquet")
    topics = pd.read_parquet(DERIVED / "forum_topics.parquet")
    docs["created_at"] = pd.to_datetime(docs["created_at"], utc=True)
    topics["created_at"] = pd.to_datetime(topics["created_at"], utc=True)
    meta = topics.set_index("id")

    corpus = docs[docs["tok_text"].str.len() > 20].copy()
    n_docs = len(corpus)
    tfidf = TfidfVectorizer(min_df=5, max_df=0.60, max_features=30000, ngram_range=(1, 1),
                            sublinear_tf=True, token_pattern=r"(?u)\b\w\w+\b")
    X = tfidf.fit_transform(corpus["tok_text"])
    terms = tfidf.get_feature_names_out().tolist()

    nmf = NMF(n_components=K, init="nndsvda", random_state=SEED, max_iter=600, beta_loss="frobenius")
    W = nmf.fit_transform(X)
    H = nmf.components_
    names, tops = name_themes(H, terms)
    corpus["theme_id"] = W.argmax(axis=1)
    corpus["theme_weight"] = W.max(axis=1) / (W.sum(axis=1) + 1e-12)

    topic_df = pd.DataFrame({"theme_id": range(K), "theme": names,
                             "top_terms": ["; ".join(t for t, _ in tops[k][:10]) for k in range(K)]})
    profile = []
    for k in range(K):
        sub = corpus[corpus.theme_id == k]
        ids = sub["topic_id"]
        m = meta.reindex(ids)
        profile.append({
            "theme_id": k, "theme": names[k], "n_topics": len(sub),
            "share_of_body_corpus": len(sub) / n_docs,
            "mean_weight": float(sub["theme_weight"].mean()),
            "top_terms": "; ".join(t for t, _ in tops[k][:10]),
            "top_category": m["category"].value_counts().idxmax() if len(m) else "",
            "top_category_share": float(m["category"].value_counts(normalize=True).iloc[0]) if len(m) else np.nan,
            "median_views": float(m["views"].median()) if len(m) else np.nan,
            "median_replies": float(m["reply_count"].median()) if len(m) else np.nan,
            "accepted_rate": float(m["has_accepted_answer"].mean()) if len(m) else np.nan,
            "zero_reply_rate": float(m["zero_reply"].mean()) if len(m) else np.nan,
            "body_chars_median": float(sub["chars"].median()),
        })
    profile = pd.DataFrame(profile).sort_values("n_topics", ascending=False)
    save_csv(profile, OUTD / "01_theme_profile.csv")

    # 版块 × 主题
    m = meta.reindex(corpus["topic_id"]).reset_index()
    m["theme"] = corpus["theme_id"].map(dict(zip(range(K), names))).values
    cat = m["category"].where(m["category"].isin(MAIN4 + ["Discussions"]), "Other")
    ct = pd.crosstab(m["theme"], cat)
    ct_share = ct.div(ct.sum(axis=1), axis=0)
    save_csv(ct.reset_index(), OUTD / "02_theme_by_category_counts.csv")
    save_csv(ct_share.reset_index(), OUTD / "02_theme_by_category_share.csv")
    chi2, p, v = cramers_v(ct)
    cat_test = {"chi2": chi2, "p": p, "cramers_v": v, "n": int(ct.to_numpy().sum())}

    # 月度份额
    month = pd.crosstab(m["month"], m["theme"])
    month_share = month.div(month.sum(axis=1), axis=0)
    save_csv(month.reset_index(), OUTD / "03_theme_month_counts.csv")
    save_csv(month_share.reset_index(), OUTD / "03_theme_month_share.csv")

    # 互动关系
    tests = {}
    groups_views = [m.loc[m.theme == t, "views"].dropna() for t in names]
    groups_reply = [m.loc[m.theme == t, "reply_count"].dropna() for t in names]
    h1, p1 = stats.kruskal(*groups_views)
    h2, p2 = stats.kruskal(*groups_reply)
    tests["views_kruskal"] = {"H": float(h1), "p": float(p1)}
    tests["replies_kruskal"] = {"H": float(h2), "p": float(p2)}
    acc = pd.crosstab(m["theme"], m["has_accepted_answer"])
    chi_a, p_a, v_a = cramers_v(acc)
    zr = pd.crosstab(m["theme"], m["zero_reply"])
    chi_z, p_z, v_z = cramers_v(zr)
    tests["accepted_chi2"] = {"chi2": chi_a, "p": p_a, "cramers_v": v_a}
    tests["zero_reply_chi2"] = {"chi2": chi_z, "p": p_z, "cramers_v": v_z}
    tests["category_chi2"] = cat_test
    tests["note"] = "互动差异是论坛行为分布，不等于需求重要性或产品质量。"

    # 覆盖敏感性：全量正文 vs 仅 2026-07 后普查
    census_mask = corpus["created_at"] >= pd.Timestamp("2026-07-01", tz="UTC")
    full_share = corpus["theme_id"].value_counts(normalize=True).reindex(range(K), fill_value=0).to_numpy()
    cens_share = corpus.loc[census_mask, "theme_id"].value_counts(normalize=True).reindex(range(K), fill_value=0).to_numpy()
    rho, rho_p = stats.spearmanr(full_share, cens_share)
    sens = pd.DataFrame({"theme_id": range(K), "theme": names,
                         "share_full_body": full_share, "share_census_only": cens_share,
                         "diff_census_minus_full": cens_share - full_share})
    save_csv(sens, OUTD / "04_sensitivity_coverage.csv")
    tests["coverage_sensitivity"] = {"spearman": float(rho), "p": float(rho_p),
                                     "n_full": int(n_docs), "n_census": int(census_mask.sum())}

    # 重新拟合（仅普查）并与主模型主题匹配
    Xc = tfidf.transform(corpus.loc[census_mask, "tok_text"])
    nmf_c = NMF(n_components=K, init="nndsvda", random_state=SEED, max_iter=600)
    Wc = nmf_c.fit_transform(Xc)
    Hc = nmf_c.components_
    names_c, tops_c = name_themes(Hc, terms)
    order, sims = cosine_match(H, Hc)
    refit = []
    for base_k, c_k in enumerate(order):
        refit.append({"base_theme_id": base_k, "base_theme": names[base_k],
                      "census_theme": names_c[c_k], "cosine_similarity": float(sims[base_k]),
                      "census_share": float((Wc.argmax(axis=1) == c_k).mean()),
                      "full_share_census_subset": float((corpus.loc[census_mask, "theme_id"] == base_k).mean())})
    refit = pd.DataFrame(refit).sort_values("cosine_similarity", ascending=False)
    save_csv(refit, OUTD / "04_sensitivity_refit.csv")
    tests["refit_match_median_cosine"] = float(refit["cosine_similarity"].median())

    # bootstrap 稳定性
    rng = np.random.default_rng(SEED)
    boot_rows = []
    for b in range(5):
        ix = rng.integers(0, n_docs, n_docs)
        Xb = X[ix]
        nb = NMF(n_components=K, init="nndsvda", random_state=SEED + b, max_iter=300)
        nb.fit_transform(Xb)
        Hb = nb.components_
        o, s = cosine_match(H, Hb)
        for base_k, b_k in enumerate(o):
            boot_rows.append({"base_theme_id": base_k, "base_theme": names[base_k],
                              "bootstrap": b, "cosine_similarity": float(s[base_k])})
    boot = pd.DataFrame(boot_rows)
    boot_sum = boot.groupby(["base_theme_id", "base_theme"])["cosine_similarity"].agg(
        mean="mean", min="min", max="max").reset_index()
    save_csv(boot, OUTD / "05_stability_bootstrap_raw.csv")
    save_csv(boot_sum, OUTD / "05_stability_bootstrap.csv")

    # 内部指标（只作模型诊断，不作真实性证明）
    token_sets = [set(x.split()) for x in corpus["tok_text"]]
    coh = umass_coherence(H, terms, token_sets, topn=10)
    topic_div = len({t for k in range(K) for t in np.argsort(H[k])[::-1][:10]}) / (K * 10)
    relative_reconstruction_error = float(nmf.reconstruction_err_ ** 2 / X.multiply(X).sum())
    mean_dominant_topic_share = float(corpus["theme_weight"].mean())
    internal = {"k": K, "n_docs": int(n_docs), "n_features": int(X.shape[1]),
                "reconstruction_error": float(nmf.reconstruction_err_),
                "relative_reconstruction_error": relative_reconstruction_error,
                "reconstruction_share": 1.0 - relative_reconstruction_error,
                "n_iter": int(nmf.n_iter_), "umass_coherence_top10": coh,
                "topic_diversity_top10": float(topic_div),
                "mean_dominant_topic_share": mean_dominant_topic_share,
                "note": "reconstruction_share=1-||X-WH||_F^2/||X||_F^2，是未中心化 TF-IDF 矩阵的相对重构比例，不等同于 PCA 方差解释率或主题准确率；mean_dominant_topic_share 表示每篇文本最大主题权重占比。内部指标不证明主题等于真实用户需求。"}
    tests["internal_metrics"] = internal

    # 与旧任务包主题的词表映射（结构复核，不覆盖旧结果）
    legacy_map = []
    # Resolve the preserved teammate delivery from the project root; never depend on
    # the original author's machine-specific checkout path.
    legacy_path = ROOT / "data" / "legacy" / "rq1" / "T9_主题关键词.csv"
    if legacy_path.exists():
        lg = pd.read_csv(legacy_path)
        for k in range(K):
            ours = {t for t, _ in tops[k][:10]}
            best, best_j = "", 0.0
            for lid, g in lg.groupby("topic_id"):
                theirs = set(g.sort_values("rank").head(10)["term"])
                j = len(ours & theirs) / len(ours | theirs) if (ours | theirs) else 0
                if j > best_j:
                    best, best_j = f"legacy_topic_{lid}", j
            legacy_map.append({"theme_id": k, "theme": names[k],
                               "best_legacy_topic": best, "jaccard_top10": best_j})
    legacy_map = pd.DataFrame(legacy_map)
    save_csv(legacy_map, OUTD / "06_legacy_theme_mapping.csv")

    # 人工验证样本：每主题权重最高的 6 个主题帖
    samples = []
    for k in range(K):
        sub = corpus[corpus.theme_id == k].sort_values("theme_weight", ascending=False).head(6)
        for _, r in sub.iterrows():
            mm = meta.loc[r["topic_id"]] if r["topic_id"] in meta.index else None
            text = (r.get("tok_text") or "")
            samples.append({
                "theme_id": k, "theme_name_proposed": names[k],
                "topic_id": int(r["topic_id"]), "title": r["title"],
                "category": mm["category"] if mm is not None else "",
                "month": r["month"], "views": mm["views"] if mm is not None else np.nan,
                "replies": mm["reply_count"] if mm is not None else np.nan,
                "accepted": mm["has_accepted_answer"] if mm is not None else np.nan,
                "top_terms": topic_df.loc[k, "top_terms"],
                "snippet": " ".join(text.split()[:60]),
                "manual_relevance": "", "manual_theme_fit": "", "boundary_note": "",
            })
    samp = pd.DataFrame(samples)
    validation = {"reviewed": 0, "note": "未找到主题人工回读表，仅输出空白样板"}
    manual_path = ROOT / "data" / "manual" / "rq1_theme_audit.csv"
    if manual_path.exists() and len(samp):
        man = pd.read_csv(manual_path).rename(columns={
            "is_product_feedback": "manual_relevance", "theme_fit": "manual_theme_fit", "note": "boundary_note"})
        samp = samp.drop(columns=["manual_relevance", "manual_theme_fit", "boundary_note"])
        samp = samp.merge(man[["topic_id", "manual_relevance", "manual_theme_fit", "boundary_note"]],
                          on="topic_id", how="left")
        aud = samp[samp.manual_theme_fit.notna()].copy()
        per_theme = []
        for k, g in aud.groupby("theme_id"):
            per_theme.append({"theme_id": int(k), "theme": g.theme_name_proposed.iloc[0],
                              "reviewed": int(len(g)), "fit": int(g.manual_theme_fit.sum()),
                              "fit_rate": float(g.manual_theme_fit.mean()),
                              "product_feedback_share": float(g.manual_relevance.mean())})
        validation = {
            "reviewed": int(len(aud)),
            "theme_fit": int(aud.manual_theme_fit.sum()),
            "theme_fit_rate": float(aud.manual_theme_fit.mean()) if len(aud) else np.nan,
            "product_feedback_share": float(aud.manual_relevance.mean()) if len(aud) else np.nan,
            "misfit_topics": aud[aud.manual_theme_fit == 0][["theme_id", "topic_id", "title", "boundary_note"]].to_dict("records"),
            "per_theme": per_theme,
            "method": "LLM 辅助的结构化回读（每主题权重最高 6 帖），不是独立人工双盲标注；名称与边界需人工抽检确认。",
        }
    save_csv(samp, OUTD / "07_theme_validation_sample.csv")
    save_json(validation, OUTD / "08_theme_validation_summary.json")

    # 图表
    p = profile.sort_values("share_of_body_corpus")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(p["theme"], p["share_of_body_corpus"] * 100, color=COLORS[0])
    ax.set_xlabel(f"占清洗后正文主题帖比例（%，n={int(p['n_topics'].sum()):,}）")
    ax.set_title("RQ1 主题结构：8 个 NMF 主题（正文覆盖 2026-03 起）")
    ax.bar_label(bars, labels=[f"{v*100:.1f}%  (n={n:,})" for v, n in zip(p["share_of_body_corpus"], p["n_topics"])], padding=3, fontsize=8)
    ax.set_xlim(0, p["share_of_body_corpus"].max() * 135)
    fig.tight_layout(); fig.savefig(FIG / "rq1_01_主题结构.png", bbox_inches="tight"); plt.close(fig)

    heat = ct_share.reindex([n for n in names if n in ct_share.index])
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow(heat.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(heat.shape[1])); ax.set_xticklabels(heat.columns, rotation=25, ha="right")
    ax.set_yticks(range(heat.shape[0])); ax.set_yticklabels(heat.index, fontsize=8)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.iloc[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.55 else "#222")
    ax.set_title("主题 × 官方版块：主题分类不是版块分类")
    fig.colorbar(im, ax=ax, fraction=0.035, label="该主题内版块占比")
    fig.tight_layout(); fig.savefig(FIG / "rq1_02_主题与版块.png", bbox_inches="tight"); plt.close(fig)

    # 编号散点 + 右侧图例，避免密集主题名称彼此覆盖。
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    plot_rows = profile.reset_index(drop=True)
    ax.scatter(plot_rows["zero_reply_rate"] * 100, plot_rows["accepted_rate"] * 100,
               s=np.sqrt(plot_rows["n_topics"]) * 7, c=COLORS[1], alpha=.8,
               edgecolors="white", linewidths=.7, zorder=3)
    offsets = [(-11, -10), (0, 12), (12, 1), (-12, -12), (-8, 13), (12, -10), (-10, 9), (0, 8)]
    for idx, r in plot_rows.iterrows():
        ax.annotate(str(idx + 1), (r["zero_reply_rate"] * 100, r["accepted_rate"] * 100),
                    fontsize=8, fontweight="bold", ha="center", va="center",
                    xytext=offsets[idx], textcoords="offset points",
                    bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="#666", lw=.6),
                    arrowprops=dict(arrowstyle="-", color="#777", lw=.65), zorder=4)
    ax.set_xlabel("零回复率（%）"); ax.set_ylabel("采纳率（%）")
    ax.set_title("主题与社区闭环：采纳率低、零回复高（论坛行为指标）")
    ax.grid(alpha=.2, zorder=0)
    ax.set_xlim(50, 103); ax.set_ylim(-1, 18)
    legend_lines = [f"{i+1}. {r.theme}  (n={int(r.n_topics):,})" for i, r in plot_rows.iterrows()]
    ax.text(1.02, .98, "主题编号", transform=ax.transAxes, ha="left", va="top",
            fontsize=9, fontweight="bold")
    ax.text(1.02, .92, "\n".join(legend_lines), transform=ax.transAxes, ha="left", va="top",
            fontsize=8, linespacing=1.55)
    fig.subplots_adjust(right=.68)
    fig.tight_layout(); fig.savefig(FIG / "rq1_03_主题与互动.png", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    mt = topics.groupby("month").size()
    bt = docs.groupby("month").size().reindex(mt.index, fill_value=0)
    axes[0].bar(mt.index, mt.values, color="#c9d6e4", label="全部主题（元数据）")
    axes[0].bar(bt.index, bt.values, color=COLORS[0], label="有正文主题")
    axes[0].set_title("正文覆盖：2026-07 后为普查，之前为分层抽样")
    axes[0].set_ylabel("主题数"); axes[0].tick_params(axis="x", rotation=60)
    axes[0].legend(fontsize=8)
    axes[1].scatter(full_share * 100, cens_share * 100, c=COLORS[2])
    for i, n in enumerate(names):
        axes[1].annotate(str(i), (full_share[i] * 100, cens_share[i] * 100), fontsize=8)
    lim = max(full_share.max(), cens_share.max()) * 105
    axes[1].plot([0, lim], [0, lim], ls="--", color="gray", lw=.8)
    axes[1].set_xlabel("完整正文样本份额（%）"); axes[1].set_ylabel("仅 2026-07 后普查份额（%）")
    axes[1].set_title(f"样本敏感性：Spearman ρ={rho:.2f}")
    axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIG / "rq1_04_覆盖与敏感性.png", bbox_inches="tight"); plt.close(fig)

    summary = {
        "corpus": {"body_topics": int(n_docs), "body_posts": int(len(pd.read_parquet(DERIVED / "forum_posts.parquet"))),
                   "body_topics_raw": int(len(docs)), "excluded_short_docs": int(len(docs) - n_docs),
                   "all_topics": int(len(topics)), "coverage": float(len(topics[topics.has_body == 1]) / len(topics)),
                   "body_start": str(docs.created_at.min()), "body_end": str(docs.created_at.max()),
                   "census_topics": int(census_mask.sum()), "pre_census_topics": int((~census_mask).sum())},
        "themes": profile.to_dict("records"),
        "tests": tests,
        "stability_bootstrap_mean": float(boot_sum["mean"].mean()) if len(boot_sum) else np.nan,
        "legacy_mapping": legacy_map.to_dict("records"),
        "manual_validation": validation,
        "naming_note": "主题名称由高频词规则生成后人工复核；名称是解释标签，不是客观类别。",
    }
    save_json(summary, OUTD / "rq1_summary.json")
    print(md_table(profile[["theme", "n_topics", "share_of_body_corpus", "median_views",
                            "accepted_rate", "zero_reply_rate"]]))
    print("internal", internal)
    print("coverage sens", rho, "refit median", tests["refit_match_median_cosine"])
    print("bootstrap", boot_sum.to_string(index=False))
    print("DONE", OUTD)


if __name__ == "__main__":
    main()
