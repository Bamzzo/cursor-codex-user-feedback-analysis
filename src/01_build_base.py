# -*- coding: utf-8 -*-
"""阶段 01：构建公共分析底座（只读原始数据，派生文件写入 data/derived）。

输出
----
data/derived/issues.parquet          一条 Issue 一行，含统一分类、状态、回复指标
data/derived/comments.parquet        关联后的评论（正文截断，保留判断所需字段）
data/derived/forum_topics.parquet    41,218 条论坛主题元数据 + 统一分类
data/derived/forum_docs.parquet      4,322 个有正文的主题（清洗后文本）
data/derived/forum_posts.parquet     19,923 条帖子（清洗后文本）
data/derived/hn_filtered.parquet     关键词过滤后的 HN 辅助样本
outputs/integrated/00_data_audit.json
docs/统一分类字典.md / 字段字典.md / 样本登记表.md / 口径登记表.md
"""
from __future__ import annotations
import sys, json, hashlib, re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import (ROOT, RAW_DIR, DELIVERY_DIR, DERIVED, OUT, DOCS, RAW_FILES,
                           SNAPSHOT_UTC, FORUM_CATEGORIES, FORUM_GROUP, MAIN4, SEED)
from common import taxonomy as tx
from common.text import html_to_text, clean_text, token_text
from common.stats import save_csv, save_json

SNAP = pd.Timestamp(SNAPSHOT_UTC)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(name: str):
    with open(RAW_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def parse_dt(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def norm_label(x) -> str:
    if isinstance(x, dict):
        x = x.get("name", "")
    return str(x).strip()


def hash_audit() -> list[dict]:
    man_path = DELIVERY_DIR / "_manifest.json"
    expected = {}
    if man_path.exists():
        mf = json.loads(man_path.read_text(encoding="utf-8"))
        for m in mf.get("files", []):
            expected[m["file"]] = m.get("sha256")
    rows = []
    for key, name in RAW_FILES.items():
        p = RAW_DIR / name
        actual = sha256(p)
        rows.append({"file": name, "bytes": p.stat().st_size, "sha256": actual,
                     "manifest_sha256": expected.get(name, ""),
                     "match_manifest": (expected.get(name) == actual) if expected.get(name) else None})
    return rows


# ---------------------------------------------------------------------------
# 统一分类
# ---------------------------------------------------------------------------
def dimension_sources(label_sig, text_sig):
    out = {}
    for dim in ["entry", "cap", "env", "type"]:
        if label_sig.get(dim):
            out[dim] = "label/tag"
        elif text_sig.get(dim):
            out[dim] = "text"
        else:
            out[dim] = "unknown"
    return out


def enrich(df: pd.DataFrame, labels_col: str, title_col: str, full_col: str | None = None,
           category: pd.Series | None = None) -> pd.DataFrame:
    """为每条记录加统一分类字段。"""
    recs = []
    for labels, title, full in zip(df[labels_col], df[title_col],
                                   df[full_col] if full_col else [""] * len(df)):
        lab_sig = tx.label_signals(labels)
        title_sig = tx.classify_text(title or "")
        full_sig = tx.classify_text((title or "") + " " + (full or "")[:8000])
        merged = tx.merge_signals(lab_sig, full_sig)
        src = dimension_sources(lab_sig, full_sig)
        rec = {}
        for dim, order, prefix in [("entry", tx.ENTRY_ORDER, "entry"),
                                   ("cap", tx.CAP_ORDER, "cap"),
                                   ("env", tx.ENV_ORDER, "env"),
                                   ("type", tx.TYPE_ORDER, "type")]:
            vals = merged[dim]
            lab_vals = tx._ordered(lab_sig.get(dim, []), order)
            rec[f"{prefix}_multi"] = ";".join(vals)
            rec[f"{prefix}_label_primary"] = tx.primary(lab_vals, order) if lab_vals else "unknown"
            # 有平台标签时，主类以标签为准；文本只用于补充多标签与缺失维度。
            rec[f"{prefix}_primary"] = (rec[f"{prefix}_label_primary"] if lab_vals
                                        else tx.primary(vals, order))
            rec[f"{prefix}_title"] = tx.primary(title_sig[dim], order)
        rec["dim_source"] = ";".join(f"{k}={v}" for k, v in src.items())
        rec["management_label"] = int(lab_sig.get("management", False))
        recs.append(rec)
    out = pd.DataFrame(recs, index=df.index)
    return pd.concat([df, out], axis=1)


# ---------------------------------------------------------------------------
# Issues + comments
# ---------------------------------------------------------------------------
def build_issues():
    raw = load_json(RAW_FILES["issues"])
    a = pd.DataFrame(raw)
    n_raw = len(a)
    a["number"] = pd.to_numeric(a["number"], errors="coerce").astype("Int64")
    a["updated_at"] = parse_dt(a["updated_at"])
    a = a.sort_values("updated_at").drop_duplicates("number", keep="last").copy()

    for c in ["created_at", "closed_at"]:
        a[c] = parse_dt(a[c])
    a["labels_display"] = a["labels"].apply(lambda xs: [norm_label(x) for x in (xs or [])])
    a["labels"] = a["labels_display"].apply(lambda xs: [x.casefold() for x in xs])
    a["label_count"] = a["labels"].apply(len)
    a["title"] = a["title"].fillna("").astype(str)
    a["body"] = a["body"].fillna("").astype(str)
    a["body_chars"] = a["body"].str.len()
    a["title_chars"] = a["title"].str.len()
    a["has_code"] = a["body"].str.contains("```|<code", case=False, regex=True).astype(int)
    a["created_month"] = a["created_at"].dt.strftime("%Y-%m")
    a["created_quarter"] = a["created_at"].dt.to_period("Q").astype(str)
    a["outcome"] = [tx.outcome_from_issue(s, r) for s, r in zip(a["state"], a["state_reason"])]
    a["closed_flag"] = a["state"].eq("closed").astype(int)
    a["close_days"] = (a["closed_at"] - a["created_at"]).dt.total_seconds() / 86400
    a["age_days"] = (SNAP - a["created_at"]).dt.total_seconds() / 86400
    a["duration_days"] = np.where(a["closed_flag"].eq(1), a["close_days"], a["age_days"])
    a["is_open_90d"] = ((a["closed_flag"].eq(0)) & (a["age_days"] >= 90)).astype(int)
    a["eligible_30d"] = (a["age_days"] >= 30).astype(int)
    a["eligible_3d"] = (a["age_days"] >= 3).astype(int)
    a["eligible_90d"] = (a["age_days"] >= 90).astype(int)
    a["closed_within_30d"] = ((a["closed_flag"].eq(1)) & (a["close_days"] <= 30)).astype(int)
    a["url"] = "https://github.com/openai/codex/issues/" + a["number"].astype(str)

    a = enrich(a, "labels", "title", "body")

    # 标签维度缺失时的显式标记：报告里不能把 text 推断当作维护者标签
    a["label_type"] = a["labels"].apply(
        lambda xs: next((t for t in ["bug", "enhancement", "performance", "documentation"]
                         if t in xs), "none"))
    a["has_problem_label"] = (~a["label_type"].eq("none")).astype(int)

    # 评论
    c = pd.DataFrame(load_json(RAW_FILES["comments"]))
    n_comments_raw = len(c)
    c["created_at"] = parse_dt(c["created_at"])
    c["updated_at"] = parse_dt(c["updated_at"])
    c = c.sort_values("updated_at").drop_duplicates("id", keep="last").copy()
    c["user"] = c["user"].fillna("")
    app = c["performed_via_github_app"]
    c["is_bot"] = app.apply(lambda x: not (x is None or x is False or str(x) in ("", "None", "nan")))
    c["is_bot"] |= c["user"].str.contains(r"\[bot\]$|-bot$", case=False, regex=True)
    valid_numbers = set(a["number"].dropna().astype(int))
    c["issue_number"] = pd.to_numeric(c["issue_number"], errors="coerce").astype("Int64")
    c["matched"] = c["issue_number"].isin(valid_numbers)
    c = c.merge(a[["number", "created_at", "user"]].rename(
        columns={"created_at": "issue_created", "user": "issue_user"}),
        left_on="issue_number", right_on="number", how="left", validate="many_to_one")
    c["after_issue"] = c["created_at"] >= c["issue_created"]
    c["before_snapshot"] = c["created_at"] <= SNAP
    c["external_human"] = (~c["is_bot"]) & (c["user"] != c["issue_user"]) & c["matched"]
    c["maintainer"] = c["external_human"] & c["author_association"].isin(
        ["MEMBER", "COLLABORATOR", "OWNER"])
    used = c[c["after_issue"] & c["before_snapshot"]].copy()

    first_ext = used[used["external_human"]].groupby("issue_number")["created_at"].min()
    first_mnt = used[used["maintainer"]].groupby("issue_number")["created_at"].min()
    a["first_external_at"] = a["number"].map(first_ext)
    a["first_maintainer_at"] = a["number"].map(first_mnt)
    a["first_external_days"] = (a["first_external_at"] - a["created_at"]).dt.total_seconds() / 86400
    a["first_maintainer_days"] = (a["first_maintainer_at"] - a["created_at"]).dt.total_seconds() / 86400
    ext_n = used[used["external_human"]].groupby("issue_number").size()
    mnt_n = used[used["maintainer"]].groupby("issue_number").size()
    all_n = used.groupby("issue_number").size()
    bot_n = used[used["is_bot"]].groupby("issue_number").size()
    a["external_reply_count"] = a["number"].map(ext_n).fillna(0).astype(int)
    a["maintainer_reply_count"] = a["number"].map(mnt_n).fillna(0).astype(int)
    a["matched_comment_count"] = a["number"].map(all_n).fillna(0).astype(int)
    a["bot_comment_count"] = a["number"].map(bot_n).fillna(0).astype(int)
    a["has_external_reply"] = (a["external_reply_count"] > 0).astype(int)
    a["has_maintainer_reply"] = (a["maintainer_reply_count"] > 0).astype(int)
    a["external_reply_3d"] = (a["first_external_days"] <= 3).fillna(False).astype(int)
    a["no_external_reply"] = (a["external_reply_count"] == 0).astype(int)

    use_cols = ["id", "issue_number", "user", "author_association", "created_at",
                "is_bot", "external_human", "maintainer", "matched"]
    cu = c[use_cols + ["body"]].copy()
    cu["body"] = cu["body"].fillna("").astype(str).str.slice(0, 1000)

    audit = {
        "issues_raw": n_raw,
        "issues_after_dedupe": len(a),
        "issue_duplicates": n_raw - len(a),
        "issue_created_min": str(a["created_at"].min()),
        "issue_created_max": str(a["created_at"].max()),
        "issue_date_na": int(a["created_at"].isna().sum()),
        "issue_invalid_closed": int(((a["state"] == "closed") & (a["closed_at"].isna())).sum()),
        "empty_body": int((a["body_chars"] == 0).sum()),
        "no_labels": int((a["label_count"] == 0).sum()),
        "comments_raw": n_comments_raw,
        "comments_after_dedupe": len(c),
        "comments_matched": int(c["matched"].sum()),
        "comments_unmatched": int((~c["matched"]).sum()),
        "comments_used": len(used),
        "comments_bot": int(used["is_bot"].sum()),
        "comments_external_human": int(used["external_human"].sum()),
        "comments_maintainer": int(used["maintainer"].sum()),
        "issues_with_external_reply": int(a["has_external_reply"].sum()),
        "issues_with_maintainer_reply": int(a["has_maintainer_reply"].sum()),
        "outcome_counts": a["outcome"].value_counts().to_dict(),
        "type_counts": a["type_primary"].value_counts().to_dict(),
        "type_label_counts": a["type_label_primary"].value_counts().to_dict(),
        "entry_counts": a["entry_primary"].value_counts().to_dict(),
        "env_counts": a["env_primary"].value_counts().to_dict(),
        "management_only_issues": int(a["labels"].apply(
            lambda xs: len(xs) > 0 and all(tx.is_management(x) for x in xs)).sum()),
        "snapshot_utc": SNAPSHOT_UTC,
    }
    save_json(audit, OUT / "integrated" / "01_issues_audit.json")

    # 只保留分析所需列，正文截断以控制体积
    out = a.drop(columns=["labels_display", "body"]).copy()
    out["body_snippet"] = a["body"].str.slice(0, 6000)
    out.to_parquet(DERIVED / "issues.parquet", index=False)
    cu.to_parquet(DERIVED / "comments.parquet", index=False)
    return a, c, used, audit


# ---------------------------------------------------------------------------
# 论坛
# ---------------------------------------------------------------------------
def build_forum():
    topics = pd.DataFrame(load_json(RAW_FILES["forum_topics"]))
    n_raw = len(topics)
    topics["created_at"] = parse_dt(topics["created_at"])
    topics["last_posted_at"] = parse_dt(topics.get("last_posted_at"))
    topics = topics.drop_duplicates("id", keep="last").copy()
    topics["tags"] = topics["tags"].apply(lambda xs: [norm_label(x) for x in (xs or [])])
    topics["category"] = topics["category_id"].map(FORUM_CATEGORIES).fillna("Unknown")
    topics["group_cn"] = topics["category"].map(FORUM_GROUP).fillna("其他")
    topics["title"] = topics["title"].fillna("").astype(str)
    topics["month"] = topics["created_at"].dt.strftime("%Y-%m")
    topics["quarter"] = topics["created_at"].dt.to_period("Q").astype(str)
    topics["year"] = topics["created_at"].dt.year
    topics["posts_count"] = pd.to_numeric(topics["posts_count"], errors="coerce").fillna(0).astype(int)
    topics["reply_count"] = pd.to_numeric(topics["reply_count"], errors="coerce").fillna(0).astype(int)
    topics["views"] = pd.to_numeric(topics["views"], errors="coerce").fillna(0)
    topics["like_count"] = pd.to_numeric(topics["like_count"], errors="coerce").fillna(0)
    topics["has_accepted_answer"] = topics["has_accepted_answer"].fillna(False).astype(int)
    topics["closed"] = topics["closed"].fillna(False).astype(int)
    topics["zero_reply"] = (topics["reply_count"] == 0).astype(int)
    topics["url"] = topics["slug"].fillna("").apply(lambda s: f"https://forum.cursor.com/t/{s}") + "/" + topics["id"].astype(str)

    topics["labels"] = topics["tags"]
    topics = enrich(topics, "labels", "title")

    # 论坛主类：版块语义优先，标签/文本补充
    cat_type = {"Bug Reports": "bug", "Help": "help", "Feature Requests": "enhancement",
                "Feedback": "feedback", "Discussions": "feedback"}
    topics["type_primary_forum"] = topics["category"].map(cat_type)
    topics["type_primary_forum"] = topics["type_primary_forum"].fillna(topics["type_primary"])
    topics["type_source_forum"] = np.where(topics["category"].isin(list(cat_type)), "category", topics["dim_source"].str.extract(r"type=(\w+)")[0])

    # 正文语料：census + early sample，按 topic_id 去重取帖数更多者
    census = load_json(RAW_FILES["forum_census"])
    sample = load_json(RAW_FILES["forum_content"])
    raw = {}
    for rec in census + sample:
        t = rec["topic"]
        tid = t["id"]
        if tid not in raw or len(rec["posts"]) > len(raw[tid]["posts"]):
            raw[tid] = rec

    post_rows, topic_rows = [], []
    for tid, rec in raw.items():
        t = rec["topic"]
        posts = sorted(rec["posts"], key=lambda p: p.get("post_number") or 0)
        cleaned, raw_chars = [], 0
        for p in posts:
            txt = clean_text(html_to_text(p.get("cooked", "")))
            raw_chars += len(p.get("cooked", "") or "")
            post_rows.append({
                "topic_id": tid, "post_number": p.get("post_number"),
                "username": p.get("username"), "created_at": parse_dt(p.get("created_at")),
                "accepted_answer": int(bool(p.get("accepted_answer"))),
                "chars": len(txt), "text": txt,
            })
            cleaned.append(txt)
        full = " ".join(x for x in cleaned if x).strip()
        topic_rows.append({
            "topic_id": tid, "title": t.get("title", ""), "slug": t.get("slug", ""),
            "category_id": t.get("category_id"), "created_at": parse_dt(t.get("created_at")),
            "views": t.get("views"), "like_count": t.get("like_count"),
            "reply_count": t.get("reply_count"), "posts_count": t.get("posts_count"),
            "has_accepted_answer": int(bool(t.get("has_accepted_answer"))),
            "tags": [norm_label(x) for x in (t.get("tags") or [])],
            "n_posts_harvested": len(posts), "raw_html_chars": raw_chars,
            "text_all": full[:60000], "text_op": (cleaned[0] if cleaned else "")[:30000],
            "chars": len(full),
        })
    docs = pd.DataFrame(topic_rows)
    posts = pd.DataFrame(post_rows)
    docs["created_at"] = parse_dt(docs["created_at"])
    posts["created_at"] = parse_dt(posts["created_at"])
    docs = docs.merge(topics[["id", "category", "group_cn", "month", "quarter", "views",
                              "reply_count", "posts_count", "has_accepted_answer", "tags"]]
                      .rename(columns={"id": "topic_id"}), on="topic_id", how="left",
                      suffixes=("", "_meta"))
    docs["tok_text"] = docs["text_all"].apply(token_text)
    docs["tok_op"] = docs["text_op"].apply(token_text)
    posts["tok_text"] = posts["text"].apply(token_text)
    docs["month"] = docs["created_at"].dt.strftime("%Y-%m")
    docs["quarter"] = docs["created_at"].dt.to_period("Q").astype(str)

    # 主题表补正文覆盖
    stats = docs.groupby("topic_id").agg(harvested_posts=("n_posts_harvested", "sum"),
                                         doc_chars=("chars", "sum")).reset_index()
    topics = topics.merge(stats, left_on="id", right_on="topic_id", how="left")
    topics["has_body"] = topics["topic_id"].notna().astype(int)
    topics["body_coverage"] = topics["has_body"]

    audit = {
        "forum_topics_raw": n_raw, "forum_topics_deduped": len(topics),
        "forum_created_min": str(topics["created_at"].min()),
        "forum_created_max": str(topics["created_at"].max()),
        "forum_body_topics": len(docs), "forum_body_posts": len(posts),
        "forum_body_coverage_topics": len(docs) / len(topics),
        "forum_census_topics": int((docs["created_at"] >= pd.Timestamp("2026-07-01", tz="UTC")).sum()),
        "forum_sample_topics": int((docs["created_at"] < pd.Timestamp("2026-07-01", tz="UTC")).sum()),
        "forum_tags": int(topics["tags"].explode().nunique()),
        "forum_duplicate_topic_ids": n_raw - len(topics),
        "forum_categories": topics["category"].value_counts().to_dict(),
        "main4_share": float(topics["category"].isin(MAIN4).mean()),
        "accepted_rate": float(topics["has_accepted_answer"].mean()),
        "zero_reply_rate": float(topics["zero_reply"].mean()),
    }
    save_json(audit, OUT / "integrated" / "02_forum_audit.json")

    topics.drop(columns=["topic_id"]).to_parquet(DERIVED / "forum_topics.parquet", index=False)
    docs.drop(columns=["text_all", "text_op"]).to_parquet(DERIVED / "forum_docs.parquet", index=False)
    posts.drop(columns=["text"]).to_parquet(DERIVED / "forum_posts.parquet", index=False)
    return topics, docs, posts, audit


# ---------------------------------------------------------------------------
# HN 辅助
# ---------------------------------------------------------------------------
def build_hn():
    hn = pd.DataFrame(load_json(RAW_FILES["hn"]))
    hn["created_at"] = parse_dt(hn["created_at"])
    text = (hn["title"].fillna("") + " " + hn["text"].fillna("")).str.lower()
    coding = r"(ide|editor|code|coding|agent|completion|developer|vs code|programming)"
    strong = r"(\bcodex\b|\bclaude code\b|\bcopilot\b|\bwindsurf\b|\bdevin\b|\baider\b|\bcline\b|agentic coding|vibe coding)"
    weak = r"\bcursor\b"
    mask = text.str.contains(strong, regex=True) | (
        text.str.contains(weak, regex=True) & text.str.contains(coding, regex=True))
    sub = hn[mask].copy()
    sub["month"] = sub["created_at"].dt.strftime("%Y-%m")
    sub["is_story"] = sub["kind"].eq("story").astype(int)
    audit = {"hn_raw": len(hn), "hn_keyword_matched": len(sub),
             "hn_stories": int(sub["is_story"].sum()),
             "hn_date_min": str(hn["created_at"].min()), "hn_date_max": str(hn["created_at"].max()),
             "note": "关键词模糊匹配，仅作辅助对照，不进入核心结论"}
    save_json(audit, OUT / "integrated" / "03_hn_audit.json")
    sub[["objectID", "kind", "created_at", "author", "title", "text", "points",
         "num_comments", "query", "month", "is_story"]].to_parquet(DERIVED / "hn_filtered.parquet", index=False)
    return audit


# ---------------------------------------------------------------------------
# 文档
# ---------------------------------------------------------------------------
FIELD_DICT = [
    ("issues.parquet", "number", "Issue 编号", "int", "GitHub 主键"),
    ("issues.parquet", "labels", "标签列表（小写）", "list[str]", "快照标签，不是创建时版本"),
    ("issues.parquet", "state / state_reason", "状态 / 关闭原因", "str", "closed 不等于 fixed"),
    ("issues.parquet", "outcome", "统一处理结果", "enum", "open/completed/duplicate/not_planned/reopened/closed_unspecified"),
    ("issues.parquet", "entry_primary", "主产品入口", "enum", "标签优先、文本补充"),
    ("issues.parquet", "cap_multi", "能力模块（多标签）", "str;", "分号分隔"),
    ("issues.parquet", "env_primary", "主运行环境", "enum", "windows/macos/linux/wsl/browser/mobile/cloud/unknown"),
    ("issues.parquet", "type_primary", "主问题类型", "enum", "bug/performance/connectivity/help/enhancement/feedback/docs/other/unknown"),
    ("issues.parquet", "type_title", "仅标题文本规则的问题类型", "enum", "用于 RQ4 跨渠道可比口径"),
    ("issues.parquet", "close_days", "创建到关闭天数", "float", "仅 closed；可能不是首次关闭"),
    ("issues.parquet", "age_days", "创建到推定快照天数", "float", "观察终点为推定值"),
    ("issues.parquet", "first_external_days", "首次外部人类回复等待天数", "float", "排除机器人与作者本人"),
    ("issues.parquet", "first_maintainer_days", "首次维护者回复等待天数", "float", "MEMBER/COLLABORATOR/OWNER"),
    ("issues.parquet", "has_external_reply", "是否有外部人类回复", "0/1", "观察窗口截至快照"),
    ("issues.parquet", "closed_within_30d", "是否在 30 天内关闭", "0/1", "仅 eligible_30d 可解释"),
    ("issues.parquet", "is_open_90d", "快照时是否已开启 ≥90 天", "0/1", "仅 eligible_90d 可解释"),
    ("forum_topics.parquet", "id", "论坛主题 ID", "int", "与帖子通过 topic_id 关联"),
    ("forum_topics.parquet", "category", "官方版块", "str", "平台事前分类，不是主题模型结果"),
    ("forum_topics.parquet", "tags", "论坛标签", "list[str]", "2026 年起才成规模"),
    ("forum_topics.parquet", "views / like_count / reply_count", "互动指标", "int", "论坛行为量，不等于需求重要性"),
    ("forum_topics.parquet", "has_accepted_answer", "是否采纳答案", "0/1", "论坛闭环代理指标"),
    ("forum_docs.parquet", "topic_id", "主题 ID", "int", "4,322 个有正文主题"),
    ("forum_docs.parquet", "tok_text", "清洗+分词后的正文", "str", "供 TF-IDF/NMF"),
    ("comments.parquet", "external_human / maintainer", "回复者身份标记", "0/1", "排除机器人与 Issue 作者"),
]


def write_docs(hashes, audits):
    (DOCS / "统一分类字典.md").write_text(tx.taxonomy_markdown(), encoding="utf-8")

    lines = ["# 字段字典（派生分析表）", "",
             "> 原始 JSON 只读；以下字段由 `src/01_build_base.py` 生成。所有时间均为 UTC。", "",
             "| 表 | 字段 | 含义 | 类型 | 口径/限制 |", "|---|---|---|---|---|"]
    for row in FIELD_DICT:
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## 原始字段参考", "",
              "- Issues: `number,title,body,state,state_reason,labels,created_at,closed_at,updated_at,comments,author_association,user`",
              "- Comments: `id,issue_number,body,user,author_association,created_at,updated_at,reactions,performed_via_github_app`",
              "- Forum topics: `id,title,slug,created_at,last_posted_at,posts_count,reply_count,like_count,views,category_id,tags,has_accepted_answer,closed,pinned`",
              "- Forum content: `topic{...},category_slug,posts[{post_number,username,created_at,cooked,accepted_answer}]`",
              "- HN: `objectID,kind,created_at,author,text,title,url,story_id,parent_id,points,num_comments,query`", ""]
    (DOCS / "字段字典.md").write_text("\n".join(lines), encoding="utf-8")

    # 样本登记表
    rows = [
        ("Cursor 论坛主题元数据", "cursor_forum_topics_full.json", 41218, "2023-08-21 ~ 2026-09-13", "全量", "41,218 个主题；不是帖子数"),
        ("Cursor 论坛正文（近期普查）", "cursor_forum_content_census.json", 3934, "2026-07-01 ~ 2026-09-13", "普查", "3,934 主题 / 15,482 帖"),
        ("Cursor 论坛正文（早期分层抽样）", "cursor_forum_content.json", 600, "2026-03 ~ 2026-09", "按热度分层 hot/mid/tail", "600 主题 / 4,441 帖"),
        ("Codex Issues", "codex_issues_full.json", 27756, "2025-04-16 ~ 2026-09-13", "全量", "27,320 有标签；25,519 至少双标签"),
        ("Codex 评论", "codex_comments.json", 115280, "2025-04-16 ~ 2026-09-13", "全量", "18,246 条无法关联到当前 Issue 表"),
        ("Hacker News 讨论", "hn_discussions.json", 80510, "2022-12-31 ~ 2026-09-13", "关键词模糊检索", "仅辅助；关键词过滤后子集另行登记"),
    ]
    lines = ["# 样本登记表", "",
             "> 顶层记录数不等于可分析记录数。论坛主题与帖子是包含关系，不得相加。", "",
             "| 数据集 | 文件 | 顶层记录 | 时间范围 | 采样方式 | 口径与限制 |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    sub = pd.DataFrame(hashes)
    lines += ["", "## 文件校验", "",
              "| 文件 | 字节 | SHA-256 | 与 manifest 一致 |", "|---|---|---|---|"]
    for _, r in sub.iterrows():
        lines.append(f"| {r['file']} | {r['bytes']:,} | `{r['sha256'][:16]}…` | {r['match_manifest']} |")
    lines += ["", "## 派生样本", "",
              f"- 论坛有正文主题：{audits['forum']['forum_body_topics']:,} / {audits['forum']['forum_topics_deduped']:,} "
              f"（{audits['forum']['forum_body_coverage_topics']:.1%}）",
              f"- 论坛正文帖子：{audits['forum']['forum_body_posts']:,}",
              f"- Issue 有标签：{audits['issues']['issues_after_dedupe'] - audits['issues']['no_labels']:,}",
              f"- HN 关键词命中：{audits['hn']['hn_keyword_matched']:,}（辅助对照）", "",
              "## 扩展样本（新增材料，独立分母）", "",
              "| 样本 | 交付包 | 分母 | 时间范围 | 采样/筛选 | 与主分析的关系 |",
              "|---|---|---:|---|---|---|",
              "| 新 RQ1 近期全量首帖 | `RQ1_20261023` | 9,498 | 2026-04 ~ 2026-09 | 9,569 候选 → 9,543 首帖 → 9,518 可用正文 → 9,498 英语 | 不与主 RQ1 的 4,307 条合并 |",
              "| RQ3 竞争风险 | `问题五` | 27,585 | 2025-04-16 ~ 2026-09-13T00:00Z | 27,756 − 17 条截点后创建 − 154 条 reopened；1 条截点后关闭按删失 | 不替换 27,756 全量状态概览 |",
              "| RQ3 敏感性 S1/S3 | `问题五` + 本项目 | 27,739 / 27,756 | 同上，reopened 作删失纳入 | 仅用于口径敏感性，不进入主结果 |",
              "", "详见 `docs/新增材料整合记录.md`。", ""]
    (DOCS / "样本登记表.md").write_text("\n".join(lines), encoding="utf-8")

    # 口径登记表
    lines = ["# 口径登记表（单一事实来源）", "",
             "> 任何报告数字必须能回溯到本表 + 输出表 + 脚本。", "",
             "## 核心数字", "",
             "| 指标 | 值 | 分母/筛选 | 来源 |", "|---|---:|---|---|",
             f"| Issue 原始顶层记录 | {audits['issues']['issues_raw']:,} | 文件全部元素 | codex_issues_full.json |",
             f"| Issue 去重后 | {audits['issues']['issues_after_dedupe']:,} | number 去重 | 01_build_base |",
             f"| 评论原始顶层记录 | {audits['issues']['comments_raw']:,} | 文件全部元素 | codex_comments.json |",
             f"| 可关联评论 | {audits['issues']['comments_matched']:,} | issue_number ∈ Issue 表 | 01_build_base |",
             f"| 无法关联评论 | {audits['issues']['comments_unmatched']:,} | 关联失败 | 01_build_base |",
             f"| 论坛主题 | {audits['forum']['forum_topics_deduped']:,} | id 去重 | cursor_forum_topics_full.json |",
             f"| 论坛有正文主题 | {audits['forum']['forum_body_topics']:,} | 正文文件去重后 | census + sample |",
             f"| 论坛正文帖子 | {audits['forum']['forum_body_posts']:,} | 展开 posts | census + sample |",
             f"| 论坛正文覆盖率 | {audits['forum']['forum_body_coverage_topics']:.1%} | 有正文主题 / 全部主题 | 01_build_base |",
             f"| HN 关键词命中 | {audits['hn']['hn_keyword_matched']:,} | 关键词正则 | hn_discussions.json |",
             "", "## 时间口径", "",
             f"- 观察终点（推定）：{SNAPSHOT_UTC}；来自 Issue 表最新时间字段，非已核实抓取时间。",
             "- 论坛正文：2026-03 起；2026-07 后为普查，之前为按热度分层抽样。",
             "- 跨渠道比较主窗口：2026-01-01 ~ 2026-09-13；正文级比较从 2026-03 起并单独标注。",
             "- 标签：论坛标签 2026 年才成规模；Issue 标签是快照值，可能是事后编辑。", "",
             "## 状态口径", "",
             "- `closed` 只表示关闭，不等于修复；必须区分 completed/duplicate/not_planned/reopened。",
             "- 论坛 `has_accepted_answer` 只表示存在采纳答案，不等于问题真正解决。",
             "- 生存分析以“创建→关闭”为事件，仍开启记录在快照处右删失。", "",
             "## 扩展材料口径（2026-10-23 新增，不并入主分母）", "",
             "| 指标 | 值 | 分母/筛选 | 来源 |", "|---|---:|---|---|",
             "| 新 RQ1 候选主题 | 9,569 | 2026-04~09 论坛首帖候选 | `RQ1_20261023` |",
             "| 新 RQ1 英文模型语料 | 9,498 | 9,518 可用正文 − 20 条非英语/不确定 | 同上 |",
             "| 新 RQ1 验证样本 | 120 + 20 | 120 按月验证 + 20 定向模糊（不混入总体精确率） | 同上 |",
             "| RQ3 竞争风险资格样本 | 27,585 | 27,756 − 17 截点后创建 − 154 reopened | `问题五` |",
             "| RQ3 竞争事件 | 删失 16,532 / completed 7,197 / duplicate 2,059 / not_planned 1,797 | 截点 2026-09-13T00:00:00Z | 同上 |",
             "| RQ3 30 天 completed（固定窗口） | 20.77% | 成熟样本 22,568，其中 4,687 | 同上 |",
             "| RQ3 30 天任意关闭（固定窗口） | 31.92% | 同一分母 | 同上 |",
             "| RQ3 30 天 completed（AJ） | 18.71% | 全 27,585，含删失 | `09_rq3_competing_risk.py` 独立复现 |",
             "| RQ3 30 天任意关闭（AJ） | 28.83% | 全 27,585，含删失 | 同上 |",
             "| reopened 敏感性 | −0.11 pp | 30 天 completed，纳入 154 条 reopened | 同上 |",
             "| 截点敏感性 | −0.01 pp | 00:00:00Z vs 02:20:37Z | 同上 |", "",
             "两套 RQ1 分母（4,307 / 9,498）、两套回复定义（3 日首次外部人类 / 7 日他人回复）、",
             "两套关闭定义（30 日关闭 / 30 日 completed）与两套 RQ3 样本（27,756 / 27,585）均不可互换。", ""]
    (DOCS / "口径登记表.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("== 01 build base ==", flush=True)
    hashes = hash_audit()
    issues, comments, used, ia = build_issues()
    print("issues done", len(issues), flush=True)
    topics, docs, posts, fa = build_forum()
    print("forum done", len(topics), len(docs), len(posts), flush=True)
    ha = build_hn()
    print("hn done", ha["hn_keyword_matched"], flush=True)
    audits = {"issues": ia, "forum": fa, "hn": ha, "hashes": hashes}
    save_json(audits, OUT / "integrated" / "00_data_audit.json")
    write_docs(hashes, audits)
    print(json.dumps(audits, ensure_ascii=False, indent=2, default=str)[:4000])
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
