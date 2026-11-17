# -*- coding: utf-8 -*-
"""阶段 07：验收核对。

检查：原始数据未被改写、关键分母一致、状态分类互斥、RQ 输出与图表齐全、
模型特征白名单不含未来信息、报告数字能回到输出表。
"""
from __future__ import annotations
import sys, json, hashlib
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import (DERIVED, OUT, FIG, REPORTS, DOCS, RAW_DIR, DELIVERY_DIR,
                           RAW_FILES, SNAPSHOT_UTC)
from common import taxonomy as tx
from common.stats import save_json, save_csv

OUTD = OUT / "integrated"
OUTD.mkdir(parents=True, exist_ok=True)

checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})
    print(("PASS " if ok else "FAIL ") + name + " :: " + str(detail), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 1. 原始数据只读：与 manifest 对照
    audit_p = OUTD / "00_data_audit.json"
    if audit_p.exists():
        audit = json.loads(audit_p.read_text(encoding="utf-8"))
        ia, fa = audit["issues"], audit["forum"]
        check("Issue 顶层记录 27,756", ia["issues_raw"] == 27756, ia["issues_raw"])
        check("Issue 去重后不减少", ia["issues_after_dedupe"] == 27756, ia["issues_after_dedupe"])
        check("评论顶层记录 115,280", ia["comments_raw"] == 115280, ia["comments_raw"])
        check("论坛主题 41,218", fa["forum_topics_deduped"] == 41218, fa["forum_topics_deduped"])
        check("正文主题 4,322", fa["forum_body_topics"] == 4322, fa["forum_body_topics"])
        check("正文帖子 18,498", fa["forum_body_posts"] == 18498, fa["forum_body_posts"])
        check("论坛四类版块合计 89.1%", abs(fa["main4_share"] - 0.8911) < 0.002, round(fa["main4_share"], 4))
        check("论坛采纳率 12.3%", abs(fa["accepted_rate"] - 0.1231) < 0.002, round(fa["accepted_rate"], 4))
        check("论坛零回复率 62.1%", abs(fa["zero_reply_rate"] - 0.6212) < 0.002, round(fa["zero_reply_rate"], 4))

    # 2. 派生表结构与状态
    a = pd.read_parquet(DERIVED / "issues.parquet")
    check("outcome 互斥且覆盖全部", a["outcome"].notna().all() and
          set(a["outcome"]) <= {"open", "completed", "duplicate", "not_planned", "reopened", "closed_unspecified"},
          a["outcome"].value_counts().to_dict())
    check("closed 与 open 之和等于总数", int((a.state == "closed").sum() + (a.state == "open").sum()) == len(a), len(a))
    check("close_days 非负（已关闭）", bool((a.loc[a.closed_flag == 1, "close_days"] >= 0).all()), "")
    check("年龄非负", bool((a["age_days"] >= 0).all()), "")
    # 验收阶段现场重算哈希，验证当前文件快照，而非复用上次审计结果。
    manifest_path = DELIVERY_DIR / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    expected = {x["file"]: x.get("sha256") for x in manifest.get("files", [])}
    hash_rows = []
    for filename in RAW_FILES.values():
        path = RAW_DIR / filename
        actual = sha256(path) if path.exists() else None
        expected_hash = expected.get(filename)
        hash_rows.append({"file": filename, "bytes": path.stat().st_size if path.exists() else None,
                          "actual_sha256": actual, "manifest_sha256": expected_hash,
                          "match": bool(actual and expected_hash and actual == expected_hash)})
    (OUTD / "raw_hash_verification.json").write_text(
        json.dumps({"files": hash_rows, "all_match": bool(hash_rows) and all(x["match"] for x in hash_rows)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    check("验收时现场重算原始文件 SHA-256", bool(hash_rows) and all(x["match"] for x in hash_rows),
          f"{sum(x['match'] for x in hash_rows)}/{len(hash_rows)}")

    taxonomy_cases = {
        "Cursor desktop app crashes": "app",
        "Cursor cannot connect to MCP": "unknown",
        "Sidebar layout is broken": "unknown",
        "VS Code extension fails": "extension",
        "Codex CLI freezes": "cli",
        "TUI rendering issue": "tui",
    }
    taxonomy_results = {text: tx.primary(tx.classify_text(text)["entry"], tx.ENTRY_ORDER)
                        for text in taxonomy_cases}
    check("入口分类语义回归样例", taxonomy_results == taxonomy_cases, taxonomy_results)

    group_path = OUT / "rq3" / "02_group_metrics_all.csv"
    if group_path.exists():
        groups = pd.read_csv(group_path)
        a["env_group"] = a["env_primary"].where(a["env_primary"].isin(["windows", "macos", "linux"]),
                                                  "other/unknown")
        a["created_month"] = pd.to_datetime(a["created_at"], utc=True).dt.strftime("%Y-%m")
        a["remote"] = a["cap_multi"].fillna("").astype(str).str.contains("remote").map(
            {True: "remote", False: "non-remote"})
        col_by_dim = {"type": "type_primary", "entry": "entry_primary", "env": "env_group",
                      "remote": "remote", "month": "created_month"}
        groups_ok, errors = True, []
        for _, row in groups.iterrows():
            col = col_by_dim.get(row.dimension)
            if not col:
                groups_ok = False; errors.append(f"未知维度:{row.dimension}"); continue
            sub = a[a[col].astype(str) == str(row.group)]
            e90 = sub[sub.eligible_90d == 1]
            n90, k90 = len(e90), int(e90.is_open_90d.sum())
            rate90 = k90 / n90 if n90 else float("nan")
            same_rate = ((pd.isna(rate90) and pd.isna(row.open90_rate)) or
                         (pd.notna(row.open90_rate) and abs(float(row.open90_rate) - rate90) < 1e-12))
            if (int(row.open90_n) != n90 or int(row.open90_observed) != k90 or not same_rate):
                groups_ok = False
                errors.append(f"{row.dimension}/{row.group}: 输出 {row.open90_observed}/{row.open90_n}/{row.open90_rate}; 应为 {k90}/{n90}/{rate90}")
        check("RQ3 分组 90 天指标分子/分母闭合", groups_ok, errors[:5] or f"{len(groups)} groups")
    f = pd.read_parquet(DERIVED / "forum_topics.parquet")
    check("论坛分类覆盖全部主题", f["type_primary_forum"].notna().all(), len(f))
    check("主题与帖子未混加", len(f) == 41218, len(f))

    # 3. 各阶段输出
    need = [
        OUT / "rq1" / "rq1_summary.json", OUT / "rq1" / "01_theme_profile.csv",
        OUT / "rq2" / "rq2_summary.json", OUT / "rq2" / "04_rules_core.csv",
        OUT / "rq3" / "rq3_summary.json", OUT / "rq3" / "10_model_metrics.csv",
        OUT / "rq4" / "rq4_summary.json", OUT / "rq4" / "02_unified_text_distribution.csv",
        REPORTS / "00_综合报告.md", DOCS / "统一分类字典.md", DOCS / "口径登记表.md",
        DOCS / "字段字典.md", DOCS / "样本登记表.md", DOCS / "复现说明.md",
    ]
    missing = [str(p.relative_to(OUT.parent)) for p in need if not p.exists()]
    check("关键输出文件齐全", not missing, missing or f"{len(need)} files")

    figs = sorted(FIG.glob("*.png"))
    check("核心图存在且非空", len(figs) >= 12 and all(p.stat().st_size > 10000 for p in figs), len(figs))
    # Figure provenance must match files actually generated by the reproducible
    # pipeline, rather than treating copied/old images as freshly regenerated.
    fig_manifest = OUT / "rq1_legacy_extensions" / "figure_manifest.json"
    expected_extensions = {
        "rq1_07_全量版块月度结构.png", "rq1_08_四类反馈长期趋势.png",
        "rq1_09_月度回应与采纳.png", "rq1_10_正文痛点词典线索.png",
    }
    if fig_manifest.exists():
        fm = json.loads(fig_manifest.read_text(encoding="utf-8"))
        ext = {x["file"] for x in fm.get("figures", [])}
        check("组员 RQ1 独立扩展图均由当前流程重建", ext == expected_extensions,
              f"{len(ext)}/{len(expected_extensions)}; missing={sorted(expected_extensions-ext)}")
        check("扩展图清单所列文件均存在且非空",
              all((FIG / n).is_file() and (FIG / n).stat().st_size > 10000 for n in ext), len(ext))
    else:
        check("组员 RQ1 独立扩展图生成清单存在", False, str(fig_manifest))

    # 4. 报告数字可回溯
    if (OUT / "rq3" / "10_model_metrics.csv").exists():
        m = pd.read_csv(OUT / "rq3" / "10_model_metrics.csv")
        check("三个预测任务均有基线对照", m["target"].nunique() == 3 and "prior_baseline" in set(m.model),
              m.groupby("target")["model"].nunique().to_dict())
        check("模型指标含 AUC/AP/Brier/召回", {"roc_auc", "average_precision", "brier", "recall"} <= set(m.columns), "")
    if (OUT / "rq2" / "05_rule_survival_2026.csv").exists():
        s = pd.read_csv(OUT / "rq2" / "05_rule_survival_2026.csv")
        check("RQ2 有时间外规则检验", len(s) > 0, len(s))
    if (OUT / "rq2" / "06_rule_stability.csv").exists():
        s = pd.read_csv(OUT / "rq2" / "06_rule_stability.csv")
        check("RQ2 有季度稳定性", len(s) > 0, len(s))
    entry_path = OUT / "rq4" / "03_entry_distribution_with_coverage.csv"
    if entry_path.exists():
        e = pd.read_csv(entry_path)
        coverage_ok = True
        conditional_ok = True
        for channel, sub in e.groupby("channel"):
            u = sub[sub.code == "unknown"]
            known = sub[sub.code != "unknown"]
            recognized = int(u.recognized_n.iloc[0]) if len(u) else 0
            total = int(sub.denominator.iloc[0])
            coverage_ok &= len(u) == 1 and int(u.n.iloc[0]) + recognized == total
            expected_coverage = recognized / total if total else 0
            coverage_ok &= abs(float(u.recognition_coverage.iloc[0]) - expected_coverage) < 1e-12
            conditional_ok &= bool(((known.share_within_recognized - known.n / recognized).abs() < 1e-12).all()) if recognized else False
        check("RQ4 入口识别覆盖率分母闭合", coverage_ok,
              {c: round(float(v), 4) for c, v in e[e.code == "unknown"].set_index("channel").recognition_coverage.items()})
        check("RQ4 多选入口条件命中率按已识别分母闭合", conditional_ok,
              f"{len(e[e.code != 'unknown'])} category-channel rows")
    if (OUT / "rq1" / "04_sensitivity_refit.csv").exists():
        s = pd.read_csv(OUT / "rq1" / "04_sensitivity_refit.csv")
        check("RQ1 有正文样本敏感性重跑", len(s) == 8, len(s))
    kscan = OUT / "rq1" / "09_k_selection.csv"
    if kscan.exists():
        ks = pd.read_csv(kscan)
        check("RQ1 K 扫描覆盖 6 个候选且保留 K=8", set(ks.k) == {4, 6, 8, 10, 12, 14}
              and int(ks.loc[ks.k == 8, "selected_for_main_analysis"].iloc[0]) == 1,
              f"K={','.join(map(str, ks.k.tolist()))}")
        check("RQ1 重构比例指标明确且范围有效",
              {"relative_reconstruction_error", "reconstruction_share"} <= set(ks.columns)
              and ks.reconstruction_share.between(0, 1).all(),
              "重构诊断不是 PCA 方差解释率")
        summary_path = OUT / "rq1" / "rq1_summary.json"
        if summary_path.exists():
            internal = json.loads(summary_path.read_text(encoding="utf-8"))["tests"]["internal_metrics"]
            check("RQ1 移除易误读的 explained_weight_mean",
                  "explained_weight_mean" not in internal
                  and 0 <= internal.get("reconstruction_share", -1) <= 1,
                  "使用相对重构比例并明确边界")

    ext = OUT / "rq3_validation" / "a5_b5_model_validation.csv"
    if ext.exists():
        ev = pd.read_csv(ext)
        temporal = ev[ev.validation.str.startswith("rolling_fold_")]
        check("RQ3 文本基线与滚动验证结果齐全",
              {"text_nb", "text_logistic_only"} <= set(ev.model)
              and temporal.validation.nunique() >= 3,
              f"rows={len(ev)}, rolling_folds={temporal.validation.nunique()}")
        check("RQ3 扩展指标有效且保留成熟标签 purge horizon",
              ev.roc_auc.between(0, 1).all() and ev.purge_days.gt(0).all()
              and ev.train_positive_rate.between(0, 1).all(),
              "滚动窗只作敏感性验证")
        cal = OUT / "rq3_validation" / "a6_calibration_diagnostic.csv"
        proto_ext = OUT / "rq3_validation" / "extension_protocol.json"
        check("RQ3 A4/A6 诊断边界已记录",
              cal.exists() and proto_ext.exists()
              and "oracle" in json.loads(proto_ext.read_text(encoding="utf-8"))["A6"],
              "A4 条件于已回复；A6 校正使用测试集率，仅回顾性诊断")

    # 5. 模型协议
    proto = OUT / "rq3" / "10_model_protocol.json"
    if proto.exists():
        protocol = json.loads(proto.read_text(encoding="utf-8"))
        forbidden = {"comments", "closed_at", "close_days", "state", "state_reason", "outcome",
                     "has_external_reply", "first_external_days", "has_maintainer_reply",
                     "first_maintainer_days", "entry_primary", "env_primary", "type_primary"}
        model_columns = [c for t in protocol.get("targets", [])
                         for cols in t.get("feature_inputs_by_model", {}).values() for c in cols]
        leaked = sorted(set(model_columns) & forbidden)
        check("模型实际白名单输入无禁止字段", not leaked and bool(model_columns), leaked or f"检查 {len(model_columns)} 项输入")

    # 6. 新增材料扩展核验（08/09/10）
    p8 = OUTD / "new_materials_validation.json"
    if p8.exists():
        v8 = json.loads(p8.read_text(encoding="utf-8"))
        check("新增材料口径核验（08）全部通过", v8["passed"] == v8["total"],
              f"{v8['passed']}/{v8['total']}")
    p9 = OUTD / "09_competing_risk.json"
    if p9.exists():
        v9 = json.loads(p9.read_text(encoding="utf-8"))
        check("竞争风险独立复现（09）全部通过", v9["passed"] == v9["total"],
              f"{v9['passed']}/{v9['total']}")
        check("竞争风险资格样本与主快照可对账",
              v9["reference"]["n"] == 27585 and v9["scenarios"][0]["n"] == 27585,
              f"N={v9['reference']['n']}，全量={27756}")
        check("30 天 completed 严格低于任意关闭",
              all(s["completed_cif_30"] < s["any_closure_cif_30"] for s in v9["scenarios"]),
              "4 个情景均成立")
    p9b = OUTD / "09b_delivery_repro.json"
    if p9b.exists():
        v9b = json.loads(p9b.read_text(encoding="utf-8"))
        check("交付脚本原样重跑（09b）与交付结果一致",
              v9b["passed"] == v9b["total"] and v9b["rerun"]["n"] == 27585,
              f"{v9b['passed']}/{v9b['total']}")
    if (OUTD / "10_rq1_profile.json").exists():
        p10 = json.loads((OUTD / "10_rq1_profile.json").read_text(encoding="utf-8"))
        check("新 RQ1 分母闭合（9,498）",
              abs(sum(t["share"] for t in p10["topics"]) - 1) < 1e-9
              and p10["corpus"]["english_model"] == 9498,
              p10["corpus"])
        a10 = OUTD / "10_rq1_agreement.json"
        if a10.exists():
            v10 = json.loads(a10.read_text(encoding="utf-8"))
            check("新 RQ1 盲审状态已记录（未填完不得报准确率）",
                  v10["status"] in {"pending", "single_rater_only", "evaluated"}, v10["status"])
    manual = OUT.parent / "data" / "manual" / "rq1_2026_review"
    check("新 RQ1 双人盲审工作簿存在且不泄露模型标签",
          all((manual / f).exists() for f in ["rater_A.csv", "rater_B.csv", "key_do_not_show_raters.csv"])
          and "model_topic_id" not in pd.read_csv(manual / "rater_A.csv").columns,
          "两表仅 review_id/title/excerpt/label/notes")

    res = pd.DataFrame(checks)
    save_csv(res, OUTD / "验收核对.csv")
    save_json({"checks": checks, "n_pass": int(res["pass"].sum()), "n_total": len(res),
               "all_pass": bool(res["pass"].all())}, OUTD / "验收核对.json")
    print(f"\n{int(res['pass'].sum())}/{len(res)} checks passed")
    if not res["pass"].all():
        sys.exit(1)


if __name__ == "__main__":
    main()
