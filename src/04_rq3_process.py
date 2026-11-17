# -*- coding: utf-8 -*-
"""阶段 04：RQ3 反馈处理过程。

把“处理”拆成三个指标：
1. 首次外部人类回复时间；
2. 首次维护者回复时间；
3. 最终关闭时间。
并分别报告：中位数、P75/P90、无回复比例、固定观察窗比例、分组比较、Kaplan-Meier、
三个时间外预测任务（30 天内关闭 / 3 天内首次外部回复 / 90 天以上长期未关闭）。
所有特征均为预测时点可知；禁止使用最终状态、closed_at、未来评论和最终标签作为特征。
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DERIVED, OUT, FIG, SEED, COLORS, setup_matplotlib
from common.stats import (save_csv, save_json, km, km_value, km_median, logrank,
                          bootstrap_metric, md_table, wilson_ci)

OUTD = OUT / "rq3"
OUTD.mkdir(parents=True, exist_ok=True)
plt = setup_matplotlib()
RNG = np.random.default_rng(SEED)


def fmt_time(df, col, prefix):
    v = df[col].dropna()
    return {f"{prefix}_n": int(len(v)), f"{prefix}_median": float(v.median()) if len(v) else np.nan,
            f"{prefix}_p75": float(v.quantile(.75)) if len(v) else np.nan,
            f"{prefix}_p90": float(v.quantile(.90)) if len(v) else np.nan}


def group_metrics(df, by, eligible_col=None, reply_days=None, rate_col=None, close30=False):
    """按分组输出样本量、比率、Wilson CI 与等待时间分位数。"""
    rows = []
    for g, sub in df.groupby(by, dropna=False):
        row = {"group": str(g), "n": len(sub)}
        if eligible_col:
            e = sub[sub[eligible_col] == 1]
            row["eligible_n"] = len(e)
            if rate_col:
                k = int(e[rate_col].sum()) if rate_col in e else 0
                row["rate"] = k / len(e) if len(e) else np.nan
                lo, hi = wilson_ci(k, len(e))
                row["ci_low"], row["ci_high"] = lo, hi
        if close30:
            e30 = sub[sub["eligible_30d"] == 1]
            k30 = int(e30["closed_within_30d"].sum())
            row["close30_n"] = len(e30); row["close30_observed"] = k30
            row["close30_rate"] = k30 / len(e30) if len(e30) else np.nan
            lo, hi = wilson_ci(k30, len(e30))
            row["close30_ci_low"], row["close30_ci_high"] = lo, hi

            # 90 天仍开启只在已完整观察满 90 天的 Issue 中计算。
            e90 = sub[sub["eligible_90d"] == 1]
            k90 = int(e90["is_open_90d"].sum())
            row["open90_n"] = len(e90); row["open90_observed"] = k90
            row["open90_rate"] = k90 / len(e90) if len(e90) else np.nan
            lo, hi = wilson_ci(k90, len(e90))
            row["open90_ci_low"], row["open90_ci_high"] = lo, hi
        if reply_days:
            v = sub[f"first_{reply_days}_days"].dropna()
            row["replied_n"] = len(v)
            row["replied_rate"] = len(v) / len(sub) if len(sub) else np.nan
            row["median_days"] = float(v.median()) if len(v) else np.nan
            row["p75_days"] = float(v.quantile(.75)) if len(v) else np.nan
            row["p90_days"] = float(v.quantile(.90)) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_km_table(df, dim, horizon=365):
    rows = []
    for g, sub in df.groupby(dim, dropna=False):
        c = km(sub["duration_days"], sub["closed_flag"])
        c = c[c.days <= horizon]
        for _, r in c.iterrows():
            rows.append({"dimension": dim, "group": str(g), "n": len(sub),
                         "days": r.days, "survival": r.survival,
                         "lower95": r.lower95, "upper95": r.upper95,
                         "at_risk": r.at_risk})
    return pd.DataFrame(rows)


def prepare_features(df):
    df = df.copy()
    df["text"] = (df["title"].fillna("") + " " + df["body_snippet"].fillna("").str.slice(0, 6000))
    df["log_body"] = np.log1p(df["body_chars"].clip(lower=0))
    df["log_title"] = np.log1p(df["title_chars"].clip(lower=0))
    df["created_hour"] = df["created_at"].dt.hour
    df["created_weekday"] = df["created_at"].dt.weekday
    df["month_index"] = (df["created_at"].dt.year - 2025) * 12 + df["created_at"].dt.month
    df["author_association"] = df["author_association"].fillna("unknown").astype(str)
    return df


def model_features(text=True):
    # 严格白名单：只使用创建时点可见的文本、长度、时间与作者身份。
    # 不使用入口/环境/类型等由最终标签派生的字段，也不使用标签数量，避免未来信息泄漏。
    num = ["log_body", "log_title", "has_code", "created_hour",
           "created_weekday", "month_index"]
    cat = ["author_association"]
    specs = [("num", StandardScaler(), num),
             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20), cat)]
    if text:
        specs.append(("text", TfidfVectorizer(min_df=5, max_df=0.9, max_features=12000,
                                              ngram_range=(1, 2), sublinear_tf=True), "text"))
    return ColumnTransformer(specs), num, cat


def eval_model(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    top = np.quantile(p, 0.9)
    pred10 = (p >= top).astype(int)
    return {"roc_auc": roc_auc_score(y, p),
            "auc_ci_low": np.nan, "auc_ci_high": np.nan,
            "average_precision": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "top10_threshold": float(top),
            "top10_precision": precision_score(y, pred10, zero_division=0),
            "top10_recall": recall_score(y, pred10, zero_division=0),
            "positive_rate": float(np.mean(y))}


def run_target(a: pd.DataFrame, target: str, eligible: str, horizon: int, label: str):
    m = a[a[eligible] == 1].sort_values("created_at").copy()
    m = m[m[target].notna()]
    boundary = m["created_at"].iloc[int(len(m) * 0.7)]
    gap = pd.Timedelta(days=horizon)
    train = m[m["created_at"] <= boundary - gap].copy()
    test = m[m["created_at"] >= boundary].copy()
    train = prepare_features(train)
    test = prepare_features(test)
    ytr, yte = train[target].astype(int), test[target].astype(int)
    rows, preds = [], pd.DataFrame({"number": test["number"], "created_at": test["created_at"],
                                    "y_true": yte.values, "title": test["title"].values})
    models = {"prior_baseline": ("dummy", None),
              "metadata_logistic": ("meta", None),
              "text_logistic": ("text", None),
              "metadata_hgb": ("hgb", None)}
    fitted = {}
    for name, (kind, _) in models.items():
        if kind == "dummy":
            from sklearn.dummy import DummyClassifier
            model = DummyClassifier(strategy="prior")
        elif kind == "hgb":
            pre, _, _ = model_features(text=False)
            model = Pipeline([("pre", pre), ("model", HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.06, max_depth=4, random_state=SEED))])
        else:
            pre, _, _ = model_features(text=(kind == "text"))
            model = Pipeline([("pre", pre), ("model", LogisticRegression(
                max_iter=2000, C=0.5, random_state=SEED, solver="liblinear"))])
        model.fit(train, ytr)
        p = model.predict_proba(test)[:, 1]
        st = eval_model(yte, p)
        st["model"] = name
        try:
            lo, hi = bootstrap_metric(yte.values, p, roc_auc_score, n_boot=200, seed=SEED)
            st["auc_ci_low"], st["auc_ci_high"] = lo, hi
        except Exception:
            pass
        rows.append(st)
        preds[name] = p
        fitted[name] = model
    # 误差案例（text_logistic）
    preds["error"] = np.where((preds["y_true"] == 1) & (preds["text_logistic"] < .5), "FN",
                              np.where((preds["y_true"] == 0) & (preds["text_logistic"] >= .5), "FP", ""))
    err = preds[preds.error != ""].copy()
    err["abs_error"] = (err["text_logistic"] - err["y_true"]).abs()
    err = err.sort_values("abs_error", ascending=False).head(30)
    err = err.merge(test[["number", "outcome", "labels", "entry_primary", "env_primary",
                          "type_primary", "age_days", "comments"]], on="number", how="left")
    err["target"] = target
    save_csv(err, OUTD / f"12_error_cases_{label}.csv")
    # 系数
    try:
        pipe = fitted["text_logistic"]
        feat = pipe["pre"].get_feature_names_out()
        coef = pd.DataFrame({"feature": feat, "coefficient": pipe["model"].coef_[0]})
        coef = pd.concat([coef.nsmallest(15, "coefficient"), coef.nlargest(15, "coefficient")])
        coef["target"] = target
        save_csv(coef, OUTD / f"13_coefficients_{label}.csv")
    except Exception as e:
        print("coef skip", e)
    feature_inputs = {}
    for model_name, fitted_model in fitted.items():
        if hasattr(fitted_model, "named_steps") and "pre" in fitted_model.named_steps:
            cols = []
            for _, _, source_cols in fitted_model.named_steps["pre"].transformers:
                cols.extend([source_cols] if isinstance(source_cols, str) else list(source_cols))
            feature_inputs[model_name] = sorted(set(cols))
        else:
            feature_inputs[model_name] = []
    info = {"target": target, "definition": label, "eligible_col": eligible,
            "eligible_n": len(m), "train_n": len(train), "test_n": len(test),
            "purged_n": len(m) - len(train) - len(test),
            "train_min": str(train.created_at.min()), "train_max": str(train.created_at.max()),
            "test_min": str(test.created_at.min()), "test_max": str(test.created_at.max()),
            "train_positive_rate": float(ytr.mean()), "test_positive_rate": float(yte.mean()),
            "threshold": 0.5, "seed": SEED,
            "feature_inputs_by_model": feature_inputs,
            "features": "创建时可见：标题/正文、长度、是否有代码、创建时间、作者身份；不含状态/关闭时间/评论/最终标签或标签派生字段"}
    return pd.DataFrame(rows), info, test, preds, fitted


def main():
    a = pd.read_parquet(DERIVED / "issues.parquet")
    c = pd.read_parquet(DERIVED / "comments.parquet")
    a["created_at"] = pd.to_datetime(a["created_at"], utc=True)
    a["closed_at"] = pd.to_datetime(a["closed_at"], utc=True)
    a["remote"] = a["cap_multi"].fillna("").str.contains("remote").map({True: "remote", False: "non-remote"})
    a["env_group"] = a["env_primary"].where(a["env_primary"].isin(["windows", "macos", "linux"]), "other/unknown")
    a["long_running_90d"] = (((a["closed_flag"] == 1) & (a["close_days"] >= 90)) |
                             ((a["closed_flag"] == 0) & (a["age_days"] >= 90))).astype(int)
    a["reply_3d"] = ((a["first_external_days"].notna()) & (a["first_external_days"] <= 3)).astype(int)
    a["maintainer_3d"] = ((a["first_maintainer_days"].notna()) & (a["first_maintainer_days"] <= 3)).astype(int)

    # ---------------- 1. 过程总览 ----------------
    closed = a[a.closed_flag == 1]
    e30 = a[a.eligible_30d == 1]
    e90 = a[a.eligible_90d == 1]
    e3 = a[a.eligible_3d == 1]
    overview = [
        {"metric": "首次外部人类回复", "n": len(a), "observed": int(a.has_external_reply.sum()),
         "ratio": a.has_external_reply.mean(), "median_days": a.first_external_days.median(),
         "p75_days": a.first_external_days.quantile(.75), "p90_days": a.first_external_days.quantile(.9),
         "definition": "排除机器人、Issue 作者本人；截至快照，含删失"},
        {"metric": "3 天内首次外部回复", "n": len(e3), "observed": int(e3.reply_3d.sum()),
         "ratio": e3.reply_3d.mean(), "median_days": e3.loc[e3.reply_3d == 1, "first_external_days"].median(),
         "p75_days": np.nan, "p90_days": np.nan,
         "definition": "仅创建满 3 天的 Issue，固定窗口"},
        {"metric": "首次维护者回复", "n": len(a), "observed": int(a.has_maintainer_reply.sum()),
         "ratio": a.has_maintainer_reply.mean(), "median_days": a.first_maintainer_days.median(),
         "p75_days": a.first_maintainer_days.quantile(.75), "p90_days": a.first_maintainer_days.quantile(.9),
         "definition": "MEMBER/COLLABORATOR/OWNER，排除机器人；不等于官方 SLA"},
        {"metric": "3 天内维护者回复", "n": len(e3), "observed": int(e3.maintainer_3d.sum()),
         "ratio": e3.maintainer_3d.mean(), "median_days": np.nan, "p75_days": np.nan, "p90_days": np.nan,
         "definition": "仅创建满 3 天的 Issue，固定窗口"},
        {"metric": "已关闭", "n": len(a), "observed": len(closed), "ratio": a.closed_flag.mean(),
         "median_days": closed.close_days.median(), "p75_days": closed.close_days.quantile(.75),
         "p90_days": closed.close_days.quantile(.9), "definition": "closed 不等于 fixed"},
        {"metric": "30 天内关闭", "n": len(e30), "observed": int(e30.closed_within_30d.sum()),
         "ratio": e30.closed_within_30d.mean(), "median_days": np.nan, "p75_days": np.nan,
         "p90_days": np.nan, "definition": "仅创建满 30 天的 Issue，固定窗口"},
        {"metric": "仍开启且 ≥90 天", "n": len(e90), "observed": int(e90.is_open_90d.sum()),
         "ratio": e90.is_open_90d.mean(), "median_days": np.nan, "p75_days": np.nan, "p90_days": np.nan,
         "definition": "仅创建满 90 天的 Issue；是快照积压，不是最终命运"},
        {"metric": "90 天以上长期未关闭", "n": len(e90), "observed": int(e90.long_running_90d.sum()),
         "ratio": e90.long_running_90d.mean(), "median_days": np.nan, "p75_days": np.nan, "p90_days": np.nan,
         "definition": "包含 90 天后才关闭与快照仍开启；固定 90 天窗口"},
    ]
    ov = pd.DataFrame(overview)
    ov["ci_low"] = [wilson_ci(int(r.observed), int(r.n))[0] for _, r in ov.iterrows()]
    ov["ci_high"] = [wilson_ci(int(r.observed), int(r.n))[1] for _, r in ov.iterrows()]
    save_csv(ov, OUTD / "01_process_overview.csv")

    # ---------------- 2. 分组描述 ----------------
    group_specs = [("type", "type_primary"), ("entry", "entry_primary"),
                   ("env", "env_group"), ("remote", "remote"), ("month", "created_month")]
    all_groups = []
    for name, col in group_specs:
        g = group_metrics(a, col, eligible_col="eligible_3d", reply_days="external",
                          rate_col="reply_3d", close30=True)
        g["dimension"] = name
        all_groups.append(g)
    save_csv(pd.concat(all_groups), OUTD / "02_group_metrics_all.csv")

    # 主分组表（问题类型 / 入口 / 环境 / 远程 / 月份）
    for name, col in [("type", "type_primary"), ("entry", "entry_primary"),
                      ("env", "env_group"), ("remote", "remote")]:
        g = group_metrics(a, col, eligible_col="eligible_3d", reply_days="external",
                          rate_col="reply_3d", close30=True)
        g = g[g.n >= 30].sort_values("n", ascending=False)
        save_csv(g, OUTD / f"02_group_{name}.csv")
    month = a.groupby("created_month").agg(
        n=("number", "size"), eligible30=("eligible_30d", "sum"),
        closed30=("closed_within_30d", "sum"), external=("has_external_reply", "sum"),
        maintainer=("has_maintainer_reply", "sum"), open90=("is_open_90d", "sum"),
        eligible90=("eligible_90d", "sum")).reset_index()
    month["close30_rate"] = month.closed30 / month.eligible30
    month["open90_rate"] = month.open90 / month.eligible90
    month["external_rate"] = month.external / month.n
    month["maintainer_rate"] = month.maintainer / month.n
    save_csv(month, OUTD / "02_group_month.csv")

    # ---------------- 3. Kaplan-Meier ----------------
    km_rows = []
    for dim in ["type_primary", "entry_primary", "env_group"]:
        sub = build_km_table(a, dim)
        km_rows.append(sub)
    km_all = pd.concat(km_rows)
    save_csv(km_all, OUTD / "07_km_stratified.csv")
    overall_curve = km(a["duration_days"], a["closed_flag"])
    save_csv(overall_curve, OUTD / "07_km_overall.csv")

    # 固定 30 天窗口按季度
    a["cohort_quarter"] = a["created_at"].dt.to_period("Q").astype(str)
    cohort_rows = []
    for q, sub in a.groupby("cohort_quarter"):
        e = sub[sub.eligible_30d == 1]
        if len(e) < 30:
            continue
        cur = km(e["duration_days"], e["closed_flag"])
        cohort_rows.append({"quarter": q, "n": len(sub), "eligible_30d": len(e),
                            "not_closed_30d": km_value(cur, 30),
                            "closed_30d_rate": 1 - km_value(cur, 30),
                            "median_close_days": km_median(cur)})
    cohort = pd.DataFrame(cohort_rows)
    save_csv(cohort, OUTD / "08_cohort_30d.csv")

    # log-rank（只作描述性检验；混合队列仍有组成差异）
    tests = []
    for dim in ["type_primary", "entry_primary", "env_group"]:
        groups = [(g["duration_days"].to_numpy(), g["closed_flag"].to_numpy())
                  for _, g in a.groupby(dim) if len(g) >= 30]
        names = [str(k) for k, g in a.groupby(dim) if len(g) >= 30]
        if len(groups) >= 2:
            chi2, p = logrank(groups)
            tests.append({"dimension": dim, "groups": "; ".join(names),
                          "chi2": chi2, "p_value": p, "n_groups": len(groups),
                          "note": "未控制创建月份与标签构成，仅作探索性比较"})
    save_csv(pd.DataFrame(tests), OUTD / "09_logrank_tests.csv")

    # ---------------- 4. 预测任务 ----------------
    metrics, infos, preds_all = [], [], []
    targets = [("closed_within_30d", "eligible_30d", 30, "30 天内关闭"),
               ("reply_3d", "eligible_3d", 3, "3 天内首次外部回复"),
               ("long_running_90d", "eligible_90d", 90, "90 天以上长期未关闭")]
    for target, elig, horizon, label in targets:
        mm, info, test, preds, fitted = run_target(a, target, elig, horizon, label)
        mm["target"] = target
        metrics.append(mm)
        infos.append(info)
        p = preds.copy(); p["target"] = target
        preds_all.append(p)
        print("model", target, mm[["model", "roc_auc", "average_precision", "brier", "f1"]].to_string(index=False), flush=True)
    metrics = pd.concat(metrics)
    save_csv(metrics, OUTD / "10_model_metrics.csv")
    save_csv(pd.concat(preds_all), OUTD / "11_model_predictions.csv")
    save_json({"targets": infos, "seed": SEED,
               "leakage_control": "特征白名单：标题/正文、长度、是否有代码、创建时间、作者身份；不含 state、state_reason、closed_at、comments、回复字段、最终标签及其派生入口/环境/类型。",
               "validation": "按创建时间 70/30 切分，并在训练/测试之间留出与预测horizon等长的 purge gap。"},
              OUTD / "10_model_protocol.json")

    # ---------------- 5. 图 ----------------
    fig, ax = plt.subplots(figsize=(9, 4.8))
    names = ["3 天内外部回复", "3 天内维护者回复", "30 天内关闭", "90 天以上长期未关闭"]
    row = [ov.loc[ov.metric == m].iloc[0] for m in ["3 天内首次外部回复", "3 天内维护者回复", "30 天内关闭", "90 天以上长期未关闭"]]
    vals = [r.ratio * 100 for r in row]; ns = [int(r.n) for r in row]
    bars = ax.barh(names[::-1], vals[::-1], color=[COLORS[0], COLORS[1], COLORS[2], COLORS[4]][::-1])
    ax.set_xlabel("比例（%）")
    ax.set_title("反馈处理过程：社区响应、维护者响应与关闭是三种不同指标")
    for b, v, n in zip(bars, vals[::-1], ns[::-1]):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.1f}%  (n={n:,})", va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.35)
    fig.tight_layout(); fig.savefig(FIG / "rq3_01_处理过程.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for color, dim in zip(COLORS, ["type_primary", "entry_primary", "env_group"]):
        pass
    for i, (g, sub) in enumerate(a.groupby("type_primary")):
        if len(sub) < 30:
            continue
        cur = km(sub["duration_days"], sub["closed_flag"])
        ax.step(np.r_[0, cur.days], np.r_[1, cur.survival] * 100, where="post",
                label=f"{g} (n={len(sub):,})", color=COLORS[i % len(COLORS)], lw=1.4)
    ax.set(xlim=(0, 240), ylim=(0, 101), xlabel="自创建起天数", ylabel="尚未关闭比例（%）",
           title="按问题类型的 Kaplan-Meier：未控制队列构成，仅作描述")
    ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(FIG / "rq3_02_生存曲线.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    mm = month[month.n >= 30]
    ax.plot(mm.created_month, mm.close30_rate * 100, marker="o", color=COLORS[0], label="30 天内关闭（固定窗口）")
    ax.plot(mm.created_month, mm.external_rate * 100, marker="s", color=COLORS[2], label="有外部回复（截至快照）")
    ax.plot(mm.created_month, mm.open90_rate * 100, marker="^", color=COLORS[4], label="仍开启且 ≥90 天")
    ax.set(ylabel="比例（%）", title="按创建月份：观察期长度不同，近期关闭率不能直接解释为效率下降")
    ax.tick_params(axis="x", rotation=60); ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(FIG / "rq3_03_月度队列.png", bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (target, _, _, label) in zip(axes, targets):
        sub = preds_all[[i for i, p in enumerate(preds_all) if p.target.iloc[0] == target][0]]
        for col, name in [("text_logistic", "文本+元数据"), ("metadata_logistic", "仅元数据"), ("prior_baseline", "先验基线")]:
            if col not in sub:
                continue
            y = sub["y_true"]; p = sub[col]
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            ax.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=.8)
        ax.set_title(label, fontsize=10); ax.set_xlabel("假阳性率"); ax.set_ylabel("真阳性率")
        ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle("时间外测试：三个任务的预测能力都比较有限", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "rq3_04_模型ROC.png", bbox_inches="tight"); plt.close(fig)

    summary = {
        "denominators": {"all_issues": len(a), "closed": int(a.closed_flag.sum()),
                         "eligible_30d": len(e30), "eligible_90d": len(e90), "eligible_3d": len(e3)},
        "outcomes": a["outcome"].value_counts().to_dict(),
        "overview": ov.to_dict("records"),
        "km": {"median_days": km_median(overall_curve),
               "not_closed_30d": km_value(overall_curve, 30),
               "not_closed_90d": km_value(overall_curve, 90),
               "not_closed_180d": km_value(overall_curve, 180)},
        "cohort": cohort.to_dict("records"),
        "logrank": tests,
        "models": infos,
        "notes": [
            "closed 不等于 fixed；39.8% 关闭率不能写成 39.8% 解决率。",
            "维护者回复覆盖率低（约 1.7%），不能解释为官方处理率。",
            "观察终点为 Issue 表最新时间字段推定值，不是已核实抓取时间。",
            "18,246 条评论无法关联到 Issue；快照标题/正文/标签可能是事后版本。",
            "模型目标是快照回顾性定义，不是修复成功或用户满意预测。",
        ],
    }
    save_json(summary, OUTD / "rq3_summary.json")
    print(md_table(ov[["metric", "n", "observed", "ratio", "median_days", "p90_days"]]))
    print(md_table(metrics[["target", "model", "roc_auc", "average_precision", "brier", "f1", "top10_recall"]]))
    print("DONE", OUTD)


if __name__ == "__main__":
    main()
