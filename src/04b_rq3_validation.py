# -*- coding: utf-8 -*-
"""RQ3 扩展验证：文本基线、滚动时间外验证及独立诊断。

新增结果均为模型稳健性/诊断，不替代 04_rq3_process.py 的主结果。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import DERIVED, OUT, FIG, SEED, setup_matplotlib
from common.stats import save_csv, save_json

OUTD = OUT / "rq3_validation"
OUTD.mkdir(parents=True, exist_ok=True)
plt = setup_matplotlib()

TARGETS = [
    ("closed_within_30d", "eligible_30d", 30, "30 天内关闭"),
    ("reply_3d", "eligible_3d", 3, "3 天内首次外部回复"),
    ("long_running_90d", "eligible_90d", 90, "90 天以上长期未关闭"),
]


def make_text_model(kind: str):
    classifier = MultinomialNB(alpha=0.5) if kind == "text_nb" else LogisticRegression(
        max_iter=2000, C=0.5, solver="liblinear", random_state=SEED)
    return Pipeline([("tfidf", TfidfVectorizer(min_df=5, max_df=0.9,
        max_features=12000, ngram_range=(1, 2), sublinear_tf=True)), ("model", classifier)])


def metrics(y, p):
    return {"roc_auc": roc_auc_score(y, p),
            "average_precision": average_precision_score(y, p),
            "brier": brier_score_loss(y, p),
            "positive_rate": float(np.mean(y)), "n": int(len(y))}


def fit_text(df, target, train_idx, test_idx, kind):
    model = make_text_model(kind)
    x_train = (df.loc[train_idx, "title"].fillna("") + " " +
               df.loc[train_idx, "body_snippet"].fillna("").str.slice(0, 6000))
    x_test = (df.loc[test_idx, "title"].fillna("") + " " +
              df.loc[test_idx, "body_snippet"].fillna("").str.slice(0, 6000))
    model.fit(x_train, df.loc[train_idx, target].astype(int))
    return model.predict_proba(x_test)[:, 1]


def model_validation(a):
    rows = []
    prediction_frames = {}
    for target, eligible, horizon, label in TARGETS:
        m = a[a[eligible].eq(1)].dropna(subset=[target]).sort_values("created_at").copy()
        m = m.reset_index(drop=True)
        if len(m) < 1000:
            continue
        # 单次比较复用主分析 70/30 时间边界及 purge gap。
        boundary = m.loc[int(len(m) * .7), "created_at"]
        tr = np.flatnonzero(m.created_at <= boundary - pd.Timedelta(days=horizon))
        te = np.flatnonzero(m.created_at >= boundary)
        y = m[target].astype(int)
        for kind in ("text_nb", "text_logistic_only"):
            p = fit_text(m, target, tr, te, kind)
            row = metrics(y.iloc[te].to_numpy(), p)
            row.update(target=target, label=label, model=kind, validation="same_70_30_time_split",
                       train_n=len(tr), test_start=str(m.loc[te, "created_at"].min().date()),
                       test_end=str(m.loc[te, "created_at"].max().date()), purge_days=horizon,
                       train_positive_rate=float(y.iloc[tr].mean()))
            rows.append(row)
            if kind == "text_logistic_only":
                prediction_frames[target] = (y.iloc[te].to_numpy(), p)

        # 3 个 expanding-window 验证段；训练标签需先成熟 horizon 天，测试段亦按成熟样本定义。
        eligible_dates = m.created_at
        q = eligible_dates.quantile([.55, .70, .82, .94]).tolist()
        for fold, (cutoff, end) in enumerate(zip(q[:3], q[1:4]), start=1):
            train_idx = np.flatnonzero(m.created_at < pd.Timestamp(cutoff) - pd.Timedelta(days=horizon))
            test_idx = np.flatnonzero((m.created_at >= pd.Timestamp(cutoff)) & (m.created_at < pd.Timestamp(end)))
            if len(train_idx) < 500 or len(test_idx) < 200 or y.iloc[test_idx].nunique() < 2:
                continue
            p = fit_text(m, target, train_idx, test_idx, "text_logistic_only")
            row = metrics(y.iloc[test_idx].to_numpy(), p)
            row.update(target=target, label=label, model="text_logistic_only", validation=f"rolling_fold_{fold}",
                       train_n=len(train_idx), test_start=str(m.loc[test_idx, "created_at"].min().date()),
                       test_end=str(m.loc[test_idx, "created_at"].max().date()), purge_days=horizon,
                       train_positive_rate=float(y.iloc[train_idx].mean()))
            rows.append(row)
    return pd.DataFrame(rows), prediction_frames


def response_time_diagnostic(a):
    # 仅描述观察到首次外部回复的记录；未回复记录属于右删失，不能按零或普通时长纳入箱线图。
    x = a.loc[a.first_external_days.notna(), ["first_external_days", "entry_primary"]].copy()
    x = x[x.first_external_days.ge(0)]
    x["entry_primary"] = x.entry_primary.fillna("unknown")
    q = x.groupby("entry_primary").first_external_days.agg(
        n="size", median="median", p25=lambda s: s.quantile(.25), p75=lambda s: s.quantile(.75),
        p90=lambda s: s.quantile(.90)).reset_index()
    save_csv(q, OUTD / "a4_observed_reply_time_summary.csv")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    grouped = list(x.groupby("entry_primary"))
    groups = [g.first_external_days.to_numpy() for _, g in grouped]
    labels = [f"{k}\n(n={len(g):,})" for k, g in grouped]
    if groups:
        ax.boxplot(groups, tick_labels=labels, showfliers=False, patch_artist=True,
                   boxprops={"facecolor":"#D7E9EC", "edgecolor":"#167D8D"},
                   medianprops={"color":"#C15A35", "linewidth":2})
        ax.set_yscale("symlog", linthresh=1)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Days to first observed external reply (symlog scale)")
        ax.set_title("Observed reply times only — non-replies are right-censored", loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        fig.savefig(FIG / "rq3_a4_observed_reply_boxplot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def calibration_diagnostic(prediction_frames):
    rows = []
    for target, (y, p) in prediction_frames.items():
        y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
        obs, pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        for i, (pi, oi) in enumerate(zip(pred, obs), 1):
            rows.append({"target":target, "bin":i, "mean_predicted":float(pi), "observed_rate":float(oi)})
        # Label-shift sensitivity only: assumes class-conditional score distributions stay fixed.
        train_rate = next((r["train_positive_rate"] for r in []), None)
        # Prefer the empirical train rate recorded in the comparison result file.
        res = pd.read_csv(OUTD / "a5_b5_model_validation.csv")
        row = res[(res.target == target) & (res.validation == "same_70_30_time_split") &
                  (res.model == "text_logistic_only")]
        if len(row):
            pi_train = float(row.iloc[0].train_positive_rate)
            pi_test = float(y.mean())
            if 0 < pi_train < 1 and 0 < pi_test < 1:
                odds = np.clip(p, 1e-6, 1-1e-6) / (1-np.clip(p, 1e-6, 1-1e-6))
                adj_odds = odds * (pi_test/(1-pi_test)) / (pi_train/(1-pi_train))
                pc = adj_odds / (1 + adj_odds)
                rows.append({"target":target, "bin":0, "mean_predicted":float(np.mean(pc)),
                             "observed_rate":float(y.mean()), "adjusted_brier":brier_score_loss(y, pc),
                             "raw_brier":brier_score_loss(y, p), "train_rate":pi_train,
                             "test_rate":pi_test, "adjustment_type":"oracle_test_prevalence",
                             "note":"retrospective upper-bound diagnostic uses held-out test prevalence; not a deployable calibration method; assumes stable class-conditional score distributions"})
        fig, ax = plt.subplots(figsize=(6.8, 6.2))
        ax.plot([0,1],[0,1],"--",color="#84939A",label="Perfect calibration")
        ax.plot(pred, obs, "o-", color="#167D8D", label="Text-only logistic")
        ax.set(xlim=(0,1), ylim=(0,1), xlabel="Mean predicted probability", ylabel="Observed event rate",
               title=f"Calibration diagnostic: {target}")
        ax.grid(alpha=.2); ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(FIG / f"rq3_a6_calibration_{target}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    save_csv(pd.DataFrame(rows), OUTD / "a6_calibration_diagnostic.csv")


def main():
    a = pd.read_parquet(DERIVED / "issues.parquet")
    a["created_at"] = pd.to_datetime(a["created_at"], utc=True)
    a["reply_3d"] = ((a["first_external_days"].notna()) & (a["first_external_days"] <= 3)).astype(int)
    a["long_running_90d"] = (((a["closed_flag"] == 1) & (a["close_days"] >= 90)) |
                             ((a["closed_flag"] == 0) & (a["age_days"] >= 90))).astype(int)
    # A4/A6 are isolated, relatively cheap diagnostics and are computed before
    # the text-only temporal validation models.
    response_time_diagnostic(a)
    validation, preds = model_validation(a)
    save_csv(validation, OUTD / "a5_b5_model_validation.csv")
    calibration_diagnostic(preds)
    save_json({"scope":"diagnostic extensions; do not replace primary RQ3 estimates",
               "A4":"boxplots include observed first external replies only; non-replies are right-censored",
               "A5":"text-only MultinomialNB compared with text-only logistic regression on same temporal split",
               "B5":"three expanding temporal folds; mature-label cohort and horizon purge applied",
               "A6":"quantile-bin calibration and retrospective oracle prior-shift sensitivity; adjustment uses held-out test prevalence and is not deployable; class-conditional score stability is unverified"},
              OUTD / "extension_protocol.json")
    print(f"Saved diagnostic extensions to {OUTD}")


if __name__ == "__main__":
    main()
