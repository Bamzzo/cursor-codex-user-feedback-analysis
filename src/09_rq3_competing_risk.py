# -*- coding: utf-8 -*-
"""RQ3 扩展：独立复现“关闭原因竞争风险”，并补做两层敏感性分析。

与 08_validate_new_materials.py 的分工：
- 08 只读交付结果，核对算术、样本流转、哈希与概率守恒（不重算曲线）；
- 09 用本项目统一底座 data/derived/issues.parquet 独立实现 Aalen–Johansen，
  把交付的竞争风险结果重算一遍，再补做：
  (1) reopened 两种处理（排除 / 作为删失纳入）；
  (2) 两个观察截点（交付的 00:00:00Z / 本项目推定快照 02:20:37Z）。

写入 outputs/integrated/09_*，不改写 outputs/rq3 与交付包。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.config import COLORS, SEED, setup_matplotlib  # noqa: E402
from common.stats import wilson_ci  # noqa: E402

OUT = ROOT / "outputs" / "integrated"
FIG = ROOT / "figures"
DELIVERED = ROOT / "data" / "delivery" / "rq3_results"

CUTOFF_DELIVERED = pd.Timestamp("2026-09-13T00:00:00Z")
CUTOFF_PROJECT = pd.Timestamp("2026-09-13T02:20:37Z")
CAUSES = {"completed": 1, "duplicate": 2, "not_planned": 3}
CAUSE_NAMES = ["completed", "duplicate", "not_planned"]


def as_float(x) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def aj_full(durations: np.ndarray, events: np.ndarray):
    """Aalen–Johansen 估计的完整逐步结果（事件竞争，右删失）。

    返回 unique 时间点、事件前生存概率 S(t-)、以及三类累计发生率。
    同一时点先处理事件、后处理删失，与交付脚本的约定一致。
    """
    order = np.argsort(durations, kind="mergesort")
    d = durations[order]
    e = events[order]
    n = len(d)
    uniq, first, cnt = np.unique(d, return_index=True, return_counts=True)
    at_risk = n - first
    gid = np.searchsorted(uniq, d, side="left")
    counts = np.bincount(gid * 4 + e, minlength=len(uniq) * 4).reshape(-1, 4)
    cause = counts[:, 1:4].astype(float)
    dtot = cause.sum(axis=1)
    haz = np.divide(dtot, at_risk, out=np.zeros_like(dtot), where=at_risk > 0)
    surv = np.cumprod(1 - haz)
    s_prev = np.r_[1.0, surv[:-1]]
    increments = cause / at_risk[:, None]
    cif = np.cumsum(s_prev[:, None] * increments, axis=0)
    return uniq, surv, cif


def at_horizon(uniq, surv, cif, h: float):
    idx = np.searchsorted(uniq, h, side="right") - 1
    if idx < 0:
        return 1.0, np.zeros(3)
    return float(surv[idx]), cif[idx].copy()


def build_sample(issues: pd.DataFrame, cutoff: pd.Timestamp, drop_reopened: bool):
    created = issues["created_at"]
    closed = issues["closed_at"]
    keep = created < cutoff
    if drop_reopened:
        keep &= issues["state_reason"].ne("reopened")
    df = issues.loc[keep].copy()
    observed = df["state"].eq("closed") & df["closed_at"].notna() & (df["closed_at"] <= cutoff)
    df["event"] = np.where(observed, df["state_reason"].map(CAUSES).fillna(0), 0).astype(int)
    dur = (df["closed_at"] - df["created_at"]).dt.total_seconds() / 86400
    age = (cutoff - df["created_at"]).dt.total_seconds() / 86400
    df["age"] = age
    df["duration"] = np.where(observed, dur, age)
    df = df[df["duration"] > 0]
    return df


def summarize(df: pd.DataFrame, horizons=(30, 90)):
    dur = df["duration"].to_numpy(float)
    ev = df["event"].to_numpy(int)
    uniq, surv, cif = aj_full(dur, ev)
    out = {"n": int(len(df)), "event_counts": pd.Series(ev).value_counts().sort_index().to_dict()}
    for h in horizons:
        s, c = at_horizon(uniq, surv, cif, h)
        out[f"open_cif_{h}"] = s
        for k, name in enumerate(CAUSE_NAMES):
            out[f"{name}_cif_{h}"] = float(c[k])
    # 任意关闭的 KM（忽略原因，用于核对交付的 km_open_at_30/90）
    order = np.argsort(dur, kind="mergesort")
    ds, es = dur[order], ev[order]
    u_all, first_all = np.unique(ds, return_index=True)
    at_risk = len(ds) - first_all
    gid = np.searchsorted(u_all, ds, side="left")
    d_any = np.bincount(gid, weights=(es > 0).astype(float), minlength=len(u_all))
    km_surv = np.cumprod(1 - np.divide(d_any, at_risk, out=np.zeros_like(d_any),
                                       where=at_risk > 0))

    def km_at(h):
        i = np.searchsorted(u_all, h, side="right") - 1
        return float(km_surv[i]) if i >= 0 else 1.0

    out["km_open_30"] = km_at(30)
    out["km_open_90"] = km_at(90)
    below = np.where(km_surv <= 0.5)[0]
    out["km_median_any_closure"] = float(u_all[below[0]]) if len(below) else None
    out["open_at_least_90d_n"] = int(((ev == 0) & (dur >= 90)).sum())
    return out, (uniq, surv, cif)


def fixed_horizon_rates(df: pd.DataFrame, horizon: int = 30):
    """固定 30 天窗口的经验比例，与交付口径一致。

    分母 = 创建时间距截点已满 30 天的样本（age，不是事件时长）；
    分子 = 该窗口内发生的 completed / 任意关闭事件。
    与交付脚本 group_summary 的 `mature = frame.loc[frame.age_days >= 30]` 一致。
    """
    event = df["event"]
    dur = df["duration"]
    mature = df["age"] >= horizon
    completed = int(((event == 1) & (dur <= horizon) & mature).sum())
    any_closed = int(((event > 0) & (dur <= horizon) & mature).sum())
    n = int(mature.sum())
    lo, hi = wilson_ci(completed, n)
    return {
        "mature_n": n, "completed_n": completed, "completed_rate": completed / n,
        "completed_wilson95": [float(lo), float(hi)],
        "closed_n": any_closed, "closed_rate": any_closed / n,
        "gap_pp": (any_closed - completed) / n * 100,
    }


def bootstrap_ci(df: pd.DataFrame, h: int, n_boot: int = 250):
    rng = np.random.default_rng(SEED)
    dur = df["duration"].to_numpy(float)
    ev = df["event"].to_numpy(int)
    n = len(df)
    vals = np.empty((n_boot, 4))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        uniq, surv, cif = aj_full(dur[idx], ev[idx])
        s, c = at_horizon(uniq, surv, cif, h)
        vals[b] = np.r_[s, c]
    return {"open": [float(np.percentile(vals[:, 0], 2.5)), float(np.percentile(vals[:, 0], 97.5))],
            **{name: [float(np.percentile(vals[:, i + 1], 2.5)), float(np.percentile(vals[:, i + 1], 97.5))]
               for i, name in enumerate(CAUSE_NAMES)}}


def main() -> None:
    issues = pd.read_parquet(ROOT / "data" / "derived" / "issues.parquet")
    issues["created_at"] = pd.to_datetime(issues["created_at"], utc=True)
    issues["closed_at"] = pd.to_datetime(issues["closed_at"], utc=True)

    delivered = json.loads((DELIVERED / "descriptive_survival.json").read_text(encoding="utf-8"))
    delivered_curves = json.loads((DELIVERED / "survival_curves.json").read_text(encoding="utf-8"))

    checks: list[dict] = []

    def check(name, passed, observed, expected, note):
        checks.append({"check": name, "passed": bool(passed), "observed": observed,
                       "expected": expected, "note": note})

    # ---------- 1) 基准复现：交付截点 + 排除 reopened ----------
    ref = build_sample(issues, CUTOFF_DELIVERED, drop_reopened=True)
    ref_sum, (uniq, surv, cif) = summarize(ref)
    ref_fixed = fixed_horizon_rates(ref, 30)

    ov = delivered["overall"]
    check("独立复现：资格样本 N", ref_sum["n"] == ov["n"], ref_sum["n"], ov["n"],
          "27,756 - 17 截点后创建 - 154 reopened")
    expected_counts = {int(k): int(v) for k, v in ov["event_counts"].items()}
    check("独立复现：四类事件计数", ref_sum["event_counts"] == expected_counts,
          ref_sum["event_counts"], expected_counts, "0=删失, 1=completed, 2=duplicate, 3=not_planned")
    check("独立复现：30 天 completed 固定窗口比例",
          abs(ref_fixed["completed_rate"] - ov["completed30_rate"]) < 1e-12,
          round(ref_fixed["completed_rate"], 12), round(ov["completed30_rate"], 12),
          f"mature30 n={ref_fixed['mature_n']}")
    check("独立复现：30 天任意关闭固定窗口比例",
          abs(ref_fixed["closed_rate"] - ov["closed30_rate"]) < 1e-12,
          round(ref_fixed["closed_rate"], 12), round(ov["closed30_rate"], 12), "固定 30 天窗口的经验比例")
    check("独立复现：30 天 completed 与任意关闭不是同一指标",
          ref_fixed["gap_pp"] > 5,
          round(ref_fixed["gap_pp"], 2), "> 5 个百分点",
          "交付口径的 30 天 completed 20.8% vs 任意关闭 31.9%")
    check("独立复现：KM 任意关闭 30 天开放概率",
          abs(ref_sum["km_open_30"] - ov["km_open_at_30"]) < 1e-9,
          round(ref_sum["km_open_30"], 12), round(ov["km_open_at_30"], 12), "")
    check("独立复现：KM 任意关闭 90 天开放概率",
          abs(ref_sum["km_open_90"] - ov["km_open_at_90"]) < 1e-9,
          round(ref_sum["km_open_90"], 12), round(ov["km_open_at_90"], 12), "")
    check("独立复现：AJ completed 累积发生率 @30 天",
          abs(ref_sum["completed_cif_30"] - ov["aj_completed_at_30"]) < 1e-9,
          round(ref_sum["completed_cif_30"], 12), round(ov["aj_completed_at_30"], 12), "")
    check("独立复现：AJ completed 累积发生率 @90 天",
          abs(ref_sum["completed_cif_90"] - ov["aj_completed_at_90"]) < 1e-9,
          round(ref_sum["completed_cif_90"], 12), round(ov["aj_completed_at_90"], 12), "")
    check("独立复现：KM 任意关闭中位时间",
          abs(ref_sum["km_median_any_closure"] - delivered["km_median_any_closure_days"]) < 1e-9,
          round(ref_sum["km_median_any_closure"], 6), round(delivered["km_median_any_closure_days"], 6), "")
    check("独立复现：90 天后仍开放的样本数",
          ref_sum["open_at_least_90d_n"] == ov["open_at_least_90d_n"],
          ref_sum["open_at_least_90d_n"], ov["open_at_least_90d_n"], "竞争风险资格样本内")

    # 与交付的 0.5 天网格逐点比对
    grid = np.asarray(delivered_curves["days"], float)
    ours_open, ours_comp = [], []
    for h in grid:
        s, c = at_horizon(uniq, surv, cif, h)
        ours_open.append(s)
        ours_comp.append(c[0])
    ours_open = np.asarray(ours_open)
    ours_comp = np.asarray(ours_comp)
    d_open = np.abs(ours_open - np.asarray(delivered_curves["overall"]["open"], float)).max()
    d_comp = np.abs(ours_comp - np.asarray(delivered_curves["overall"]["completed"], float)).max()
    check("独立复现：0–180 天开放曲线逐点一致", d_open < 1e-9, float(d_open), "< 1e-9",
          "0.5 天网格，362 个点")
    check("独立复现：0–180 天 completed 曲线逐点一致", d_comp < 1e-9, float(d_comp), "< 1e-9", "")
    conservation = abs(ref_sum["open_cif_30"] + sum(ref_sum[f"{k}_cif_30"] for k in CAUSE_NAMES) - 1)
    check("曲线概率守恒（复现值）", conservation < 1e-9, float(conservation), "< 1e-9",
          "开放概率 + 三类累计发生率 = 1")

    # ---------- 2) 敏感性：reopened 处理 × 观察截点 ----------
    scenarios = {
        "S0_交付口径": (CUTOFF_DELIVERED, True, "截点 00:00:00Z；排除 154 条 reopened（交付基准）"),
        "S1_纳入reopened": (CUTOFF_DELIVERED, False, "截点 00:00:00Z；154 条 reopened 作为右删失纳入"),
        "S2_项目截点": (CUTOFF_PROJECT, True, "截点 02:20:37Z；排除 reopened（本项目统一快照）"),
        "S3_项目截点含reopened": (CUTOFF_PROJECT, False, "截点 02:20:37Z；reopened 作为删失纳入"),
    }
    rows, boot_cache = [], {}
    for name, (cut, drop, note) in scenarios.items():
        df = build_sample(issues, cut, drop)
        s, _ = summarize(df)
        f30 = fixed_horizon_rates(df, 30)
        boot = bootstrap_ci(df, 30)
        boot_cache[name] = boot
        rows.append({
            "scenario": name, "cutoff_utc": cut.isoformat(), "drop_reopened": drop,
            "n": s["n"], "censored": s["event_counts"].get(0, 0),
            "completed_n": s["event_counts"].get(1, 0),
            "duplicate_n": s["event_counts"].get(2, 0),
            "not_planned_n": s["event_counts"].get(3, 0),
            "completed_cif_30": s["completed_cif_30"], "completed_cif_90": s["completed_cif_90"],
            "any_closure_cif_30": 1 - s["open_cif_30"], "any_closure_cif_90": 1 - s["open_cif_90"],
            "duplicate_cif_30": s["duplicate_cif_30"], "not_planned_cif_30": s["not_planned_cif_30"],
            "fixed_completed30_rate": f30["completed_rate"],
            "fixed_closed30_rate": f30["closed_rate"],
            "completed_cif30_boot95": [boot["completed"][0], boot["completed"][1]],
            "note": note,
        })
    scen = pd.DataFrame(rows)
    ref_row = scen[scen.scenario == "S0_交付口径"].iloc[0]
    scen["delta_completed_cif_30_pp"] = (scen["completed_cif_30"] - ref_row["completed_cif_30"]) * 100
    scen["delta_completed_cif_90_pp"] = (scen["completed_cif_90"] - ref_row["completed_cif_90"]) * 100
    scen.to_csv(OUT / "09_scenario_sensitivity.csv", index=False, encoding="utf-8-sig")

    d31 = abs(scen.loc[scen.scenario == "S1_纳入reopened", "completed_cif_30"].iloc[0]
              - ref_row["completed_cif_30"]) * 100
    d32 = abs(scen.loc[scen.scenario == "S2_项目截点", "completed_cif_30"].iloc[0]
              - ref_row["completed_cif_30"]) * 100
    d33 = abs(scen.loc[scen.scenario == "S3_项目截点含reopened", "completed_cif_30"].iloc[0]
              - ref_row["completed_cif_30"]) * 100
    check("敏感性：reopened 处理对 30 天 completed 的影响 < 1 个百分点", max(d31, d33) < 1.0,
          {"纳入reopened": round(d31, 3), "项目截点含reopened": round(d33, 3)}, "< 1 pp",
          "仅 154/27,756 条，点估计变化小；区间仍需同时报告")
    check("敏感性：观察截点对 30 天 completed 的影响 < 0.2 个百分点", max(d32, d33) < 0.2,
          {"项目截点": round(d32, 3)}, "< 0.2 pp", "02:20:37Z 与 00:00:00Z 仅差 2 小时 20 分")
    check("敏感性：所有情景 completed 均低于任意关闭", bool(
        (scen["completed_cif_30"] < scen["any_closure_cif_30"]).all()),
          scen[["scenario", "completed_cif_30", "any_closure_cif_30"]].round(4).to_dict("records"),
          "completed < any closure", "竞争结局不可合并")

    # ---------- 3) 曲线导出与图 ----------
    grid180 = np.arange(0, 180.5, 0.5)
    curve_rows = []
    for h in grid180:
        s, c = at_horizon(uniq, surv, cif, h)
        curve_rows.append({"days": h, "open": s, "completed": c[0],
                           "duplicate": c[1], "not_planned": c[2]})
    pd.DataFrame(curve_rows).to_csv(OUT / "09_competing_risk_curves.csv", index=False, encoding="utf-8-sig")

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    plot_open = np.asarray([r["open"] for r in curve_rows])
    ax.step(grid180, plot_open * 100, where="post", color=COLORS[0], label="仍未关闭")
    for k, color, label in [("completed", COLORS[2], "completed"),
                            ("duplicate", COLORS[4], "duplicate"),
                            ("not_planned", COLORS[5], "not_planned")]:
        vals = [r[k] for r in curve_rows]
        ax.step(grid180, np.asarray(vals) * 100, where="post", color=color, label=f"{label}（累积发生）")
    ax.set_xlabel("自创建起的天数"); ax.set_ylabel("概率（%）")
    ax.set_title("竞争风险：关闭不等于完成（N=27,585）", fontsize=10)
    ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=.15)
    ax.set_xlim(0, 180); ax.set_ylim(0, 100)

    ax = axes[1]
    x = np.arange(len(scen))
    ax.bar(x - .2, scen["completed_cif_30"] * 100, width=.4, color=COLORS[2], label="30 天 completed（AJ）")
    ax.bar(x + .2, scen["any_closure_cif_30"] * 100, width=.4, color=COLORS[0], label="30 天任意关闭")
    ax.set_xticks(x)
    ax.set_xticklabels(["交付口径", "纳入\nreopened", "项目\n截点", "项目截点\n含 reopened"], fontsize=8)
    ax.set_ylabel("累积发生率（%）")
    ax.set_title("敏感性：口径变化约 0–1 个百分点", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.15)
    fig.suptitle("RQ3 扩展：三种关闭原因作为互斥竞争结局", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "rq3_05_竞争风险.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 4) 写 JSON ----------
    result = {
        "generated_local": pd.Timestamp.now().isoformat(timespec="seconds"),
        "purpose": "独立复现交付的竞争风险结果，并补做 reopened/截点敏感性；不覆盖主分析。",
        "seed": SEED, "bootstrap_n": 250,
        "cutoff_delivered": CUTOFF_DELIVERED.isoformat(),
        "cutoff_project": CUTOFF_PROJECT.isoformat(),
        "reference": {"n": ref_sum["n"], "event_counts": ref_sum["event_counts"],
                      "fixed_horizon_30d": ref_fixed,
                      "aj": {k: v for k, v in ref_sum.items() if k.endswith(("_cif_30", "_cif_90"))},
                      "km": {"open_30": ref_sum["km_open_30"], "open_90": ref_sum["km_open_90"],
                             "median_any_closure": ref_sum["km_median_any_closure"],
                             "open_at_least_90d_n": ref_sum["open_at_least_90d_n"]}},
        "bootstrap95_reference": bootstrap_ci(ref, 30),
        "scenarios": json.loads(scen.to_json(orient="records", force_ascii=False)),
        "checks": checks,
        "passed": sum(c["passed"] for c in checks), "total": len(checks),
        "boundaries": [
            "只重算交付口径下的竞争风险估计，不重训主题模型、不重跑回归与文本预测。",
            "两个截点相差约 2 小时 20 分，差异很小；但报告必须固定写清用的是哪一个。",
            "reopened 的早先关闭时间不可见，纳入时只能按当前 open 状态右删失，属于保守处理。",
            "Aalen–Johansen 为未调整估计；creation cohort 构成变化会影响曲线，不能解释为处理效率。",
        ],
    }
    (OUT / "09_competing_risk.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"checks {result['passed']}/{result['total']}")
    print(scen[["scenario", "n", "completed_cif_30", "any_closure_cif_30",
                "delta_completed_cif_30_pp"]].to_string(index=False))
    for c in checks:
        if not c["passed"]:
            print("FAILED:", c["check"], c["observed"], "expected", c["expected"])


if __name__ == "__main__":
    main()
