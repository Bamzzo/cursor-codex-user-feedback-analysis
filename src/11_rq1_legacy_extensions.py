# -*- coding: utf-8 -*-
"""Rebuild selected, non-duplicative long-horizon RQ1 charts from preserved delivery tables.

These are clearly labeled extensions: the source tables describe all forum topics
(metadata) or the original sampled body corpus, and are not merged into the unified
4,307-post NMF result. The teammate delivery itself remains read-only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.config import ROOT, FIG
from common.config import setup_matplotlib

plt = setup_matplotlib()
SRC = ROOT / "data" / "legacy" / "rq1"
OUT = ROOT / "outputs" / "rq1_legacy_extensions"
OUT.mkdir(parents=True, exist_ok=True)


def read_table(name: str) -> pd.DataFrame:
    path = SRC / name
    if not path.is_file():
        raise FileNotFoundError(f"缺少已交付的 RQ1 源结果表：{path}")
    return pd.read_csv(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    outputs = []

    # All-topic metadata has a longer horizon than the cleaned body-topic model.
    monthly = read_table("T2_月度版块量.csv")
    required_groups = ["缺陷报告", "使用求助", "功能需求", "体验反馈", "观点讨论", "生态与教程", "官方公告/版本", "地区社群"]
    missing = [c for c in ["month", "合计", *required_groups] if c not in monthly]
    if missing:
        raise ValueError(f"月度版块表字段不完整：{missing}")
    monthly = monthly.sort_values("month").reset_index(drop=True)
    x = np.arange(len(monthly))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [0.8, 1.2]})
    complete = monthly.get("是否完整月", pd.Series(True, index=monthly.index)).astype(bool)
    axes[0].bar(x, monthly["合计"], color=np.where(complete, "#2563a6", "#c7d4e5"), width=.82)
    axes[0].set_ylabel("新建主题数")
    axes[0].set_title("Cursor 论坛主题量与版块结构（全量主题元数据，非正文模型样本）", loc="left")
    share_cols = [f"{g}_占比" for g in required_groups]
    if set(share_cols).issubset(monthly.columns):
        shares = [monthly[c].to_numpy() * 100 for c in share_cols]
    else:
        shares = [monthly[g].to_numpy() / monthly["合计"].replace(0, np.nan).to_numpy() * 100 for g in required_groups]
    axes[1].stackplot(x, shares, labels=required_groups,
                      colors=["#c0504d", "#2563a6", "#e2b93b", "#299b89", "#8172b3", "#5b9bd5", "#7f7f7f", "#dc8551"],
                      alpha=.92)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("版块占比（%）")
    axes[1].legend(ncol=4, fontsize=8, loc="upper left", frameon=False)
    step = max(1, len(monthly) // 12)
    axes[1].set_xticks(x[::step], monthly["month"].iloc[::step], rotation=40, ha="right")
    fig.tight_layout()
    p = FIG / "rq1_07_全量版块月度结构.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); outputs.append(p)

    # Long-run movement among the four product-feedback categories.
    core = ["缺陷报告", "使用求助", "功能需求", "体验反馈"]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    palette = ["#c0504d", "#2563a6", "#e2b93b", "#299b89"]
    for col, color in zip(core, palette):
        rate = (monthly[col] / monthly["合计"].replace(0, np.nan)).rolling(3, min_periods=1).mean() * 100
        ax.plot(x, rate, label=col, color=color, lw=2.2)
    ax.set_xticks(x[::step], monthly["month"].iloc[::step], rotation=40, ha="right")
    ax.set_ylabel("占全部主题比例（%，3 月滑动平均）")
    ax.set_title("论坛四类产品反馈主题的长期变化（全量主题元数据）", loc="left")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    p = FIG / "rq1_08_四类反馈长期趋势.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); outputs.append(p)

    # Community response/acceptance are observed outcomes, not causal measures.
    outcomes = read_table("T7_解决率月度.csv").sort_values("month").reset_index(drop=True)
    required = {"month", "accepted_rate", "zero_reply_rate", "topics"}
    if not required.issubset(outcomes.columns):
        raise ValueError(f"月度互动表字段不完整：{sorted(required - set(outcomes.columns))}")
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ox = np.arange(len(outcomes))
    ax.plot(ox, outcomes.accepted_rate * 100, marker="o", ms=3.5, lw=2, label="采纳答案率", color="#299b89")
    ax.plot(ox, outcomes.zero_reply_rate * 100, marker="o", ms=3.5, lw=2, label="零回复率", color="#c0504d")
    ax.set_xticks(ox[::step], outcomes["month"].iloc[::step], rotation=40, ha="right")
    ax.set_ylabel("占当月主题比例（%）")
    ax.set_title("论坛主题的回应与采纳情况随时间变化（不代表产品解决因果）", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = FIG / "rq1_09_月度回应与采纳.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); outputs.append(p)

    # The original pain-point rates have a distinct operational lens, but are
    # lexicon matches and must not be interpreted as mutually exclusive classes.
    pain = read_table("T6_痛点词.csv").sort_values("占正文语料比").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    bars = ax.barh(pain["痛点"], pain["占正文语料比"] * 100, color="#dc8551", height=.72)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in pain["占正文语料比"] * 100], padding=3, fontsize=8)
    ax.set_xlim(0, float(pain["占正文语料比"].max()) * 125)
    ax.set_xlabel("匹配该词典的正文主题比例（%，类别可重叠）")
    ax.set_title("正文中常见产品痛点线索（词典匹配，探索性）", loc="left")
    fig.tight_layout()
    p = FIG / "rq1_10_正文痛点词典线索.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); outputs.append(p)

    manifest = {
        "status": "generated_from_preserved_delivery_tables",
        "source_dir": str(SRC.relative_to(ROOT)),
        "source_tables": [{"file": name, "sha256": sha256(SRC / name)}
                          for name in ["T2_月度版块量.csv", "T7_解决率月度.csv", "T6_痛点词.csv"]],
        "figures": [{"file": p.name, "bytes": p.stat().st_size,
                     "sha256": sha256(p), "source": "outputs from teammate RQ1 delivery; distinct sampling/time scopes"}
                    for p in outputs],
        "scope_note": "These extension figures are not refits from raw text. They reproduce selected non-duplicative teammate analyses from their delivered result tables. Do not combine denominators with the unified RQ1 body NMF corpus.",
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"RQ1 legacy extensions: {len(outputs)} figures rebuilt; manifest={OUT / 'figure_manifest.json'}")


if __name__ == "__main__":
    main()
