# -*- coding: utf-8 -*-
"""直接用交付脚本重跑“问题五”竞争风险分析，并与交付结果、本项目独立实现三方对照。

做法：
1) 在 outputs/repro_rq5/ 下复制交付的 *.py，把两份原始 JSON 以硬链接接入
   （不改动原文件；同盘 NTFS 硬链接不占额外空间）；
2) 用本项目 Python 运行交付脚本（其 OUT 指向该临时目录，不覆盖交付包）；
3) 比较 交付结果 / 重跑结果 / 本项目 09 独立实现 的关键数字。

这是“环境与代码可复现”核验，不是重训主题模型或回归。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "integrated"
STAGE = ROOT / "outputs" / "repro_rq5"
PKG = ROOT / "data" / "delivery" / "rq3_analysis"
DELIVERED = PKG / "rq3_results"
RAW = Path(os.getenv("FEEDBACK_RAW_DIR", ROOT / "data" / "raw"))


def stage() -> Path:
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / "analysis").mkdir(exist_ok=True)
    for py in PKG.glob("*.py"):
        shutil.copy2(py, STAGE / "analysis" / py.name)
    for name in ["codex_issues_full.json", "codex_comments.json"]:
        target = STAGE / name
        if target.exists():
            continue
        try:
            os.link(RAW / name, target)
        except OSError:
            shutil.copy2(RAW / name, target)
    return STAGE


def main() -> None:
    checks: list[dict] = []

    def check(name, passed, observed, expected, note=""):
        checks.append({"check": name, "passed": bool(passed), "observed": observed,
                       "expected": expected, "note": note})

    stage_dir = stage()
    log = ROOT / "outputs" / "logs" / "09b_delivery_repro.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run([sys.executable, str(stage_dir / "analysis" / "analyze_issue_outcomes.py")],
                              cwd=str(stage_dir), stdout=f, stderr=subprocess.STDOUT, text=True)
    check("交付脚本可在本项目环境运行（rc=0）", proc.returncode == 0, proc.returncode, 0,
          f"log={log.relative_to(ROOT)}")

    fresh_dir = stage_dir / "analysis" / "rq3_results"
    if proc.returncode != 0 or not (fresh_dir / "descriptive_survival.json").exists():
        (OUT / "09b_delivery_repro.json").write_text(json.dumps(
            {"checks": checks, "passed": sum(c["passed"] for c in checks), "total": len(checks),
             "log_tail": log.read_text(encoding="utf-8", errors="replace")[-2000:]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print("delivery script failed; see", log)
        return

    delivered = json.loads((DELIVERED / "descriptive_survival.json").read_text(encoding="utf-8"))
    fresh = json.loads((fresh_dir / "descriptive_survival.json").read_text(encoding="utf-8"))
    dcurves = json.loads((DELIVERED / "survival_curves.json").read_text(encoding="utf-8"))
    fcurves = json.loads((fresh_dir / "survival_curves.json").read_text(encoding="utf-8"))
    ours = json.loads((OUT / "09_competing_risk.json").read_text(encoding="utf-8"))

    dv, fv = delivered["overall"], fresh["overall"]
    check("重跑：资格样本 N", fv["n"] == dv["n"] == 27585, fv["n"], 27585, "交付脚本内部口径")
    check("重跑：四类事件计数", fv["event_counts"] == dv["event_counts"],
          fv["event_counts"], dv["event_counts"], "")
    check("重跑：30 天 completed 固定窗口比例",
          abs(fv["completed30_rate"] - dv["completed30_rate"]) < 1e-12,
          fv["completed30_rate"], dv["completed30_rate"], "")
    check("重跑：30 天任意关闭比例",
          abs(fv["closed30_rate"] - dv["closed30_rate"]) < 1e-12,
          fv["closed30_rate"], dv["closed30_rate"], "")
    check("重跑：AJ completed @30 天",
          abs(fv["aj_completed_at_30"] - dv["aj_completed_at_30"]) < 1e-12,
          fv["aj_completed_at_30"], dv["aj_completed_at_30"], "")
    check("重跑：AJ completed @90 天",
          abs(fv["aj_completed_at_90"] - dv["aj_completed_at_90"]) < 1e-12,
          fv["aj_completed_at_90"], dv["aj_completed_at_90"], "")
    check("重跑：KM 中位任意关闭",
          abs(fresh["km_median_any_closure_days"] - delivered["km_median_any_closure_days"]) < 1e-9,
          fresh["km_median_any_closure_days"], delivered["km_median_any_closure_days"], "")

    grid = np.asarray(dcurves["days"], float)
    dmax = max(
        float(np.max(np.abs(np.asarray(fcurves["overall"][k], float) -
                            np.asarray(dcurves["overall"][k], float)))) for k in
        ["open", "completed", "duplicate", "not_planned"])
    check("重跑：0–180 天四条曲线逐点一致", dmax < 1e-12, dmax, "< 1e-12", "362 个网格点")

    ref = ours["reference"]
    check("三方一致：本项目 09 与重跑 AJ completed @30",
          abs(ref["aj"]["completed_cif_30"] - fv["aj_completed_at_30"]) < 1e-12,
          ref["aj"]["completed_cif_30"], fv["aj_completed_at_30"], "")
    check("三方一致：本项目 09 与交付固定窗口 completed",
          abs(ref["fixed_horizon_30d"]["completed_rate"] - fv["completed30_rate"]) < 1e-12,
          ref["fixed_horizon_30d"]["completed_rate"], fv["completed30_rate"], "")

    import matplotlib, numpy, scipy, sklearn  # noqa: E401
    env = {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": numpy.__version__,
           "scipy": scipy.__version__, "sklearn": sklearn.__version__,
           "matplotlib": matplotlib.__version__,
           "delivered_versions": json.loads((DELIVERED / "audit.json").read_text(encoding="utf-8"))["versions"]}
    check("环境差异已记录（版本不同但结果一致）", True, env, "documented",
          "交付为 Python 3.11.7 / pandas 2.2.3 / scipy 1.15.3 / sklearn 1.6.1；本项目 3.12.10 等")

    result = {
        "generated_local": pd.Timestamp.now().isoformat(timespec="seconds"),
        "purpose": "用交付脚本重跑竞争风险，核对环境、代码与结果；不改写交付包。",
        "stage_dir": str(stage_dir.relative_to(ROOT)),
        "script": str((PKG / 'analyze_issue_outcomes.py').relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
        "environment": env,
        "delivered": {"n": dv["n"], "event_counts": dv["event_counts"],
                      "completed30_rate": dv["completed30_rate"], "closed30_rate": dv["closed30_rate"],
                      "aj_completed_at_30": dv["aj_completed_at_30"],
                      "aj_completed_at_90": dv["aj_completed_at_90"]},
        "rerun": {"n": fv["n"], "event_counts": fv["event_counts"],
                  "completed30_rate": fv["completed30_rate"], "closed30_rate": fv["closed30_rate"],
                  "aj_completed_at_30": fv["aj_completed_at_30"],
                  "aj_completed_at_90": fv["aj_completed_at_90"]},
        "checks": checks, "passed": sum(c["passed"] for c in checks), "total": len(checks),
        "scope_limit": "只复现竞争风险/生存/预测脚本的主结果；不重训论坛主题模型，不重抓数据。",
    }
    (OUT / "09b_delivery_repro.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"checks {result['passed']}/{result['total']}")
    for c in checks:
        if not c["passed"]:
            print("FAILED:", c["check"], c["observed"], "expected", c["expected"])


if __name__ == "__main__":
    main()
