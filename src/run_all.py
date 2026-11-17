# -*- coding: utf-8 -*-
r"""一键复现入口。

用法：
    python src\run_all.py
    python src\run_all.py --from 02 --to 05
"""
from __future__ import annotations
import argparse, subprocess, sys, time, json, platform, importlib.metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LOGS = ROOT / "outputs" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

STAGES = [
    ("01_build_base.py", "公共数据底座：清洗、统一分类、回复指标、审计"),
    ("02_rq1_topics.py", "RQ1 主题建模、主题验证抽样、覆盖敏感性与互动关系"),
    ("02b_rq1_k_selection.py", "RQ1 候选主题数多指标扫描（诊断，不自动选 K）"),
    ("03_rq2_rules.py", "RQ2 标签组合、规则 P/R/覆盖、季度稳定性与治理"),
    ("04_rq3_process.py", "RQ3 首次回复/维护者回复/关闭、KM 与三个预测任务"),
    ("04b_rq3_validation.py", "RQ3 文本朴素贝叶斯、滚动时间验证与评估敏感性"),
    ("05_rq4_compare.py", "RQ4 统一口径下的跨渠道比较"),
    ("08_validate_new_materials.py", "新增材料口径/哈希/算术核验（只读交付包）"),
    ("09_rq3_competing_risk.py", "竞争风险独立复现与 reopened/截点敏感性"),
    ("09b_repro_delivery_script.py", "用交付脚本原样重跑竞争风险（隔离目录）"),
    ("10_rq1_review_agreement.py", "新 RQ1 双人盲审工作簿与一致性评分"),
    ("11_rq1_legacy_extensions.py", "重建组员 RQ1 长期版块/互动/痛点扩展图（口径独立）"),
    ("12_validate_issue_ml_extension.py", "Issue 类型分流与 completed 预测扩展核验（只读交付结果）"),
    ("06_build_reports.py", "综合报告、研究简报与汇报大纲（读取扩展核验结果）"),
    ("07_verify_outputs.py", "验收核对与输出清单"),
]

OPTIONAL_STAGES = {
    "08_validate_new_materials.py",
    "09_rq3_competing_risk.py",
    "09b_repro_delivery_script.py",
    "10_rq1_review_agreement.py",
    "11_rq1_legacy_extensions.py",
    "12_validate_issue_ml_extension.py",
}


def env_info():
    pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pyarrow"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = importlib.metadata.version(p)
        except Exception:
            versions[p] = "missing"
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": versions,
            "seed": 42, "timezone": "UTC for all timestamps"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="01")
    ap.add_argument("--to", dest="end", default="12")
    args = ap.parse_args()
    manifest = {"environment": env_info(), "stages": []}
    for script, desc in STAGES:
        stem = Path(script).stem
        num = stem[:2]
        if not (args.start <= num <= args.end):
            continue
        if script in OPTIONAL_STAGES:
            prerequisites = {
                "08_validate_new_materials.py": ROOT / "data" / "derived" / "issues.parquet",
                "09_rq3_competing_risk.py": ROOT / "data" / "derived" / "issues.parquet",
                "09b_repro_delivery_script.py": ROOT / "data" / "raw" / "codex_issues_full.json",
                "10_rq1_review_agreement.py": ROOT / "data" / "manual" / "rq1_2026_review",
                "11_rq1_legacy_extensions.py": ROOT / "data" / "legacy" / "rq1",
                "12_validate_issue_ml_extension.py": ROOT / "issue_ml",
            }
            required = prerequisites.get(script)
            if required is not None and not required.exists():
                print(f"==> {script} skipped: prerequisite not found ({required})", flush=True)
                manifest.setdefault("skipped", []).append({"script": script, "reason": f"missing {required.relative_to(ROOT)}"})
                continue
        t0 = time.time()
        log = LOGS / f"{stem}.log"
        print(f"\n=== {stem} {desc} ===", flush=True)
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.run([sys.executable, str(SRC / script)], cwd=str(ROOT),
                               stdout=f, stderr=subprocess.STDOUT, text=True)
        dt = time.time() - t0
        manifest["stages"].append({"stage": stem.split("_")[0], "script": script, "description": desc,
                                   "returncode": p.returncode, "seconds": round(dt, 1),
                                   "log": str(log.relative_to(ROOT))})
        print(f"==> {script} rc={p.returncode} {dt:.0f}s log={log}", flush=True)
        if p.returncode != 0:
            break
    (ROOT / "outputs" / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
