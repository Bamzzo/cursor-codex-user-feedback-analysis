# -*- coding: utf-8 -*-
"""Validate the delivered Issue ML extension without retraining models."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "issue_ml" / "artifacts" / "full_v1"
COMP = ROOT / "issue_ml" / "artifacts" / "completion_rf_v1"
OUT = ROOT / "outputs" / "integrated"
OUT.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    checks: list[dict] = []

    def check(name: str, passed: bool, observed, expected, note: str):
        checks.append({"check": name, "passed": bool(passed), "observed": observed,
                       "expected": expected, "note": note})

    metrics = load_json(FULL / "metrics.json")
    refactor = load_json(FULL / "refactor_verification.json")
    test_metrics = pd.read_csv(COMP / "test_metrics.csv")
    verification = load_json(COMP / "verification.json")

    lr = metrics["logistic_regression"]
    check("classification split sizes", metrics["train_n"] == 21343 and metrics["test_n"] == 5287,
          {"train_n": metrics["train_n"], "test_n": metrics["test_n"]},
          {"train_n": 21343, "test_n": 5287}, "Time-ordered split from the delivered artifact.")
    check("classification metrics", np.isclose(lr["accuracy"], 0.9612256478, atol=1e-9)
          and np.isclose(lr["macro_f1"], 0.9194264880, atol=1e-9),
          {"accuracy": lr["accuracy"], "macro_f1": lr["macro_f1"]},
          {"accuracy": 0.9612256478, "macro_f1": 0.9194264880},
          "TF-IDF + logistic regression; labels are existing maintainer tags.")
    check("classification ranking metrics", np.isclose(lr["roc_auc"], 0.9836587462, atol=1e-9)
          and np.isclose(lr["average_precision"], 0.9419062666, atol=1e-9),
          {"roc_auc": lr["roc_auc"], "average_precision": lr["average_precision"]},
          {"roc_auc": 0.9836587462, "average_precision": 0.9419062666},
          "Ranking metrics are reported alongside class metrics because the labels are imbalanced.")
    check("classification artifact verification", bool(refactor.get("metrics_match"))
          and bool(refactor.get("saved_model_verified")), refactor,
          "metrics_match=true and saved_model_verified=true",
          "Saved predictions and smoke test agree with the delivered model.")

    selected = test_metrics[(test_metrics["h"] == 24) & (test_metrics["d"] == 7)
                            & (test_metrics["variant"] == "structure_early")
                            & (test_metrics["model"] == "random_forest")]
    row = selected.iloc[0] if len(selected) == 1 else None
    check("completion selected row exists", row is not None, int(len(selected)), 1,
          "24-hour observation window and 7-day completed outcome.")
    if row is not None:
        check("completion test size and positives", int(row["n"]) == 1395 and int(row["positives"]) == 22,
              {"n": int(row["n"]), "positives": int(row["positives"])},
              {"n": 1395, "positives": 22}, "Positive class is rare in the temporal test set.")
        check("completion metrics", np.isclose(row["roc_auc"], 0.5269813944, atol=1e-9)
              and np.isclose(row["average_precision"], 0.0997498161, atol=1e-9)
              and np.isclose(row["brier"], 0.0211517971, atol=1e-9),
              {"roc_auc": row["roc_auc"], "average_precision": row["average_precision"],
               "brier": row["brier"]},
              {"roc_auc": 0.5269813944, "average_precision": 0.0997498161, "brier": 0.0211517971},
              "The selected model provides weak ranking signal and is not an automatic decision tool.")
    early_30 = test_metrics[(test_metrics["h"] == 24) & (test_metrics["d"] == 30)
                            & (test_metrics["variant"] == "early_only")
                            & (test_metrics["model"] == "random_forest")]
    check("30-day early interaction comparison", len(early_30) == 1,
          int(len(early_30)), 1, "Retains the delivered comparison rather than selecting on the test set.")
    if len(early_30) == 1:
        check("30-day early interaction AUC", np.isclose(float(early_30.iloc[0]["roc_auc"]), 0.6133896831, atol=1e-9),
              float(early_30.iloc[0]["roc_auc"]), 0.6133896831,
              "Useful as a research comparison, not evidence of deployable performance.")
    check("completion artifact verification", all(bool(v) for v in verification.values()), verification,
          "all verification flags true", "Recomputed metrics and saved predictions agree.")

    result = {
        "generated_local": pd.Timestamp.now().isoformat(timespec="seconds"),
        "purpose": "Read-only validation of the delivered Issue ML extension; no retraining.",
        "passed": sum(c["passed"] for c in checks),
        "total": len(checks),
        "checks": checks,
        "interpretation": {
            "classification": "辅助 bug/enhancement 标签分流，不代表严重程度或真实故障率。",
            "completion_prediction": "24h 观察预测 7d completed 的时间外信号很弱；不用于自动决策。",
            "integration": "作为 RQ2 的分流扩展与 RQ3 的处理结果预测扩展接入。",
        },
    }
    (OUT / "issue_ml_extension_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(checks).to_csv(OUT / "issue_ml_extension_validation_checks.csv", index=False,
                                encoding="utf-8-sig")
    print(f"Issue ML extension checks passed: {result['passed']}/{result['total']}")
    for item in checks:
        if not item["passed"]:
            print("FAILED:", item["check"], "observed=", item["observed"], "expected=", item["expected"])


if __name__ == "__main__":
    main()
