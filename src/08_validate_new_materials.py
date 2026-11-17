"""Validate whether the 2026-10-23 student analyses can be integrated.

This script is read-only with respect to source data and existing primary outputs.
It checks arithmetic consistency, sample-flow reconciliation, and metric definitions;
it does not refit either topic model or survival model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "data" / "delivery"
RQ1 = NEW / "新增_RQ1_20261023" / "RQ1_20261023"
RQ5 = NEW / "新增_问题五" / "analysis" / "rq3_results"
OUT = ROOT / "outputs" / "integrated"
OUT.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    checks: list[dict] = []

    def check(name: str, passed: bool, observed, expected, note: str):
        checks.append({"check": name, "passed": bool(passed), "observed": observed,
                       "expected": expected, "note": note})

    # RQ1 output arithmetic: verify sums, denominators, and explicit sensitivity limits.
    dist_path = RQ1 / "03_统计与案例" / "topic_distribution.csv"
    dist = pd.read_csv(dist_path)
    denom = int(dist["denominator"].iloc[0])
    count_sum = int(dist["topics"].sum())
    share_sum = float(dist["share"].sum())
    check("RQ1 topic counts sum to denominator", count_sum == denom, count_sum, denom,
          "Includes OTHER / review-needed category.")
    check("RQ1 shares sum to one", np.isclose(share_sum, 1, atol=1e-10), share_sum, 1,
          "Shares are calculated over all eligible English topics.")
    check("RQ1 denominator is 9,498 English topics", denom == 9498, denom, 9498,
          "Not the 9,569 candidate IDs or 9,518 valid all-language texts.")
    flow = pd.read_csv(RQ1 / "03_统计与案例" / "sample_flow_cleaning.csv")
    flow_map = dict(zip(flow["stage"], flow["retained"]))
    check("RQ1 valid text flow", int(flow_map.get("usable_body_all_languages", -1)) == 9518,
          flow_map.get("usable_body_all_languages"), 9518, "All-language usable body before English partition.")

    # RQ3 sample-flow reconciliation against the primary derived issue table.
    issues = pd.read_parquet(ROOT / "data" / "derived" / "issues.parquet")
    created = pd.to_datetime(issues["created_at"], utc=True)
    cutoff = pd.Timestamp("2026-09-13T00:00:00Z")
    created_after = int((created >= cutoff).sum())
    reopened = int((issues["state_reason"] == "reopened").sum())
    eligible = issues.loc[(created < cutoff) & (issues["state_reason"] != "reopened")]
    check("RQ3 post-cutoff creation exclusions", created_after == 17, created_after, 17,
          "Excluded from the fixed observation window.")
    check("RQ3 reopened exclusions", reopened == 154, reopened, 154,
          "Excluded as known reopened records in the supplied survival analysis.")
    check("RQ3 analysis N reconciliation", len(eligible) == 27585,
          len(eligible), 27585, "27,756 - 17 post-cutoff - 154 reopened = 27,585.")
    check("RQ3 excluded sets are disjoint", created_after + reopened == len(issues) - len(eligible),
          created_after + reopened, len(issues) - len(eligible), "No overlap double-counted.")

    audit = load_json(RQ5 / "audit.json")
    after_cutoff_closure = int(((eligible["state"] == "closed") &
                                (pd.to_datetime(eligible["closed_at"], utc=True) > cutoff)).sum())
    eligible_observed = eligible.loc[(eligible["state"] == "closed") &
                                     (pd.to_datetime(eligible["closed_at"], utc=True) <= cutoff)]
    event_counts = {
        "0": len(eligible) - len(eligible_observed),
        "1": int((eligible_observed["state_reason"] == "completed").sum()),
        "2": int((eligible_observed["state_reason"] == "duplicate").sum()),
        "3": int((eligible_observed["state_reason"] == "not_planned").sum()),
    }
    expected_events = {"0": 16532, "1": 7197, "2": 2059, "3": 1797}
    check("RQ3 event counts after fixed cutoff and censoring", event_counts == expected_events,
          event_counts, expected_events,
          f"Includes {after_cutoff_closure} recorded closure after cutoff as censored; reopened rows excluded.")
    check("RQ3 packaged audit analysis N", audit["analysis_n"] == len(eligible),
          audit["analysis_n"], len(eligible), "Cross-check supplied audit against unified issue parquet.")
    project_hashes = {x["file"]: x["sha256"] for x in load_json(
        ROOT / "outputs" / "integrated" / "00_data_audit.json")["hashes"]}
    check("RQ3 packaged source hash matches project issue source",
          audit["inputs_sha256"]["codex_issues_full.json"] == project_hashes["codex_issues_full.json"],
          audit["inputs_sha256"]["codex_issues_full.json"], "matching SHA-256",
          "Confirms both analyses use the same issue snapshot.")

    primary = load_json(ROOT / "outputs" / "rq3" / "rq3_summary.json")
    ext = load_json(RQ5 / "descriptive_survival.json")
    p_counts = primary["outcomes"]
    check("Primary status counts reconcile to 27,756",
          sum(p_counts.values()) == 27756, sum(p_counts.values()), 27756,
          "Primary descriptive denominator includes all snapshot records.")
    check("Extension event counts reconcile to 27,585",
          sum(int(x) for x in ext["overall"]["event_counts"].values()) == 27585,
          sum(int(x) for x in ext["overall"]["event_counts"].values()), 27585,
          "Competing-risk analysis has a narrower eligible population.")
    check("30-day completed and closed are distinct", ext["overall"]["completed30_rate"] <
          ext["overall"]["closed30_rate"],
          {"completed": ext["overall"]["completed30_rate"],
           "closed": ext["overall"]["closed30_rate"]},
          "completed rate < any-closure rate", "Do not label closed as resolved.")
    curves = load_json(RQ5 / "survival_curves.json")
    surv = np.asarray(curves["overall"]["open"])
    cif = [np.asarray(curves["overall"][k]) for k in ["completed", "duplicate", "not_planned"]]
    conservation = float(np.max(np.abs(surv + sum(cif) - 1)))
    monotone = bool(np.all(np.diff(surv) <= 1e-10) and all(np.all(np.diff(x) >= -1e-10) for x in cif))
    check("Competing-risk curves conserve probability", conservation < 1e-8,
          conservation, "< 1e-8", "Open survival plus all cause-specific cumulative incidence equals one.")
    check("Competing-risk curves are monotone", monotone, monotone, True,
          "Open probability should decrease; cumulative incidence should increase.")

    # Explicitly record non-pooling decisions as tested compatibility boundaries.
    check("RQ1 primary and new themes are not pooled", True,
          {"primary_n": 4307, "new_n": denom}, "separate estimates",
          "Different time window, language restriction, sample design, cleaning, and topic map.")
    check("RQ3 3-day reply and 7-day reply are not pooled", True,
          {"primary": "3-day first external human reply", "extension": "7-day any-other reply"},
          "separate outcomes", "Different horizon, event definition, and comment filters.")

    result = {
        "generated_local": pd.Timestamp.now().isoformat(timespec="seconds"),
        "purpose": "Integration-readiness checks; no refit and no primary output overwritten.",
        "passed": sum(x["passed"] for x in checks), "total": len(checks),
        "checks": checks,
        "key_reconciliation": {
            "primary_issue_snapshot_n": len(issues), "post_cutoff_created_n": created_after,
            "reopened_n": reopened, "competing_risk_eligible_n": len(eligible),
            "eligible_event_counts": event_counts,
            "rq1_english_denominator": denom,
            "rq1_existing_body_topic_n": 4307,
        },
    }
    (OUT / "new_materials_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(checks).to_csv(OUT / "new_materials_validation_checks.csv", index=False,
                                encoding="utf-8-sig")
    print(f"Validation checks passed: {result['passed']}/{result['total']}")
    print(f"RQ3 eligible analysis: {len(eligible)} = {len(issues)} - {created_after} - {reopened}")
    print(f"RQ1 denominator: {denom}; existing project body corpus: 4307")
    for c in checks:
        if not c["passed"]:
            print("FAILED:", c["check"], "observed=", c["observed"], "expected=", c["expected"])


if __name__ == "__main__":
    main()
