"""Offline diagnosis for semantic-risk reflections; no model training required."""

import argparse
import glob
import json
import os

from src.risk_analysis import RiskForgettingAnalyzer


def _best(records, key):
    valid = [item for item in records if item.get(key) is not None]
    if not valid:
        return None
    return max(valid, key=lambda item: float(item[key]))


def _pairwise_diagnosis(records):
    groups = {}
    for record in records:
        groups.setdefault((record["task_id"], record["source_new"]), []).append(record)

    comparisons = []
    for (task_id, source), items in sorted(groups.items()):
        actual = _best(items, "observed_forgetting")
        llm = _best(items, "llm_risk")
        teacher = _best(items, "teacher_confusion_risk")
        final = _best(items, "final_risk")
        comparisons.append({
            "task_id": task_id,
            "source_new": source,
            "actual_most_forgotten": _describe(actual, "observed_forgetting"),
            "llm_top_1": _describe(llm, "llm_risk"),
            "teacher_top_1": _describe(teacher, "teacher_confusion_risk"),
            "final_top_1": _describe(final, "final_risk"),
            "llm_hit": bool(llm and actual and llm["target_old"] == actual["target_old"]),
            "teacher_hit": bool(teacher and actual and teacher["target_old"] == actual["target_old"]),
            "final_hit": bool(final and actual and final["target_old"] == actual["target_old"])
        })
    return comparisons


def _task_target_diagnosis(records):
    from src.risk_analysis import _aggregate_task_target

    groups = {}
    for record in _aggregate_task_target(records):
        groups.setdefault(record["task_id"], []).append(record)
    comparisons = []
    for task_id, items in sorted(groups.items()):
        actual = _best(items, "observed_forgetting")
        llm = _best(items, "llm_risk")
        teacher = _best(items, "teacher_confusion_risk")
        final = _best(items, "final_risk")
        comparisons.append({
            "task_id": task_id,
            "actual_most_forgotten": _describe(actual, "observed_forgetting"),
            "llm_top_1": _describe(llm, "llm_risk"),
            "teacher_top_1": _describe(teacher, "teacher_confusion_risk"),
            "final_top_1": _describe(final, "final_risk"),
            "llm_hit": bool(llm and actual and llm["target_old"] == actual["target_old"]),
            "teacher_hit": bool(teacher and actual and teacher["target_old"] == actual["target_old"]),
            "final_hit": bool(final and actual and final["target_old"] == actual["target_old"])
        })
    return comparisons


def _interference_diagnosis(records):
    groups = {}
    for record in records:
        groups.setdefault(record["task_id"], []).append(record)
    comparisons = []
    for task_id, items in sorted(groups.items()):
        actual = _best(items, "observed_interference")
        llm = _best(items, "llm_risk")
        prototype = _best(items, "prototype_similarity")
        final = _best(items, "final_risk")
        comparisons.append({
            "task_id": task_id,
            "actual_highest_interference": _describe(actual, "observed_interference"),
            "llm_top_1": _describe(llm, "llm_risk"),
            "prototype_top_1": _describe(prototype, "prototype_similarity"),
            "final_top_1": _describe(final, "final_risk"),
            "llm_hit": bool(llm and actual and _pair(llm) == _pair(actual)),
            "prototype_hit": bool(prototype and actual and _pair(prototype) == _pair(actual)),
            "final_hit": bool(final and actual and _pair(final) == _pair(actual))
        })
    return comparisons


def _describe(record, score_key):
    if record is None:
        return None
    return {
        "target_old": record["target_old"],
        "score": round(float(record[score_key]), 6),
        "observed_forgetting": round(float(record["observed_forgetting"]), 6)
    }


def _pair(record):
    return record["source_new"], record["target_old"]


def main():
    parser = argparse.ArgumentParser(description="Diagnose saved semantic-risk reflections without training")
    parser.add_argument("--reflection_dir", required=True, help="Directory containing <domain>_task_*.json")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    pattern = os.path.join(args.reflection_dir, "%s_task_*.json" % args.domain)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError("No reflection files matched %s" % pattern)

    analyzer = RiskForgettingAnalyzer(args.output_dir)
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            analyzer.add_reflection(json.load(f))

    records = analyzer._all_records()
    metrics = analyzer._build_metrics(records)
    report = {
        "reflection_files": paths,
        "metrics": metrics,
        "per_task_interference_top_1": _interference_diagnosis(records),
        "per_task_target_top_1": _task_target_diagnosis(records),
        "edge_level_source_diagnostic": _pairwise_diagnosis(records)
    }
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "risk_diagnosis_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=2)

    print("Saved offline risk diagnosis to %s" % report_path)
    for item in report["per_task_interference_top_1"]:
        print(
            "Task %(task_id)s interference: actual=%(actual_highest_interference)s "
            "llm=%(llm_top_1)s hit=%(llm_hit)s prototype=%(prototype_top_1)s "
            "hit=%(prototype_hit)s final=%(final_top_1)s "
            "hit=%(final_hit)s" % item
        )


if __name__ == "__main__":
    main()
