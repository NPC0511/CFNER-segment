import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class RiskForgettingAnalyzer(object):
    """Aggregate task-end reflections into risk-prediction diagnostics."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.records_by_task = {}

    def add_reflection(self, reflection):
        task_id = int(reflection["task_id"])
        self.records_by_task[task_id] = self._records_from_reflection(reflection)
        return self.save()

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        records = self._all_records()
        self._save_csv(records)
        metrics = self._build_metrics(records)
        self._save_json(metrics)
        self._save_plot(records, "risk_vs_forgetting.png", "All Tasks")
        for task_id in sorted(self.records_by_task):
            task_records = self.records_by_task[task_id]
            self._save_plot(
                task_records,
                "risk_vs_forgetting_task_%d.png" % task_id,
                "Task %d" % task_id
            )
        return metrics

    def _records_from_reflection(self, reflection):
        confusion_rates = {}
        interference_by_pair = {}
        for item in reflection.get("observed_confusion", []):
            source = item.get("source")
            target = item.get("target")
            if target:
                confusion_rates[target] = max(confusion_rates.get(target, 0.0), float(item.get("rate", 0.0)))
            if source and target:
                interference_by_pair[(source, target)] = float(item.get("rate", 0.0))

        records = []
        for edge in reflection.get("risk_edges", []):
            target = edge.get("target")
            if target not in reflection.get("observed_forgetting", {}):
                continue
            records.append({
                "task_id": int(reflection["task_id"]),
                "source_new": edge.get("source", ""),
                "target_old": target,
                # Rule-only or partial-fallback edges must not be counted as
                # LLM predictions in risk-quality metrics.
                "llm_risk": _float_or_none(edge.get("llm_risk")),
                "rule_risk": _float_or_none(edge.get("rule_risk")),
                "semantic_prior_delta": _float_or_none(edge.get("semantic_prior_delta")),
                "initial_risk": _float_or_none(edge.get("initial_risk", edge.get("risk"))),
                "final_risk": _float_or_none(edge.get("risk")),
                "observed_forgetting": _float_or_none(reflection["observed_forgetting"].get(target)),
                # Test-time old->new confusion is the identifiable edge-level
                # consequence of adding a new class.  Missing pairs have zero
                # observed confusion because the trainer stores nonzero pairs.
                "observed_interference": interference_by_pair.get(
                    (edge.get("source"), target), 0.0
                ),
                "confusion_rate": confusion_rates.get(target, 0.0),
                "teacher_confusion_risk": _nested_float(edge, "model_confusion", "risk"),
                "prototype_similarity": _nested_float(edge, "prototype_similarity", "cosine_similarity"),
                "reflection_empirical_risk": _nested_float(edge, "reflection_update", "empirical_risk"),
                "vulnerability_score": _nested_float(edge, "vulnerability_update", "score")
            })
        return records

    def _all_records(self):
        records = []
        for task_id in sorted(self.records_by_task):
            records.extend(self.records_by_task[task_id])
        return records

    def _save_csv(self, records):
        path = os.path.join(self.output_dir, "risk_vs_forgetting.csv")
        fieldnames = [
            "task_id", "source_new", "target_old", "llm_risk", "rule_risk",
            "semantic_prior_delta", "teacher_confusion_risk", "prototype_similarity", "reflection_empirical_risk",
            "vulnerability_score", "initial_risk", "final_risk", "observed_forgetting",
            "observed_interference", "confusion_rate"
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record)

    def _build_metrics(self, records):
        target_records = _aggregate_task_target(records)
        target_metrics = _component_metrics(target_records)
        edge_metrics = _component_metrics(records)
        interference_metrics = _component_metrics(records, outcome_key="observed_interference")
        metrics = {
            "evaluation_unit": "task_target_max_over_new_sources",
            "num_edges": len(records),
            "num_task_target_records": len(target_records),
            "num_rankable_tasks": target_metrics["final_risk"]["top_1"]["num_tasks"],
            # Headline results use the identifiable task-target unit.
            "spearman": {key: value["spearman"] for key, value in target_metrics.items()},
            "top_1": {key: value["top_1"] for key, value in target_metrics.items()},
            "top_3": {key: value["top_3"] for key, value in target_metrics.items()},
            "random_baseline": {
                "top_1": _random_top_k_baseline(target_records, 1),
                "top_3": _random_top_k_baseline(target_records, 3)
            },
            "component_metrics": target_metrics,
            "interference_evaluation": {
                "evaluation_unit": "source_target",
                "outcome": "test_old_to_new_confusion_rate",
                "spearman": {key: value["spearman"] for key, value in interference_metrics.items()},
                "top_1": {key: value["top_1"] for key, value in interference_metrics.items()},
                "top_3": {key: value["top_3"] for key, value in interference_metrics.items()},
                "random_baseline": {
                    "top_1": _random_top_k_baseline(records, 1, outcome_key="observed_interference"),
                    "top_3": _random_top_k_baseline(records, 3, outcome_key="observed_interference")
                },
                "component_metrics": interference_metrics
            },
            # Kept only as a diagnostic.  It is not valid pairwise ground
            # truth when multiple new types are learned in one task.
            "edge_level_source_diagnostic": edge_metrics
        }
        return metrics

    def _save_json(self, metrics):
        path = os.path.join(self.output_dir, "risk_prediction_metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=True, indent=2)

    def _save_plot(self, records, filename, title_suffix):
        if not records:
            return
        x_values = [record["final_risk"] for record in records]
        y_values = [record["observed_forgetting"] for record in records]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x_values, y_values, s=70, c="#c9464b", edgecolors="#222222", linewidths=0.6)
        for record in records:
            ax.annotate(
                "%s->%s" % (record["source_new"], record["target_old"]),
                (record["final_risk"], record["observed_forgetting"]),
                xytext=(4, 4), textcoords="offset points", fontsize=8
            )
        ax.set_title("Risk vs Forgetting - %s" % title_suffix)
        ax.set_xlabel("Final predicted risk")
        ax.set_ylabel("Observed old-class F1 drop")
        ax.grid(alpha=0.25)
        ax.set_xlim(-0.02, 1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename), dpi=200, bbox_inches="tight")
        plt.close(fig)


def _component_metrics(records, outcome_key="observed_forgetting"):
        component_keys = [
            "llm_risk", "rule_risk", "semantic_prior_delta", "teacher_confusion_risk", "prototype_similarity",
            "reflection_empirical_risk", "vulnerability_score", "initial_risk", "final_risk"
        ]
        return {
            key: {
                "spearman": _spearman(records, key, outcome_key=outcome_key),
                "top_1": _top_k(records, key, 1, outcome_key=outcome_key),
                "top_3": _top_k(records, key, 3, outcome_key=outcome_key),
                "top_1_by_source": _top_k(records, key, 1, by_source=True, outcome_key=outcome_key),
                "top_3_by_source": _top_k(records, key, 3, by_source=True, outcome_key=outcome_key)
            }
            for key in component_keys
        }


def _aggregate_task_target(records):
    """Aggregate max predicted risk across simultaneously added new types."""
    grouped = {}
    score_keys = [
        "llm_risk", "rule_risk", "semantic_prior_delta", "teacher_confusion_risk", "prototype_similarity",
        "reflection_empirical_risk", "vulnerability_score", "initial_risk", "final_risk"
    ]
    for record in records:
        group_key = (record["task_id"], record["target_old"])
        aggregate = grouped.setdefault(group_key, {
            "task_id": record["task_id"],
            "source_new": "ANY_NEW_SOURCE",
            "target_old": record["target_old"],
            "observed_forgetting": record["observed_forgetting"],
            "observed_interference": record.get("observed_interference", 0.0),
            "confusion_rate": record.get("confusion_rate", 0.0)
        })
        for key in score_keys:
            value = record.get(key)
            if value is not None:
                aggregate[key] = max(aggregate.get(key, value), value)
    return [grouped[key] for key in sorted(grouped)]

def _float_or_none(value):
    if value is None:
        return None
    return float(value)


def _nested_float(record, parent_key, child_key):
    parent = record.get(parent_key) or {}
    return _float_or_none(parent.get(child_key))


def _spearman(records, risk_key, outcome_key="observed_forgetting"):
    valid_records = [
        record for record in records
        if record.get(risk_key) is not None and record.get(outcome_key) is not None
    ]
    if len(valid_records) < 3:
        return {"correlation": None, "p_value": None, "num_edges": len(valid_records)}
    risk_values = [record[risk_key] for record in valid_records]
    forgetting_values = [record[outcome_key] for record in valid_records]
    if len(set(risk_values)) < 2 or len(set(forgetting_values)) < 2:
        return {"correlation": None, "p_value": None, "num_edges": len(valid_records)}
    try:
        from scipy.stats import spearmanr

        correlation, p_value = spearmanr(risk_values, forgetting_values)
        if math.isnan(correlation):
            correlation = None
        if math.isnan(p_value):
            p_value = None
        return {
            "correlation": None if correlation is None else round(float(correlation), 6),
            "p_value": None if p_value is None else round(float(p_value), 6),
            "num_edges": len(valid_records)
        }
    except Exception as exc:
        return {"correlation": None, "p_value": None, "num_edges": len(valid_records), "error": str(exc)}


def _top_k(records, risk_key, k, by_source=False, outcome_key="observed_forgetting"):
    records_by_task = {}
    for record in records:
        if record.get(risk_key) is None or record.get(outcome_key) is None:
            continue
        group_key = (record["task_id"], record["source_new"]) if by_source else record["task_id"]
        records_by_task.setdefault(group_key, []).append(record)

    hits = []
    for task_records in records_by_task.values():
        if len(task_records) < 2:
            continue
        actual_k = min(k, len(task_records))
        predicted = sorted(task_records, key=lambda item: item[risk_key], reverse=True)[:actual_k]
        observed = sorted(task_records, key=lambda item: item[outcome_key], reverse=True)[:actual_k]
        predicted_targets = {item["target_old"] for item in predicted}
        observed_targets = {item["target_old"] for item in observed}
        hits.append(len(predicted_targets & observed_targets) / float(actual_k))
    return {
        "hit_rate": None if not hits else round(sum(hits) / float(len(hits)), 6),
        "num_tasks": len(hits),
        "k": k
    }


def _random_top_k_baseline(records, k, outcome_key="observed_forgetting"):
    """Expected Top-k overlap for uniform random ranking on each task."""
    records_by_task = {}
    for record in records:
        if record.get(outcome_key) is not None:
            records_by_task.setdefault(record["task_id"], []).append(record)
    expected = []
    for task_records in records_by_task.values():
        targets = {record["target_old"] for record in task_records}
        if len(targets) < 2:
            continue
        actual_k = min(k, len(targets))
        expected.append(actual_k / float(len(targets)))
    return {
        "hit_rate": None if not expected else round(sum(expected) / float(len(expected)), 6),
        "num_tasks": len(expected),
        "k": k
    }
