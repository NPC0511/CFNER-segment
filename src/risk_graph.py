import json
import os


class RiskGraph(object):
    """Task-level semantic risk graph for continual NER."""

    def __init__(self, task_id, domain_name, new_types, old_types,
                 edges=None, evidence_summary=None, training_policy=None):
        self.task_id = int(task_id)
        self.domain_name = domain_name
        self.new_types = list(new_types)
        self.old_types = list(old_types)
        self.edges = list(edges or [])
        self.evidence_summary = dict(evidence_summary or {})
        self.training_policy = dict(training_policy or {})

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "domain_name": self.domain_name,
            "new_types": self.new_types,
            "old_types": self.old_types,
            "edges": self.edges,
            "evidence_summary": self.evidence_summary,
            "training_policy": self.training_policy
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            task_id=payload.get("task_id", 0),
            domain_name=payload.get("domain_name", ""),
            new_types=payload.get("new_types", []),
            old_types=payload.get("old_types", []),
            edges=payload.get("edges", []),
            evidence_summary=payload.get("evidence_summary", {}),
            training_policy=payload.get("training_policy", {})
        )


def build_risk_graph_filename(domain_name, task_id):
    return "%s_task_%d.json" % (domain_name, int(task_id))


def save_risk_graph(risk_graph, cache_dir, filename=None):
    if not cache_dir:
        return ""
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    payload = risk_graph.to_dict() if isinstance(risk_graph, RiskGraph) else dict(risk_graph)
    if not filename:
        filename = build_risk_graph_filename(
            payload.get("domain_name", "unknown"),
            payload.get("task_id", 0)
        )

    path = os.path.join(cache_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return path


def load_risk_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return RiskGraph.from_dict(payload)
