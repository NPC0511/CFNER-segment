import numpy as np

from src.reflection_memory import ReflectionMemory


class RiskEdgeCalibrator(object):
    """Fit task-data evidence to observed old-to-new interference rates."""

    FEATURE_NAMES = (
        "semantic_candidate", "prototype_similarity", "teacher_confusion",
        "historical_interference", "vulnerability", "rule_risk"
    )

    def __init__(self, reflection_cache_dir, min_tasks=3, min_edges=30,
                 ridge_alpha=1.0, risk_scale=3.0):
        self.memory = ReflectionMemory(reflection_cache_dir)
        self.min_tasks = max(int(min_tasks), 1)
        self.min_edges = max(int(min_edges), 1)
        self.ridge_alpha = max(float(ridge_alpha), 1e-8)
        self.risk_scale = max(float(risk_scale), 0.0)

    def apply(self, risk_edges, domain_name, current_iteration):
        # Reflections are saved only after a task completes.  Reading files
        # already present on disk is therefore causal for the current run, and
        # also permits a separately completed diagnostic run to act as history.
        history = self.memory.load_history(domain_name)
        x_rows, y_values = self._history_rows(history)
        if len(history) < self.min_tasks or len(x_rows) < self.min_edges:
            return risk_edges, {
                "mode": "warmup", "history_tasks": len(history), "history_edges": len(x_rows)
            }

        x = np.asarray(x_rows, dtype=np.float64)
        y = np.asarray(y_values, dtype=np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
        normalized = (x - mean) / std
        design = np.column_stack([np.ones(len(normalized)), normalized])
        penalty = np.eye(design.shape[1]) * self.ridge_alpha
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T.dot(design) + penalty, design.T.dot(y))

        updated = []
        for edge in risk_edges:
            row = np.asarray(self._features(edge), dtype=np.float64)
            prediction = float(np.dot(np.r_[1.0, (row - mean) / std], coefficients))
            evidence_risk = min(max(prediction * self.risk_scale, 0.0), 1.0)
            item = dict(edge)
            # Evidence may add protection, but cannot erase a verified rule or
            # reflection risk already present on the edge.
            item["risk"] = round(max(float(edge.get("risk", 0.0)), evidence_risk), 4)
            item["risk_calibration"] = {
                "predicted_interference": round(prediction, 6),
                "evidence_risk": round(evidence_risk, 4),
                "features": dict(zip(self.FEATURE_NAMES, [round(value, 6) for value in row]))
            }
            updated.append(item)
        return updated, {
            "mode": "ridge", "history_tasks": len(history), "history_edges": len(x_rows),
            "feature_names": list(self.FEATURE_NAMES),
            "coefficients": dict(zip(["intercept"] + list(self.FEATURE_NAMES),
                                      [round(float(value), 6) for value in coefficients]))
        }

    def _history_rows(self, reflections):
        rows, targets = [], []
        for reflection in reflections:
            observed = {
                (item.get("source"), item.get("target")): float(item.get("rate", 0.0))
                for item in reflection.get("observed_confusion", [])
            }
            for edge in reflection.get("risk_edges", []):
                rows.append(self._features(edge))
                targets.append(observed.get((edge.get("source"), edge.get("target")), 0.0))
        return rows, targets

    def _features(self, edge):
        prototype = (edge.get("prototype_similarity") or {}).get("cosine_similarity", 0.0)
        reflection = edge.get("reflection_update") or {}
        vulnerability = (edge.get("vulnerability_update") or {}).get("score", 0.0)
        teacher = (edge.get("model_confusion") or {}).get("risk", 0.0)
        return [
            1.0 if float(edge.get("llm_risk") or 0.0) >= 0.525 else 0.0,
            min(max((float(prototype) + 1.0) / 2.0, 0.0), 1.0),
            min(max(float(teacher), 0.0), 1.0),
            min(max(float(reflection.get("empirical_risk", 0.0)), 0.0), 1.0),
            min(max(float(vulnerability), 0.0), 1.0),
            min(max(float(edge.get("rule_risk", edge.get("risk", 0.0))), 0.0), 1.0)
        ]
