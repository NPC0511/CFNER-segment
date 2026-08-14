import csv
import os


class ContinualResultSummary(object):
    """Write per-task continual NER metrics and their stage averages to CSV."""

    def __init__(self, output_path, entity_list):
        self.output_path = output_path
        self.entity_list = list(entity_list)
        self.rows = []

    def add_task_result(self, task_id, new_entity_list, seen_entity_list,
                        micro_f1, macro_f1, class_f1):
        class_scores = {
            entity_name: class_f1.get(entity_name)
            for entity_name in self.entity_list
        }
        seen_scores = [
            class_scores[entity_name]
            for entity_name in seen_entity_list
            if class_scores.get(entity_name) is not None
        ]
        old_entity_list = [
            entity_name for entity_name in seen_entity_list
            if entity_name not in new_entity_list
        ]
        old_scores = [
            class_scores[entity_name]
            for entity_name in old_entity_list
            if class_scores.get(entity_name) is not None
        ]
        new_scores = [
            class_scores[entity_name]
            for entity_name in new_entity_list
            if class_scores.get(entity_name) is not None
        ]
        self.rows.append({
            "stage": int(task_id) + 1,
            "new_entity_types": ";".join(new_entity_list),
            "seen_entity_types": ";".join(seen_entity_list),
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
            "seen_class_avg_f1": _mean(seen_scores),
            "old_class_avg_f1": _mean(old_scores),
            "new_class_avg_f1": _mean(new_scores),
            "class_scores": class_scores
        })
        self.save()

    def save(self):
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fieldnames = [
            "stage", "new_entity_types", "seen_entity_types", "micro_f1", "macro_f1",
            "seen_class_avg_f1", "old_class_avg_f1", "new_class_avg_f1"
        ] + ["f1_%s" % entity_name for entity_name in self.entity_list]

        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(self._to_csv_row(row))
            if self.rows:
                writer.writerow(self._average_csv_row())

    def _to_csv_row(self, row):
        output_row = {
            "stage": row["stage"],
            "new_entity_types": row["new_entity_types"],
            "seen_entity_types": row["seen_entity_types"],
            "micro_f1": _format_score(row["micro_f1"]),
            "macro_f1": _format_score(row["macro_f1"]),
            "seen_class_avg_f1": _format_score(row["seen_class_avg_f1"]),
            "old_class_avg_f1": _format_score(row["old_class_avg_f1"]),
            "new_class_avg_f1": _format_score(row["new_class_avg_f1"])
        }
        for entity_name, score in row["class_scores"].items():
            output_row["f1_%s" % entity_name] = _format_score(score)
        return output_row

    def _average_csv_row(self):
        average_row = {
            "stage": "avg",
            "new_entity_types": "",
            "seen_entity_types": "",
            "micro_f1": _format_score(_mean([row["micro_f1"] for row in self.rows])),
            "macro_f1": _format_score(_mean([row["macro_f1"] for row in self.rows])),
            "seen_class_avg_f1": _format_score(_mean([row["seen_class_avg_f1"] for row in self.rows])),
            "old_class_avg_f1": _format_score(_mean([row["old_class_avg_f1"] for row in self.rows])),
            "new_class_avg_f1": _format_score(_mean([row["new_class_avg_f1"] for row in self.rows]))
        }
        for entity_name in self.entity_list:
            scores = [
                row["class_scores"][entity_name]
                for row in self.rows
                if row["class_scores"].get(entity_name) is not None
            ]
            average_row["f1_%s" % entity_name] = _format_score(_mean(scores))
        return average_row


def _mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def _format_score(value):
    return "" if value is None else "%.4f" % float(value)
