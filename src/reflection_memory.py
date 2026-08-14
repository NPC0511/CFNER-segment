import json
import os


class ReflectionMemory(object):
    """Persistent task-end evidence for validating predicted risk edges."""

    def __init__(self, cache_dir="semantic_cache/reflection"):
        self.cache_dir = cache_dir

    def save(self, domain_name, task_id, reflection):
        if not self.cache_dir:
            return ""
        os.makedirs(self.cache_dir, exist_ok=True)
        task_path = os.path.join(self.cache_dir, "%s_task_%d.json" % (domain_name, int(task_id)))
        self._write_json(task_path, reflection)
        latest_path = os.path.join(self.cache_dir, "%s_latest.json" % domain_name)
        self._write_json(latest_path, reflection)
        return task_path

    def _write_json(self, path, payload):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def load_latest(self, domain_name):
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, "%s_latest.json" % domain_name)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_history(self, domain_name, before_task_id=None):
        """Load completed task reflections, optionally excluding the current task.

        This is intentionally task-end evidence only.  The semantic controller
        uses it to calibrate the *next* task and never observes future results.
        """
        if not self.cache_dir or not os.path.isdir(self.cache_dir):
            return []
        prefix = "%s_task_" % domain_name
        reflections = []
        for filename in os.listdir(self.cache_dir):
            if not filename.startswith(prefix) or not filename.endswith(".json"):
                continue
            path = os.path.join(self.cache_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                reflection = json.load(f)
            task_id = int(reflection.get("task_id", -1))
            if before_task_id is not None and task_id >= int(before_task_id):
                continue
            reflections.append(reflection)
        return sorted(reflections, key=lambda item: int(item.get("task_id", -1)))
