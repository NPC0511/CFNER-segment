import os

import torch


class PrototypeMemory(object):
    """Persistent task-level prototype summaries for old labels."""

    def __init__(self, cache_dir="semantic_cache/prototype_memory"):
        self.cache_dir = cache_dir

    def build_filename(self, domain_name, task_id):
        return "%s_task_%d.pt" % (domain_name, int(task_id))

    def save(self, task_id, domain_name, label_list, old_label_indices,
             prototype_vectors, feature_counts):
        if not self.cache_dir:
            return ""
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        payload = {
            "task_id": int(task_id),
            "domain_name": domain_name,
            "label_list": list(label_list),
            "old_label_indices": list(old_label_indices),
            "prototype_vectors": prototype_vectors.detach().cpu(),
            "feature_counts": feature_counts.detach().cpu()
        }
        path = os.path.join(self.cache_dir, self.build_filename(domain_name, task_id))
        torch.save(payload, path)
        return path

    def load(self, path):
        return torch.load(path)

    def load_latest(self, domain_name, before_task_id=None):
        """Load the newest prototype snapshot from an earlier task, if present."""
        if not self.cache_dir or not os.path.isdir(self.cache_dir):
            return None, ""
        prefix = "%s_task_" % domain_name
        candidates = []
        for filename in os.listdir(self.cache_dir):
            if not filename.startswith(prefix) or not filename.endswith(".pt"):
                continue
            task_text = filename[len(prefix):-3]
            try:
                task_id = int(task_text)
            except ValueError:
                continue
            if before_task_id is not None and task_id >= int(before_task_id):
                continue
            candidates.append((task_id, os.path.join(self.cache_dir, filename)))
        if not candidates:
            return None, ""
        _, path = max(candidates, key=lambda item: item[0])
        return self.load(path), path
