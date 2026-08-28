import json
import os


class SemanticRetriever(object):
    """Local rule-based retriever for semantic evidence."""

    def __init__(self, entity_memory_path, rules_path=None, examples_path=None):
        self.entity_memory_path = entity_memory_path
        self.rules_path = rules_path or self._sibling_path(entity_memory_path, "annotation_rules.json")
        self.examples_path = examples_path or self._sibling_path(entity_memory_path, "examples.jsonl")

        self.entity_memory = self._load_json(self.entity_memory_path)
        self.annotation_rules = self._load_json(self.rules_path)
        self.examples = self._load_jsonl(self.examples_path)

    def retrieve(self, domain_name, new_entity_list, old_entity_list):
        new_entity_list = list(new_entity_list)
        old_entity_list = list(old_entity_list)
        relevant_types = sorted(set(new_entity_list + old_entity_list))

        entity_rules = self._get_entity_rules(domain_name)
        rules = self._get_annotation_rules(domain_name)
        confusion_rules = self._get_confusion_rules(domain_name)

        evidence = {
            "domain_name": domain_name,
            "new_entity_list": new_entity_list,
            "old_entity_list": old_entity_list,
            "definitions": {},
            "annotation_rules": {},
            "examples": [],
            "confusion_evidence": []
        }

        for entity_type in relevant_types:
            entity_info = entity_rules.get(entity_type, {})
            if "definition" in entity_info:
                evidence["definitions"][entity_type] = entity_info["definition"]
            if entity_type in rules:
                evidence["annotation_rules"][entity_type] = rules[entity_type]

        relevant_type_set = set(relevant_types)
        for example in self.examples:
            if example.get("dataset") != domain_name:
                continue
            if example.get("entity_type") in relevant_type_set or example.get("label") in relevant_type_set:
                evidence["examples"].append(example)

        old_type_set = set(old_entity_list)
        direct_pairs = set()
        for rule in confusion_rules:
            if rule.get("source_type") in new_entity_list and rule.get("target_type") in old_type_set:
                evidence["confusion_evidence"].append(rule)
                direct_pairs.add((rule.get("source_type"), rule.get("target_type")))

        # Keep explicit directional rules strongest, but use their reverse as a
        # weak fallback when the annotation memory has no rule for that pair.
        for rule in confusion_rules:
            source_type = rule.get("source_type")
            target_type = rule.get("target_type")
            if source_type not in old_type_set or target_type not in new_entity_list:
                continue
            reversed_pair = (target_type, source_type)
            if reversed_pair in direct_pairs:
                continue
            reversed_rule = dict(rule)
            reversed_rule["source_type"] = target_type
            reversed_rule["target_type"] = source_type
            reversed_rule["reason"] = "Symmetric fallback from %s -> %s: %s" % (
                source_type, target_type, rule.get("reason", "")
            )
            reversed_rule["id"] = "symmetric_%s_to_%s" % reversed_pair
            reversed_rule["is_symmetric_fallback"] = True
            evidence["confusion_evidence"].append(reversed_rule)
            direct_pairs.add(reversed_pair)

        return evidence

    def _get_entity_rules(self, domain_name):
        datasets = self.entity_memory.get("datasets", {})
        dataset_memory = datasets.get(domain_name, {})
        return dataset_memory.get("entity_types", {})

    def _get_annotation_rules(self, domain_name):
        datasets = self.annotation_rules.get("datasets", {})
        dataset_memory = datasets.get(domain_name, {})
        return dataset_memory.get("rules", {})

    def _get_confusion_rules(self, domain_name):
        datasets = self.annotation_rules.get("datasets", {})
        dataset_memory = datasets.get(domain_name, {})
        return dataset_memory.get("confusion_rules", [])

    def _load_json(self, path):
        if not path or not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_jsonl(self, path):
        if not path or not os.path.isfile(path):
            return []
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _sibling_path(self, source_path, filename):
        if not source_path:
            return filename
        return os.path.join(os.path.dirname(source_path), filename)
