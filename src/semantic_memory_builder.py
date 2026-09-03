"""Offline construction and freezing of LLM-assisted semantic memory."""
import hashlib
import json
import os
from datetime import datetime, timezone

from src.local_llm_agent import LocalLLMAgent


class SemanticMemoryBuilder(object):
    def __init__(self, model_path, max_new_tokens=1024, torch_dtype="float32"):
        self.agent = LocalLLMAgent(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            torch_dtype=torch_dtype
        )

    def build_and_freeze(self, source_spec, output_dir, version, base_memory_path=""):
        dataset = str(source_spec.get("dataset", "")).strip()
        entity_types = source_spec.get("entity_types", {})
        if not dataset or not isinstance(entity_types, dict) or not entity_types:
            raise ValueError("source_spec requires dataset and non-empty entity_types")
        base_artifacts = self._load_base_artifacts(base_memory_path) if base_memory_path else None
        prompt_spec = self._with_human_memory_context(source_spec, base_artifacts)
        result = self.agent.generate_semantic_memory(prompt_spec)
        if not result.get("ok", False):
            raise RuntimeError("LLM semantic-memory generation failed: %s" % result.get("error", ""))
        llm_candidates = self._validate_and_normalize(dataset, entity_types, result["memory"])
        # A generated candidate must never silently replace reviewed annotation
        # boundaries. The frozen training memory remains the human backbone.
        artifacts = base_artifacts or llm_candidates
        version_dir = os.path.join(output_dir, dataset, version)
        if os.path.exists(version_dir):
            raise FileExistsError("Frozen memory version already exists: %s" % version_dir)
        os.makedirs(version_dir)
        self._write_json(os.path.join(version_dir, "entity_types.json"), artifacts["entity_types"])
        self._write_json(os.path.join(version_dir, "annotation_rules.json"), artifacts["annotation_rules"])
        self._write_jsonl(os.path.join(version_dir, "examples.jsonl"), artifacts["examples"])
        self._write_json(os.path.join(version_dir, "llm_candidates.json"), llm_candidates)
        with open(os.path.join(version_dir, "llm_raw_output.txt"), "w", encoding="utf-8") as handle:
            handle.write(result.get("raw_output", ""))
        manifest = {
            "dataset": dataset,
            "version": version,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_spec_sha256": self._sha256_json(source_spec),
            "base_memory_path": base_memory_path,
            "base_memory_sha256": self._sha256_json(base_artifacts) if base_artifacts else "",
            "artifact_sha256": self._sha256_json(artifacts),
            "llm_candidates_sha256": self._sha256_json(llm_candidates),
            "review_status": "pending_human_review",
            "training_memory_source": "human_backbone" if base_artifacts else "llm_draft",
            "training_memory_path": os.path.join(version_dir, "entity_types.json")
        }
        self._write_json(os.path.join(version_dir, "manifest.json"), manifest)
        return manifest

    def _load_base_artifacts(self, entity_memory_path):
        if not os.path.isfile(entity_memory_path):
            raise FileNotFoundError("Base entity memory not found: %s" % entity_memory_path)
        directory = os.path.dirname(entity_memory_path)
        rules_path = os.path.join(directory, "annotation_rules.json")
        examples_path = os.path.join(directory, "examples.jsonl")
        if not os.path.isfile(rules_path) or not os.path.isfile(examples_path):
            raise FileNotFoundError("Base memory requires sibling annotation_rules.json and examples.jsonl")
        with open(entity_memory_path, "r", encoding="utf-8") as handle:
            entity_types = json.load(handle)
        with open(rules_path, "r", encoding="utf-8") as handle:
            annotation_rules = json.load(handle)
        examples = []
        with open(examples_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return {
            "entity_types": entity_types,
            "annotation_rules": annotation_rules,
            "examples": examples
        }

    @staticmethod
    def _with_human_memory_context(source_spec, base_artifacts):
        if not base_artifacts:
            return source_spec
        prompt_spec = json.loads(json.dumps(source_spec))
        prompt_spec["reviewed_human_memory"] = base_artifacts
        prompt_spec["task_mode"] = (
            "Propose candidate additions only. Do not remove or contradict reviewed human definitions, "
            "annotation rules, confusion rules, or examples."
        )
        return prompt_spec

    def _validate_and_normalize(self, dataset, source_types, draft):
        if not isinstance(draft, dict):
            raise ValueError("LLM semantic memory must be a JSON object")
        allowed_types = set(source_types.keys())
        draft_types = draft.get("entity_types", {})
        if set(draft_types.keys()) != allowed_types:
            raise ValueError("LLM memory must define exactly the source entity types")
        normalized_types = {}
        examples = []
        for type_name in sorted(allowed_types):
            item = draft_types[type_name]
            # Some instruction models compress a type object to a plain
            # definition string. Treat that as a valid minimal draft rather
            # than failing after an expensive offline generation call.
            if isinstance(item, str):
                item = {"definition": item}
            if not isinstance(item, dict):
                raise ValueError("Entity memory for %s must be an object or definition string" % type_name)
            source_item = source_types.get(type_name, {})
            definition = str(item.get("definition") or source_item.get("source_definition", "")).strip()
            if not definition:
                raise ValueError("Missing definition for %s" % type_name)
            positive = self._clean_examples(item.get("positive_examples", []))
            negative = self._clean_examples(item.get("negative_examples", []))
            confusable = {
                other: min(max(float(weight), 1.0), 2.0)
                for other, weight in item.get("confusable_with", {}).items()
                if other in allowed_types and other != type_name
            }
            normalized_types[type_name] = {
                "definition": definition,
                "positive_examples": positive,
                "negative_examples": negative,
                "confusable_with": confusable
            }
            for text in positive:
                examples.append({
                    "dataset": dataset, "entity_type": type_name, "text": text,
                    "span": text, "label": type_name, "example_type": "positive"
                })
            for text in negative:
                examples.append({
                    "dataset": dataset, "entity_type": type_name, "text": text,
                    "span": text, "label": "O", "example_type": "negative"
                })
        rules = {
            key: [str(value).strip()]
            for key, value in draft.get("annotation_rules", {}).items()
            if key in allowed_types and str(value).strip()
        }
        confusion_rules = []
        for index, rule in enumerate(draft.get("confusion_rules", [])):
            source, target = rule.get("source_type"), rule.get("target_type")
            if source not in allowed_types or target not in allowed_types or source == target:
                continue
            confusion_rules.append({
                "id": "llm_%s_%s_to_%s_%d" % (dataset, source, target, index),
                "source_type": source,
                "target_type": target,
                "weight": min(max(float(rule.get("weight", 1.0)), 1.0), 2.0),
                "reason": str(rule.get("reason", "")).strip()
            })
        return {
            "entity_types": {"datasets": {dataset: {"entity_types": normalized_types}}},
            "annotation_rules": {"datasets": {dataset: {"rules": rules, "confusion_rules": confusion_rules}}},
            "examples": examples
        }

    @staticmethod
    def _clean_examples(values):
        return [str(value).strip() for value in values[:5] if str(value).strip()]

    @staticmethod
    def _write_json(path, payload):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)

    @staticmethod
    def _write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    @staticmethod
    def _sha256_json(payload):
        value = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
        return hashlib.sha256(value).hexdigest()
