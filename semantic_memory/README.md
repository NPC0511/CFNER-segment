# Semantic Memory

Each dataset has one reviewed, immutable semantic-memory directory:

```text
semantic_memory/<dataset>/
  entity_types.json
  annotation_rules.json
  examples.jsonl
  manifest.json
```

Training points `semantic_memory_path` to `entity_types.json`. The retriever
automatically loads the sibling rules and examples files. Do not edit these
files during an experiment; duplicate the dataset directory before a reviewed
update.
