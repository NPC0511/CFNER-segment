import argparse
import json

from src.semantic_memory_builder import SemanticMemoryBuilder


def main():
    parser = argparse.ArgumentParser(description="Build and freeze LLM-assisted NER semantic memory.")
    parser.add_argument("--source_spec", required=True, help="JSON with dataset, entity_types, guidelines, and examples")
    parser.add_argument("--output_dir", default="semantic_memory/generated")
    parser.add_argument("--version", required=True, help="Immutable version name, e.g. v1")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_memory", default="",
                        help="Reviewed entity_types.json to preserve as the frozen training backbone")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--torch_dtype", default="float32", choices=["float16", "bfloat16", "float32", "auto"])
    args = parser.parse_args()
    with open(args.source_spec, "r", encoding="utf-8") as handle:
        source_spec = json.load(handle)
    builder = SemanticMemoryBuilder(args.model_path, args.max_new_tokens, args.torch_dtype)
    manifest = builder.build_and_freeze(
        source_spec, args.output_dir, args.version, base_memory_path=args.base_memory
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
