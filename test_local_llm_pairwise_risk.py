"""Fast pairwise LLM risk-ranking probe without continual-model training."""

import argparse
import json
import os
import sys

from src.local_llm_agent import LocalLLMAgent
from src.semantic_retriever import SemanticRetriever


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--domain", default="ontonotes5")
    parser.add_argument("--memory_path", default="semantic_memory/ontonotes5/entity_types.json")
    parser.add_argument("--source", required=True, help="New entity type")
    parser.add_argument("--candidates", nargs="+", required=True, help="Old entity types to rank")
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16", "float32", "auto"], default="float32")
    parser.add_argument("--output", default="semantic_cache/llm_tests/pairwise_risk_test.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.candidates) < 2:
        raise ValueError("At least two candidate old types are required")

    retriever = SemanticRetriever(args.memory_path)
    evidence = retriever.retrieve(args.domain, [args.source], args.candidates)
    agent = LocalLLMAgent(model_path=args.model_path, torch_dtype=args.torch_dtype)
    result = agent.rank_risk_targets(args.source, args.candidates, evidence)
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "pairwise ranking failed"))
    ranking = result["ranking"]
    scores = result["scores"]
    payload = {
        "source": args.source,
        "candidates": args.candidates,
        "ranking": ranking,
        "scores": scores,
        "comparisons": result["comparisons"]
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Pairwise risk ranking for %s: %s" % (args.source, " > ".join(ranking)))
    print("Scores: %s" % json.dumps(scores, ensure_ascii=False, sort_keys=True))
    print("Saved pairwise test result to %s" % args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Pairwise risk test failed: %s" % exc, file=sys.stderr)
        sys.exit(1)
