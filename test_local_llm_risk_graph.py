import argparse
import json
import os
import sys

from src.local_llm_agent import LocalLLMAgent
from src.semantic_retriever import SemanticRetriever


def main():
    parser = argparse.ArgumentParser(description="Test local LLM semantic risk-graph generation.")
    parser.add_argument("--model_path", required=True, help="Local Qwen model directory")
    parser.add_argument("--domain", default="conll2003")
    parser.add_argument("--memory_path", default="semantic_memory/conll2003/entity_types.json")
    parser.add_argument("--new_types", nargs="+", default=["organisation"])
    parser.add_argument("--old_types", nargs="+", default=["location", "misc"])
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16", "float32", "auto"], default="float32")
    parser.add_argument("--output", default="semantic_cache/llm_tests/risk_graph_test.json")
    parser.add_argument("--allow_partial", action="store_true",
                        help="Exit successfully when only a subset of requested pairs is generated")
    args = parser.parse_args()

    retriever = SemanticRetriever(args.memory_path)
    evidence = retriever.retrieve(args.domain, args.new_types, args.old_types)
    # The test evaluates LLM reasoning from semantic memory, not rule copying.
    evidence["confusion_evidence"] = []

    agent = LocalLLMAgent(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        torch_dtype=args.torch_dtype
    )
    result = agent.generate_risk_graph(
        retrieved_evidence=evidence,
        new_entity_list=args.new_types,
        old_entity_list=args.old_types
    )
    risk_graph = result.get("risk_graph", {})
    risk_edges = risk_graph.get("risk_edges", [])
    failed_pairs = risk_graph.get("failed_pairs", [])
    expected_pairs = len(args.new_types) * len(args.old_types)
    complete = result["ok"] and len(risk_edges) == expected_pairs and not failed_pairs
    component_profiles = {}
    top_ranked_targets = {}
    for edge in risk_edges:
        components = edge.get("risk_components", {})
        profile = "%s/%s/%s" % (
            components.get("semantic_overlap", "?"),
            components.get("annotation_conflict", "?"),
            components.get("context_overlap", "?")
        )
        component_profiles[profile] = component_profiles.get(profile, 0) + 1
        top_ranked_targets.setdefault(edge.get("source", ""), []).append(edge.get("target", ""))
    top_ranked_targets = {
        source: targets[:4]
        for source, targets in top_ranked_targets.items()
    }
    payload = {
        "domain": args.domain,
        "new_types": args.new_types,
        "old_types": args.old_types,
        "semantic_evidence": evidence,
        "ok": result["ok"],
        "complete": complete,
        "expected_edge_count": expected_pairs,
        "returned_edge_count": len(risk_edges),
        "failed_pairs": failed_pairs,
        "component_profiles": component_profiles,
        "top_ranked_targets": top_ranked_targets,
        "error": result["error"],
        "risk_graph": risk_graph,
        "raw_output": result["raw_output"]
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Saved LLM risk-graph test result to %s" % args.output)
    print("Risk-edge coverage: %d/%d" % (len(risk_edges), expected_pairs))
    print("Risk-component profiles: %s" % json.dumps(component_profiles, ensure_ascii=False, sort_keys=True))
    print("Top-ranked old types: %s" % json.dumps(top_ranked_targets, ensure_ascii=False, sort_keys=True))
    if failed_pairs:
        print("Failed pairs: %s" % json.dumps(failed_pairs, ensure_ascii=False))
    if complete or (result["ok"] and args.allow_partial):
        print(json.dumps(risk_graph, ensure_ascii=False, indent=2))
        return

    print("LLM risk-graph generation failed or was incomplete: %s" % result["error"], file=sys.stderr)
    if result["raw_output"]:
        print("Raw model output:\n%s" % result["raw_output"], file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
