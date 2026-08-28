"""Run reproducible CoNLL2003 ablations with isolated caches per mode and seed."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


MODE_OVERRIDES = {
    "baseline": {
        "is_use_semantic_agent": False,
        "is_use_risk_graph": False,
        "is_use_prototype_anchor": False,
        "is_use_local_llm_agent": False,
        "is_use_reflection_memory": False,
        "is_save_risk_analysis": False,
        "is_use_risk_aware_pseudo_label": False,
        "is_use_risk_contrastive": False,
        "is_use_llm_verifier": False
    },
    "anchor_only": {
        "is_use_semantic_agent": False,
        "is_use_risk_graph": False,
        "is_use_prototype_anchor": True,
        "is_use_local_llm_agent": False,
        "is_use_reflection_memory": False,
        "is_save_risk_analysis": False,
        "is_use_risk_aware_pseudo_label": False,
        "is_use_risk_contrastive": False,
        "is_use_llm_verifier": False
    },
    "semantic_only": {
        "is_use_semantic_agent": True,
        "is_use_risk_graph": True,
        "is_use_prototype_anchor": False,
        "is_use_local_llm_agent": False,
        "is_use_reflection_memory": False,
        "is_save_risk_analysis": False,
        "is_use_risk_aware_pseudo_label": False,
        "is_use_risk_contrastive": False,
        "is_use_llm_verifier": False
    },
    "semantic_pseudo": {
        "is_use_semantic_agent": True,
        "is_use_risk_graph": True,
        "is_use_prototype_anchor": False,
        "is_use_local_llm_agent": False,
        "is_use_reflection_memory": False,
        "is_save_risk_analysis": False,
        "is_use_risk_aware_pseudo_label": True,
        "is_use_risk_contrastive": False,
        "is_use_llm_verifier": False
    },
    "semantic_contrastive": {
        "is_use_semantic_agent": True,
        "is_use_risk_graph": True,
        "is_use_prototype_anchor": False,
        "is_use_local_llm_agent": False,
        "is_use_reflection_memory": False,
        "is_save_risk_analysis": False,
        "is_use_risk_aware_pseudo_label": True,
        "is_use_risk_contrastive": True,
        "is_use_llm_verifier": False
    },
    "rule": {
        "is_use_semantic_agent": True,
        "is_use_risk_graph": True,
        "is_use_prototype_anchor": True,
        "is_use_local_llm_agent": False,
        "is_use_reflection_memory": True,
        "is_save_risk_analysis": True,
        "is_use_risk_aware_pseudo_label": True,
        "is_use_risk_contrastive": True,
        "is_use_llm_verifier": False
    },
    "hybrid": {
        "is_use_semantic_agent": True,
        "is_use_risk_graph": True,
        "is_use_prototype_anchor": True,
        "is_use_local_llm_agent": True,
        "is_use_reflection_memory": True,
        "is_save_risk_analysis": True,
        "is_use_risk_aware_pseudo_label": True,
        "is_use_risk_contrastive": True,
        "is_use_llm_verifier": False,
        "local_llm_torch_dtype": "float32"
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_config",
        default="config/conll2003/fg_1_pg_1/RDP_semantic_agent.yaml",
        help="Shared training hyperparameter source"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--modes", nargs="+", choices=sorted(MODE_OVERRIDES),
                        default=["baseline", "anchor_only", "semantic_only", "semantic_pseudo", "semantic_contrastive", "rule", "hybrid"])
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--dry_run", action="store_true", help="Generate configs and print commands only")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path.cwd().resolve()
    base_config_path = (root / args.base_config).resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError("Base config not found: %s" % base_config_path)
    if not _is_within(root, base_config_path):
        raise ValueError("Base config must be inside the project root")

    with base_config_path.open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}

    for mode in args.modes:
        for seed in args.seeds:
            exp_name = "conll2003_ablation_%s" % mode
            run_name = "%s_seed%d" % (mode, seed)
            generated_config = root / "ablation_configs" / run_name / "config.yaml"
            cache_root = root / "semantic_cache" / "ablations" / run_name
            experiment_path = root / "experiments" / exp_name / str(seed)

            run_config = _build_config(base_config, mode, cache_root, seed)
            _write_config(generated_config, run_config)
            command = [
                sys.executable,
                "main_CL.py",
                "--exp_name", exp_name,
                "--exp_id", str(seed),
                "--seed", str(seed),
                "--cfg", str(generated_config.relative_to(root))
            ]
            print("\n[%s] %s" % (run_name, " ".join(command)))
            if args.dry_run:
                continue

            _safe_remove(root, experiment_path)
            _safe_remove(root, cache_root)
            _write_config(generated_config, run_config)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.gpu
            result = subprocess.run(command, cwd=str(root), env=env)
            if result.returncode != 0 and not args.continue_on_error:
                raise RuntimeError("Run failed: %s" % run_name)


def _build_config(base_config, mode, cache_root, seed):
    config = dict(base_config)
    config.update(MODE_OVERRIDES[mode])
    config["seed"] = int(seed)

    # Force each run to train from scratch instead of loading old checkpoints.
    config["is_load_ckpt_if_exists"] = False
    config["is_load_common_first_model"] = False

    config["semantic_cache_dir"] = str(cache_root / "semantic")
    config["risk_graph_cache_dir"] = str(cache_root / "risk_graph")
    config["prototype_memory_cache_dir"] = str(cache_root / "prototype_memory")
    config["reflection_cache_dir"] = str(cache_root / "reflection")
    config["llm_verifier_cache_dir"] = str(cache_root / "llm_verifier")
    return config


def _write_config(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _safe_remove(root, path):
    if not path.exists():
        return
    if not _is_within(root, path):
        raise ValueError("Refusing to remove path outside the project root: %s" % path)
    shutil.rmtree(path)


def _is_within(root, path):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
