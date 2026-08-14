import yaml
import argparse
import os


def _load_yaml_config(path):
    """Load a YAML config with an optional relative ``base_cfg`` parent."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_cfg = config.pop("base_cfg", None)
    if not base_cfg:
        return config
    if not os.path.isabs(base_cfg):
        base_cfg = os.path.join(os.path.dirname(path), base_cfg)
    merged = _load_yaml_config(base_cfg)
    merged.update(config)
    return merged

# ===========================================================================
# LOAD CONFIGURATION
def get_params():
    parser = argparse.ArgumentParser(description="Continual Learning for NER")


    # =========================================================================================
    # Experimental Settings
    # =========================================================================================
    # Debug
    parser.add_argument("--debug", default=False, action="store_true", help="if skipping the test on training and validation set")
    
    # Config
    parser.add_argument("--cfg", default="./config/default.yaml", help="Hyper-parameters") # 超参数配置文件

    # Path
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="train.log")
    parser.add_argument("--dump_path", type=str, default="experiments", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")
    parser.add_argument("--seed", type=int, default=None, help="Random Seed")

    # Model
    parser.add_argument("--model_name", type=str, default="bert-base-cased", help="model name (e.g., bert-base-cased, roberta-base or wide_resnet)")
    parser.add_argument("--is_load_ckpt_if_exists", default=False, action='store_true', help="Loading the ckpt if best finetuned ckpt exists")
    parser.add_argument("--is_load_common_first_model", default=False, action='store_true', help="Loading the common first ckpt if best finetuned ckpt exists")
    parser.add_argument("--ckpt", type=str, default=None, help="the pretrained lauguage model")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden layer dimension")
    parser.add_argument("--alpha", type=float, default=0, help="Trade-off parameter")
    parser.add_argument("--none_idx", type=int, default=103, help="None token index(103=[mask])")

    # Data
    parser.add_argument("--data_path", type=str, default="./datasets/NER_data/ai/", help="source domain")
    parser.add_argument("--n_samples", type=int, default=-1, help="conduct few-shot learning (10, 25, 40, 55, 70, 85, 100)")
    parser.add_argument("--entity_list", type=str, default="", help="entity list")
    parser.add_argument("--schema", type=str, default="BIO", choices=['IO','BIO','BIOES'], help="Lable schema")
    parser.add_argument("--is_filter_O", default=False, action='store_true', help="If filter out samples contains only O labels")
    parser.add_argument("--is_load_disjoin_train", default=False, action='store_true', help="If loading the join ckpt for training dat (only for CL)")

    # =========================================================================================
    # Training Settings
    # =========================================================================================
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size") 
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max length for each sentence") 
    
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate") 
    parser.add_argument("--is_train_by_steps", default=False, action='store_true', help="If the scheduer and evaluation is meausured by steps")
    parser.add_argument("--training_epochs", type=int, default=0, help="Number of training epochs")
    parser.add_argument("--first_training_epochs", type=int, default=0, help="Number of training epochs in first iteration (will be set as training_epochs by default)")
    parser.add_argument("--training_steps", type=int, default=0, help="Number of training steps")
    parser.add_argument("--first_training_steps", type=int, default=0, help="Number of training steps in first iteration (will be set as training_steps by default)")
    
    parser.add_argument("--schedule", type=str, default='(20, 40)', help="Multistep scheduler")
    parser.add_argument("--stable_lr", type=float, default=4e-4, help="Stable learning rate")
    parser.add_argument("--gamma", type=float, default=0.2, help="Factor of the learning rate decay")

    parser.add_argument("--mu", type=float, default=0.9, help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")

    parser.add_argument("--info_per_epochs", type=int, default=1, help="Print information every how many epochs")
    parser.add_argument("--info_per_steps", type=int, default=0, help="Print information every how many steps")
    parser.add_argument("--save_per_epochs", type=int, default=0, help="Save checkpoints every how many epochs")
    parser.add_argument("--save_per_steps", type=int, default=0, help="Save checkpoints every how many steps")
    parser.add_argument("--evaluate_interval", type=int, default=3, help="Evaluation interval")
    parser.add_argument("--early_stop", type=int, default=5, help="No improvement after several epoch, we stop training")
    parser.add_argument("--checkpoint_selection_metric", choices=["new_micro", "cumulative_macro", "balanced"],
                        default="new_micro", help="Validation metric used to choose the checkpoint")
    parser.add_argument("--checkpoint_selection_new_weight", type=float, default=0.5,
                        help="New-class weight when checkpoint_selection_metric is balanced")


    # =========================================================================================
    # Incremental Learning Settings
    # =========================================================================================
    parser.add_argument("--nb_class_pg", type=int, default=2, help="number of classes in each group")
    parser.add_argument("--nb_class_fg", type=int, default=4, help="number of classes in the first group")
    
    parser.add_argument("--is_rescale_new_weight", default=False, action='store_true', help="If rescale the new weight matrix")
    parser.add_argument("--is_fix_trained_classifier", default=False, action='store_true', help="If fix the trained classifer")
    parser.add_argument("--is_unfix_O_classifier", default=False, action='store_true', help="If not fix the O classifer")
    
    parser.add_argument("--is_MTL", default=False, action='store_true', help="If using multi-task learning")
    parser.add_argument("--extra_annotate_type", type=str, default='none', choices=['none','current','all'] , help="Simulate mannual annotation in each data split")
    parser.add_argument("--is_from_scratch", default=False, action='store_true', help="If training from scratch for multi-task learning")

    parser.add_argument("--reserved_ratio", type=float, default=0, help="the ratio of reserved samples")

    # =========================================================================================
    # Baseline Settings
    # =========================================================================================
    # RDP
    # pseudo label
    parser.add_argument('--proto_temperature', type=float, default=1.0)

    # soft_sharp soft label
    parser.add_argument("--soft_param", default=1, type=float)
    parser.add_argument("--regular_param", default=1., type=float)

    # logits KD 
    parser.add_argument("--distill_logits_weight", type=float, default=1, help="distillation logits weight for loss")
    parser.add_argument("--temperature", type=int, default=1, help="temperature of the student model")
    parser.add_argument("--ref_temperature", type=int, default=1, help="temperature of the teacher model")

    # =========================================================================================
    # Semantic Agent Settings
    # =========================================================================================
    parser.add_argument("--is_use_semantic_agent", default=False, action="store_true",
                        help="If using rule-based semantic agent to guide continual training")
    parser.add_argument("--semantic_memory_path", type=str, default="semantic_memory/conll2003/entity_types.json",
                        help="Path to semantic entity memory")
    parser.add_argument("--semantic_distill_base_weight", type=float, default=1.0,
                        help="Minimum class weight for an old label selected as high risk")
    parser.add_argument("--semantic_risk_threshold", type=float, default=0.5,
                        help="Only old labels at or above this risk receive semantic protection")
    parser.add_argument("--semantic_max_weight", type=float, default=1.6,
                        help="Maximum semantic weight assigned to a selected old label")
    parser.add_argument("--is_use_model_risk", default=False, action="store_true",
                        help="Use teacher confusion on the development set as a semantic risk signal")
    parser.add_argument("--model_confusion_risk_scale", type=float, default=2.0,
                        help="Multiplier that converts teacher new-to-old confusion rate into risk")
    parser.add_argument("--is_collect_prototype_similarity", default=False, action="store_true",
                        help="Record frozen-encoder new-to-old prototype similarity on the development set")
    parser.add_argument("--is_use_instance_context_risk", default=False, action="store_true",
                        help="Provide gold-labelled development contexts to the local LLM risk scorer")
    parser.add_argument("--instance_context_examples_per_type", type=int, default=2,
                        help="Maximum development contexts retained per entity type for LLM risk scoring")
    parser.add_argument("--is_use_risk_calibrator", default=False, action="store_true",
                        help="Calibrate risk edges from historical observed interference")
    parser.add_argument("--risk_calibrator_history_dir", type=str, default="",
                        help="Reflection directory used as calibrator history; defaults to current reflection cache")
    parser.add_argument("--risk_calibrator_min_tasks", type=int, default=3)
    parser.add_argument("--risk_calibrator_min_edges", type=int, default=30)
    parser.add_argument("--risk_calibrator_ridge_alpha", type=float, default=1.0)
    parser.add_argument("--risk_calibrator_risk_scale", type=float, default=3.0)
    parser.add_argument("--is_use_risk_aware_pseudo_label", default=False, action="store_true",
                        help="Reject low-confidence pseudo labels for high-risk old labels")
    parser.add_argument("--pseudo_label_min_confidence", type=float, default=0.55,
                        help="Minimum confidence required for a selected high-risk pseudo label")
    parser.add_argument("--pseudo_label_max_confidence", type=float, default=0.80,
                        help="Confidence required when the pseudo-label risk is maximal")
    parser.add_argument("--risk_pseudo_label_fallback_confidence", type=float, default=1.0,
                        help="Below this confidence a risk-filtered pseudo label is dropped; higher-confidence candidates remain available for verification")
    parser.add_argument("--is_use_risk_contrastive", default=False, action="store_true",
                        help="Separate high-risk new and old entity representations")
    parser.add_argument("--risk_contrastive_weight", type=float, default=0.1,
                        help="Global coefficient for the risk-guided contrastive loss")
    parser.add_argument("--risk_contrastive_margin", type=float, default=0.2,
                        help="Cosine-similarity margin between new features and old prototypes")
    parser.add_argument("--is_use_risk_graph", default=False, action="store_true",
                        help="If building and saving task-level risk graphs")
    parser.add_argument("--risk_graph_cache_dir", type=str, default="semantic_cache/risk_graph",
                        help="Directory for cached task-level risk graphs")
    parser.add_argument("--is_save_risk_graph_plot", default=False, action="store_true",
                        help="If saving task-level risk graph plots to the experiment directory")
    parser.add_argument("--is_use_reflection_memory", default=False, action="store_true",
                        help="If saving task-end forgetting and risk-validation reflections")
    parser.add_argument("--reflection_cache_dir", type=str, default="semantic_cache/reflection",
                        help="Directory for task-end reflection memory")
    parser.add_argument("--reflection_risk_threshold", type=float, default=0.5,
                        help="Risk at or above which an edge is predicted high risk")
    parser.add_argument("--reflection_forgetting_threshold", type=float, default=1.0,
                        help="Old-class F1 drop at or above which forgetting is high")
    parser.add_argument("--rule_risk_weight", type=float, default=0.8,
                        help="Weight of rule risk in the initial hybrid risk")
    parser.add_argument("--llm_risk_weight", type=float, default=0.2,
                        help="Weight of local LLM risk in the initial hybrid risk")
    parser.add_argument("--llm_risk_min_weight", type=float, default=0.05,
                        help="Lower bound for calibrated local-LLM risk weight")
    parser.add_argument("--llm_reliability_min_tasks", type=int, default=2,
                        help="Completed tasks required before calibrating LLM influence")
    parser.add_argument("--llm_reliability_floor", type=float, default=0.25,
                        help="Minimum historical reliability multiplier for LLM risk")
    parser.add_argument("--semantic_prior_max_delta", type=float, default=0.15,
                        help="Maximum amount the LLM semantic prior can raise one edge risk")
    parser.add_argument("--reflection_update_weight", type=float, default=0.3,
                        help="Weight of historical reflection in the final risk")
    parser.add_argument("--reflection_forgetting_weight", type=float, default=0.6,
                        help="Weight of normalized old-class forgetting in empirical risk")
    parser.add_argument("--reflection_confusion_weight", type=float, default=0.4,
                        help="Weight of normalized new-to-old confusion in empirical risk")
    parser.add_argument("--reflection_forgetting_scale", type=float, default=10.0,
                        help="F1 drop in points corresponding to maximum empirical forgetting risk")
    parser.add_argument("--is_use_old_class_vulnerability", default=False, action="store_true",
                        help="Raise next-task protection for historically fragile old entity types")
    parser.add_argument("--old_class_vulnerability_f1_floor", type=float, default=80.0,
                        help="Cumulative per-class F1 below this value contributes vulnerability")
    parser.add_argument("--old_class_vulnerability_weight", type=float, default=0.3,
                        help="Maximum fraction of remaining risk range added by old-class vulnerability")
    parser.add_argument("--is_save_risk_analysis", default=False, action="store_true",
                        help="If saving risk-versus-forgetting CSV, plots, and metrics")
    parser.add_argument("--prototype_memory_cache_dir", type=str, default="semantic_cache/prototype_memory",
                        help="Directory for cached task-level prototype memory")
    parser.add_argument("--is_use_prototype_anchor", default=False, action="store_true",
                        help="If using prototype anchor loss for old labels")
    parser.add_argument("--prototype_anchor_weight", type=float, default=0.5,
                        help="Global coefficient for prototype anchor loss")
    parser.add_argument("--is_use_local_llm_agent", default=False, action="store_true",
                        help="If using local LLM to generate semantic training plans")
    parser.add_argument("--local_llm_model_path", type=str, default="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct",
                        help="Path to local instruction LLM")
    parser.add_argument("--semantic_cache_dir", type=str, default="semantic_cache",
                        help="Directory for cached semantic plans")
    parser.add_argument("--local_llm_max_new_tokens", type=int, default=512,
                        help="Max new tokens for local LLM semantic plan generation")
    parser.add_argument("--local_llm_temperature", type=float, default=0.0,
                        help="Sampling temperature for local LLM semantic plan generation")
    parser.add_argument("--local_llm_torch_dtype", choices=["float16", "bfloat16", "float32", "auto"],
                        default="float32", help="Torch dtype for the local LLM")
    parser.add_argument("--is_use_llm_verifier", default=False, action="store_true",
                        help="If using local LLM to verify high-risk pseudo labels")
    parser.add_argument("--llm_verifier_model_path", type=str, default="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct",
                        help="Path to local instruction LLM for pseudo-label verification")
    parser.add_argument("--llm_verifier_cache_dir", type=str, default="semantic_cache/llm_verifier",
                        help="Directory for cached LLM verifier decisions")
    parser.add_argument("--llm_verifier_max_new_tokens", type=int, default=256,
                        help="Max new tokens for local LLM verifier output")
    parser.add_argument("--llm_verifier_temperature", type=float, default=0.0,
                        help="Sampling temperature for local LLM verifier")
    parser.add_argument("--llm_verifier_torch_dtype", choices=["float16", "bfloat16", "float32", "auto"],
                        default="float32", help="Torch dtype for the local LLM verifier")
    parser.add_argument("--llm_verifier_max_checks_per_batch", type=int, default=2,
                        help="Safety cap for LLM checks in one batch")
    parser.add_argument("--llm_verifier_task_budget", type=int, default=100,
                        help="Maximum LLM verification decisions for one incremental task")
    parser.add_argument("--llm_verifier_min_risk", type=float, default=0.7,
                        help="Only schedule labels whose semantic risk reaches this value")

    params = parser.parse_args() # 默认配置

    with open(params.cfg) as f: # 读取yaml文件 进行配置覆盖
        config = _load_yaml_config(params.cfg)
        for k, v in config.items():
            # for parameters set in the args
            if k in ['none_idx']:
                continue
            params.__setattr__(k,v)

    return params
