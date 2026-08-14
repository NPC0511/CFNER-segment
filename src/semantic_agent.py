import json
import os

from src.local_llm_agent import LocalLLMAgent
from src.reflection_memory import ReflectionMemory
from src.risk_calibrator import RiskEdgeCalibrator
from src.risk_graph import RiskGraph
from src.semantic_retriever import SemanticRetriever


class SemanticAgent(object):
    """Semantic training controller backed by local retrieved evidence."""

    def __init__(self, memory_path, enabled=False, base_weight=1.0,
                 use_local_llm=False, local_llm_model_path="",
                 cache_dir="semantic_cache", llm_max_new_tokens=512,
                 llm_temperature=0.0, local_llm_torch_dtype="float32",
                 risk_threshold=0.5, max_weight=1.6,
                 model_confusion_risk_scale=2.0,
                 pseudo_label_min_confidence=0.55,
                 pseudo_label_max_confidence=0.80,
                 reflection_cache_dir="semantic_cache/reflection",
                 rule_risk_weight=0.6, llm_risk_weight=0.4,
                 reflection_update_weight=0.3,
                 reflection_forgetting_weight=0.6,
                 reflection_confusion_weight=0.4,
                 reflection_forgetting_scale=10.0,
                 use_old_class_vulnerability=False,
                 old_class_vulnerability_weight=0.3,
                 llm_risk_min_weight=0.05,
                 llm_reliability_min_tasks=2,
                 llm_reliability_floor=0.25,
                 semantic_prior_max_delta=0.15,
                 use_risk_calibrator=False, risk_calibrator_history_dir="",
                 risk_calibrator_min_tasks=3, risk_calibrator_min_edges=30,
                 risk_calibrator_ridge_alpha=1.0, risk_calibrator_risk_scale=3.0):
        self.memory_path = memory_path
        self.enabled = bool(enabled)
        self.base_weight = float(base_weight)
        self.use_local_llm = bool(use_local_llm)
        self.local_llm_model_path = local_llm_model_path
        self.cache_dir = cache_dir
        self.llm_max_new_tokens = int(llm_max_new_tokens)
        self.llm_temperature = float(llm_temperature)
        self.local_llm_torch_dtype = local_llm_torch_dtype
        self.risk_threshold = min(max(float(risk_threshold), 0.0), 1.0)
        self.max_weight = max(float(max_weight), 1.0)
        self.model_confusion_risk_scale = max(float(model_confusion_risk_scale), 0.0)
        self.pseudo_label_min_confidence = min(max(float(pseudo_label_min_confidence), 0.0), 1.0)
        self.pseudo_label_max_confidence = min(
            max(float(pseudo_label_max_confidence), self.pseudo_label_min_confidence), 1.0
        )
        self.reflection_memory = ReflectionMemory(reflection_cache_dir)
        self.rule_risk_weight = float(rule_risk_weight)
        self.llm_risk_weight = float(llm_risk_weight)
        self.reflection_update_weight = float(reflection_update_weight)
        self.reflection_forgetting_weight = float(reflection_forgetting_weight)
        self.reflection_confusion_weight = float(reflection_confusion_weight)
        self.reflection_forgetting_scale = float(reflection_forgetting_scale)
        self.use_old_class_vulnerability = bool(use_old_class_vulnerability)
        self.old_class_vulnerability_weight = min(max(float(old_class_vulnerability_weight), 0.0), 1.0)
        self.llm_risk_min_weight = max(float(llm_risk_min_weight), 0.0)
        self.llm_reliability_min_tasks = max(int(llm_reliability_min_tasks), 1)
        self.llm_reliability_floor = min(max(float(llm_reliability_floor), 0.0), 1.0)
        self.semantic_prior_max_delta = min(max(float(semantic_prior_max_delta), 0.0), 1.0)
        self.risk_calibrator = None
        if use_risk_calibrator:
            self.risk_calibrator = RiskEdgeCalibrator(
                reflection_cache_dir=risk_calibrator_history_dir or reflection_cache_dir,
                min_tasks=risk_calibrator_min_tasks,
                min_edges=risk_calibrator_min_edges,
                ridge_alpha=risk_calibrator_ridge_alpha,
                risk_scale=risk_calibrator_risk_scale
            )
        self.memory = self._load_memory(memory_path) if self.enabled else {}
        self.retriever = SemanticRetriever(memory_path) if self.enabled else None

    def _load_memory(self, memory_path):
        if not memory_path:
            return {}
        if not os.path.isfile(memory_path):
            raise FileNotFoundError("Semantic memory file not found: %s" % memory_path)
        with open(memory_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_plan(self, domain_name, new_entity_list, all_seen_entity_list, schema="BIO", iteration=None,
                   model_confusion_pairs=None):
        old_entity_list = [e for e in all_seen_entity_list if e not in new_entity_list]
        model_confusion_pairs = list(model_confusion_pairs or [])
        plan = {
            "enabled": False,
            "domain_name": domain_name,
            "schema": schema,
            "new_entity_list": list(new_entity_list),
            "old_entity_list": old_entity_list,
            "important_old_types": [],
            "old_type_weights": {},
            "old_label_weights": {},
            "risk_graph": {},
            "risk_edges": [],
            "training_policy": {
                "old_type_weights": {},
                "old_label_weights": {},
                "prototype_anchor_weights": {},
                "pseudo_label_thresholds": {},
                "contrastive_pairs": [],
                "verifier_label_risks": {},
                "old_type_vulnerability": {},
                "plan_source": "disabled"
            },
            "retrieved_evidence": {},
            "model_confusion_pairs": model_confusion_pairs,
            "plan_source": "disabled",
            "llm_raw_output": "",
            "llm_error": ""
        }
        if (not self.enabled) or len(old_entity_list) == 0:
            return plan

        retrieved_evidence = self.retriever.retrieve(
            domain_name=domain_name,
            new_entity_list=new_entity_list,
            old_entity_list=old_entity_list
        )
        rule_plan = self._build_rule_plan(
            plan=plan,
            retrieved_evidence=retrieved_evidence,
            old_entity_list=old_entity_list,
            schema=schema,
            plan_source="rule_rag",
            iteration=iteration,
            model_confusion_pairs=model_confusion_pairs
        )
        if not self.use_local_llm:
            return rule_plan

        cached_plan = self._load_cached_plan(
            domain_name=domain_name,
            new_entity_list=new_entity_list,
            old_entity_list=old_entity_list,
            schema=schema,
            iteration=iteration
        )
        if cached_plan is not None and not model_confusion_pairs:
            self._ensure_verifier_policy(cached_plan, schema)
            cached_plan["plan_source"] = "cache"
            return cached_plan

        llm_evidence = dict(retrieved_evidence)
        # The LLM estimates interference from semantic memory, not copied rule scores.
        llm_evidence["confusion_evidence"] = []
        llm_result = LocalLLMAgent(
            model_path=self.local_llm_model_path,
            max_new_tokens=self.llm_max_new_tokens,
            temperature=self.llm_temperature,
            torch_dtype=self.local_llm_torch_dtype
        ).generate_risk_graph(
            retrieved_evidence=llm_evidence,
            new_entity_list=new_entity_list,
            old_entity_list=old_entity_list
        )

        if llm_result.get("ok", False):
            try:
                llm_plan = self._build_llm_plan(
                    base_plan=plan,
                    retrieved_evidence=retrieved_evidence,
                    llm_output=llm_result.get("risk_graph", {}),
                    raw_output=llm_result.get("raw_output", ""),
                    old_entity_list=old_entity_list,
                    schema=schema,
                    iteration=iteration,
                    model_confusion_pairs=model_confusion_pairs
                )
                self._save_cached_plan(
                    llm_plan,
                    domain_name=domain_name,
                    new_entity_list=new_entity_list,
                    old_entity_list=old_entity_list,
                    schema=schema,
                    iteration=iteration
                )
                return llm_plan
            except Exception as exc:
                rule_plan["plan_source"] = "rule_fallback"
                rule_plan["llm_raw_output"] = llm_result.get("raw_output", "")
                rule_plan["llm_error"] = str(exc)
                self._save_cached_plan(
                    rule_plan,
                    domain_name=domain_name,
                    new_entity_list=new_entity_list,
                    old_entity_list=old_entity_list,
                    schema=schema,
                    iteration=iteration
                )
                return rule_plan

        rule_plan["plan_source"] = "rule_fallback"
        rule_plan["llm_raw_output"] = llm_result.get("raw_output", "")
        rule_plan["llm_error"] = llm_result.get("error", "")
        self._save_cached_plan(
            rule_plan,
            domain_name=domain_name,
            new_entity_list=new_entity_list,
            old_entity_list=old_entity_list,
            schema=schema,
            iteration=iteration
        )
        return rule_plan

    def _build_rule_plan(self, plan, retrieved_evidence, old_entity_list, schema, plan_source, iteration,
                         model_confusion_pairs):
        risk_edges = self._build_rule_risk_edges(
            retrieved_evidence=retrieved_evidence,
            new_entity_list=plan["new_entity_list"],
            old_entity_list=old_entity_list
        )
        risk_edges = self._apply_reflection_updates(
            risk_edges=risk_edges,
            domain_name=plan["domain_name"],
            current_iteration=iteration
        )
        risk_edges = self._apply_model_confusion_updates(risk_edges, model_confusion_pairs)
        important_old_types = self._risk_edges_to_type_weights(risk_edges)

        old_label_weights = {}
        for old_entity, weight in sorted(important_old_types.items()):
            for label_name in self._entity_to_labels(old_entity, schema):
                old_label_weights[label_name] = weight
        pseudo_label_thresholds = self._risk_edges_to_pseudo_label_thresholds(risk_edges, schema)
        contrastive_pairs = self._risk_edges_to_contrastive_pairs(risk_edges)
        verifier_label_risks = self._risk_edges_to_verifier_label_risks(risk_edges, schema)
        old_type_vulnerability = self._risk_edges_to_old_type_vulnerability(risk_edges)

        training_policy = {
            "old_type_weights": important_old_types,
            "old_label_weights": old_label_weights,
            "prototype_anchor_weights": dict(old_label_weights),
            "pseudo_label_thresholds": pseudo_label_thresholds,
            "contrastive_pairs": contrastive_pairs,
            "verifier_label_risks": verifier_label_risks,
            "old_type_vulnerability": old_type_vulnerability,
            "plan_source": plan_source
        }
        risk_graph = self._build_risk_graph(
            domain_name=plan["domain_name"],
            new_entity_list=plan["new_entity_list"],
            old_entity_list=old_entity_list,
            risk_edges=risk_edges,
            retrieved_evidence=retrieved_evidence,
            training_policy=training_policy,
            iteration=None
        )

        plan["enabled"] = len(old_label_weights) > 0
        plan["important_old_types"] = sorted(important_old_types.keys())
        plan["old_type_weights"] = important_old_types
        plan["old_label_weights"] = old_label_weights
        plan["risk_graph"] = risk_graph
        plan["risk_edges"] = risk_edges
        plan["training_policy"] = training_policy
        plan["retrieved_evidence"] = retrieved_evidence
        plan["plan_source"] = plan_source
        return plan

    def _build_llm_plan(self, base_plan, retrieved_evidence, llm_output, raw_output,
                        old_entity_list, schema, iteration, model_confusion_pairs):
        llm_edges = list(llm_output.get("risk_edges", []))
        expected_pairs = {
            (new_entity, old_entity)
            for new_entity in base_plan["new_entity_list"]
            for old_entity in old_entity_list
        }
        rule_edges = self._build_rule_risk_edges(
            retrieved_evidence=retrieved_evidence,
            new_entity_list=base_plan["new_entity_list"],
            old_entity_list=old_entity_list
        )
        llm_by_pair = {(edge.get("source"), edge.get("target")): edge for edge in llm_edges}
        rule_by_pair = {(edge.get("source"), edge.get("target")): edge for edge in rule_edges}
        missing_pairs = expected_pairs - set(llm_by_pair)
        risk_edges = self._merge_rule_and_llm_risks(rule_edges, llm_edges)
        for pair in sorted(missing_pairs):
            rule_edge = dict(rule_by_pair[pair])
            rule_edge["initial_risk"] = float(rule_edge["risk"])
            rule_edge["llm_risk"] = None
            rule_edge["risk_type"] = list(rule_edge.get("risk_type", [])) + ["llm_batch_fallback"]
            rule_edge["reason"] = "%s LLM batch fallback." % rule_edge.get("reason", "")
            risk_edges.append(rule_edge)
        risk_edges.sort(key=lambda edge: (edge["source"], edge["target"]))
        risk_edges = self._apply_reflection_updates(
            risk_edges=risk_edges,
            domain_name=base_plan["domain_name"],
            current_iteration=iteration
        )
        risk_edges = self._apply_model_confusion_updates(risk_edges, model_confusion_pairs)
        old_type_weights = self._risk_edges_to_type_weights(risk_edges)
        important_old_types = sorted(old_type_weights.keys())

        old_label_weights = {}
        for old_entity, weight in sorted(old_type_weights.items()):
            for label_name in self._entity_to_labels(old_entity, schema):
                old_label_weights[label_name] = weight
        pseudo_label_thresholds = self._risk_edges_to_pseudo_label_thresholds(risk_edges, schema)
        contrastive_pairs = self._risk_edges_to_contrastive_pairs(risk_edges)
        verifier_label_risks = self._risk_edges_to_verifier_label_risks(risk_edges, schema)
        old_type_vulnerability = self._risk_edges_to_old_type_vulnerability(risk_edges)

        training_policy = {
            "old_type_weights": old_type_weights,
            "old_label_weights": old_label_weights,
            "prototype_anchor_weights": dict(old_label_weights),
            "pseudo_label_thresholds": pseudo_label_thresholds,
            "contrastive_pairs": contrastive_pairs,
            "verifier_label_risks": verifier_label_risks,
            "old_type_vulnerability": old_type_vulnerability,
            "plan_source": "local_llm" if not missing_pairs else "local_llm_partial"
        }
        plan = dict(base_plan)
        plan["enabled"] = len(old_label_weights) > 0
        plan["important_old_types"] = important_old_types
        plan["old_type_weights"] = old_type_weights
        plan["old_label_weights"] = old_label_weights
        plan["risk_edges"] = risk_edges
        plan["training_policy"] = training_policy
        plan["risk_graph"] = self._build_risk_graph(
            domain_name=base_plan["domain_name"],
            new_entity_list=base_plan["new_entity_list"],
            old_entity_list=old_entity_list,
            risk_edges=risk_edges,
            retrieved_evidence=retrieved_evidence,
            training_policy=training_policy,
            iteration=None
        )
        plan["retrieved_evidence"] = retrieved_evidence
        plan["plan_source"] = training_policy["plan_source"]
        plan["llm_raw_output"] = raw_output
        plan["llm_error"] = ""
        plan["llm_reason"] = "Pairwise semantic risk graph generated by local LLM."
        plan["llm_calibration"] = calibration
        plan["llm_missing_pairs"] = [
            {"source": source, "target": target}
            for source, target in sorted(missing_pairs)
        ]
        return plan

    def _build_rule_risk_edges(self, retrieved_evidence, new_entity_list, old_entity_list):
        """Create one risk edge for every new-old type pair.

        Explicit annotation confusion rules override the low background risk used
        for pairs for which the rule memory has no direct evidence.
        """
        evidence_by_pair = {}
        for evidence in retrieved_evidence.get("confusion_evidence", []):
            pair = (evidence.get("source_type"), evidence.get("target_type"))
            if pair not in evidence_by_pair:
                evidence_by_pair[pair] = evidence

        risk_edges = []
        for new_entity in new_entity_list:
            for old_entity in old_entity_list:
                evidence = evidence_by_pair.get((new_entity, old_entity))
                if evidence is None:
                    risk_edges.append({
                        "source": new_entity,
                        "target": old_entity,
                        "risk": 0.10,
                        "risk_type": ["background_risk"],
                        "reason": "No direct confusion rule was found; retain a low background risk.",
                        "evidence_ids": []
                    })
                    continue
                risk_edges.append({
                    "source": new_entity,
                    "target": old_entity,
                    "risk": self._map_rule_weight_to_risk(float(evidence.get("weight", 1.0))),
                    "risk_type": ["rule_confusion"],
                    "reason": evidence.get("reason", ""),
                    "evidence_ids": [evidence.get("id", "%s_to_%s" % (new_entity, old_entity))]
                })
        return risk_edges

    def _build_risk_evidence_cards(self, domain_name, new_entity_list, old_entity_list,
                                   model_confusion_pairs, prototype_similarity_pairs,
                                   current_iteration):
        """Summarize only pre-training evidence for each LLM risk decision."""
        teacher_by_pair = {
            (item.get("source"), item.get("target")): item
            for item in model_confusion_pairs
        }
        prototype_by_pair = {
            (item.get("source"), item.get("target")): item
            for item in prototype_similarity_pairs
        }
        previous = self.reflection_memory.load_latest(domain_name) or {}
        if current_iteration is not None and int(previous.get("task_id", -1)) >= int(current_iteration):
            previous = {}
        previous_forgetting = previous.get("observed_forgetting", {})
        previous_vulnerability = previous.get("old_class_vulnerability", {})
        previous_interference = {}
        for item in previous.get("observed_confusion", []):
            target = item.get("target")
            if target:
                previous_interference[target] = max(
                    previous_interference.get(target, 0.0), float(item.get("rate", 0.0))
                )

        cards = {}
        for source in new_entity_list:
            for target in old_entity_list:
                teacher = teacher_by_pair.get((source, target), {})
                prototype = prototype_by_pair.get((source, target), {})
                cards["%s->%s" % (source, target)] = {
                    "teacher_new_to_old_rate": round(float(teacher.get("rate", 0.0)), 6),
                    "teacher_new_to_old_count": int(teacher.get("count", 0)),
                    "teacher_source_token_count": int(teacher.get("source_token_count", 0)),
                    "prototype_cosine_similarity": round(float(prototype.get("cosine_similarity", 0.0)), 6),
                    "prototype_source_token_count": int(prototype.get("source_token_count", 0)),
                    "prototype_target_token_count": int(prototype.get("target_token_count", 0)),
                    "previous_old_forgetting_f1": round(float(previous_forgetting.get(target, 0.0)), 4),
                    "previous_old_vulnerability": round(float(previous_vulnerability.get(target, 0.0)), 4),
                    "previous_old_to_new_interference": round(float(previous_interference.get(target, 0.0)), 6)
                }
        return cards

    def _merge_rule_and_llm_risks(self, rule_edges, llm_edges, calibration=None):
        rule_by_pair = {(edge["source"], edge["target"]): edge for edge in rule_edges}
        merged_edges = []
        total_weight = self.rule_risk_weight + self.llm_risk_weight
        if total_weight <= 0:
            raise ValueError("At least one initial risk weight must be positive")
        for llm_edge in llm_edges:
            pair = (llm_edge["source"], llm_edge["target"])
            rule_edge = rule_by_pair.get(pair)
            if rule_edge is None:
                raise ValueError("Rule risk is missing for %s -> %s" % pair)
            rule_risk = float(rule_edge["risk"])
            llm_risk = float(llm_edge["risk"])
            # Restore the original hybrid controller: semantic and rule evidence
            # jointly determine the edge before any task-end reflection update.
            initial_risk = (
                self.rule_risk_weight * rule_risk +
                self.llm_risk_weight * llm_risk
            ) / total_weight
            merged_edge = dict(llm_edge)
            merged_edge["risk"] = round(initial_risk, 4)
            merged_edge["initial_risk"] = round(initial_risk, 4)
            merged_edge["rule_risk"] = rule_risk
            merged_edge["llm_risk"] = llm_risk
            merged_edge["semantic_candidate"] = bool(llm_risk >= 0.525)
            merged_edge["semantic_prior_delta"] = round(initial_risk - rule_risk, 4)
            merged_edge["risk_type"] = sorted(set(
                list(rule_edge.get("risk_type", [])) + list(llm_edge.get("risk_type", [])) + ["hybrid_rule_llm"]
            ))
            merged_edge["evidence_ids"] = list(rule_edge.get("evidence_ids", [])) + list(llm_edge.get("evidence_ids", []))
            merged_edge["reason"] = "LLM: %s Rule: %s" % (
                llm_edge.get("reason", ""), rule_edge.get("reason", "")
            )
            merged_edges.append(merged_edge)
        return merged_edges

    def _apply_risk_calibration(self, risk_edges, domain_name, iteration):
        if self.risk_calibrator is None:
            return risk_edges, {"mode": "disabled"}
        return self.risk_calibrator.apply(risk_edges, domain_name, iteration)

    def _get_llm_calibration(self, domain_name, current_iteration):
        """Calibrate LLM influence from strictly earlier completed tasks."""
        history = self.reflection_memory.load_history(domain_name, current_iteration)
        rankable = []
        top1_hits = []
        top3_hits = []
        for reflection in history:
            forgetting = reflection.get("observed_forgetting", {})
            # A task can introduce several new types at once, while the
            # task-end F1 drop is observed only per old target type.  Aggregate
            # source edges by target before judging the semantic prior; doing
            # otherwise would incorrectly assign one task-level drop to every
            # new source type.
            by_target = {}
            for edge in reflection.get("risk_edges", []):
                if edge.get("llm_risk") is None or edge.get("target") not in forgetting:
                    continue
                target = edge["target"]
                by_target[target] = max(by_target.get(target, 0.0), float(edge["llm_risk"]))
            if len(by_target) < 2:
                continue
            rankable.append(by_target)
            observed = max(by_target, key=lambda target: float(forgetting[target]))
            predicted = sorted(by_target, key=by_target.get, reverse=True)
            top1_hits.append(float(observed == predicted[0]))
            top3_hits.append(float(observed in set(predicted[:3])))

        configured = max(self.llm_risk_weight, 0.0)
        if len(rankable) < self.llm_reliability_min_tasks:
            return {
                "mode": "warmup", "history_tasks": len(rankable),
                "top_1": None, "top_3": None,
                "configured_llm_weight": configured,
                "effective_llm_weight": configured
            }
        top1 = sum(top1_hits) / float(len(top1_hits))
        top3 = sum(top3_hits) / float(len(top3_hits))
        # Top-3 is included because the semantic graph is a candidate generator,
        # while Top-1 measures its ability to drive a precise intervention.
        reliability = max(self.llm_reliability_floor, 0.5 * top1 + 0.5 * top3)
        effective = max(self.llm_risk_min_weight, configured * reliability)
        return {
            "mode": "history_calibrated", "history_tasks": len(rankable),
            "top_1": round(top1, 4), "top_3": round(top3, 4),
            "reliability": round(reliability, 4),
            "configured_llm_weight": configured,
            "effective_llm_weight": round(effective, 4)
        }

    def _apply_reflection_updates(self, risk_edges, domain_name, current_iteration):
        reflection = self.reflection_memory.load_latest(domain_name)
        if not reflection:
            return risk_edges
        reflection_task_id = int(reflection.get("task_id", -1))
        if current_iteration is None or reflection_task_id >= int(current_iteration):
            return risk_edges

        forgetting = reflection.get("observed_forgetting", {})
        vulnerability = reflection.get("old_class_vulnerability", {})
        confusion_rates = {}
        for item in reflection.get("observed_confusion", []):
            target = item.get("target")
            if target:
                confusion_rates[target] = max(confusion_rates.get(target, 0.0), float(item.get("rate", 0.0)))
        updated_edges = []
        for edge in risk_edges:
            target = edge.get("target")
            if target not in forgetting and target not in confusion_rates and target not in vulnerability:
                updated_edges.append(edge)
                continue
            initial_risk = float(edge.get("initial_risk", edge.get("risk", 0.0)))
            forgetting_scale = max(self.reflection_forgetting_scale, 1e-8)
            forgetting_score = min(max(float(forgetting.get(target, 0.0)) / forgetting_scale, 0.0), 1.0)
            confusion_score = min(max(confusion_rates.get(target, 0.0), 0.0), 1.0)
            empirical_weight = self.reflection_forgetting_weight + self.reflection_confusion_weight
            if empirical_weight <= 0:
                updated_edges.append(edge)
                continue
            empirical_risk = (
                self.reflection_forgetting_weight * forgetting_score +
                self.reflection_confusion_weight * confusion_score
            ) / empirical_weight
            blended_risk = (
                (1.0 - self.reflection_update_weight) * initial_risk +
                self.reflection_update_weight * empirical_risk
            )
            # A validated forgetting signal should strengthen a weak semantic
            # prior, never dilute it.  This is especially important for broad
            # labels such as ``misc`` which may lack a hand-written rule.
            final_risk = max(initial_risk, blended_risk, empirical_risk)
            updated_edge = dict(edge)
            updated_edge["initial_risk"] = round(initial_risk, 4)
            updated_edge["risk"] = round(min(max(final_risk, 0.0), 1.0), 4)
            updated_edge["reflection_update"] = {
                "source_task_id": reflection_task_id,
                "forgetting_score": round(forgetting_score, 4),
                "confusion_score": round(confusion_score, 4),
                "empirical_risk": round(empirical_risk, 4),
                "update_weight": self.reflection_update_weight
            }
            vulnerability_score = min(max(float(vulnerability.get(target, 0.0)), 0.0), 1.0)
            if self.use_old_class_vulnerability and vulnerability_score > 0:
                vulnerable_risk = initial_risk + (1.0 - initial_risk) * (
                    self.old_class_vulnerability_weight * vulnerability_score
                )
                updated_edge["risk"] = round(min(max(max(final_risk, vulnerable_risk), 0.0), 1.0), 4)
                updated_edge["vulnerability_update"] = {
                    "score": round(vulnerability_score, 4),
                    "weight": self.old_class_vulnerability_weight,
                    "source_task_id": reflection_task_id
                }
            updated_edges.append(updated_edge)
        return updated_edges

    def _apply_model_confusion_updates(self, risk_edges, model_confusion_pairs):
        """Raise risk when the old teacher mistakes a new type for an old type."""
        confusion_by_pair = {}
        for item in model_confusion_pairs:
            source = item.get("source")
            target = item.get("target")
            if source and target:
                confusion_by_pair[(source, target)] = item

        updated_edges = []
        for edge in risk_edges:
            confusion = confusion_by_pair.get((edge.get("source"), edge.get("target")))
            if confusion is None:
                updated_edges.append(edge)
                continue
            rate = min(max(float(confusion.get("rate", 0.0)), 0.0), 1.0)
            model_risk = min(rate * self.model_confusion_risk_scale, 1.0)
            updated_edge = dict(edge)
            updated_edge["model_confusion"] = {
                "rate": round(rate, 6),
                "count": int(confusion.get("count", 0)),
                "source_token_count": int(confusion.get("source_token_count", 0)),
                "risk": round(model_risk, 4)
            }
            if model_risk > float(edge.get("risk", 0.0)):
                updated_edge["risk"] = round(model_risk, 4)
                updated_edge["risk_type"] = sorted(set(
                    list(edge.get("risk_type", [])) + ["teacher_new_to_old_confusion"]
                ))
            updated_edges.append(updated_edge)
        return updated_edges

    def _attach_prototype_similarity(self, risk_edges, prototype_similarity_pairs):
        """Attach frozen-encoder contextual evidence without changing risk."""
        similarity_by_pair = {
            (item.get("source"), item.get("target")): item
            for item in prototype_similarity_pairs
            if item.get("source") and item.get("target")
        }
        updated_edges = []
        for edge in risk_edges:
            similarity = similarity_by_pair.get((edge.get("source"), edge.get("target")))
            if similarity is None:
                updated_edges.append(edge)
                continue
            updated_edge = dict(edge)
            updated_edge["prototype_similarity"] = dict(similarity)
            updated_edges.append(updated_edge)
        return updated_edges

    def _build_llm_risk_edges(self, llm_output, old_type_weights):
        risk_edges = []
        source_name = ",".join(llm_output.get("important_old_types", [])) or "llm_plan"
        for idx, old_entity in enumerate(sorted(old_type_weights.keys())):
            risk_edges.append({
                "source": source_name,
                "target": old_entity,
                "risk": min(max(float(old_type_weights[old_entity]) / 2.0, 0.0), 1.0),
                "risk_type": ["llm_semantic_risk"],
                "reason": llm_output.get("reason", ""),
                "evidence_ids": ["llm_%d" % idx]
            })
        return risk_edges

    def _risk_edges_to_type_weights(self, risk_edges):
        """Convert only sufficiently risky edges into extra old-label protection.

        Previously every new-to-old pair received at least ``base_weight``.  That
        made a missing rule (background risk 0.10) strengthen the corresponding
        old class just like an evidence-backed relation.  Low-risk edges now have
        no training effect; selected edges are smoothly scaled by their risk.
        """
        important_old_types = {}
        min_weight = min(max(self.base_weight, 1.0), self.max_weight)
        for edge in risk_edges:
            old_entity = edge.get("target")
            if not old_entity:
                continue
            risk = min(max(float(edge.get("risk", 0.0)), 0.0), 1.0)
            if risk < self.risk_threshold:
                continue
            if self.risk_threshold >= 1.0:
                progress = 1.0
            else:
                progress = (risk - self.risk_threshold) / (1.0 - self.risk_threshold)
            mapped_weight = min_weight + progress * (self.max_weight - min_weight)
            important_old_types[old_entity] = max(
                important_old_types.get(old_entity, 1.0),
                mapped_weight
            )
        return important_old_types

    def _risk_edges_to_old_type_vulnerability(self, risk_edges):
        vulnerabilities = {}
        for edge in risk_edges:
            update = edge.get("vulnerability_update", {})
            target = edge.get("target")
            if target and update:
                vulnerabilities[target] = max(vulnerabilities.get(target, 0.0), float(update.get("score", 0.0)))
        return vulnerabilities

    def _risk_edges_to_pseudo_label_thresholds(self, risk_edges, schema):
        """Require higher teacher confidence for pseudo labels on risky old types."""
        thresholds = {}
        for edge in risk_edges:
            risk = min(max(float(edge.get("risk", 0.0)), 0.0), 1.0)
            if risk < self.risk_threshold:
                continue
            if self.risk_threshold >= 1.0:
                progress = 1.0
            else:
                progress = (risk - self.risk_threshold) / (1.0 - self.risk_threshold)
            threshold = self.pseudo_label_min_confidence + progress * (
                self.pseudo_label_max_confidence - self.pseudo_label_min_confidence
            )
            for label_name in self._entity_to_labels(edge.get("target"), schema):
                thresholds[label_name] = max(thresholds.get(label_name, 0.0), round(threshold, 4))
        return thresholds

    def _risk_edges_to_contrastive_pairs(self, risk_edges):
        """Expose only high-risk new-to-old edges to the contrastive loss."""
        pairs = []
        for edge in risk_edges:
            risk = min(max(float(edge.get("risk", 0.0)), 0.0), 1.0)
            if risk < self.risk_threshold:
                continue
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                pairs.append({"source": source, "target": target, "risk": round(risk, 4)})
        return pairs

    def _risk_edges_to_verifier_label_risks(self, risk_edges, schema):
        """Expose the strongest new-to-old risk for each old BIO label.

        The trainer uses this map only to schedule expensive LLM verification;
        it does not turn every old pseudo label into an LLM request.
        """
        label_risks = {}
        for edge in risk_edges:
            risk = min(max(float(edge.get("risk", 0.0)), 0.0), 1.0)
            if risk < self.risk_threshold:
                continue
            for label_name in self._entity_to_labels(edge.get("target"), schema):
                label_risks[label_name] = max(label_risks.get(label_name, 0.0), round(risk, 4))
        return label_risks

    def _ensure_verifier_policy(self, plan, schema):
        """Upgrade pre-scheduler cached plans in memory without invalidating them."""
        training_policy = plan.setdefault("training_policy", {})
        if "verifier_label_risks" not in training_policy:
            training_policy["verifier_label_risks"] = self._risk_edges_to_verifier_label_risks(
                plan.get("risk_edges", []), schema
            )
        return plan

    def _build_risk_graph(self, domain_name, new_entity_list, old_entity_list,
                          risk_edges, retrieved_evidence, training_policy, iteration):
        graph = RiskGraph(
            task_id=0 if iteration is None else int(iteration),
            domain_name=domain_name,
            new_types=new_entity_list,
            old_types=old_entity_list,
            edges=risk_edges,
            evidence_summary={
                "definition_count": len(retrieved_evidence.get("definitions", {})),
                "annotation_rule_count": len(retrieved_evidence.get("annotation_rules", {})),
                "example_count": len(retrieved_evidence.get("examples", [])),
                "confusion_evidence_count": len(retrieved_evidence.get("confusion_evidence", []))
            },
            training_policy=training_policy
        )
        return graph.to_dict()

    def _map_rule_weight_to_risk(self, weight):
        if weight >= 1.7:
            return 0.85
        if weight >= 1.5:
            return 0.75
        if weight >= 1.1:
            return 0.55
        return 0.35

    def _cache_path(self, domain_name, new_entity_list, old_entity_list, schema, iteration):
        iter_name = "iter_%s" % str(iteration) if iteration is not None else "iter_unknown"
        new_part = "-".join(new_entity_list)
        old_part = "-".join(old_entity_list) if old_entity_list else "none"
        filename = "%s_%s_risk_v4_hybrid_reflection_%s_new_%s_old_%s.json" % (
            domain_name,
            schema,
            iter_name,
            new_part,
            old_part
        )
        return os.path.join(self.cache_dir, filename)

    def get_llm_raw_output_path(self, domain_name, new_entity_list, old_entity_list, schema, iteration):
        iter_name = "iter_%s" % str(iteration) if iteration is not None else "iter_unknown"
        new_part = "-".join(new_entity_list)
        old_part = "-".join(old_entity_list) if old_entity_list else "none"
        filename = "%s_%s_%s_new_%s_old_%s.txt" % (
            domain_name,
            schema,
            iter_name,
            new_part,
            old_part
        )
        raw_dir = os.path.join(self.cache_dir, "llm_raw")
        return os.path.join(raw_dir, filename)

    def save_llm_raw_output(self, raw_output, domain_name, new_entity_list, old_entity_list, schema, iteration):
        if not raw_output:
            return ""
        path = self.get_llm_raw_output_path(
            domain_name=domain_name,
            new_entity_list=new_entity_list,
            old_entity_list=old_entity_list,
            schema=schema,
            iteration=iteration
        )
        raw_dir = os.path.dirname(path)
        if not os.path.isdir(raw_dir):
            os.makedirs(raw_dir)
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(raw_output))
        return path

    def _load_cached_plan(self, domain_name, new_entity_list, old_entity_list, schema, iteration):
        path = self._cache_path(domain_name, new_entity_list, old_entity_list, schema, iteration)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_cached_plan(self, plan, domain_name, new_entity_list, old_entity_list, schema, iteration):
        if not self.cache_dir:
            return
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        path = self._cache_path(domain_name, new_entity_list, old_entity_list, schema, iteration)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=True, indent=2)

    def _get_entity_rules(self, domain_name):
        datasets = self.memory.get("datasets", {})
        dataset_memory = datasets.get(domain_name, {})
        return dataset_memory.get("entity_types", {})

    def _entity_to_labels(self, entity_name, schema):
        if schema == "IO":
            return ["I-" + entity_name]
        if schema == "BIO":
            return ["B-" + entity_name, "I-" + entity_name]
        if schema == "BIOES":
            return [
                "B-" + entity_name,
                "I-" + entity_name,
                "E-" + entity_name,
                "S-" + entity_name
            ]
        raise ValueError("Unsupported schema for semantic agent: %s" % schema)
