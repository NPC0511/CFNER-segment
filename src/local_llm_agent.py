import json
import re
import itertools


class LocalLLMAgent(object):
    """Local Qwen-based semantic planner.

    The model is loaded only inside generate_plan() and then released so that
    BERT training does not keep competing with the local LLM for GPU memory.
    """

    def __init__(self, model_path, max_new_tokens=512, temperature=0.0,
                 torch_dtype="float32"):
        self.model_path = model_path
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.torch_dtype = torch_dtype
        self._tokenizer = None
        self._model = None

    def generate_plan(self, retrieved_evidence, new_entity_list, old_entity_list):
        raw_output = ""
        try:
            raw_output = self._generate_text(retrieved_evidence, new_entity_list, old_entity_list)
            parsed = self._extract_json(raw_output)
            return {
                "ok": True,
                "raw_output": raw_output,
                "plan": parsed,
                "error": ""
            }
        except Exception as exc:
            fallback_plan = self._extract_text_plan(
                text=raw_output if 'raw_output' in locals() else "",
                old_entity_list=old_entity_list
            )
            if fallback_plan is not None:
                return {
                    "ok": True,
                    "raw_output": raw_output,
                    "plan": fallback_plan,
                    "error": ""
                }
            return {
                "ok": False,
                "raw_output": raw_output,
                "plan": {},
                "error": str(exc)
            }

    def generate_risk_graph(self, retrieved_evidence, new_entity_list, old_entity_list):
        """Rank old types per new type, then map rank to compact risk levels.

        The local model was unable to provide calibrated independent scores:
        batched scores were position-sensitive and singleton scores collapsed
        to the maximum. Relative ranking is a smaller and better-defined task.
        """
        all_edges = []
        raw_outputs = []
        failed_pairs = []
        errors = []

        for source_new in new_entity_list:
            source_pairs = [(source_new, old_entity) for old_entity in old_entity_list]
            result = self.rank_risk_targets(
                source_new=source_new,
                old_entity_list=old_entity_list,
                retrieved_evidence=retrieved_evidence
            )
            if result.get("ok"):
                all_edges.extend(self._ranked_targets_to_edges(source_new, result["ranking"]))
                raw_outputs.append(json.dumps(result, ensure_ascii=True))
            else:
                failed_pairs.extend(source_pairs)
                errors.append("%s: %s" % (source_new, result.get("error", "unknown error")))
                raw_outputs.append("")

        return {
            "ok": bool(all_edges),
            "raw_output": "\n\n".join(raw_outputs),
            "risk_graph": {
                "risk_edges": all_edges,
                "failed_pairs": [
                    {"source": source, "target": target}
                    for source, target in failed_pairs
                ]
            },
            "error": "; ".join(errors)
        }

    def rank_risk_targets(self, source_new, old_entity_list, retrieved_evidence):
        """Rank old targets through position-balanced pairwise preferences."""
        if len(old_entity_list) < 2:
            return {"ok": True, "ranking": list(old_entity_list), "scores": {}, "comparisons": []}
        scores = {target: 0.0 for target in old_entity_list}
        comparisons = []
        try:
            for left, right in itertools.combinations(old_entity_list, 2):
                forward = self.compare_risk_targets(source_new, left, right, retrieved_evidence)
                reverse = self.compare_risk_targets(source_new, right, left, retrieved_evidence)
                if not forward.get("ok") or not reverse.get("ok"):
                    error = forward.get("error") or reverse.get("error")
                    raise ValueError("%s vs %s: %s" % (left, right, error))

                # Both values express preference for `left` over `right`.
                left_advantage = (forward["preference"] - reverse["preference"]) / 2.0
                winner = left if left_advantage >= 0 else right
                scores[winner] += 1.0
                scores[left] += left_advantage / 2.0
                scores[right] -= left_advantage / 2.0
                comparisons.append({
                    "left": left,
                    "right": right,
                    "left_advantage": left_advantage,
                    "winner": winner
                })
            ranking = [
                target for target, _ in sorted(
                    scores.items(), key=lambda item: item[1], reverse=True
                )
            ]
            return {
                "ok": True,
                "ranking": ranking,
                "scores": scores,
                "comparisons": comparisons,
                "error": ""
            }
        except Exception as exc:
            return {"ok": False, "ranking": [], "scores": scores, "comparisons": comparisons, "error": str(exc)}

    def generate_semantic_memory(self, source_spec):
        """Create a dataset-level semantic-memory draft in a one-off offline job."""
        raw_output = ""
        try:
            raw_output = self._generate_semantic_memory_text(source_spec)
            return {
                "ok": True,
                "raw_output": raw_output,
                "memory": self._extract_json(raw_output),
                "error": ""
            }
        except Exception as exc:
            return {"ok": False, "raw_output": raw_output, "memory": {}, "error": str(exc)}

    def compare_risk_targets(self, source_new, candidate_a, candidate_b, retrieved_evidence):
        """Return a log-probability preference between two old-type targets.

        This avoids asking a small local model to calibrate an absolute score or
        emit a long structured ranking. Positive preference means candidate_a
        is judged more vulnerable than candidate_b.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer, model = self._load_backend(AutoTokenizer, AutoModelForCausalLM, torch)
            prompt = self._build_pairwise_risk_prompt(
                source_new=source_new,
                candidate_a=candidate_a,
                candidate_b=candidate_b,
                retrieved_evidence=retrieved_evidence
            )
            messages = [
                {"role": "system", "content": "You compare two NER interference risks. Output only A or B."},
                {"role": "user", "content": prompt}
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = prompt
            inputs = tokenizer([text], return_tensors="pt")
            model_device = next(model.parameters()).device
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
            token_a = self._single_token_id(tokenizer, "A")
            token_b = self._single_token_id(tokenizer, "B")
            with torch.inference_mode():
                logits = model(**inputs).logits[0, -1].float()
                log_probs = torch.log_softmax(logits, dim=-1)
            score_a = float(log_probs[token_a].item())
            score_b = float(log_probs[token_b].item())
            return {
                "ok": True,
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "score_a": score_a,
                "score_b": score_b,
                "preference": score_a - score_b,
                "preferred": candidate_a if score_a >= score_b else candidate_b,
                "prompt": prompt,
                "error": ""
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _generate_text(self, retrieved_evidence, new_entity_list, old_entity_list):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer, model = self._load_backend(AutoTokenizer, AutoModelForCausalLM, torch)
        prompt = self._build_prompt(retrieved_evidence, new_entity_list, old_entity_list)
        messages = [
            {
                "role": "system",
                "content": "You are a strict JSON generator for continual NER training plans."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = tokenizer([text], return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        do_sample = self.temperature > 0
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = 0.8
            generation_kwargs["top_k"] = 20
        generated_ids = model.generate(**inputs, **generation_kwargs)
        new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
        return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    def _generate_risk_ranking_text(self, retrieved_evidence, source_new, old_entity_list):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer, model = self._load_backend(AutoTokenizer, AutoModelForCausalLM, torch)
        prompt = self._build_risk_ranking_prompt(
            retrieved_evidence=retrieved_evidence,
            source_new=source_new,
            old_entity_list=old_entity_list
        )
        messages = [
            {
                "role": "system",
                "content": "You are a strict JSON generator for semantic risk graphs in continual NER."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = tokenizer([text], return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = 0.8
            generation_kwargs["top_k"] = 20
        generated_ids = model.generate(**inputs, **generation_kwargs)
        new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
        return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    def _generate_semantic_memory_text(self, source_spec):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer, model = self._load_backend(AutoTokenizer, AutoModelForCausalLM, torch)
        prompt = self._build_semantic_memory_prompt(source_spec)
        messages = [
            {"role": "system", "content": "You are a strict JSON generator for frozen NER semantic memory."},
            {"role": "user", "content": prompt}
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = tokenizer([text], return_tensors="pt")
        model_device = next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        new_tokens = generated_ids[:, inputs["input_ids"].shape[1]:]
        return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

    def _load_backend(self, auto_tokenizer_cls, auto_model_cls, torch_module):
        if self._tokenizer is None:
            self._tokenizer = auto_tokenizer_cls.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        if self._model is None:
            dtype_map = {
                "float16": torch_module.float16,
                "bfloat16": torch_module.bfloat16,
                "float32": torch_module.float32,
                "auto": "auto"
            }
            if self.torch_dtype not in dtype_map:
                raise ValueError("Unsupported local LLM torch dtype: %s" % self.torch_dtype)
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": dtype_map[self.torch_dtype]
            }
            if torch_module.cuda.is_available():
                model_kwargs["device_map"] = "auto"
            self._model = auto_model_cls.from_pretrained(self.model_path, **model_kwargs)
        return self._tokenizer, self._model

    def _build_prompt(self, retrieved_evidence, new_entity_list, old_entity_list):
        payload = {
            "task": "Generate a semantic continual-learning plan for NER.",
            "new_entity_list": list(new_entity_list),
            "old_entity_list": list(old_entity_list),
            "retrieved_evidence": retrieved_evidence,
            "instructions": [
                "Choose only old entity types that are semantically important for preserving old knowledge.",
                "Assign each chosen old type a float weight between 1.0 and 2.0.",
                "Use higher weights for stronger confusion or semantic dependency.",
                "Return exactly one valid JSON object.",
                "Do not output markdown fences, explanations, or any extra text before or after the JSON.",
                "If uncertain, still return a valid JSON object with empty lists or empty objects.",
                "If you cannot produce JSON, output one old type per line as old_type: weight."
            ],
            "required_json_schema": {
                "important_old_types": ["old_type_name"],
                "old_type_weights": {
                    "old_type_name": 1.0
                },
                "reason": "short reason"
            }
        }
        return json.dumps(payload, ensure_ascii=True, indent=2)

    def _build_pairwise_risk_prompt(self, source_new, candidate_a, candidate_b, retrieved_evidence):
        definitions = retrieved_evidence.get("definitions", {})
        rules = retrieved_evidence.get("annotation_rules", {})
        contexts = retrieved_evidence.get("instance_context_examples", {})
        cards = retrieved_evidence.get("risk_evidence_cards", {})
        payload = {
            "task": "Choose the old entity type more likely to be confused with the new NER type in this dataset.",
            "source_new": source_new,
            "source_definition": definitions.get(source_new, ""),
            "source_labelled_contexts": self._context_examples_for_type(contexts, source_new),
            "candidate_A": {
                "type": candidate_a,
                "definition": definitions.get(candidate_a, ""),
                "rules": rules.get(candidate_a, []),
                "labelled_contexts": self._context_examples_for_type(contexts, candidate_a),
                "pre_training_evidence": cards.get("%s->%s" % (source_new, candidate_a), {})
            },
            "candidate_B": {
                "type": candidate_b,
                "definition": definitions.get(candidate_b, ""),
                "rules": rules.get(candidate_b, []),
                "labelled_contexts": self._context_examples_for_type(contexts, candidate_b),
                "pre_training_evidence": cards.get("%s->%s" % (source_new, candidate_b), {})
            },
            "instruction": "Use all pre-training evidence: teacher behaviour, contextual prototype similarity, old-class history, labelled contexts, and annotation boundaries. Do not infer post-training outcomes. Answer A if candidate_A is more likely to be confused with source_new; answer B otherwise. Answer exactly one letter."
        }
        return json.dumps(payload, ensure_ascii=True, indent=2)

    @staticmethod
    def _context_examples_for_type(contexts, entity_type, max_examples=2, max_chars=280):
        compact = []
        for item in contexts.get(entity_type, [])[:max_examples]:
            compact.append({
                "span": str(item.get("span", ""))[:80],
                "context": str(item.get("context", ""))[:max_chars]
            })
        return compact

    @staticmethod
    def _single_token_id(tokenizer, text):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError("Expected %r to be represented by one tokenizer token, got %s" % (text, token_ids))
        return token_ids[0]

    def _build_risk_ranking_prompt(self, retrieved_evidence, source_new, old_entity_list):
        payload = {
            "task": "Rank old NER entity types by likely interference from one new entity type.",
            "source_new": source_new,
            "candidate_old_types": list(old_entity_list),
            "semantic_evidence": retrieved_evidence,
            "instructions": [
                "Rank every candidate_old_types item from most likely to least likely to be interfered with when learning source_new.",
                "Use annotation-boundary overlap, shared contexts, and semantic similarity; do not rank alphabetically.",
                "Return a JSON array only, containing every candidate type exactly once, from highest risk to lowest risk.",
                "Do not use a wrapper object, markdown fence, explanation, score, or any text outside the JSON array."
            ],
            "valid_output_example": ["MOST_LIKELY_OLD_TYPE", "NEXT_OLD_TYPE", "LEAST_LIKELY_OLD_TYPE"]
        }
        return json.dumps(payload, ensure_ascii=True, indent=2)

    def _filter_risk_evidence(self, evidence, required_pairs):
        relevant_types = set()
        for source, target in required_pairs:
            relevant_types.add(source)
            relevant_types.add(target)
        filtered = dict(evidence)
        filtered["definitions"] = {
            key: value for key, value in evidence.get("definitions", {}).items()
            if key in relevant_types
        }
        filtered["annotation_rules"] = {
            key: value for key, value in evidence.get("annotation_rules", {}).items()
            if key in relevant_types
        }
        filtered["examples"] = [
            item for item in evidence.get("examples", [])
            if item.get("entity_type") in relevant_types
        ][:8]
        filtered["confusion_evidence"] = [
            item for item in evidence.get("confusion_evidence", [])
            if (item.get("source_type"), item.get("target_type")) in set(required_pairs)
        ]
        return filtered

    @staticmethod
    def _format_pairs(pairs):
        return ", ".join("%s->%s" % (source, target) for source, target in pairs)

    def _normalize_risk_ranking(self, parsed, source_new, old_entity_list):
        if isinstance(parsed, list):
            ranking = parsed
        elif isinstance(parsed, dict):
            ranking = parsed.get("ranked_targets", parsed.get("ranking", parsed.get("targets")))
        else:
            raise ValueError("LLM ranking must be a JSON array or object")
        if not isinstance(ranking, list):
            raise ValueError("LLM JSON must contain a ranking list")
        ranking = [str(item) for item in ranking]
        expected = list(old_entity_list)
        if len(ranking) != len(expected) or set(ranking) != set(expected):
            raise ValueError("LLM ranking must contain every old type exactly once for %s" % source_new)
        return ranking

    def _ranked_targets_to_edges(self, source_new, ranking):
        edges = []
        for rank, target in enumerate(ranking):
            if rank == 0:
                level = 2
            elif rank < min(4, len(ranking)):
                level = 1
            else:
                level = 0
            components = {
                "semantic_overlap": level,
                "annotation_conflict": level,
                "context_overlap": level
            }
            edges.append({
                "source": source_new,
                "target": target,
                "risk": self._risk_from_components(components),
                "risk_components": components,
                "risk_type": ["llm_semantic_rank"],
                "reason": "LLM relative-rank position %d." % (rank + 1),
                "evidence_ids": ["llm_%s_to_%s" % (source_new, target)]
            })
        return edges

    def _build_semantic_memory_prompt(self, source_spec):
        payload = {
            "task": "Create a frozen semantic-memory draft for one NER dataset.",
            "source_spec": source_spec,
            "instructions": [
                "Use only entity types present in source_spec.entity_types.",
                "Every value under entity_types must be a JSON object with a definition field; never use a plain string as an entity_types value.",
                "When reviewed_human_memory is provided, preserve it as the authoritative baseline and propose only candidate additions consistent with it.",
                "Do not invent dataset facts, annotation rules, or examples not supported by source_spec.",
                "For each type write a short operational definition, at most 5 positive examples, and at most 5 negative examples.",
                "Create a confusion rule only when the supplied guidelines or definitions support an annotation ambiguity.",
                "Confusion weights must be between 1.0 and 2.0.",
                "Return exactly one JSON object with no markdown or explanation."
            ],
            "required_json_schema": {
                "entity_types": {
                    "type_name": {
                        "definition": "short operational definition",
                        "positive_examples": ["example"],
                        "negative_examples": ["example"],
                        "confusable_with": {"other_type": 1.0}
                    }
                },
                "annotation_rules": {"type_name": "short annotation boundary rule"},
                "confusion_rules": [{
                    "source_type": "new_or_current_type",
                    "target_type": "other_type",
                    "weight": 1.0,
                    "reason": "guideline-grounded ambiguity"
                }]
            }
        }
        return json.dumps(payload, ensure_ascii=True, indent=2)

    def _extract_json(self, text):
        try:
            return json.loads(text)
        except Exception:
            pass

        fenced_match = re.search(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", text, flags=re.DOTALL)
        if fenced_match:
            return json.loads(fenced_match.group(1))

        array_candidate = self._find_balanced_json_array(text)
        if array_candidate is not None:
            return json.loads(array_candidate)
        candidate = self._find_balanced_json_object(text)
        if candidate is None:
            raise ValueError("No JSON object found in local LLM output")
        return json.loads(candidate)

    def _find_balanced_json_object(self, text):
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:idx+1]
        return None

    def _find_balanced_json_array(self, text):
        start = text.find("[")
        if start == -1:
            return None
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start:idx+1]
        return None

    def _normalize_risk_graph(self, parsed, new_entity_list=None, old_entity_list=None,
                              required_pairs=None):
        if isinstance(parsed, list):
            raw_edges = parsed
        elif isinstance(parsed, dict):
            raw_edges = parsed.get("risk_edges", parsed.get("edges", parsed.get("risk_graph")))
        else:
            raise ValueError("LLM risk graph must be a JSON object or edge list")
        if not isinstance(raw_edges, list):
            raise ValueError("LLM JSON must contain a risk_edges list")

        if required_pairs is None:
            expected_pairs = {
                (new_entity, old_entity)
                for new_entity in new_entity_list
                for old_entity in old_entity_list
            }
        else:
            expected_pairs = set(required_pairs)
        normalized_edges = {}
        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue
            pair = (edge.get("source"), edge.get("target"))
            if pair not in expected_pairs:
                continue
            if pair in normalized_edges:
                raise ValueError("LLM returned a duplicate risk edge: %s -> %s" % pair)
            raw_components = edge.get("risk_components")
            components = self._normalize_risk_components(
                raw_components=raw_components,
                pair=pair,
                risk_level=edge.get("risk_level")
            )
            risk = self._risk_from_components(components)
            risk_type = edge.get("risk_type", ["llm_semantic_risk"])
            if not isinstance(risk_type, list):
                risk_type = [str(risk_type)]
            if raw_components is None and edge.get("risk_level") is not None:
                risk_type.append("llm_compact_score")
            elif isinstance(raw_components, list):
                risk_type.append("llm_repaired_component_array")
            normalized_edges[pair] = {
                "source": pair[0],
                "target": pair[1],
                "risk": risk,
                "risk_components": components,
                "risk_type": risk_type or ["llm_semantic_risk"],
                "reason": str(edge.get("reason", "")),
                "evidence_ids": ["llm_%s_to_%s" % pair]
            }

        missing_pairs = expected_pairs - set(normalized_edges.keys())
        if missing_pairs:
            missing_text = ", ".join("%s->%s" % pair for pair in sorted(missing_pairs))
            raise ValueError("LLM omitted required risk edges: %s" % missing_text)
        return {"risk_edges": [normalized_edges[pair] for pair in sorted(expected_pairs)]}

    def _normalize_risk_components(self, raw_components, pair, risk_level=None):
        if raw_components is None and risk_level is not None:
            level = self._normalize_risk_level(risk_level, pair)
            return {
                "semantic_overlap": level,
                "annotation_conflict": level,
                "context_overlap": level
            }
        if isinstance(raw_components, list):
            values = [self._normalize_risk_level(value, pair) for value in raw_components]
            if not values:
                raise ValueError("LLM returned an empty risk_components array for %s -> %s" % pair)
            if len(values) == 3:
                return {
                    "semantic_overlap": values[0],
                    "annotation_conflict": values[1],
                    "context_overlap": values[2]
                }
            level = int(round(sum(values) / float(len(values))))
            return {
                "semantic_overlap": level,
                "annotation_conflict": level,
                "context_overlap": level
            }
        if not isinstance(raw_components, dict):
            raise ValueError("LLM must return risk_level or risk_components for %s -> %s" % pair)
        component_names = ["semantic_overlap", "annotation_conflict", "context_overlap"]
        normalized = {}
        for component_name in component_names:
            value = raw_components.get(component_name)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                raise ValueError("Invalid %s for %s -> %s" % (component_name, pair[0], pair[1]))
            if numeric_value not in (0.0, 1.0, 2.0):
                raise ValueError("%s must be 0, 1, or 2 for %s -> %s" % (
                    component_name, pair[0], pair[1]
                ))
            normalized[component_name] = int(numeric_value)
        return normalized

    @staticmethod
    def _normalize_risk_level(value, pair):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise ValueError("Invalid risk level for %s -> %s" % pair)
        if numeric_value not in (0.0, 1.0, 2.0):
            raise ValueError("Risk level must be 0, 1, or 2 for %s -> %s" % pair)
        return int(numeric_value)

    def _risk_from_components(self, components):
        weighted_score = (
            0.30 * components["semantic_overlap"] +
            0.45 * components["annotation_conflict"] +
            0.25 * components["context_overlap"]
        ) / 2.0
        return round(0.05 + 0.95 * weighted_score, 4)

    def _extract_text_plan(self, text, old_entity_list):
        if not text:
            return None

        old_type_weights = {}
        important_old_types = []
        old_type_set = set(old_entity_list)
        for line in text.splitlines():
            line = line.strip().strip("-").strip("*").strip()
            if ":" not in line:
                continue
            left, right = line.split(":", 1)
            old_type = left.strip().strip('"').strip("'")
            if old_type not in old_type_set:
                continue
            try:
                weight_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", right)
                if not weight_match:
                    continue
                weight = float(weight_match.group(1))
            except Exception:
                continue
            weight = min(max(weight, 1.0), 2.0)
            old_type_weights[old_type] = weight
            if old_type not in important_old_types:
                important_old_types.append(old_type)

        if not old_type_weights:
            return None

        return {
            "important_old_types": important_old_types,
            "old_type_weights": old_type_weights,
            "reason": "parsed_from_text_fallback"
        }
