import hashlib
import json
import os
import re


class LocalLLMVerifier(object):
    """Cached local-LLM verifier for high-risk pseudo labels."""

    def __init__(self, model_path, cache_dir="semantic_cache/llm_verifier",
                 max_new_tokens=256, temperature=0.0, torch_dtype="float32"):
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.torch_dtype = torch_dtype
        self._tokenizer = None
        self._model = None

    def verify(self, sentence, candidate_span, candidate_type, retrieved_evidence,
               teacher_label="", teacher_confidence=0.0):
        payload = {
            "verifier_prompt_version": 3,
            "verifier_torch_dtype": self.torch_dtype,
            "sentence": sentence,
            "candidate_span": candidate_span,
            "candidate_type": candidate_type,
            "teacher_label": teacher_label,
            "teacher_confidence": float(teacher_confidence),
            "retrieved_evidence": retrieved_evidence
        }
        cached = self._load_cache(payload)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

        raw_output = ""
        try:
            raw_output = self._generate_text(payload)
            parsed = self._extract_json(raw_output)
            result = self._normalize_result(parsed, raw_output)
        except Exception as exc:
            result = self._normalize_text_result(raw_output, str(exc))

        self._save_cache(payload, result)
        return result

    def _generate_text(self, payload):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer, model = self._load_backend(AutoTokenizer, AutoModelForCausalLM, torch)
        prompt = self._build_prompt(payload)
        messages = [
            {
                "role": "system",
                "content": "You verify high-risk NER pseudo labels and return strict JSON only."
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
                raise ValueError("Unsupported verifier torch dtype: %s" % self.torch_dtype)
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": dtype_map[self.torch_dtype]
            }
            if torch_module.cuda.is_available():
                model_kwargs["device_map"] = "auto"
            self._model = auto_model_cls.from_pretrained(self.model_path, **model_kwargs)
        return self._tokenizer, self._model

    def _build_prompt(self, payload):
        request = {
            "task": "Verify whether candidate_span should be labeled as candidate_type in the sentence.",
            "sentence": payload["sentence"],
            "candidate_span": payload["candidate_span"],
            "candidate_type": payload["candidate_type"],
            "teacher_label": payload["teacher_label"],
            "teacher_confidence": payload["teacher_confidence"],
            "retrieved_evidence": payload["retrieved_evidence"],
                "instructions": [
                "Use the entity definitions, annotation rules, examples, and confusion evidence.",
                "Return accept only if the span clearly belongs to the candidate type in context.",
                "Return reject if the span is not an entity, belongs to another type, or evidence contradicts the label.",
                "Return uncertain if the evidence is insufficient.",
                "Your first line must be exactly: DECISION: accept, DECISION: reject, or DECISION: uncertain.",
                "Then return one valid JSON object with the same decision, confidence, and reason.",
                "Do not output markdown fences."
            ],
            "required_json_schema": {
                "decision": "accept|reject|uncertain",
                "confidence": 0.0,
                "reason": "short reason"
            }
        }
        return json.dumps(request, ensure_ascii=True, indent=2)

    def _extract_json(self, text):
        try:
            return json.loads(text)
        except Exception:
            pass

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced_match:
            return json.loads(fenced_match.group(1))

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

    def _normalize_result(self, parsed, raw_output):
        decision = str(parsed.get("decision", "uncertain")).lower().strip()
        if decision not in ["accept", "reject", "uncertain"]:
            decision = "uncertain"
        confidence = parsed.get("confidence", 0.0)
        try:
            confidence = min(max(float(confidence), 0.0), 1.0)
        except Exception:
            confidence = 0.0
        return {
            "ok": True,
            "decision": decision,
            "confidence": confidence,
            "reason": str(parsed.get("reason", "")),
            "raw_output": raw_output,
            "error": "",
            "cache_hit": False
        }

    def _normalize_text_result(self, raw_output, parse_error):
        """Recover a verdict from a non-JSON local-LLM response.

        Local instruction models often follow the requested decision but add a
        prose explanation before JSON. A clear textual verdict is still useful
        for this bounded verifier; ambiguous output remains an explicit error.
        """
        text = str(raw_output or "").strip()
        decision_match = re.search(
            r"(?im)^\s*(?:decision\s*[:=-]\s*)?(accept|reject|uncertain)\b", text
        )
        if decision_match is None:
            decision_match = re.search(r"(?i)\b(accept|reject|uncertain)\b", text)
        if decision_match is not None:
            decision = decision_match.group(1).lower()
            return {
                "ok": True,
                "decision": decision,
                "confidence": 0.5,
                "reason": text[:500],
                "raw_output": text,
                "error": "",
                "cache_hit": False,
                "parsed_from_text": True
            }
        return {
            "ok": False,
            "decision": "uncertain",
            "confidence": 0.0,
            "reason": "",
            "raw_output": text,
            "error": parse_error,
            "cache_hit": False
        }

    def _cache_path(self, payload):
        if not self.cache_dir:
            return ""
        text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, key + ".json")

    def _load_cache(self, payload):
        path = self._cache_path(payload)
        if not path or not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_cache(self, payload, result):
        path = self._cache_path(payload)
        if not path:
            return
        cache_dir = os.path.dirname(path)
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=True, indent=2)
