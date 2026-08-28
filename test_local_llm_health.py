import argparse
import json
import os
import sys


def generate_reply(tokenizer, model, torch_module, instruction, max_new_tokens):
    messages = [
        {"role": "system", "content": "Follow the user instruction exactly."},
        {"role": "user", "content": instruction}
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt = instruction

    inputs = tokenizer(prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    with torch_module.inference_mode():
        logits = model(**inputs).logits[0, -1].float()
        top_logits, top_token_ids = torch_module.topk(logits, k=8)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    new_token_ids = generated_ids[0, inputs["input_ids"].shape[1]:].detach().cpu().tolist()
    output = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    return {
        "instruction": instruction,
        "rendered_prompt": prompt,
        "output": output,
        "generated_token_ids": new_token_ids[:64],
        "next_token_topk": [
            {
                "token_id": int(token_id),
                "text": tokenizer.decode([int(token_id)]),
                "logit": float(logit)
            }
            for token_id, logit in zip(top_token_ids.detach().cpu().tolist(), top_logits.detach().cpu().tolist())
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal local Qwen health check.")
    parser.add_argument("--model_path", required=True, help="Local model directory")
    parser.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto",
                        help="Model loading dtype; use auto first for Qwen checkpoints")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output", default="semantic_cache/llm_tests/health_check.json")
    args = parser.parse_args()

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype_map[args.torch_dtype]
        }
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
        model.eval()

        checks = [
            generate_reply(tokenizer, model, torch, "Reply with exactly this text: OK", args.max_new_tokens),
            generate_reply(tokenizer, model, torch, "Return exactly this JSON object and no other text: {\"status\": \"ok\"}", args.max_new_tokens)
        ]
        json_ok = parse_json_reply(checks[1]["output"]) == {"status": "ok"}

        payload = {
            "ok": checks[0]["output"] == "OK" and json_ok,
            "model_path": args.model_path,
            "transformers_version": transformers.__version__,
            "requested_torch_dtype": args.torch_dtype,
            "cuda_available": torch.cuda.is_available(),
            "tokenizer_class": tokenizer.__class__.__name__,
            "model_class": model.__class__.__name__,
            "model_device": str(next(model.parameters()).device),
            "model_dtype": str(next(model.parameters()).dtype),
            "eos_token_id": tokenizer.eos_token_id,
            "model_files": collect_model_files(args.model_path),
            "checks": checks
        }
    except Exception as exc:
        payload = {"ok": False, "model_path": args.model_path, "error": repr(exc)}

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("Health-check result saved to %s" % args.output)
    if not payload.get("ok", False):
        sys.exit(1)


def collect_model_files(model_path):
    files = []
    if not os.path.isdir(model_path):
        return files
    for filename in sorted(os.listdir(model_path)):
        if filename.endswith((".safetensors", ".bin", ".json")):
            path = os.path.join(model_path, filename)
            files.append({"name": filename, "bytes": os.path.getsize(path)})
    return files


def parse_json_reply(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
    try:
        return json.loads(text)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
