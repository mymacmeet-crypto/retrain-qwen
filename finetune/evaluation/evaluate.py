#!/usr/bin/env python3
"""
Evaluate the fine-tuned LoRA adapter on the validation set.

Metrics reported:
  - Perplexity        : how surprised the model is by ground-truth answers
                        (lower is better; compare to base model baseline)
  - ROUGE-L F1        : recall-oriented n-gram overlap with reference answers
  - Average response  : mean character length of generated answers
  - Exact match rate  : fraction of answers containing key phrases

Usage:
  # Evaluate adapter (merges on-the-fly for generation)
  python finetune/evaluation/evaluate.py

  # Evaluate the base model as a baseline
  python finetune/evaluation/evaluate.py --base-only

  # Evaluate a specific adapter
  python finetune/evaluation/evaluate.py --adapter finetune/adapters/my-adapter
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from finetune.training.config import ADAPTER_DIR, VAL_FILE, DEFAULTS


def load_val_set(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def compute_perplexity(
    model, tokenizer, examples: list[dict], max_samples: int = 200
) -> float:
    """
    Compute perplexity on the assistant turns.

    We tokenise the full conversation, then set labels = -100 for all tokens
    except the assistant turn so the loss is computed only on what the model
    should generate (not the prompt).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for ex in tqdm(examples[:max_samples], desc="Perplexity"):
        msgs = ex["messages"]
        full_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        # Find where the assistant turn starts
        prompt_only = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )

        full_ids   = tokenizer(full_text,   return_tensors="pt").input_ids.to(model.device)
        prompt_len = tokenizer(prompt_only, return_tensors="pt").input_ids.shape[-1]

        labels = full_ids.clone()
        labels[:, :prompt_len] = -100   # mask prompt tokens from loss

        with torch.no_grad():
            loss = model(input_ids=full_ids, labels=labels).loss
        n_tokens = (labels != -100).sum().item()
        total_nll    += loss.item() * n_tokens
        total_tokens += n_tokens

    return math.exp(total_nll / total_tokens) if total_tokens else float("inf")


def generate_answers(
    model, tokenizer, examples: list[dict], max_new_tokens: int = 512,
    max_samples: int = 50
) -> list[tuple[str, str]]:
    """Return list of (generated_answer, reference_answer) pairs."""
    model.eval()
    pairs = []

    for ex in tqdm(examples[:max_samples], desc="Generating"):
        msgs = ex["messages"]
        reference = msgs[-1]["content"]

        prompt_msgs = msgs[:-1]
        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()
        pairs.append((generated, reference))

    return pairs


def rouge_l_f1(generated: str, reference: str) -> float:
    """Compute ROUGE-L F1 between two strings (word-level LCS)."""
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()

    if not gen_tokens or not ref_tokens:
        return 0.0

    # LCS length via DP
    m, n = len(ref_tokens), len(gen_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == gen_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]

    precision = lcs / n if n else 0.0
    recall    = lcs / m if m else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate(
    adapter_dir: str | None,
    base_model_id: str,
    val_file: str,
    base_only: bool = False,
) -> None:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    examples = load_val_set(val_file)
    print(f"Validation set: {len(examples)} examples")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\nLoading tokenizer from {base_model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if not base_only and adapter_dir:
        from peft import PeftModel
        print(f"Attaching LoRA adapter from {adapter_dir} …")
        model = PeftModel.from_pretrained(model, adapter_dir)

    label = "base model" if base_only else "fine-tuned model"
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}\n")

    # ── Perplexity ────────────────────────────────────────────────────────────
    ppl = compute_perplexity(model, tokenizer, examples, max_samples=200)
    print(f"Perplexity        : {ppl:.2f}")

    # ── Generation metrics ────────────────────────────────────────────────────
    pairs = generate_answers(model, tokenizer, examples, max_samples=50)

    rouge_scores = [rouge_l_f1(gen, ref) for gen, ref in pairs]
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    avg_len   = sum(len(gen) for gen, _ in pairs) / len(pairs) if pairs else 0.0

    print(f"ROUGE-L F1 (avg)  : {avg_rouge:.4f}")
    print(f"Avg response len  : {avg_len:.0f} chars")

    # ── Show 3 sample predictions ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*60}")
    for i, (gen, ref) in enumerate(pairs[:3]):
        msgs = examples[i]["messages"]
        question = next(m["content"] for m in msgs if m["role"] == "user")
        print(f"\n[{i+1}] Q: {question[:150]}")
        print(f"     Reference : {ref[:200]}")
        print(f"     Generated : {gen[:200]}")
        print(f"     ROUGE-L   : {rouge_scores[i]:.4f}")

    print(f"\n{'='*60}")
    print(f"Summary ({label})")
    print(f"  Perplexity : {ppl:.2f}")
    print(f"  ROUGE-L    : {avg_rouge:.4f}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--adapter",    default=str(ADAPTER_DIR))
    parser.add_argument("--model",      default=DEFAULTS["base_model"])
    parser.add_argument("--val-file",   default=str(VAL_FILE))
    parser.add_argument("--base-only",  action="store_true",
                        help="Evaluate the base model without any adapter")
    args = parser.parse_args()

    evaluate(
        adapter_dir=args.adapter,
        base_model_id=args.model,
        val_file=args.val_file,
        base_only=args.base_only,
    )


if __name__ == "__main__":
    main()
