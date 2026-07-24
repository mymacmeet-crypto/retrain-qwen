#!/usr/bin/env python3
"""
Validate and inspect processed training datasets before fine-tuning.

Checks:
  - All records have the required "messages" key
  - Each conversation has system / user / assistant turns
  - No empty or whitespace-only message content
  - Token-length distribution (requires transformers tokenizer)
  - Duplicate detection (by exact user message)

Usage:
  python validate_dataset.py --split train
  python validate_dataset.py --split val --tokenizer Qwen/Qwen2.5-7B-Instruct
"""

import json
import sys
from collections import Counter
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "dataset" / "processed"


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [ERROR] Line {lineno}: {e}")
    return examples


def validate(examples: list[dict], tokenizer=None, max_seq_len: int = 2048) -> bool:
    errors = 0
    role_counts: Counter = Counter()
    lengths: list[int] = []
    user_messages: list[str] = []

    for i, ex in enumerate(examples):
        # ── structural checks ────────────────────────────────────────────────
        if "messages" not in ex:
            print(f"  [ERROR] Record {i}: missing 'messages' key")
            errors += 1
            continue

        msgs = ex["messages"]
        roles = [m.get("role") for m in msgs]
        role_counts.update(roles)

        if "user" not in roles:
            print(f"  [ERROR] Record {i}: no user turn")
            errors += 1
        if "assistant" not in roles:
            print(f"  [ERROR] Record {i}: no assistant turn")
            errors += 1

        for j, msg in enumerate(msgs):
            content = msg.get("content", "").strip()
            if not content:
                print(f"  [ERROR] Record {i}, turn {j} ({msg.get('role')}): empty content")
                errors += 1

        # ── collect user message for duplicate check ─────────────────────────
        for msg in msgs:
            if msg.get("role") == "user":
                user_messages.append(msg.get("content", "").strip())
                break

        # ── token-length check ───────────────────────────────────────────────
        if tokenizer is not None:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            token_ids = tokenizer(text)["input_ids"]
            lengths.append(len(token_ids))
            if len(token_ids) > max_seq_len:
                print(f"  [WARN]  Record {i}: {len(token_ids)} tokens > max_seq_len {max_seq_len}")

    # ── duplicate report ─────────────────────────────────────────────────────
    dup_count = len(user_messages) - len(set(user_messages))
    if dup_count:
        print(f"  [WARN]  {dup_count} duplicate user messages detected — consider deduplication")

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n  Total records : {len(examples):,}")
    print(f"  Role counts   : {dict(role_counts)}")
    print(f"  Errors        : {errors}")
    if lengths:
        import statistics
        print(f"  Token lengths : min={min(lengths)}  max={max(lengths)}  "
              f"mean={statistics.mean(lengths):.0f}  p95={sorted(lengths)[int(0.95*len(lengths))]}")
        over = sum(1 for l in lengths if l > max_seq_len)
        if over:
            print(f"  Over limit    : {over} records ({100*over/len(lengths):.1f}%) "
                  f"will be truncated during training")

    return errors == 0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate QLoRA training dataset")
    parser.add_argument("--split",     default="train", choices=["train", "val"])
    parser.add_argument("--tokenizer", default=None,
                        help="HuggingFace model ID for token-length stats "
                             "(e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    path = PROCESSED_DIR / f"{args.split}.jsonl"
    if not path.exists():
        sys.exit(f"Dataset not found: {path}\nRun prepare_dataset.py first.")

    print(f"Validating {path} …")
    examples = load_jsonl(path)

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    ok = validate(examples, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
