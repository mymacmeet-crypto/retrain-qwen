#!/usr/bin/env python3
"""
Dataset preparation for LoRA fine-tuning of Qwen2.5-7B-Instruct.

Two modes:
  1. generate  -- queries your existing ChromaDB + Ollama LLM to auto-create
                  Q&A pairs from every stored chunk.
  2. convert   -- converts manually curated raw/*.jsonl files into the
                  canonical training format.

RESUMABLE GENERATION
--------------------
Every run saves a progress file at:
  finetune/dataset/processed/generation_progress.json

This file remembers exactly which chunk index was processed last.
Three ways to continue where you left off:

  --resume              Read offset automatically from the progress file.
  --offset N            Skip the first N chunks manually.
  --append              Write to existing train/val files instead of overwriting.

Typical multi-day workflow:
  # Day 1: chunks 0-199
  python prepare_dataset.py generate --n-pairs 3 --max-chunks 200

  # Day 2: pick up from chunk 200 automatically
  python prepare_dataset.py generate --n-pairs 3 --max-chunks 200 --resume

  # Day 3: pick up from chunk 400 automatically
  python prepare_dataset.py generate --n-pairs 3 --max-chunks 200 --resume

  # Any day: check progress
  python prepare_dataset.py status

Output: finetune/dataset/processed/train.jsonl
        finetune/dataset/processed/val.jsonl
        finetune/dataset/processed/generation_progress.json
"""

import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT     = Path(__file__).resolve().parents[2]
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "dataset" / "processed"
RAW_DIR       = Path(__file__).resolve().parents[1] / "dataset" / "raw"
PROGRESS_FILE = PROCESSED_DIR / "generation_progress.json"

sys.path.insert(0, str(REPO_ROOT))
from src.config import get_config
from src.vector_store import VectorStore

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert Salesforce OmniStudio assistant. "
    "Answer questions accurately and concisely based on your knowledge of "
    "Salesforce OmniStudio, Lightning Web Components, Apex, and the broader "
    "Salesforce platform. Use numbered steps for procedures, bullet points "
    "for lists, and include relevant code snippets where helpful."
)

QA_GENERATION_PROMPT = """\
You are a dataset curator creating training examples for a Salesforce AI assistant.

Given the following documentation excerpt, generate {n_pairs} distinct question-answer pairs.

Rules:
- Questions must be answerable ONLY from the excerpt — do not invent facts.
- Answers should be complete, accurate, and 2-10 sentences long.
- Vary question styles: how-to, what-is, why, troubleshooting, comparison.
- Output ONLY a JSON array. No preamble, no explanation.

Format:
[
  {{"question": "...", "answer": "..."}},
  ...
]

DOCUMENTATION EXCERPT:
{chunk}
"""

# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {
        "next_chunk_index": 0,
        "total_chunks": 0,
        "total_examples": 0,
        "train_examples": 0,
        "val_examples": 0,
        "batches": [],
    }


def _save_progress(progress: dict) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def _print_status(progress: dict) -> None:
    total = progress.get("total_chunks", 0)
    done  = progress.get("next_chunk_index", 0)
    remaining = max(0, total - done) if total else "unknown"

    print("=" * 50)
    print("Dataset generation progress")
    print("=" * 50)
    print(f"  Chunks processed : {done} / {total or '?'}")
    print(f"  Chunks remaining : {remaining}")
    print(f"  Total examples   : {progress.get('total_examples', 0):,}")
    print(f"  Train examples   : {progress.get('train_examples', 0):,}")
    print(f"  Val examples     : {progress.get('val_examples', 0):,}")
    print(f"  Batches run      : {len(progress.get('batches', []))}")

    for i, b in enumerate(progress.get("batches", []), 1):
        print(f"    Batch {i}: chunks {b['offset']}-{b['offset']+b['limit']-1}  "
              f"→ {b['examples']} examples  ({b['timestamp']})")

    if total and done < total:
        print(f"\n  To continue: python prepare_dataset.py generate "
              f"--n-pairs 3 --max-chunks 200 --resume")
    elif total and done >= total:
        print("\n  All chunks have been processed.")
    print("=" * 50)

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str, base_url: str, timeout: int) -> str:
    resp = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.7, "num_ctx": 4096}},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _parse_qa_json(text: str) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        pairs = json.loads(match.group())
        return [p for p in pairs if "question" in p and "answer" in p]
    except json.JSONDecodeError:
        return []


def _to_training_example(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ]
    }


def _append_jsonl(examples: list[dict], path: Path) -> None:
    """Append examples to an existing JSONL file (or create it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:                 # "a" = append, not overwrite
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def _write_jsonl(examples: list[dict], path: Path) -> None:
    """Write (overwrite) a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def _train_val_split(
    examples: list[dict], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)               # local RNG — doesn't affect global state
    shuffled = examples[:]
    rng.shuffle(shuffled)
    split = max(1, int(len(shuffled) * val_ratio))
    return shuffled[split:], shuffled[:split]


# ---------------------------------------------------------------------------
# Mode 1: generate from ChromaDB (resumable)
# ---------------------------------------------------------------------------

def generate_from_chromadb(
    n_pairs_per_chunk: int,
    max_chunks: int,
    offset: int,
    append: bool,
) -> None:
    """
    Generate Q&A pairs from ChromaDB chunks.

    offset     : skip the first N chunks (resume support).
    max_chunks : how many chunks to process this run (0 = all remaining).
    append     : if True, append to existing train/val files; otherwise overwrite.
    """
    cfg    = get_config()
    ollama = cfg["ollama"]
    vs_cfg = cfg["vector_store"]

    store = VectorStore(
        persist_directory=vs_cfg["persist_directory"],
        collection_name=vs_cfg["collection_name"],
    )
    total_in_db = store.count()
    if total_in_db == 0:
        sys.exit("ChromaDB is empty — run ingest.py first.")

    # Load all chunk texts (ChromaDB returns them in stable insertion order)
    raw   = store._collection.get(include=["documents", "metadatas"])
    docs  = raw["documents"] or []
    metas = raw["metadatas"] or []

    # Apply offset (skip already-processed chunks)
    if offset > len(docs):
        print(f"Offset {offset} exceeds total chunks ({len(docs)}). Nothing to do.")
        return

    docs  = docs[offset:]
    metas = metas[offset:]

    # Apply limit
    if max_chunks and max_chunks < len(docs):
        docs  = docs[:max_chunks]
        metas = metas[:max_chunks]

    actual_limit = len(docs)
    end_index    = offset + actual_limit

    print("=" * 55)
    print(f"Generating Q&A pairs")
    print("=" * 55)
    print(f"  Total chunks in DB  : {total_in_db:,}")
    print(f"  Starting at chunk   : {offset}")
    print(f"  Processing          : {actual_limit} chunks (up to chunk {end_index - 1})")
    print(f"  Pairs per chunk     : {n_pairs_per_chunk}")
    print(f"  Mode                : {'append to existing files' if append else 'overwrite files'}")
    print(f"  LLM                 : {ollama['llm_model']}")
    print("=" * 55)

    if not append and (PROCESSED_DIR / "train.jsonl").exists():
        confirm = input("\nThis will OVERWRITE existing train.jsonl and val.jsonl.\n"
                        "Type 'yes' to confirm, or press Enter to cancel: ").strip()
        if confirm.lower() != "yes":
            print("Cancelled.")
            return

    examples: list[dict] = []
    skipped = 0

    for chunk_text, meta in tqdm(zip(docs, metas), total=actual_limit, desc="Generating"):
        if len(chunk_text.strip()) < 100:
            skipped += 1
            continue

        prompt = QA_GENERATION_PROMPT.format(
            chunk=chunk_text[:3000],
            n_pairs=n_pairs_per_chunk,
        )
        try:
            raw_output = _call_ollama(
                prompt,
                model=ollama["llm_model"],
                base_url=ollama["base_url"],
                timeout=ollama["timeout"],
            )
            pairs = _parse_qa_json(raw_output)
        except Exception as exc:
            tqdm.write(f"  [WARN] chunk {offset + len(examples)//n_pairs_per_chunk} skipped — {exc}")
            skipped += 1
            continue

        for pair in pairs:
            examples.append(_to_training_example(pair["question"], pair["answer"]))

    print(f"\nGenerated {len(examples):,} examples ({skipped} chunks skipped).")

    if not examples:
        print("No examples generated — nothing written.")
        return

    # Split this batch 90/10 train/val
    train_ex, val_ex = _train_val_split(examples, seed=42 + offset)

    train_path = PROCESSED_DIR / "train.jsonl"
    val_path   = PROCESSED_DIR / "val.jsonl"

    if append:
        _append_jsonl(train_ex, train_path)
        _append_jsonl(val_ex,   val_path)
        print(f"  Appended {len(train_ex):,} → {train_path.name}")
        print(f"  Appended {len(val_ex):,}   → {val_path.name}")
    else:
        _write_jsonl(train_ex, train_path)
        _write_jsonl(val_ex,   val_path)
        print(f"  Wrote {len(train_ex):,} → {train_path.name}")
        print(f"  Wrote {len(val_ex):,}   → {val_path.name}")

    # Count totals across the whole file
    def _count_lines(p: Path) -> int:
        return sum(1 for ln in p.open() if ln.strip()) if p.exists() else 0

    # Update progress file
    progress = _load_progress()
    progress["total_chunks"]    = total_in_db
    progress["next_chunk_index"] = end_index
    progress["total_examples"]  = _count_lines(train_path) + _count_lines(val_path)
    progress["train_examples"]  = _count_lines(train_path)
    progress["val_examples"]    = _count_lines(val_path)
    progress["batches"].append({
        "offset":    offset,
        "limit":     actual_limit,
        "examples":  len(examples),
        "skipped":   skipped,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    _save_progress(progress)

    # Final summary
    remaining = total_in_db - end_index
    print(f"\nProgress saved → {PROGRESS_FILE.name}")
    print(f"  Total train examples so far : {progress['train_examples']:,}")
    print(f"  Total val examples so far   : {progress['val_examples']:,}")
    if remaining > 0:
        print(f"\n  {remaining} chunks remaining.")
        print(f"  Tomorrow, run:")
        print(f"    python finetune/preprocessing/prepare_dataset.py "
              f"generate --n-pairs {n_pairs_per_chunk} "
              f"--max-chunks {max_chunks or 200} --resume")
    else:
        print("\n  All chunks processed! Dataset is complete.")


# ---------------------------------------------------------------------------
# Mode 2: convert manually curated raw JSONL files
# ---------------------------------------------------------------------------

def convert_raw_jsonl(append: bool) -> None:
    raw_files = list(RAW_DIR.glob("*.jsonl"))
    if not raw_files:
        sys.exit(f"No *.jsonl files found in {RAW_DIR}")

    examples: list[dict] = []
    for fpath in raw_files:
        print(f"  Loading {fpath.name}")
        with fpath.open() as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  [WARN] {fpath.name}:{lineno} — {e}")
                    continue

                if "messages" in obj:
                    examples.append(obj)
                elif "question" in obj and "answer" in obj:
                    examples.append(_to_training_example(obj["question"], obj["answer"]))
                elif "instruction" in obj and "output" in obj:
                    examples.append(_to_training_example(obj["instruction"], obj["output"]))
                else:
                    print(f"  [WARN] {fpath.name}:{lineno} — unrecognised shape, skipped.")

    print(f"\nLoaded {len(examples):,} examples from {len(raw_files)} file(s).")
    train_ex, val_ex = _train_val_split(examples)

    train_path = PROCESSED_DIR / "train.jsonl"
    val_path   = PROCESSED_DIR / "val.jsonl"

    if append:
        _append_jsonl(train_ex, train_path)
        _append_jsonl(val_ex,   val_path)
    else:
        _write_jsonl(train_ex, train_path)
        _write_jsonl(val_ex,   val_path)

    print(f"  {'Appended' if append else 'Wrote'} {len(train_ex):,} → {train_path.name}")
    print(f"  {'Appended' if append else 'Wrote'} {len(val_ex):,}   → {val_path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare LoRA training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First run — chunks 0-199
  python prepare_dataset.py generate --n-pairs 3 --max-chunks 200

  # Resume automatically (reads progress file)
  python prepare_dataset.py generate --n-pairs 3 --max-chunks 200 --resume

  # Resume manually from a specific chunk
  python prepare_dataset.py generate --n-pairs 3 --max-chunks 200 --offset 400 --append

  # Check how far you've got
  python prepare_dataset.py status

  # Convert hand-crafted examples AND append to existing dataset
  python prepare_dataset.py convert --append
        """,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── generate ─────────────────────────────────────────────────────────────
    gen = sub.add_parser("generate", help="Auto-generate Q&A from ChromaDB chunks")
    gen.add_argument("--n-pairs",   type=int, default=3,
                     help="Q&A pairs per chunk (default: 3)")
    gen.add_argument("--max-chunks", type=int, default=0,
                     help="Max chunks to process this run (0 = all remaining, default: 0)")

    offset_group = gen.add_mutually_exclusive_group()
    offset_group.add_argument(
        "--resume", action="store_true",
        help="Auto-read starting offset from generation_progress.json and append to files",
    )
    offset_group.add_argument(
        "--offset", type=int, default=0,
        help="Manually skip the first N chunks (requires --append on subsequent runs)",
    )
    gen.add_argument(
        "--append", action="store_true",
        help="Append to existing train/val files instead of overwriting",
    )

    # ── convert ──────────────────────────────────────────────────────────────
    conv = sub.add_parser("convert", help="Convert raw/*.jsonl files to training format")
    conv.add_argument(
        "--append", action="store_true",
        help="Append to existing train/val files instead of overwriting",
    )

    # ── status ───────────────────────────────────────────────────────────────
    sub.add_parser("status", help="Show generation progress")

    args = parser.parse_args()

    if args.mode == "status":
        _print_status(_load_progress())
        return

    if args.mode == "convert":
        convert_raw_jsonl(append=args.append)
        return

    # ── generate mode ─────────────────────────────────────────────────────────
    if args.resume:
        progress = _load_progress()
        offset   = progress.get("next_chunk_index", 0)
        append   = True          # --resume always appends
        if offset == 0:
            print("No previous progress found — starting from chunk 0.")
            append = False
        else:
            print(f"Resuming from chunk {offset} (read from progress file).")
    else:
        offset = args.offset
        append = args.append or (offset > 0)   # non-zero offset implies append

    generate_from_chromadb(
        n_pairs_per_chunk=args.n_pairs,
        max_chunks=args.max_chunks,
        offset=offset,
        append=append,
    )


if __name__ == "__main__":
    main()
