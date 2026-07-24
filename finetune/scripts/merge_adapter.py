#!/usr/bin/env python3
"""
Merge the trained MLX LoRA adapter into the base model weights.

WHY MERGING IS REQUIRED
-----------------------
After training, you have:
  1. The MLX base model   (finetune/mlx_model/  — 4-bit or fp16 weights)
  2. The LoRA adapter     (finetune/adapters/qwen2.5-salesforce/)

These are separate files.  The adapter matrices (A and B) are tiny (~50 MB)
but must be applied on every forward pass:
    W_effective = W_base + scale * (B @ A)

Merging computes this once at export time, producing a single clean model
with no adapter dependency.  Merged models:
  - Load faster (one set of weights, no adapter overhead)
  - Are simpler to deploy (single directory to copy)
  - Are required for GGUF conversion (llama.cpp expects a single checkpoint)

HOW MLX FUSE WORKS
-------------------
`mlx_lm fuse` calls m.fuse() on every LoRA-wrapped linear layer, which:
  1. Dequantises W_base to fp16 if the model was 4-bit quantised
  2. Computes W_merged = W_base + scale * (B @ A)
  3. Replaces the LoRA layer with a plain nn.Linear containing W_merged
  4. Saves the result as HuggingFace-compatible safetensors

IMPORTANT: --export-gguf IS NOT USED HERE
-----------------------------------------
mlx_lm fuse has a --export-gguf flag, but it only supports model types
"llama", "mixtral", and "mistral".  Qwen2.5's model_type is "qwen2".
Passing --export-gguf with Qwen2.5 raises ValueError at runtime.

The GGUF conversion is handled by convert_gguf.sh using llama.cpp, which
supports qwen2 fully.  This script only handles the merge step.

Usage:
  python finetune/scripts/merge_adapter.py
  python finetune/scripts/merge_adapter.py --adapter finetune/adapters/my-run
  python finetune/scripts/merge_adapter.py --mlx-model finetune/mlx_model
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT     = Path(__file__).resolve().parents[2]
FINETUNE_ROOT = REPO_ROOT / "finetune"
ADAPTER_DIR   = FINETUNE_ROOT / "adapters" / "qwen2.5-salesforce"
MLX_MODEL_DIR = FINETUNE_ROOT / "mlx_model"
MERGED_DIR    = FINETUNE_ROOT / "merged_model"


def merge(mlx_model: str, adapter_dir: str, output_dir: str) -> None:
    mlx_path     = Path(mlx_model)
    adapter_path = Path(adapter_dir)
    out_path     = Path(output_dir)

    # ── Validation ────────────────────────────────────────────────────────
    if not mlx_path.exists():
        sys.exit(
            f"MLX model not found: {mlx_path}\n"
            "Run: python finetune/training/train.py  (it converts the model first)"
        )

    if not adapter_path.exists():
        sys.exit(
            f"Adapter not found: {adapter_path}\n"
            "Run: python finetune/training/train.py"
        )

    adapter_weights = list(adapter_path.glob("*.safetensors")) + \
                      list(adapter_path.glob("adapters.npz"))
    if not adapter_weights:
        sys.exit(
            f"No adapter weights found in {adapter_path}\n"
            "The adapter directory exists but contains no weight files.\n"
            "Training may not have completed — check finetune/training/train.py output."
        )

    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Merging MLX LoRA adapter into base model")
    print("=" * 60)
    print(f"  Base model : {mlx_path}")
    print(f"  Adapter    : {adapter_path}")
    print(f"  Output     : {out_path}")

    # ── Run mlx_lm fuse ───────────────────────────────────────────────────
    #
    # --dequantize is REQUIRED if the base model was loaded in 4-bit.
    # Without it, the fused model is still quantised and llama.cpp's
    # convert_hf_to_gguf.py may not correctly handle the quantised weights.
    # With --dequantize, the merged model is clean fp16 — ideal for llama.cpp.
    #
    # Memory note: dequantising a 4-bit 7B model during fuse produces a
    # ~14 GB fp16 model in memory.  On M1 Max with 32 GB, this is fine.
    #
    # The fused model is saved in HuggingFace-compatible format:
    #   config.json, tokenizer.json, model-XXXXX-of-XXXXX.safetensors, ...
    # This is exactly what llama.cpp's convert_hf_to_gguf.py expects.

    print("\nRunning mlx_lm fuse (this takes 2-5 minutes) …")
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model",        str(mlx_path),
        "--adapter-path", str(adapter_path),
        "--save-path",    str(out_path),
        "--dequantize",   # convert 4-bit weights back to fp16 for clean GGUF export
    ]
    subprocess.run(cmd, check=True)

    # ── Verify output ─────────────────────────────────────────────────────
    safetensors = list(out_path.glob("*.safetensors"))
    config_file = out_path / "config.json"

    if not safetensors or not config_file.exists():
        sys.exit(
            f"Merge seemed to complete but expected files are missing in {out_path}.\n"
            "Check the mlx_lm fuse output above for errors."
        )

    size_gb = sum(f.stat().st_size for f in safetensors) / 1e9
    print(f"\n  Safetensors : {len(safetensors)} shard(s), {size_gb:.1f} GB total")
    print(f"  config.json : {'found' if config_file.exists() else 'MISSING'}")

    print("\n" + "=" * 60)
    print("Merge complete.")
    print(f"  Merged model: {out_path}")
    print("  Next step → run: bash finetune/scripts/convert_gguf.sh")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge MLX LoRA adapter into base model weights"
    )
    parser.add_argument(
        "--mlx-model", default=str(MLX_MODEL_DIR),
        help="Path to the MLX base model (produced by train.py's conversion step)"
    )
    parser.add_argument(
        "--adapter", default=str(ADAPTER_DIR),
        help="Path to the trained LoRA adapter directory"
    )
    parser.add_argument(
        "--output", default=str(MERGED_DIR),
        help="Where to save the merged fp16 model"
    )
    args = parser.parse_args()

    merge(args.mlx_model, args.adapter, args.output)


if __name__ == "__main__":
    main()
