#!/usr/bin/env python3
"""
LoRA fine-tuning via Apple MLX (mlx-lm) — M1/M2/M3 native.

HOW MLX DIFFERS FROM THE PYTORCH/QLORA APPROACH
-------------------------------------------------
The original train.py used HuggingFace TRL + BitsAndBytes (CUDA-only).
This version uses mlx-lm, Apple's fine-tuning framework built on MLX:

  • No BitsAndBytes — MLX has its own 4-bit quantisation that runs on Metal.
  • No flash-attn — MLX uses fused Metal attention kernels automatically.
  • No paged_adamw_8bit — MLX's AdamW manages Metal memory natively.
  • LoRA rank/dropout/scale are identical concepts; only the YAML key for
    "alpha" changes: mlx-lm calls it "scale" where scale ≈ alpha / r.
    Default scale=10 ≈ alpha=80 with r=8, which is more aggressive than the
    HF convention of alpha=2r. We use scale=10 for domain fine-tuning.

WHAT THIS SCRIPT DOES
---------------------
  1. (Optional) Convert Qwen2.5-7B-Instruct from HuggingFace → MLX format
     with 4-bit quantisation so training fits in ≈8 GB of unified memory.
     Skip this if you have ≥24 GB free and want fp16 precision.
  2. Create a symlink so mlx-lm finds valid.jsonl (it requires that exact name;
     our preprocessing saves val.jsonl).
  3. Calculate total training iterations from dataset size + epochs.
  4. Write a lora_config.yaml with all hyperparameters documented.
  5. Launch `mlx_lm lora --config lora_config.yaml`.

MEMORY USAGE ON M1 MAX (32 GB unified memory)
---------------------------------------------
  Mode                 | Model VRAM | Optimizer | Total
  ---------------------|-----------|-----------|-------
  4-bit quantised (q4) | ~4 GB      | ~2 GB     | ~8 GB   ← recommended
  fp16 (no quant)      | ~14 GB     | ~5 GB     | ~21 GB  ← possible but tight

Usage:
  python finetune/training/train.py
  python finetune/training/train.py --no-quantize       # train in fp16
  python finetune/training/train.py --epochs 5 --rank 16 --lr 1e-5
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT     = Path(__file__).resolve().parents[2]
FINETUNE_ROOT = REPO_ROOT / "finetune"
PROCESSED_DIR = FINETUNE_ROOT / "dataset" / "processed"
ADAPTER_DIR   = FINETUNE_ROOT / "adapters" / "qwen2.5-salesforce"
MLX_MODEL_DIR = FINETUNE_ROOT / "mlx_model"
CONFIG_PATH   = FINETUNE_ROOT / "training" / "lora_config.yaml"

# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

DEFAULTS = {
    "base_model":      "Qwen/Qwen2.5-7B-Instruct",
    "epochs":          3,
    "rank":            8,
    # scale ≈ alpha / r  (mlx-lm's name for the LoRA scaling factor)
    # scale=10 is equivalent to alpha≈80 at rank=8.
    # For conservative fine-tuning start at 5; for fast adaptation use 20.
    "scale":           10.0,
    "dropout":         0.05,
    "num_layers":      16,      # how many transformer layers get LoRA adapters
    "batch_size":      4,       # M1 Max with 4-bit model handles 4 easily
    "lr":              2e-5,
    "max_seq_length":  2048,
    "grad_accum":      4,       # effective batch = 4 × 4 = 16
    "steps_per_report": 10,
    "save_every":      100,
    "seed":            42,
    "quantize":        True,    # 4-bit quant for training (recommended on M1)
    "q_bits":          4,
}


# ---------------------------------------------------------------------------
# Step 1: convert HF model → MLX format
# ---------------------------------------------------------------------------

def convert_to_mlx(base_model: str, mlx_path: Path, quantize: bool, q_bits: int) -> Path:
    """
    Convert Qwen2.5-7B-Instruct from HuggingFace to MLX format.

    With --quantize (-q), the model is loaded in 4-bit which takes ~4 GB on
    disk and ~4 GB Metal memory during training.  Without quantisation, the
    fp16 model is ~14 GB.

    mlx_lm convert downloads the model from HuggingFace on first run and
    caches it in ~/.cache/huggingface.  Subsequent runs are instant.
    """
    if (mlx_path / "config.json").exists():
        print(f"MLX model already exists at {mlx_path} — skipping conversion.")
        return mlx_path

    # mlx_lm convert refuses to write into an existing directory.
    # Remove it if it exists but is incomplete (no config.json).
    if mlx_path.exists():
        import shutil
        shutil.rmtree(mlx_path)

    print(f"\n[Step 1/4] Converting {base_model} → MLX format …")
    print(f"  Output : {mlx_path}")
    print(f"  4-bit  : {quantize}")

    cmd = [
        sys.executable, "-m", "mlx_lm", "convert",
        "--hf-path",  base_model,
        "--mlx-path", str(mlx_path),
        "--dtype",    "float16",
    ]
    if quantize:
        cmd += ["-q", "--q-bits", str(q_bits)]

    subprocess.run(cmd, check=True)
    print(f"  Conversion complete → {mlx_path}")
    return mlx_path


# ---------------------------------------------------------------------------
# Step 2: prepare data directory (mlx-lm requires "valid.jsonl")
# ---------------------------------------------------------------------------

def prepare_data_dir(processed_dir: Path) -> Path:
    """
    mlx-lm looks for {data}/train.jsonl and {data}/valid.jsonl.
    Our preprocessing script saves val.jsonl (matching HuggingFace convention).
    We create a symlink so both names work without copying data.
    """
    train_file = processed_dir / "train.jsonl"
    val_file   = processed_dir / "val.jsonl"
    valid_file = processed_dir / "valid.jsonl"

    if not train_file.exists():
        sys.exit(f"train.jsonl not found at {processed_dir}\n"
                 "Run: python finetune/preprocessing/prepare_dataset.py generate")
    if not val_file.exists():
        sys.exit(f"val.jsonl not found at {processed_dir}\n"
                 "Run: python finetune/preprocessing/prepare_dataset.py generate")

    # Create symlink valid.jsonl → val.jsonl if it doesn't exist
    if not valid_file.exists():
        valid_file.symlink_to("val.jsonl")
        print(f"  Created symlink: valid.jsonl → val.jsonl")

    # Count training examples for iters calculation
    with train_file.open() as f:
        n_train = sum(1 for line in f if line.strip())
    with val_file.open() as f:
        n_val = sum(1 for line in f if line.strip())

    print(f"\n[Step 2/4] Data directory ready")
    print(f"  Train : {n_train:,} examples")
    print(f"  Val   : {n_val:,} examples")

    return n_train


# ---------------------------------------------------------------------------
# Step 3: build lora_config.yaml
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace, model_path: str, n_train: int) -> dict:
    """
    mlx-lm reads all training parameters from a single YAML file.

    Key differences from HuggingFace SFTConfig:
      - 'iters' instead of 'num_train_epochs' — we calculate it here
      - 'scale' instead of 'lora_alpha'
      - 'num_layers' controls how many transformer layers receive adapters
        (vs HuggingFace's target_modules which names specific weight names)
      - No per-device batch size — MLX handles device placement automatically
    """
    steps_per_epoch = max(1, n_train // args.batch_size)
    total_iters     = steps_per_epoch * args.epochs
    val_batches     = min(50, max(10, n_train // (args.batch_size * 10)))

    cfg = {
        # ── Model ──────────────────────────────────────────────────────────
        "model": model_path,

        # ── Training mode ──────────────────────────────────────────────────
        "train":          True,
        "fine_tune_type": "lora",
        "optimizer":      "adamw",

        # ── Data ───────────────────────────────────────────────────────────
        "data": str(PROCESSED_DIR),

        # ── LoRA architecture ──────────────────────────────────────────────
        # num_layers: how many transformer layers (from the output end) get
        # LoRA adapters.  16 = last 16 layers, -1 = all layers.
        # More layers → more parameters → slower but more expressive.
        # For 7B models with 28 layers, 16 covers the top 57%.
        "num_layers": args.num_layers,

        # ── LoRA hyperparameters ───────────────────────────────────────────
        "lora_parameters": {
            # rank: same concept as HuggingFace LoRA r.
            # 8 is conservative and fast; use 16 for richer adaptation.
            "rank": args.rank,
            # scale: equivalent to lora_alpha/r in HuggingFace.
            # scale=10 at rank=8 means effective alpha≈80.
            # Lower values (5) = more stable; higher (20) = faster adaptation.
            "scale": args.scale,
            # dropout: regularisation on LoRA outputs.
            "dropout": args.dropout,
        },

        # ── Training schedule ─────────────────────────────────────────────
        # iters: total gradient steps (not epochs).
        # Formula: (n_train / batch_size) * epochs
        "iters":            total_iters,
        "batch_size":       args.batch_size,
        "learning_rate":    args.lr,
        "grad_accumulation_steps": args.grad_accum,

        # ── LR schedule ───────────────────────────────────────────────────
        # Cosine decay with linear warmup for 5% of training.
        "lr_schedule": {
            "name":        "cosine_decay",
            "warmup":      max(1, int(total_iters * 0.05)),
            "warmup_init": 1e-7,
            "arguments":   [args.lr, total_iters],
        },

        # ── Sequence ──────────────────────────────────────────────────────
        "max_seq_length": args.max_seq_length,
        # mask_prompt=True: loss computed only on assistant tokens,
        # identical to the label=-100 masking in our HuggingFace version.
        "mask_prompt":    True,

        # ── Checkpointing ─────────────────────────────────────────────────
        "adapter_path":      str(ADAPTER_DIR),
        "save_every":        args.save_every,
        "steps_per_report":  args.steps_per_report,
        "steps_per_eval":    max(10, total_iters // 20),
        "val_batches":       val_batches,

        # ── Memory ────────────────────────────────────────────────────────
        # grad_checkpoint: recompute activations during backward pass to save
        # Metal memory.  Small throughput cost (~15%), large memory saving.
        "grad_checkpoint":       True,
        # clear_cache_threshold: free the MLX allocator cache when it grows
        # past this many bytes.  0 = always clear (safest on M1 Max 32 GB).
        "clear_cache_threshold": 0,

        # ── Reproducibility ───────────────────────────────────────────────
        "seed": args.seed,
    }

    return cfg, total_iters, steps_per_epoch


# ---------------------------------------------------------------------------
# Step 4: run training
# ---------------------------------------------------------------------------

def run_training(config_path: Path) -> None:
    print(f"\n[Step 4/4] Starting mlx-lm LoRA training …")
    print(f"  Config : {config_path}")
    print("  (Press Ctrl-C to stop; training resumes via --resume-adapter-file)")

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--config", str(config_path),
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="MLX LoRA fine-tuning for Qwen2.5")
    parser.add_argument("--model",        default=DEFAULTS["base_model"],
                        help="HuggingFace model ID or local MLX model path")
    parser.add_argument("--epochs",       type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--rank",         type=int,   default=DEFAULTS["rank"],
                        help="LoRA rank (4, 8, 16, 32)")
    parser.add_argument("--scale",        type=float, default=DEFAULTS["scale"],
                        help="LoRA scale (≈ alpha/r); default 10.0")
    parser.add_argument("--dropout",      type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--num-layers",   type=int,   default=DEFAULTS["num_layers"],
                        dest="num_layers",
                        help="Number of transformer layers to apply LoRA to (-1=all)")
    parser.add_argument("--batch-size",   type=int,   default=DEFAULTS["batch_size"],
                        dest="batch_size")
    parser.add_argument("--lr",           type=float, default=DEFAULTS["lr"])
    parser.add_argument("--seq-len",      type=int,   default=DEFAULTS["max_seq_length"],
                        dest="max_seq_length")
    parser.add_argument("--grad-accum",   type=int,   default=DEFAULTS["grad_accum"],
                        dest="grad_accum")
    parser.add_argument("--save-every",   type=int,   default=DEFAULTS["save_every"],
                        dest="save_every")
    parser.add_argument("--steps-per-report", type=int,
                        default=DEFAULTS["steps_per_report"],
                        dest="steps_per_report")
    parser.add_argument("--seed",         type=int,   default=DEFAULTS["seed"])
    parser.add_argument("--no-quantize",  action="store_false", dest="quantize",
                        help="Train in fp16 instead of 4-bit (needs ~21 GB)")
    parser.add_argument("--q-bits",       type=int,   default=DEFAULTS["q_bits"],
                        dest="q_bits",
                        help="Bits for MLX quantisation (4 or 8)")
    parser.add_argument("--skip-convert", action="store_true", dest="skip_convert",
                        help="Skip model conversion (use if MLX model already exists)")
    args = parser.parse_args()

    print("=" * 60)
    print("MLX LoRA Fine-tuning — Qwen2.5 Salesforce Assistant")
    print("=" * 60)
    print(f"  Base model  : {args.model}")
    print(f"  Rank        : {args.rank}  |  Scale : {args.scale}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}  "
          f"(effective: {args.batch_size * args.grad_accum})")
    print(f"  Quantise    : {'4-bit' if args.quantize else 'fp16'}")

    # ── Step 1: convert to MLX ────────────────────────────────────────────
    if args.skip_convert and MLX_MODEL_DIR.exists():
        model_path = str(MLX_MODEL_DIR)
        print(f"\n[Step 1/4] Using existing MLX model at {model_path}")
    else:
        mlx_path   = convert_to_mlx(args.model, MLX_MODEL_DIR, args.quantize, args.q_bits)
        model_path = str(mlx_path)

    # ── Step 2: prepare data ──────────────────────────────────────────────
    n_train = prepare_data_dir(PROCESSED_DIR)

    # ── Step 3: build config ──────────────────────────────────────────────
    cfg, total_iters, steps_per_epoch = build_config(args, model_path, n_train)

    print(f"\n[Step 3/4] Writing training config → {CONFIG_PATH}")
    print(f"  Total iters : {total_iters} "
          f"({steps_per_epoch} steps/epoch × {args.epochs} epochs)")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Also save a human-readable training summary
    summary = {
        "base_model":    args.model,
        "rank":          args.rank,
        "scale":         args.scale,
        "epochs":        args.epochs,
        "total_iters":   total_iters,
        "lr":            args.lr,
        "batch_size":    args.batch_size,
        "effective_batch": args.batch_size * args.grad_accum,
        "quantize":      args.quantize,
        "n_train":       n_train,
    }
    (FINETUNE_ROOT / "training" / "training_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # ── Step 4: train ─────────────────────────────────────────────────────
    run_training(CONFIG_PATH)

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Adapter saved to : {ADAPTER_DIR}")
    print("  Next step → run : python finetune/scripts/merge_adapter.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
