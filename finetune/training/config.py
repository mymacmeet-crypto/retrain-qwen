"""
Centralised configuration for MLX LoRA fine-tuning on Apple Silicon.

train.py reads DEFAULTS from here, then merges CLI overrides and writes
the final values into lora_config.yaml for mlx_lm.

evaluate.py and merge_adapter.py also import the path constants.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT     = Path(__file__).resolve().parents[2]
FINETUNE_ROOT = REPO_ROOT / "finetune"
ADAPTER_DIR   = FINETUNE_ROOT / "adapters" / "qwen2.5-salesforce"
MERGED_DIR    = FINETUNE_ROOT / "merged_model"
MLX_MODEL_DIR = FINETUNE_ROOT / "mlx_model"
TRAIN_FILE    = FINETUNE_ROOT / "dataset" / "processed" / "train.jsonl"
VAL_FILE      = FINETUNE_ROOT / "dataset" / "processed" / "val.jsonl"

# ---------------------------------------------------------------------------
# Default hyperparameters — all consumed by train.py
# ---------------------------------------------------------------------------

DEFAULTS = {
    # ── Model ─────────────────────────────────────────────────────────────
    # HuggingFace model ID.  train.py converts this to MLX format on first run.
    "base_model": "Qwen/Qwen2.5-7B-Instruct",

    # ── LoRA architecture ─────────────────────────────────────────────────
    # num_layers: how many transformer layers (counting from the output end)
    # receive LoRA adapters.
    #   8  — fast, minimal memory increase, good for tone/style tuning
    #   16 — balanced, covers top 57% of a 28-layer 7B model (recommended)
    #   -1 — all layers, maximum adaptation, slower + more memory
    "num_layers": 16,

    # rank (r): dimension of the LoRA low-rank matrices A and B.
    # Each adapted linear layer adds r × (input_dim + output_dim) parameters.
    #   4  — minimal, ~30 M trainable params
    #   8  — recommended for domain adaptation on M1 (~60 M trainable params)
    #   16 — richer, use if you have >24 GB free memory
    "rank": 8,

    # scale: mlx-lm's name for the LoRA scaling factor.
    # Relationship to HuggingFace convention: scale ≈ alpha / r
    # At rank=8, scale=10 → effective alpha≈80.
    # Lower (5) = conservative, stable; Higher (20) = aggressive, fast adaptation.
    "scale": 10.0,

    # dropout: applied to LoRA outputs for regularisation.
    "dropout": 0.05,

    # ── Training schedule ─────────────────────────────────────────────────
    # epochs: train.py converts this to total iteration count automatically.
    "epochs": 3,

    # batch_size: samples per gradient step.
    # With 4-bit quantised 7B model on M1 Max (32 GB):
    #   batch_size=4 is comfortable (~8 GB Metal memory used)
    #   batch_size=8 is possible but leaves little headroom
    "batch_size": 4,

    # grad_accum: gradient accumulation steps.
    # Effective batch = batch_size × grad_accum = 4 × 4 = 16.
    "grad_accum": 4,

    # lr: peak learning rate after warmup.
    # mlx-lm uses cosine decay by default.
    # 2e-5 is more conservative than HuggingFace's typical 2e-4 because
    # mlx-lm's scale parameter already amplifies gradient updates.
    "lr": 2e-5,

    # ── Sequence ──────────────────────────────────────────────────────────
    "max_seq_length": 2048,

    # ── Checkpointing ─────────────────────────────────────────────────────
    "save_every":       100,   # save adapter checkpoint every N steps
    "steps_per_report": 10,    # print loss every N steps

    # ── Quantisation ──────────────────────────────────────────────────────
    # quantize=True: convert Qwen2.5-7B to 4-bit MLX format before training.
    # This reduces Metal memory from ~14 GB (fp16) to ~4 GB.
    # Recommended for M1 Max 32 GB.  Set False only if you need fp16 precision.
    "quantize": True,
    "q_bits":   4,

    # ── Reproducibility ───────────────────────────────────────────────────
    "seed": 42,
}
