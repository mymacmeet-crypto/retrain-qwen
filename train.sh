#!/usr/bin/env bash
# =============================================================================
# train.sh — Salesforce QLoRA daily runner
#
# DAILY USAGE (dataset generation — 200 chunks at a time):
#   bash train.sh generate     ← run this each day until all chunks are done
#   bash train.sh status       ← check how many chunks have been processed
#
# AFTER ALL CHUNKS ARE DONE (run once):
#   bash train.sh train        ← train + merge + convert + deploy in one go
#
# INDIVIDUAL STEPS (if you need to re-run one part):
#   bash train.sh merge        ← merge LoRA adapter into base model
#   bash train.sh convert      ← convert merged model to GGUF
#   bash train.sh deploy       ← register GGUF with Ollama
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-generate}"

PYTHON="python3"
PREPARE="$PYTHON $REPO_ROOT/finetune/preprocessing/prepare_dataset.py"
TRAIN_PY="$PYTHON $REPO_ROOT/finetune/training/train.py"
MERGE_PY="$PYTHON $REPO_ROOT/finetune/scripts/merge_adapter.py"
CONVERT_SH="bash $REPO_ROOT/finetune/scripts/convert_gguf.sh"
DEPLOY_SH="bash $REPO_ROOT/finetune/scripts/deploy_ollama.sh"
PROGRESS_FILE="$REPO_ROOT/finetune/dataset/processed/generation_progress.json"

# ── Colour helpers ────────────────────────────────────────────────────────────
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
blue()  { printf '\033[34m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[33m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
sep()   { echo ""; bold "════════════════════════════════════════════════════════"; }

# ── Check whether all ChromaDB chunks have been processed ────────────────────
all_chunks_done() {
    [ ! -f "$PROGRESS_FILE" ] && return 1
    local next total
    next=$($PYTHON -c "import json; d=json.load(open('$PROGRESS_FILE')); print(d.get('next_chunk_index', 0))")
    total=$($PYTHON -c "import json; d=json.load(open('$PROGRESS_FILE')); print(d.get('total_chunks', 0))")
    [ "$total" -gt 0 ] && [ "$next" -ge "$total" ]
}

# ── Check Ollama is running ───────────────────────────────────────────────────
check_ollama() {
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        red "ERROR: Ollama is not running."
        echo "       Start it in a separate terminal: ollama serve"
        exit 1
    fi
}

# =============================================================================
case "$MODE" in

# ─────────────────────────────────────────────────────────────────────────────
# generate — run one 200-chunk batch (call this daily)
# ─────────────────────────────────────────────────────────────────────────────
generate)
    sep
    bold "  Dataset Generation — 200 chunks"
    sep
    check_ollama

    # --resume is safe on the first run too: if there is no progress file,
    # it starts from chunk 0 and creates fresh train.jsonl / val.jsonl.
    $PREPARE generate --n-pairs 3 --max-chunks 200 --resume

    echo ""
    sep
    bold "  Progress"
    sep
    $PREPARE status

    echo ""
    if all_chunks_done; then
        green "All chunks have been processed — dataset is complete."
        echo ""
        yellow "Next step: bash train.sh train"
    else
        yellow "Run again tomorrow to process the next 200 chunks:"
        echo "  bash train.sh generate"
    fi
    ;;

# ─────────────────────────────────────────────────────────────────────────────
# status — show dataset generation progress only
# ─────────────────────────────────────────────────────────────────────────────
status)
    $PREPARE status
    if all_chunks_done; then
        echo ""
        green "Dataset is complete. Ready to train."
        echo "  bash train.sh train"
    fi
    ;;

# ─────────────────────────────────────────────────────────────────────────────
# train — full pipeline: train → merge → convert → deploy
# Run once after all chunks are processed.
# ─────────────────────────────────────────────────────────────────────────────
train)
    if ! all_chunks_done && [ -f "$PROGRESS_FILE" ]; then
        yellow "Warning: dataset generation is not complete."
        read -rp "Train anyway with partial dataset? [y/N] " resp
        [[ "${resp,,}" != "y" ]] && { echo "Aborted."; exit 0; }
    fi

    check_ollama

    # ── Step 1: Training ─────────────────────────────────────────────────────
    sep
    bold "  STEP 1 / 4 — LoRA Training  (mlx-lm)"
    bold "  This takes ~50 min per 200-chunk batch on M1 Max"
    sep
    $TRAIN_PY

    # ── Step 2: Merge ─────────────────────────────────────────────────────────
    sep
    bold "  STEP 2 / 4 — Merge LoRA adapter into base model"
    bold "  (mlx_lm fuse —  ~2-5 min)"
    sep
    $MERGE_PY

    # ── Step 3: GGUF conversion ───────────────────────────────────────────────
    sep
    bold "  STEP 3 / 4 — Convert to GGUF + Q4_K_M quantisation"
    bold "  (llama.cpp — ~5-10 min)"
    sep
    $CONVERT_SH

    # ── Step 4: Ollama deploy ─────────────────────────────────────────────────
    sep
    bold "  STEP 4 / 4 — Deploy to Ollama"
    sep
    $DEPLOY_SH

    sep
    green "Pipeline complete."
    echo ""
    echo "  Fine-tuned model is live in Ollama as: qwen2.5-salesforce"
    echo ""
    echo "  To switch your RAG server to the fine-tuned model:"
    echo "    1. Open config.yaml"
    echo "    2. Set:  llm_model: \"qwen2.5-salesforce\""
    echo "    3. Restart: python server.py"
    sep
    ;;

# ─────────────────────────────────────────────────────────────────────────────
# Individual steps (re-run one part if something failed)
# ─────────────────────────────────────────────────────────────────────────────
merge)
    sep; bold "  Merge adapter"; sep
    $MERGE_PY
    ;;

convert)
    sep; bold "  GGUF conversion"; sep
    $CONVERT_SH
    ;;

deploy)
    sep; bold "  Ollama deploy"; sep
    $DEPLOY_SH
    ;;

# ─────────────────────────────────────────────────────────────────────────────
*)
    bold "Usage: bash train.sh <command>"
    echo ""
    echo "  generate   Run next 200-chunk batch  (repeat daily until done)"
    echo "  status     Show dataset generation progress"
    echo "  train      Train + merge + convert + deploy  (run once when done)"
    echo ""
    echo "  merge      Merge LoRA adapter into base model"
    echo "  convert    Convert merged model to GGUF"
    echo "  deploy     Register GGUF with Ollama"
    ;;

esac
