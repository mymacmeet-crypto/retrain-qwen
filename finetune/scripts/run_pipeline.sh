#!/usr/bin/env bash
# =============================================================================
# End-to-end QLoRA pipeline runner
#
# Runs all steps in sequence.  Safe to re-run — each step checks whether
# its output already exists before doing expensive work.
#
# Usage:
#   bash finetune/scripts/run_pipeline.sh              # full pipeline
#   bash finetune/scripts/run_pipeline.sh --from train # start from training
#
# Steps:
#   1. generate   — auto-generate Q&A dataset from ChromaDB
#   2. validate   — check dataset quality
#   3. train      — QLoRA fine-tuning
#   4. evaluate   — compare base vs fine-tuned perplexity & ROUGE
#   5. merge      — merge LoRA adapter into base weights
#   6. convert    — convert merged model to GGUF
#   7. deploy     — build and deploy to Ollama
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FROM_STEP="${1:-generate}"

steps=(generate validate train evaluate merge convert deploy)
run=false

log() { echo ""; echo "══════════════════════════════════════════════════════"; echo "  STEP: $1"; echo "══════════════════════════════════════════════════════"; }

for step in "${steps[@]}"; do
    [ "$step" = "$FROM_STEP" ] && run=true
    [ "$run" = false ] && continue

    case "$step" in
      generate)
        log "Dataset Generation"
        python "$REPO_ROOT/finetune/preprocessing/prepare_dataset.py" generate --n-pairs 3
        ;;
      validate)
        log "Dataset Validation"
        python "$REPO_ROOT/finetune/preprocessing/validate_dataset.py" --split train
        python "$REPO_ROOT/finetune/preprocessing/validate_dataset.py" --split val
        ;;
      train)
        log "QLoRA Training"
        python "$REPO_ROOT/finetune/training/train.py"
        ;;
      evaluate)
        log "Evaluation"
        python "$REPO_ROOT/finetune/evaluation/evaluate.py" --base-only
        python "$REPO_ROOT/finetune/evaluation/evaluate.py"
        ;;
      merge)
        log "Adapter Merge"
        python "$REPO_ROOT/finetune/scripts/merge_adapter.py"
        ;;
      convert)
        log "GGUF Conversion"
        bash "$REPO_ROOT/finetune/scripts/convert_gguf.sh"
        ;;
      deploy)
        log "Ollama Deployment"
        bash "$REPO_ROOT/finetune/scripts/deploy_ollama.sh"
        ;;
    esac
done

echo ""
echo "Pipeline complete."
