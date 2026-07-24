#!/usr/bin/env bash
# =============================================================================
# Build and deploy the fine-tuned model to your local Ollama instance.
#
# What this script does:
#   1. Verifies the GGUF file exists.
#   2. Removes any existing model with the same name from Ollama.
#   3. Builds the Ollama model from the Modelfile.
#   4. Runs a quick smoke test.
#   5. Updates config.yaml to use the new model name.
#
# Usage:
#   bash finetune/scripts/deploy_ollama.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GGUF_DIR="$REPO_ROOT/finetune/gguf"
MODELFILE="$REPO_ROOT/finetune/ollama/Modelfile"
MODEL_NAME="qwen2.5-salesforce"
QUANT="${QUANT:-Q4_K_M}"
GGUF_FILE="$GGUF_DIR/${MODEL_NAME}-${QUANT}.gguf"

echo "============================================================"
echo "Ollama Deployment"
echo "  Model name : $MODEL_NAME"
echo "  GGUF file  : $GGUF_FILE"
echo "============================================================"

# ── Validate prerequisites ────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    echo "ERROR: ollama is not installed or not in PATH."
    echo "       Install from: https://ollama.ai/download"
    exit 1
fi

if [ ! -f "$GGUF_FILE" ]; then
    echo "ERROR: GGUF file not found: $GGUF_FILE"
    echo "       Run: bash finetune/scripts/convert_gguf.sh"
    exit 1
fi

# ── Remove existing model (if any) ───────────────────────────────────────────
echo ""
echo "[1/3] Removing existing '$MODEL_NAME' model from Ollama (if present) …"
ollama rm "$MODEL_NAME" 2>/dev/null && echo "      Removed." || echo "      (none found — skipping)"

# ── Build model ───────────────────────────────────────────────────────────────
# ollama create reads the Modelfile, loads the GGUF, and registers the model.
# The Modelfile's FROM line must point to an accessible GGUF path.
# We create from the Modelfile directory so the relative path resolves correctly.
echo ""
echo "[2/3] Building Ollama model from Modelfile …"
(cd "$REPO_ROOT/finetune/ollama" && ollama create "$MODEL_NAME" -f Modelfile)
echo "      Build complete."

# ── Smoke test ────────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Running smoke test …"
RESPONSE=$(ollama run "$MODEL_NAME" "What is an OmniScript in Salesforce OmniStudio?" --nowordwrap 2>&1 | head -20)
echo ""
echo "Smoke test response (first 20 lines):"
echo "--------------------------------------"
echo "$RESPONSE"
echo "--------------------------------------"

# ── Show active models ────────────────────────────────────────────────────────
echo ""
echo "All Ollama models:"
ollama list

echo ""
echo "============================================================"
echo "Deployment complete."
echo ""
echo "To use the fine-tuned model:"
echo "  1. Update config.yaml:  llm_model: \"$MODEL_NAME\""
echo "  2. Restart your server: python server.py"
echo ""
echo "Or run directly:"
echo "  ollama run $MODEL_NAME"
echo "============================================================"
