#!/usr/bin/env bash
# =============================================================================
# Convert merged model → GGUF → quantised GGUF for Ollama
#
# WHY THIS SCRIPT STILL NEEDS LLAMA.CPP
# --------------------------------------
# mlx_lm fuse has a --export-gguf flag, but it only handles model_type values
# "llama", "mixtral", and "mistral".  Qwen2.5 has model_type = "qwen2", so
# that flag raises ValueError.  llama.cpp's convert_hf_to_gguf.py supports
# Qwen2 fully and is the correct tool for this conversion.
#
# WHAT CHANGED VERSUS THE PYTORCH VERSION
# ----------------------------------------
# Old approach (CUDA):
#   1. llama.cpp convert_hf_to_gguf.py  (heavy Python + safetensors parsing)
#   2. llama-quantize
#
# New approach (MLX):
#   1. mlx_lm fuse --dequantize    ← merge_adapter.py already did this
#   2. llama.cpp convert_hf_to_gguf.py  (same as before — reads the merged safetensors)
#   3. llama-quantize              ← same binary, same Q4_K_M target
#
# The merged model produced by mlx_lm fuse --dequantize is in HuggingFace
# safetensors format (config.json + *.safetensors shards), identical to what
# you get from saving a HuggingFace model.  llama.cpp reads it natively.
#
# PREREQUISITES
# -------------
# Build llama.cpp with Metal support (uses your M1 GPU for fast quantisation):
#
#   git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
#   cd ~/llama.cpp
#   cmake -B build -DGGML_METAL=ON
#   cmake --build build -j10
#   pip install -r requirements.txt      # inside your project venv
#
# cmake and git are already installed on your system.
# -DGGML_METAL=ON enables the Metal backend so llama-quantize runs on M1 GPU.
#
# QUANTISATION OPTIONS
# --------------------
#   Q4_K_M  : 4-bit mixed  — ~4.1 GB, ~97% quality  (RECOMMENDED)
#   Q5_K_M  : 5-bit mixed  — ~4.8 GB, ~98% quality
#   Q8_0    : 8-bit         — ~7.2 GB, ~99.9% quality (near lossless)
#   Q2_K    : 2-bit         — ~2.7 GB, noticeable quality loss
#
# Usage:
#   bash finetune/scripts/convert_gguf.sh
#   bash finetune/scripts/convert_gguf.sh Q5_K_M
#   bash finetune/scripts/convert_gguf.sh Q8_0
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MERGED_DIR="${REPO_ROOT}/finetune/merged_model"
GGUF_DIR="${REPO_ROOT}/finetune/gguf"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${HOME}/llama.cpp}"
QUANT="${1:-Q4_K_M}"

MODEL_NAME="qwen2.5-salesforce"
FP16_GGUF="${GGUF_DIR}/${MODEL_NAME}.fp16.gguf"
QUANT_GGUF="${GGUF_DIR}/${MODEL_NAME}-${QUANT}.gguf"

# ── Colour helpers ────────────────────────────────────────────────────────────
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }

bold "============================================================"
bold "GGUF Conversion  (Qwen2.5 Salesforce Assistant)"
bold "============================================================"
echo "  Merged model : ${MERGED_DIR}"
echo "  Output GGUF  : ${QUANT_GGUF}"
echo "  Quantisation : ${QUANT}"
echo ""

# ── Validate prerequisites ────────────────────────────────────────────────────
if [ ! -d "${MERGED_DIR}" ] || [ -z "$(ls -A "${MERGED_DIR}" 2>/dev/null)" ]; then
    red "ERROR: Merged model not found at ${MERGED_DIR}"
    echo "       Run: python finetune/scripts/merge_adapter.py"
    exit 1
fi

if [ ! -f "${MERGED_DIR}/config.json" ]; then
    red "ERROR: ${MERGED_DIR}/config.json missing."
    echo "       The merged model directory is incomplete."
    echo "       Re-run: python finetune/scripts/merge_adapter.py"
    exit 1
fi

if [ ! -d "${LLAMA_CPP_DIR}" ]; then
    red "ERROR: llama.cpp not found at ${LLAMA_CPP_DIR}"
    echo ""
    echo "Build it with:"
    echo "  git clone https://github.com/ggerganov/llama.cpp ${LLAMA_CPP_DIR}"
    echo "  cd ${LLAMA_CPP_DIR}"
    echo "  cmake -B build -DGGML_METAL=ON    # Metal = M1 GPU acceleration"
    echo "  cmake --build build -j10"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "Then re-run this script.  Estimated build time: 3-5 minutes."
    exit 1
fi

CONVERT_SCRIPT="${LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
QUANTIZE_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"

if [ ! -f "${CONVERT_SCRIPT}" ]; then
    red "ERROR: ${CONVERT_SCRIPT} not found."
    echo "       llama.cpp exists but convert_hf_to_gguf.py is missing."
    echo "       Pull the latest: cd ${LLAMA_CPP_DIR} && git pull"
    exit 1
fi

if [ ! -f "${QUANTIZE_BIN}" ]; then
    red "ERROR: llama-quantize binary not found at ${QUANTIZE_BIN}"
    echo "       Build llama.cpp first:"
    echo "  cd ${LLAMA_CPP_DIR} && cmake -B build -DGGML_METAL=ON && cmake --build build -j10"
    exit 1
fi

mkdir -p "${GGUF_DIR}"

# ── Step 1: HuggingFace safetensors → fp16 GGUF ──────────────────────────────
#
# convert_hf_to_gguf.py reads all *.safetensors shards and config.json from
# the merged model directory, maps weight names to GGUF format, and writes a
# single binary file.  This step does NOT quantise — it is a lossless format
# conversion at fp16 precision.
#
# The output file is large (~14 GB for 7B fp16) but is only an intermediate.
# We delete it after quantisation to reclaim disk space.

if [ -f "${FP16_GGUF}" ]; then
    echo "[1/2] fp16 GGUF already exists ($(du -sh "${FP16_GGUF}" | cut -f1)) — skipping."
else
    bold "[1/2] Converting safetensors → fp16 GGUF …"
    python3 "${CONVERT_SCRIPT}" \
        "${MERGED_DIR}" \
        --outtype f16 \
        --outfile "${FP16_GGUF}"
    green "      Done: ${FP16_GGUF} ($(du -sh "${FP16_GGUF}" | cut -f1))"
fi

# ── Step 2: fp16 GGUF → quantised GGUF ───────────────────────────────────────
#
# llama-quantize reads the fp16 GGUF and produces a smaller quantised version.
# Q4_K_M uses a mixed strategy:
#   - Q6_K for attention output and FFN down projections (more sensitive)
#   - Q4_K for all other weight matrices
# This preserves quality better than uniform Q4_0 quantisation.
#
# On M1 Max with -DGGML_METAL=ON, quantisation runs on the GPU and completes
# in approximately 2-5 minutes for a 7B model.

bold "[2/2] Quantising to ${QUANT} …"
"${QUANTIZE_BIN}" \
    "${FP16_GGUF}" \
    "${QUANT_GGUF}" \
    "${QUANT}"
green "      Done: ${QUANT_GGUF} ($(du -sh "${QUANT_GGUF}" | cut -f1))"

# ── Clean up intermediate fp16 GGUF ──────────────────────────────────────────
echo ""
echo "The fp16 GGUF ($(du -sh "${FP16_GGUF}" | cut -f1)) is only needed for quantisation."
read -rp "Delete it now to reclaim disk space? [Y/n] " resp
resp="${resp:-Y}"
if [[ "${resp,,}" == "y" ]]; then
    rm -f "${FP16_GGUF}"
    green "Deleted ${FP16_GGUF}"
fi

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
bold "============================================================"
green "Conversion complete."
echo ""
echo "  Final GGUF : ${QUANT_GGUF}"
echo "  Size       : $(du -sh "${QUANT_GGUF}" | cut -f1)"
echo ""
echo "Next step → run: bash finetune/scripts/deploy_ollama.sh"
bold "============================================================"
