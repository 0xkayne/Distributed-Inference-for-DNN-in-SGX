#!/bin/bash
# 
# BERT Profiling Modes Comparison Script
#
# This script demonstrates the difference between batched and per-head profiling modes.
#

set -e

echo "========================================================================"
echo "BERT Multi-Head Attention Profiling Modes Comparison"
echo "========================================================================"
echo ""
echo "This script will run BERT profiling in two modes:"
echo "  1. Batched Mode (standard): All heads computed together"
echo "  2. Per-Head Mode (fine-grained): Each head profiled separately"
echo ""
echo "Using BERT-mini for faster execution (~5 minutes total)"
echo "========================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "experiments/models/profile_bert_enclave.py" ]; then
    echo "Error: Please run this script from the TAOISM project root directory"
    exit 1
fi

# Check if enclave library exists
if [ ! -f "App/bin/enclave_bridge.so" ]; then
    echo "Warning: Enclave library not found. Run 'make' first."
    echo "Continuing anyway (may fail)..."
fi

# Set LD_PRELOAD for libstdc++ compatibility
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Activate conda environment if needed
# conda activate taoism

echo "========================================================================"
echo "Mode 1: Batched Profiling (Standard)"
echo "========================================================================"
echo "Command: python -m experiments.models.profile_bert_enclave --model mini"
echo ""

python -m experiments.models.profile_bert_enclave \
    --model mini \
    --seq-len 128 \
    --iterations 5 \
    --warmup 2

echo ""
echo "✓ Batched profiling completed"
echo "  Output: experiments/data/bert_mini_enclave_layers.csv"
echo ""

# Count layers in batched mode
BATCHED_LAYERS=$(wc -l < experiments/data/bert_mini_enclave_layers.csv)
echo "  Total layers (batched): $BATCHED_LAYERS"
echo ""

echo "========================================================================"
echo "Mode 2: Per-Head Profiling (Fine-Grained)"
echo "========================================================================"
echo "Command: python -m experiments.models.profile_bert_enclave --model mini --per-head"
echo ""

python -m experiments.models.profile_bert_enclave \
    --model mini \
    --seq-len 128 \
    --iterations 5 \
    --warmup 2 \
    --per-head

echo ""
echo "✓ Per-head profiling completed"
echo "  Output: experiments/data/bert_mini_enclave_per_head_layers.csv"
echo ""

# Count layers in per-head mode
PERHEAD_LAYERS=$(wc -l < experiments/data/bert_mini_enclave_per_head_layers.csv)
echo "  Total layers (per-head): $PERHEAD_LAYERS"
echo ""

echo "========================================================================"
echo "Comparison Summary"
echo "========================================================================"
echo ""
echo "Layer Count:"
echo "  - Batched mode:  $BATCHED_LAYERS layers"
echo "  - Per-head mode: $PERHEAD_LAYERS layers"
echo "  - Ratio:         $(echo "scale=1; $PERHEAD_LAYERS / $BATCHED_LAYERS" | bc)x more layers in per-head mode"
echo ""
echo "Files Generated:"
echo "  1. experiments/data/bert_mini_enclave_layers.csv (batched)"
echo "  2. experiments/data/bert_mini_enclave_per_head_layers.csv (per-head)"
echo ""
echo "Next Steps:"
echo "  - Compare CSV files to see per-head details"
echo "  - Analyze performance differences between heads"
echo "  - Use per-head data for fine-grained optimization"
echo ""
echo "Example analysis:"
echo "  grep 'head[0-9]*_qk_matmul' experiments/data/bert_mini_enclave_per_head_layers.csv | head -10"
echo ""
echo "========================================================================"
echo "Comparison Complete!"
echo "========================================================================"
