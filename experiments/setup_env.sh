#!/bin/bash
#
# Setup Environment for TAOISM Experiments
# 设置实验环境
#

echo "Setting up TAOISM experiment environment..."

# 1. Activate conda environment
echo "1. Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate taoism

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate taoism environment"
    echo "Please create it first: conda create -n taoism python=3.7"
    exit 1
fi

# 2. Source SGX SDK
echo "2. Sourcing SGX SDK..."
if [ -f /opt/intel/sgxsdk/environment ]; then
    source /opt/intel/sgxsdk/environment
    echo "   ✓ SGX SDK loaded"
else
    echo "   ⚠ SGX SDK not found (optional for CPU-only tests)"
fi

# 3. Set library path
echo "3. Setting library path..."
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH
echo "   ✓ LD_LIBRARY_PATH configured"

# 4. Change to TAOISM directory
echo "4. Changing to TAOISM directory..."
cd /root/exp_DNN_SGX/TAOISM
echo "   ✓ Current directory: $(pwd)"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Environment ready!"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Quick commands:"
echo ""
echo "  Test environment:"
echo "    python experiments/quick_test.py"
echo ""
echo "  Measure communication (verified working):"
echo "    python experiments/measurement/measure_communication.py --single-model NiN"
echo ""
echo "  Use existing baseline (recommended):"
echo "    cd teeslice && python -m sgx_resnet_cifar --arch resnet18 --mode CPU --num_repeat 10"
echo ""
echo "═══════════════════════════════════════════════════"
echo ""

# Keep the environment active
exec bash

