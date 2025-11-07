#!/bin/bash

# SGX2 EDMM ResNet 测试脚本（带库路径修复）
# 解决 conda libstdc++ 版本冲突问题

set -e

echo "========================================="
echo " ResNet SGX2 EDMM 测试"
echo "========================================="
echo ""

# 设置 SGX 环境
echo "[1] 设置 SGX 环境..."
source /opt/intel/sgxsdk/environment
echo "✓ SGX SDK 环境已加载"

# 关键：使用 LD_PRELOAD 强制加载系统 libstdc++
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH
echo "✓ 库路径已配置（使用系统 libstdc++）"
echo ""

# 运行 ResNet 测试
echo "[2] 运行 ResNet18 测试..."
echo ""

echo "===== Enclave 模式（SGX2 EDMM）====="
python -m teeslice.sgx_resnet_cifar --arch resnet18 --mode Enclave --batch_size 4

# echo ""
# echo "===== GPU 模式（对比基准）====="
# python -m teeslice.sgx_resnet_cifar --arch resnet18 --mode GPU --batch_size 1

echo ""
echo "===== Verifier 模式（对比基准）====="
python -m teeslice.sgx_resnet_cifar --arch resnet18 --mode CPU --batch_size 4

echo ""
echo "========================================="
echo " 测试完成"
echo "========================================="

