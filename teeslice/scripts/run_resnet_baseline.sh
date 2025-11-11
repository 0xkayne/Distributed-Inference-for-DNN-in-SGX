#!/bin/bash

# 设置 SGX 环境
source /opt/intel/sgxsdk/environment

# 设置库路径，优先使用系统的 libstdc++，避免 conda 环境中的版本冲突
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH

# 运行 ResNet 测试
# Enclave 模式
python -m teeslice.sgx_resnet_cifar --arch resnet18 --mode Enclave --batch_size 1

# GPU 模式对比
# python -m teeslice.sgx_resnet_cifar --arch resnet18 --mode GPU --batch_size 1

# Verifier 模式对比
# python -m teeslice.sgx_resnet_cifar --arch resnet18 --mode Verifier --batch_size 1
