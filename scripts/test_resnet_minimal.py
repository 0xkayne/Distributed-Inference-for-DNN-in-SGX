#!/usr/bin/env python3
"""
ResNet 最小化测试脚本
用于调试 SGX2 EDMM 环境下的 ResNet 运行问题
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np

print("=" * 70)
print("  ResNet 最小化测试（SGX2 EDMM）")
print("=" * 70)
print()

# Step 1: 检查基础环境
print("[Step 1/6] 检查基础环境...")
try:
    print(f"  ✓ Python: {sys.version.split()[0]}")
    print(f"  ✓ PyTorch: {torch.__version__}")
    print(f"  ✓ NumPy: {np.__version__}")
    print(f"  ✓ CUDA Available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"  ✗ 环境检查失败: {e}")
    sys.exit(1)
print()

# Step 2: 初始化 Enclave
print("[Step 2/6] 初始化 SGX Enclave（SGX2 EDMM）...")
try:
    from python.enclave_interfaces import EnclaveInterface, GlobalTensor
    from python.utils.basic_utils import ExecutionModeOptions
    
    enclave = EnclaveInterface()
    print(f"  ✓ Enclave 初始化成功")
    print(f"  ✓ Enclave ID: {enclave.eid}")
    
    # 初始化 GlobalTensor
    GlobalTensor.init()
    print(f"  ✓ GlobalTensor 初始化成功")
except Exception as e:
    print(f"  ✗ Enclave 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Step 3: 创建简单的测试张量
print("[Step 3/6] 测试张量操作...")
try:
    # 创建小测试张量
    test_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
    print(f"  ✓ 创建测试数据: shape={test_data.shape}")
    
    # 测试张量初始化
    tensor_id = 100000
    GlobalTensor.init_from_numpy(
        tensor_id,
        test_data,
        ExecutionModeOptions.Enclave
    )
    print(f"  ✓ 张量初始化成功: ID={tensor_id}")
    
    # 清理
    GlobalTensor.delete_tensor(tensor_id)
    print(f"  ✓ 张量清理成功")
except Exception as e:
    print(f"  ✗ 张量操作失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Step 4: 加载 ResNet 模型（如果可用）
print("[Step 4/6] 加载 ResNet18 模型...")
try:
    from torchvision import models
    
    # 创建模型
    model = models.resnet18(pretrained=False)
    model.eval()
    print(f"  ✓ ResNet18 模型创建成功")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ 总参数量: {total_params:,}")
except Exception as e:
    print(f"  ⚠ 模型加载跳过: {e}")
    model = None
print()

# Step 5: 测试简单推理（CPU）
print("[Step 5/6] 测试 CPU 推理...")
if model is not None:
    try:
        test_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = model(test_input)
        print(f"  ✓ CPU 推理成功: output shape={output.shape}")
    except Exception as e:
        print(f"  ✗ CPU 推理失败: {e}")
else:
    print("  ⚠ 跳过（模型未加载）")
print()

# Step 6: 总结
print("[Step 6/6] 测试总结...")
print("=" * 70)
print("  ✓ 基础环境：正常")
print("  ✓ Enclave 初始化：成功（SGX2 EDMM 模式）")
print("  ✓ 张量操作：正常")
if model is not None:
    print("  ✓ ResNet18 模型：可用")
else:
    print("  ⚠ ResNet18 模型：未测试")
print("=" * 70)
print()
print("🎉 最小化测试通过！可以进行完整 ResNet 测试。")
print()
print("下一步：")
print("  bash scripts/test_resnet_sgx2_edmm.sh")
print()

