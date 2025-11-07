#!/usr/bin/env python3
"""
调试 Enclave 初始化问题
"""

import sys
import os
sys.path.insert(0, '.')

print("=" * 70)
print("  Enclave 初始化调试")
print("=" * 70)
print()

print("[1] 导入模块...")
try:
    from ctypes import cdll
    import ctypes
    print(f"  ✓ ctypes 导入成功")
except Exception as e:
    print(f"  ✗ ctypes 导入失败: {e}")
    sys.exit(1)

print()
print("[2] 加载 enclave_bridge.so...")
try:
    lib_path = "App/bin/enclave_bridge.so"
    if not os.path.exists(lib_path):
        print(f"  ✗ 文件不存在: {lib_path}")
        sys.exit(1)
    
    print(f"  路径: {lib_path}")
    lib = cdll.LoadLibrary(lib_path)
    print(f"  ✓ 加载成功: {lib}")
except Exception as e:
    print(f"  ✗ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("[3] 调用 initialize_enclave...")
try:
    # 设置返回类型
    lib.initialize_enclave.restype = ctypes.c_uint64
    
    print("  开始初始化...")
    eid = lib.initialize_enclave()
    print(f"  ✓ 初始化成功!")
    print(f"  ✓ Enclave ID: {eid}")
    
    print()
    print("[4] 销毁 Enclave...")
    lib.destroy_enclave(eid)
    print(f"  ✓ 销毁成功")
    
except Exception as e:
    print(f"  ✗ 初始化失败: {e}")
    print(f"  异常类型: {type(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("  ✓ 所有测试通过")
print("=" * 70)

