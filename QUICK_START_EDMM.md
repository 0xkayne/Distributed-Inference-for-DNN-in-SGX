# TAOISM SGX2 EDMM Quick Start Guide

## 快速开始

### 1. 验证硬件支持

```bash
# 检查 SGX2/EDMM 支持
bash scripts/check_sgx2_edmm.sh
```

期望输出：`✓ System appears to support SGX2 with EDMM`

### 2. 编译项目

```bash
# 加载 SGX SDK 环境
source /opt/intel/sgxsdk/environment

# 清理并编译
make clean && make
```

期望输出：
- `App/bin/enclave_bridge.so` (~156KB)
- `enclave.signed.so` (~448KB)

### 3. 运行测试

```bash
# 激活 conda 环境
conda activate taoism

# 设置库路径（避免版本冲突）
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH

# 运行基础测试
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from python.enclave_interfaces import EnclaveInterface

print("Initializing SGX2 EDMM Enclave...")
enclave = EnclaveInterface()
print("✓ SUCCESS: Enclave initialized with EDMM support!")
print(f"  Enclave ID: {enclave.eid}")
EOF
```

### 4. 运行完整模型测试

```bash
# 确保在 taoism 环境中
conda activate taoism

# 设置环境变量
source /opt/intel/sgxsdk/environment
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH

# 运行 ResNet 基准测试
bash teeslice/scripts/run_resnet_baseline.sh
```

## 关键文件说明

| 文件 | 用途 |
|------|------|
| `scripts/check_sgx2_edmm.sh` | 硬件能力检测 |
| `Enclave/Enclave.config.xml` | Enclave 内存配置 |
| `Include/sgx_edmm_wrapper.h` | EDMM API 封装 |
| `SGXDNN/chunk_manager.cpp` | 动态内存管理实现 |
| `App/enclave_bridge.cpp` | Enclave 初始化逻辑 |

## 验证 EDMM 启用

### 方法 1: 查看启动日志

Enclave 初始化时应显示：
```
Initializing Enclave with SGX2 EDMM support...
✓ SGX2 EDMM features detected and enabled
```

### 方法 2: 检查内存配置

```bash
cat Enclave/Enclave.config.xml
```

应包含：
- `<HeapMaxSize>0x40000000</HeapMaxSize>` (1GB)
- `<MiscSelect>1</MiscSelect>`
- `<MiscMask>0xFFFFFFFE</MiscMask>`

### 方法 3: 监控 EPC 使用

```bash
# 运行模型时监控 EPC 事件
sudo perf stat -e sgx:* python your_model.py
```

## 常见问题

### Q: 编译时找不到 EDMM API

**A:** 升级 SGX SDK 到 2.19 或更高版本。

### Q: 运行时提示 "EDMM features not detected"

**A:** 
1. 检查 CPU 是否支持 SGX2 (Ice Lake+)
2. 验证 BIOS 中 SGX 已启用
3. 确认安装了 DCAP 驱动而非旧版驱动

### Q: libstdc++ 版本冲突

**A:** 
```bash
# 使用系统 libstdc++
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH
```

### Q: Out of EPC memory

**A:** 
1. 增加 `HeapMaxSize` 在 `Enclave.config.xml`
2. 减少 `STORE_CHUNK_ELEM` 在 `Include/common_with_enclaves.h`
3. 关闭其他 SGX 应用

## 性能优化建议

### 1. 启用 Chunk 统计
```bash
export PRINT_CHUNK_INFO=1
python your_model.py
```

### 2. 调整 Chunk 大小

编辑 `Include/common_with_enclaves.h`:
```c
// 当前值
#define STORE_CHUNK_ELEM 401408
#define WORK_CHUNK_ELEM 409600

// 根据模型调整 - 更大的 chunk 减少 commit 次数，但增加内存占用
```

### 3. 激进回收（可选）

在 `SGXDNN/chunk_manager.cpp` 的 `return_chunk_id()` 中取消注释 decommit 代码。

## 下一步

1. **性能分析**: 使用 `perf` 分析 EPC page fault
2. **模型优化**: 调整层分配策略以减少 Enclave 内存使用
3. **批处理测试**: 运行多批次推理验证内存正确释放
4. **对比测试**: 与 SGX1 版本对比延迟和吞吐量

## 获取帮助

详细文档：
- `README.md` - 完整安装和使用指南
- `EDMM_MIGRATION_SUMMARY.md` - 迁移技术细节
- `README.md#troubleshooting` - 故障排除

## 注意事项

⚠️ **重要**: 这是研究原型，不应用于生产环境。

✓ **测试**: 在真实 SGX2 硬件上测试
✓ **监控**: 密切关注 EPC 使用和性能指标
✓ **备份**: 保留 SGX1 版本作为备份

---

**最后更新**: 2025年11月5日

