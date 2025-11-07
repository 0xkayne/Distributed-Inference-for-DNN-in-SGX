# SGX2 EDMM ResNet 运行问题解决方案

## 问题分析

在 SGX2 EDMM 环境下运行 ResNet 测试时遇到的主要问题：

```
OSError: libstdc++.so.6: version `GLIBCXX_3.4.32' not found
```

**根本原因**：
- `enclave_bridge.so` 使用系统 GCC 13 编译，需要 `GLIBCXX_3.4.32`
- Conda 环境中的 `libstdc++.so.6` 版本较旧（Python 3.7）
- Python ctypes 加载动态库时优先使用 conda 环境的库

## 解决方案

### 方案 1：使用系统 Python（推荐）

不使用 conda 环境，直接使用系统 Python：

```bash
# 安装必要的 Python 包到系统
sudo apt-get install python3-pip python3-numpy python3-torch python3-torchvision

# 设置环境
source /opt/intel/sgxsdk/environment
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH

# 运行测试
cd /root/exp_DNN_SGX/TAOISM
python3 -m teeslice.sgx_resnet_cifar --arch resnet18 --mode Enclave --batch_size 1
```

### 方案 2：更新 Conda 环境的 libstdc++

```bash
# 激活环境
conda activate taoism

# 从系统复制新版 libstdc++
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $CONDA_PREFIX/lib/libstdc++.so.6.new
mv $CONDA_PREFIX/lib/libstdc++.so.6 $CONDA_PREFIX/lib/libstdc++.so.6.backup
mv $CONDA_PREFIX/lib/libstdc++.so.6.new $CONDA_PREFIX/lib/libstdc++.so.6

# 运行测试
bash scripts/test_resnet_sgx2_edmm.sh
```

### 方案 3：使用 LD_PRELOAD 强制加载系统库

```bash
# 激活环境
conda activate taoism
source /opt/intel/sgxsdk/environment

# 使用 LD_PRELOAD 强制加载系统 libstdc++
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH

# 运行测试
python3 -m teeslice.sgx_resnet_cifar --arch resnet18 --mode Enclave --batch_size 1
```

### 方案 4：创建新的 Python 3.10+ 环境（最佳长期方案）

```bash
# 创建新环境（Python 3.10 有更新的 libstdc++）
conda create -n taoism_sgx2 python=3.10 numpy -y
conda activate taoism_sgx2

# 安装 PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 或使用 pip 安装（避免 conda 的库）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 设置环境并运行
source /opt/intel/sgxsdk/environment
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH
bash scripts/test_resnet_sgx2_edmm.sh
```

## 验证步骤

### 1. 验证硬件支持

```bash
bash scripts/check_sgx2_edmm.sh
```

预期输出：`✓ System appears to support SGX2 with EDMM`

### 2. 验证编译

```bash
source /opt/intel/sgxsdk/environment
make clean && make
```

预期输出：
- `enclave.signed.so` 生成成功
- 包含：`Enclave can run on both SGX1 and SGX2 platforms. Only on SGX2 platform can it take advantage of dynamic features.`

### 3. 最小化测试

```bash
# 使用上述任一解决方案设置环境后
python3 scripts/test_resnet_minimal.py
```

预期输出：
```
✓ 基础环境：正常
✓ Enclave 初始化：成功（SGX2 EDMM 模式）
✓ 张量操作：正常
✓ ResNet18 模型：可用
```

### 4. 完整 ResNet 测试

```bash
bash scripts/test_resnet_sgx2_edmm.sh
```

预期输出：
```
✓ SGX2 EDMM 硬件支持：通过
✓ Enclave 初始化：通过
✓ ResNet18 推理：通过
🎉 SGX2 EDMM 功能验证成功！
```

## 快速测试脚本

已创建以下测试脚本：

1. **`scripts/check_sgx2_edmm.sh`** - 硬件能力检测
2. **`scripts/test_resnet_sgx2_edmm.sh`** - 完整 ResNet 测试
3. **`scripts/test_resnet_minimal.py`** - 最小化功能测试
4. **`scripts/test_enclave_init_debug.py`** - Enclave 初始化调试

## 常见问题

### Q1: 如何确认 SGX2 EDMM 真正启用？

查看 Enclave 初始化输出：
```
✓ SGX2 EDMM features detected and enabled
```

或在代码中检查 `misc_attr.misc_select & MISC_EXINFO`。

### Q2: 如何监控 EDMM 内存使用？

在 Enclave 代码中调用：
```cpp
auto& edmm_mgr = EdmmManager::getInstance();
edmm_mgr.print_stats();
```

或通过 OCALL：
```cpp
ocall_print_edmm_stats(...);
```

### Q3: Enclave 内存不足怎么办？

编辑 `Enclave/Enclave.config.xml`：
```xml
<HeapMaxSize>0x80000000</HeapMaxSize>  <!-- 增加到 2GB -->
```

然后重新编译：
```bash
make clean && make
```

### Q4: 如何对比 SGX1 vs SGX2 EDMM 性能？

1. 记录当前 SGX2 EDMM 配置的性能
2. 修改 `Enclave.config.xml`，将 `HeapMaxSize` 改小（模拟 SGX1）
3. 重新编译运行，对比结果

## 后续实验建议

1. **测试不同 batch_size**
   ```bash
   python3 -m teeslice.sgx_resnet_cifar --arch resnet18 --mode Enclave --batch_size 4
   python3 -m teeslice.sgx_resnet_cifar --arch resnet18 --mode Enclave --batch_size 8
   ```

2. **测试更深的模型**
   ```bash
   python3 -m teeslice.sgx_resnet_cifar --arch resnet50 --mode Enclave --batch_size 1
   python3 -m teeslice.sgx_resnet_cifar --arch resnet101 --mode Enclave --batch_size 1
   ```

3. **监控 EDMM 统计信息**
   - 在测试中添加 EDMM 统计输出
   - 观察内存 commit/decommit 次数
   - 分析峰值内存使用

4. **性能对比实验**
   - Enclave vs GPU vs Verifier 模式
   - SGX1（小内存）vs SGX2 EDMM（大内存）
   - 不同模型深度/宽度的影响

## 参考文档

- SGX2 EDMM API: `/opt/intel/sgxsdk/include/sgx_rsrv_mem_mngr.h`
- 配置说明: `Enclave/Enclave.config.xml` 注释
- 迁移总结: `EDMM_MIGRATION_SUMMARY.md`
- 快速开始: `QUICK_START_EDMM.md`

