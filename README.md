# 基于 DNN 拓扑结构的分布式并行推理加速研究

<div align="center">
  <a href="https://opensource.org/license/mit/">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch->=v1.7.0-EE4C2C.svg?style=flat-square">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/SGX2-EDMM_Enabled-orange">
  </a>
</div>

---

## 目录

- [1. 研究概述](#1-研究概述)
- [2. 方法论](#2-方法论)
- [3. 环境配置](#3-环境配置)
- [4. 基本使用](#4-基本使用)
- [5. 实验指南](#5-实验指南)
- [6. 代码结构](#6-代码结构)
- [7. STORE_CHUNK_ELEM 配置参考](#7-store_chunk_elem-配置参考)
- [8. 故障排除](#8-故障排除)
- [9. 主要结果](#9-主要结果)
- [10. 假设与限制](#10-假设与限制)
- [11. 引用](#11-引用)

---

## 1. 研究概述

### 1.1 研究背景与动机

在可信执行环境（TEE，如 Intel SGX）中执行 DNN 推理是实现隐私保护机器学习的重要手段。然而，SGX Enclave 的执行开销显著高于普通 CPU/GPU 执行。

### 1.2 核心研究问题

> **能否通过对 DNN 模型进行适当分割，利用网络拓扑中的并行结构，采用分布式方式实现端到端推理加速？**

### 1.3 研究贡献

1. **成本测量框架**：系统化测量 6 种 DNN 模型在 CPU/Enclave 环境下的层级性能
2. **分布式推理引擎**：支持任意 DAG 拓扑结构的多分区并行执行框架
3. **实验验证**：在 ResNet-18 上实现 **35% 端到端加速**

---

## 2. 方法论

### 2.1 核心思想

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        传统串行推理 vs 分布式并行推理                      │
├─────────────────────────────────────────────────────────────────────────┤
│   【传统方式】整个模型在 Enclave 中串行执行                                │
│   Input → [Layer1] → [Layer2] → ... → [LayerN] → Output                │
│                                                                         │
│   【本研究】利用网络并行结构，分区并行执行                                  │
│                    ┌─[Partition-A: Enclave]─┐                          │
│   Input → Split → │  Layer1 → Layer2        │ → Merge → Output         │
│                    └────────────────────────┘                          │
│                    ┌─[Partition-B: CPU]─────┐                          │
│                    │  Layer3 → Layer4        │ (并行执行)               │
│                    └────────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 分布式推理框架设计

- **FlexibleGraphWorker**：通用图执行线程，处理跨分区依赖
- **拓扑分析**：自动识别"切边"并创建通信队列
- **多线程安全**：共享模型实例，避免 GlobalTensor 冲突

---

## 3. 环境配置

### 3.1 硬件要求

| 组件 | 要求 |
|------|------|
| **CPU** | Intel SGX2 支持 (Ice Lake+) |
| **内存** | ≥16GB |
| **EPC** | ≥128MB（BIOS 中配置） |

### 3.2 软件要求

| 软件 | 版本 |
|------|------|
| Ubuntu | 20.04 LTS |
| Intel SGX SDK | ≥2.19 |
| Python | 3.7+ |
| PyTorch | ≥1.7.0 |

### 3.3 安装步骤

```bash
# 1. 克隆仓库
git clone <repository-url>
cd TAOISM

# 2. 检查 SGX 支持
bash scripts/check_sgx2_edmm.sh

# 3. 创建 Python 环境
conda create -n taoism python=3.7 -y
conda activate taoism
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch

# 4. 编译框架
source /opt/intel/sgxsdk/environment
make clean && make

# 预期输出：
#   - App/bin/enclave_bridge.so (~156KB)
#   - enclave.signed.so (~448KB)
```

---

## 4. 基本使用

### 4.1 环境激活（每次使用前必须执行）

```bash
# 完整环境设置脚本
source /opt/intel/sgxsdk/environment
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate taoism
```

**建议**：将上述命令添加到 `~/.bashrc` 或创建快捷脚本：

```bash
# 创建快捷脚本
cat > activate_taoism.sh << 'EOF'
#!/bin/bash
source /opt/intel/sgxsdk/environment
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate taoism
echo "✓ TAOISM 环境已激活"
EOF
chmod +x activate_taoism.sh

# 使用
source ./activate_taoism.sh
```

### 4.2 验证安装

```bash
# 验证 Enclave 初始化
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from python.enclave_interfaces import EnclaveInterface
print("Initializing SGX2 EDMM Enclave...")
enclave = EnclaveInterface()
print(f"✓ SUCCESS: Enclave ID = {enclave.eid}")
EOF
```

### 4.3 运行基本测试

```bash
# 快速验证
python experiments/quick_test.py

# ResNet-18 基线测试
bash teeslice/scripts/run_resnet_baseline.sh
```

### 4.4 重新编译（修改配置后）

**重要**：修改 `STORE_CHUNK_ELEM` 或 `Enclave.config.xml` 后，必须重新编译：

```bash
# 完整重新编译命令
rm -rf SGXDNN/bin_sgx && make clean && make SGX_MODE=HW all
```

**常用编译命令**：

| 场景 | 命令 |
|------|------|
| 普通编译 | `make` |
| 清理编译 | `make clean && make` |
| 修改 STORE_CHUNK_ELEM 后 | `rm -rf SGXDNN/bin_sgx && make clean && make SGX_MODE=HW all` |
| 检查 EDMM 支持 | `make check-edmm` |

---

## 5. 实验指南

### 5.1 实验总览

| 实验 | 脚本 | 输出 |
|------|------|------|
| **成本模型测量** | `experiments/measurement/*.py` | JSON 数据 |
| **分布式推理** | `experiments/models/distributed_resnet.py` | 推理延迟 |
| **分割策略对比** | `experiments/models/resnet_partition_benchmark.py` | 加速比 |

### 5.2 实验一：成本模型测量

```bash
cd /root/exp_DNN_SGX/TAOISM

# 快速测试
python experiments/quick_test.py

# 单模型测量
python experiments/measurement/measure_computation.py \
    --single-model NiN \
    --devices CPU Enclave \
    --batch-sizes 1 \
    --iterations 100

# 批量测量（30-60 分钟）
python experiments/run_all_measurements.py --models NiN ResNet18
```

**测量类型**：

| 脚本 | 测量内容 | 输出文件 |
|------|---------|---------|
| `measure_computation.py` | 每层执行时间 | `computation_cost_*.json` |
| `measure_communication.py` | 数据传输开销 | `communication_cost_*.json` |
| `measure_security.py` | CPU vs Enclave 对比 | `security_cost_*.json` |
| `measure_paging.py` | EPC 换页开销 | `paging_cost_*.json` |

### 5.3 实验二：分布式推理

```bash
# 运行分布式 ResNet-18 推理
python -m experiments.models.distributed_resnet
```

**预期输出**：
```
[Topology] Found cut edge: input->conv1 (CPU -> Enclave)
[Topology] Found cut edge: layer2_block1_relu2->layer3_block0_conv1 (Enclave -> CPU)
...
Total Latency: 49.193 ms
```

### 5.4 实验三：分割策略对比

```bash
# 运行分割策略基准测试
python experiments/models/resnet_partition_benchmark.py
```

**预期输出**：
```
================================================================================
策略                        延迟 (ms)         vs基线
--------------------------------------------------------------------------------
all_cpu                         66.468    1.00x
pipeline_half                   49.193    1.35x  ← 最佳策略
================================================================================
```

### 5.5 自定义分割策略

```python
from experiments.models.distributed_resnet import run_distributed_inference
from python.utils.basic_utils import ExecutionModeOptions

# 自定义分割：前半在 Enclave，后半在 CPU
custom_overrides = {
    "input": ExecutionModeOptions.CPU,  # 必须
    # Layer3, Layer4 在 CPU
    **{f"layer{li}_block{bi}_{suffix}": ExecutionModeOptions.CPU
       for li in [3, 4]
       for bi in range(2)
       for suffix in ["conv1", "relu1", "conv2", "skip", "downsample", "add", "relu2"]},
    "avgpool": ExecutionModeOptions.CPU,
    "fc": ExecutionModeOptions.CPU,
    "output": ExecutionModeOptions.CPU,
}

result = run_distributed_inference(layer_mode_overrides=custom_overrides)
print(f"延迟: {result['latency_ms']:.3f} ms")
```

---

## 6. 代码结构

```
TAOISM/
│
├── 📁 experiments/                    # 【核心研究代码】
│   ├── models/                        # 模型定义与分布式推理
│   │   ├── sgx_resnet.py              # ⭐ 可分割 ResNet-18
│   │   ├── distributed_resnet.py      # ⭐ 分布式推理框架
│   │   ├── resnet_partition_benchmark.py  # 分割策略基准
│   │   ├── nin.py, vgg16.py, ...      # 其他模型
│   ├── measurement/                   # 成本测量脚本
│   ├── data/                          # 输出：测量数据
│   └── figures/                       # 输出：图表
│
├── 📁 python/                         # Python 层接口
├── 📁 App/                            # Host 端代码
├── 📁 Enclave/                        # SGX Enclave 代码
│   └── Enclave.config.xml             # Enclave 内存配置
├── 📁 SGXDNN/                         # Enclave 内 DNN 算子
├── 📁 Include/
│   └── common_with_enclaves.h         # STORE_CHUNK_ELEM 配置
├── 📁 scripts/                        # 系统脚本
│   └── check_sgx2_edmm.sh             # 硬件检测
│
├── Makefile
└── README.md
```

---

## 7. STORE_CHUNK_ELEM 配置参考

`STORE_CHUNK_ELEM` 是 Enclave 内存管理的关键参数，必须根据模型输入尺寸正确配置。

### 7.1 常用模型配置

| 模型 | 输入尺寸 | STORE_CHUNK_ELEM | 说明 |
|------|---------|------------------|------|
| **NiN, ResNet18** | 32×32 | 409600 | CIFAR 数据集 |
| **VGG16, AlexNet** | 224×224 | 802816 | ImageNet |
| **Inception V3** | 299×299 | 见下表 | 分组配置 |

### 7.2 Inception V3 分组配置

由于 Inception V3 结构复杂，需要分组配置：

| 组名 | STORE_CHUNK_ELEM | 内存/Chunk | 关键约束 |
|------|-----------------|-----------|---------|
| **Stem** | 130560500 | 498 MB | MaxPool(35×35, 73×73) |
| **Inception-A** | 940800 | 3.6 MB | MaxPool(35×35) |
| **Reduction-A** | 134175475 | 512 MB | MaxPool(35×35, 17×17) |
| **Inception-B** | 221952 | 0.85 MB | MaxPool(17×17) |
| **Reduction-B** | 1109760 | 4.2 MB | MaxPool(17×17, 8×8) |
| **Inception-C** | 30720 | 0.12 MB | MaxPool(8×8) |
| **Classifier** | 256000 | 0.98 MB | Linear(2048) |

### 7.3 修改 STORE_CHUNK_ELEM

**方法 1：手动编辑**

```bash
# 编辑配置文件
vim Include/common_with_enclaves.h

# 修改以下行：
#define STORE_CHUNK_ELEM 409600
#define WORK_CHUNK_ELEM 409600
```

**方法 2：使用 sed 命令**

```bash
# 设置为 409600（ResNet 32x32）
sed -i 's/#define STORE_CHUNK_ELEM [0-9]*/#define STORE_CHUNK_ELEM 409600/' Include/common_with_enclaves.h
sed -i 's/#define WORK_CHUNK_ELEM [0-9]*/#define WORK_CHUNK_ELEM 409600/' Include/common_with_enclaves.h

# 设置为 802816（VGG 224x224）
sed -i 's/#define STORE_CHUNK_ELEM [0-9]*/#define STORE_CHUNK_ELEM 802816/' Include/common_with_enclaves.h
sed -i 's/#define WORK_CHUNK_ELEM [0-9]*/#define WORK_CHUNK_ELEM 802816/' Include/common_with_enclaves.h
```

**方法 3：重新编译（必须）**

```bash
# 修改配置后必须执行
rm -rf SGXDNN/bin_sgx && make clean && make SGX_MODE=HW all
```

### 7.4 约束条件

| 约束类型 | 条件 | 说明 |
|---------|------|------|
| **MaxPool（强制）** | `STORE_CHUNK_ELEM % (H × W) == 0` | 不满足会导致错误 |
| **Conv（警告）** | `STORE_CHUNK_ELEM % (row_size × stride) == 0` | 打印警告但可运行 |
| **Linear（重要）** | `STORE_CHUNK_ELEM % input_features == 0` | 可能影响性能 |

### 7.5 内存计算

```
每个 chunk 内存 = STORE_CHUNK_ELEM × 4 bytes (float32)
总内存 = 8 chunks × 每个 chunk 内存
```

---

## 8. 故障排除

### 8.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| **Enclave 创建失败** | SGX 驱动未安装 | `ls /dev/sgx_enclave` 验证 |
| **libstdc++ 版本错误** | Conda 环境库冲突 | 见 8.2 节 |
| **Out of EPC memory** | Enclave 内存不足 | 调整 `Enclave.config.xml` |
| **EDMM not detected** | 硬件不支持 SGX2 | 需要 Ice Lake+ CPU |
| **MaxPool 返回错误** | STORE_CHUNK_ELEM 配置错误 | 见第 7 节 |

### 8.2 libstdc++ 版本冲突

**问题**：
```
OSError: libstdc++.so.6: version `GLIBCXX_3.4.32' not found
```

**解决方案**：

```bash
# 方案 1：设置正确的 LD_LIBRARY_PATH（推荐）
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH

# 方案 2：使用 LD_PRELOAD
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# 方案 3：更新 Conda 环境的 libstdc++
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $CONDA_PREFIX/lib/
```

### 8.3 Enclave 内存配置

编辑 `Enclave/Enclave.config.xml`：

```xml
<EnclaveConfiguration>
  <!-- 调整堆大小 -->
  <HeapMaxSize>0x80000000</HeapMaxSize>  <!-- 2GB -->
  <StackMaxSize>0x2000000</StackMaxSize> <!-- 32MB -->
  <TCSNum>4</TCSNum>
</EnclaveConfiguration>
```

修改后重新编译：
```bash
make clean && make
```

### 8.4 诊断命令

```bash
# 检查 SGX2/EDMM 支持
bash scripts/check_sgx2_edmm.sh

# 检查 Enclave 配置
cat Enclave/Enclave.config.xml

# 监控 EPC 使用
export PRINT_CHUNK_INFO=1
python teeslice/sgx_resnet_cifar.py --mode Enclave

# 检查内核 SGX 事件
sudo dmesg | grep -i sgx
```

### 8.5 分布式推理常见问题

| 问题 | 解决方案 |
|------|---------|
| **"Tags must linked before tensor initialization"** | 使用最新的 `distributed_resnet.py`（共享模型实例） |
| **"Trying to create tensor with negative dimension"** | 输入尺寸太小，ResNet-18 需要至少 64×64 |
| **程序卡住** | 可能是死锁，检查依赖关系是否正确 |
| **LayerA 报 0x1006 错误** | 始终将 `input` 层设为 CPU 模式 |

---

## 9. 主要结果

### 9.1 ResNet-18 分割策略对比

| 策略 | 延迟 (ms) | 加速比 | 说明 |
|------|-----------|--------|------|
| `all_cpu` | 66.468 | 1.00× | 基线 |
| `pipeline_quarter` | 50.391 | 1.32× | 前 1/4 Enclave |
| `pipeline_half` | **49.193** | **1.35×** | **最优：前 1/2 Enclave** |
| `pipeline_three_quarter` | 65.184 | 1.02× | 前 3/4 Enclave |

### 9.2 关键发现

1. **分割点选择至关重要**：最佳分割点在网络中间，使两分区负载均衡
2. **并行结构带来显著加速**：ResNet 残差连接提供天然并行机会
3. **通信开销可控**：切边数量有限（3 条），通信开销远小于并行收益

---

## 10. 假设与限制

### 10.1 研究假设

- DNN 表示为有向无环图（DAG）
- 分区粒度为层级
- 当前支持 CPU/Enclave 两分区

### 10.2 系统限制

| 限制 | 可能的扩展 |
|------|-----------|
| 两分区 | 扩展到 CPU/GPU/Enclave 三分区 |
| ResNet 验证 | 扩展到 Inception、DenseNet |
| 手动分割 | 自动最优分割算法 |

---

## 11. 引用

本研究基于 TAOISM 框架构建：

```bibtex
@inproceedings{zhang2024no,
  title={No Privacy Left Outside: On the (In-)Security of TEE-Shielded DNN Partition for On-Device ML},
  author={Zhang, Ziqi and Gong, Chen and Cai, Yifeng and Yuan, Yuanyuan and Liu, Bingyan and Li, Ding and Guo, Yao and Chen, Xiangqun},
  booktitle={2024 IEEE Symposium on Security and Privacy (SP)},
  year={2024}
}

@inproceedings{zhang2022teeslice,
  title={TEESlice: Slicing DNN Models for Secure and Efficient Deployment},
  author={Zhang, Ziqi and Ng, Lucien KL and Liu, Bingyan and Cai, Yifeng and Li, Ding and Guo, Yao and Chen, Xiangqun},
  booktitle={AISTA '22},
  year={2022}
}
```

---

## 附录 A：快速命令参考

```bash
# ========== 环境激活 ==========
source /opt/intel/sgxsdk/environment
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate taoism

# ========== 编译 ==========
make                                          # 普通编译
make clean && make                            # 清理编译
rm -rf SGXDNN/bin_sgx && make clean && make SGX_MODE=HW all  # 完整重编译

# ========== 测试 ==========
python experiments/quick_test.py              # 快速验证
bash teeslice/scripts/run_resnet_baseline.sh  # ResNet 基线

# ========== 实验 ==========
python -m experiments.models.distributed_resnet              # 分布式推理
python experiments/models/resnet_partition_benchmark.py      # 分割策略对比
python experiments/run_all_measurements.py --quick-test      # 成本测量

# ========== 诊断 ==========
bash scripts/check_sgx2_edmm.sh               # 检查 SGX2 支持
cat Enclave/Enclave.config.xml                # 查看 Enclave 配置
cat Include/common_with_enclaves.h | grep CHUNK  # 查看 chunk 配置
```

## 附录 B：文件索引

| 目的 | 文件位置 |
|------|---------|
| **分布式推理框架** | `experiments/models/distributed_resnet.py` |
| **可分割 ResNet** | `experiments/models/sgx_resnet.py` |
| **分割策略基准** | `experiments/models/resnet_partition_benchmark.py` |
| **成本测量脚本** | `experiments/measurement/*.py` |
| **Chunk 配置** | `Include/common_with_enclaves.h` |
| **Enclave 配置** | `Enclave/Enclave.config.xml` |
| **硬件检测** | `scripts/check_sgx2_edmm.sh` |

---

<p align="center">
  <em>Last Updated: December 2025</em>
</p>
