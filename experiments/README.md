# TAOISM Thesis Experiments - Phase 1

本目录包含毕业论文阶段一的所有实验代码：理论建模与基础测量。

## 📁 目录结构

```
experiments/
├── models/              # DNN模型定义
│   ├── nin.py          # Network in Network
│   ├── vgg16.py        # VGG16
│   ├── resnet18.py     # ResNet18
│   ├── alexnet.py      # AlexNet
│   ├── inception_v3.py # Inception V3
│   └── inception_v4.py # Inception V4
├── measurement/        # 测量脚本
│   ├── measure_computation.py    # 计算开销测量
│   ├── measure_communication.py  # 通信开销测量
│   ├── measure_security.py       # 安全开销测量
│   └── measure_paging.py         # EPC换页开销测量
├── utils/              # 工具类
│   ├── layer_profiler.py   # 层级性能分析器
│   └── data_collector.py   # 数据收集器
├── data/               # 测量数据（自动生成）
├── figures/            # 生成的图表（自动生成）
├── run_all_measurements.py  # 批量运行所有测量
├── quick_test.py       # 快速测试脚本
└── README.md           # 本文件
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate taoism

# 设置SGX环境
source /opt/intel/sgxsdk/environment

# 确保在TAOISM根目录
cd /root/exp_DNN_SGX/TAOISM
```

### 2. 快速测试

首先运行快速测试确保所有组件正常：

```bash
python experiments/quick_test.py
```

预期输出：所有测试通过 (✓)

### 3. 单个模型测试

测试单个模型的计算开销（推荐先从CPU模式开始）：

```bash
# NiN模型，CPU模式，10次迭代（快速测试）
python experiments/measurement/measure_computation.py \
    --single-model NiN \
    --devices CPU \
    --batch-sizes 1 \
    --iterations 10
```

### 4. 完整测量流程

```bash
# 快速测试模式（减少迭代次数）
python experiments/run_all_measurements.py --quick-test

# 或完整测试（约需30-60分钟）
python experiments/run_all_measurements.py --models NiN ResNet18
```

## 📊 测量内容

### 1. 计算开销测量 (measure_computation.py)

**目标**：测量每层在不同设备上的执行时间

**参数**：
- `--models`: 要测试的模型列表
- `--devices`: 设备类型 (CPU/GPU/Enclave)
- `--batch-sizes`: 批大小列表
- `--iterations`: 迭代次数

**输出文件**：
- `computation_cost_{model}_{device}.json`
- `computation_cost_{model}_aggregated.json`

**示例**：
```bash
python experiments/measurement/measure_computation.py \
    --models NiN ResNet18 \
    --devices CPU \
    --batch-sizes 1 4 8 \
    --iterations 100
```

### 2. 通信开销测量 (measure_communication.py)

**目标**：测量层间数据传输开销

**参数**：
- `--models`: 要测试的模型列表
- `--bandwidths`: 带宽列表 (Mbps)
- `--iterations`: 迭代次数

**输出文件**：
- `communication_cost_{model}.json`

**示例**：
```bash
python experiments/measurement/measure_communication.py \
    --models NiN \
    --bandwidths 10 100 1000 \
    --iterations 100
```

### 3. 安全开销测量 (measure_security.py)

**目标**：测量TEE安全开销（CPU vs Enclave）

**参数**：
- `--models`: 要测试的模型列表
- `--batch-size`: 批大小
- `--iterations`: 迭代次数

**输出文件**：
- `security_cost_{model}.json`

**示例**：
```bash
# 需要SGX Enclave支持
python experiments/measurement/measure_security.py \
    --models NiN \
    --batch-size 1 \
    --iterations 100
```

## 📈 数据格式

### 计算开销数据示例

```json
{
  "model": "NiN",
  "devices": {
    "CPU": {
      "batch_1": {
        "layers": [
          {
            "index": 0,
            "name": "input",
            "type": "SecretInputLayer",
            "mean_ms": 0.52,
            "std_ms": 0.03,
            "param_count": 0,
            "memory_mb": 0.01
          },
          ...
        ],
        "summary": {
          "total_time_ms": 45.23,
          "total_params": 966986,
          "total_memory_mb": 12.5
        }
      }
    }
  }
}
```

### 通信开销数据示例

```json
{
  "model": "NiN",
  "layers": [
    {
      "layer_index": 0,
      "layer_name": "conv1_1",
      "output_shape": [1, 192, 32, 32],
      "output_size_mb": 0.75,
      "serialize_mean_ms": 0.15,
      "transfer_times": {
        "10Mbps": 600.0,
        "100Mbps": 60.0,
        "1000Mbps": 6.0
      },
      "total_comm_cost": {
        "10Mbps": 600.3,
        "100Mbps": 60.3,
        "1000Mbps": 6.3
      }
    },
    ...
  ]
}
```

### 安全开销数据示例

```json
{
  "model": "NiN",
  "layers": [
    {
      "layer_index": 0,
      "layer_name": "conv1_1",
      "cpu_time_ms": 2.5,
      "enclave_time_ms": 3.2,
      "overhead_ms": 0.7,
      "overhead_percent": 28.0
    },
    ...
  ],
  "summary": {
    "cpu_total_ms": 45.2,
    "enclave_total_ms": 58.7,
    "total_overhead_ms": 13.5,
    "total_overhead_percent": 29.9
  }
}
```

## ⚙️ 配置说明

### 模型配置

不同模型需要不同的chunk配置（在`Include/common_with_enclaves.h`中）：

- **NiN, ResNet18** (32x32输入): `STORE_CHUNK_ELEM 409600`
- **VGG16, AlexNet** (224x224输入): `STORE_CHUNK_ELEM 802816`

### Enclave配置

在`Enclave/Enclave.config.xml`中调整：

```xml
<!-- 小模型 -->
<HeapMaxSize>0x20000000</HeapMaxSize>  <!-- 512MB -->

<!-- 大模型 -->
<HeapMaxSize>0x40000000</HeapMaxSize>  <!-- 1GB -->
```

## 🐛 常见问题

### 1. Import错误

```bash
# 确保在TAOISM根目录运行
cd /root/exp_DNN_SGX/TAOISM
python experiments/quick_test.py
```

### 2. Enclave初始化失败

```bash
# 检查SGX环境
source /opt/intel/sgxsdk/environment
bash scripts/check_sgx2_edmm.sh
```

### 3. 内存不足错误

- 调整`Enclave.config.xml`中的`HeapMaxSize`
- 减小`STORE_CHUNK_ELEM`
- 减小batch size

### 4. 测量时间过长

```bash
# 使用快速测试模式
python experiments/run_all_measurements.py --quick-test

# 或减少迭代次数
python experiments/measurement/measure_computation.py \
    --single-model NiN --iterations 10
```

## 📝 下一步

完成阶段一测量后：

1. **查看数据**：`experiments/data/`目录中的JSON文件
2. **数据分析**：使用收集的数据建立成本模型
3. **阶段二**：实现安全等级自动标注器
4. **阶段三**：实现DNN分割优化算法

## 📧 帮助

如有问题，请查看：
- 主README：`/root/exp_DNN_SGX/TAOISM/README.md`
- EDMM文档：`/root/exp_DNN_SGX/TAOISM/QUICK_START_EDMM.md`

