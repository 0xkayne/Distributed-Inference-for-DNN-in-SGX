# 分布式推理框架使用指南

## 快速开始

### 1. 运行 ParallelToyNet 示例

```bash
# 激活环境
source /opt/intel/sgxsdk/environment
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate taoism

# 运行简单示例
cd /root/exp_DNN_SGX/TAOISM
python -m experiments.models.distributed_inference
```

**输出示例：**
```
Input tensor shape: (1, 3, 4, 4)
[Topology] Found cut edge: LayerA_Pool->LayerA_Pool_to_LayerB (CPU -> Enclave)
[Topology] Found cut edge: LayerB->LayerB_to_LayerD (Enclave -> CPU)
...
[Parallel] Total Latency: 10355.067 ms
Sequential vs parallel: 10418.895 ms vs 10355.067 ms (speedup 1.01x)
```

---

### 2. 运行 ResNet-18 分布式推理

```bash
# 基准测试（包含多种分割策略）
python experiments/models/resnet_partition_benchmark.py
```

**输出示例：**
```
性能对比报告
================================================================================
策略                        延迟 (ms)         vs基线            
--------------------------------------------------------------------------------
all_cpu                         66.468    1.00x           
pipeline_half                   49.193    1.35x  ← 最佳策略，35%加速
================================================================================
```

---

## 文件说明

### 核心文件

1. **`distributed_inference.py`** - ParallelToyNet 分布式推理
   - 适合：概念验证、理解框架原理
   - 模型：简单的 6 层网络（A→C in Enclave, D→F in CPU）

2. **`sgx_resnet.py`** - ResNet-18 模型定义
   - 56 层完整 ResNet-18
   - 支持 `layer_mode_overrides` 灵活分割
   - BasicBlock 残差结构

3. **`distributed_resnet.py`** - ResNet-18 分布式推理框架
   - 复用 `FlexibleGraphWorker` 通用执行器
   - 自动拓扑分析
   - 多线程安全

4. **`resnet_partition_benchmark.py`** - 性能基准测试
   - 预定义 4 种分割策略
   - 自动对比性能
   - 生成分析报告

---

## 自定义分割策略

### 方法 1: 使用预定义策略

```python
from experiments.models.resnet_partition_benchmark import get_partition_strategy

# 获取预定义策略
overrides = get_partition_strategy("pipeline_half")

# 运行
from experiments.models.distributed_resnet import run_distributed_inference
result = run_distributed_inference(layer_mode_overrides=overrides)
```

**可用策略：**
- `all_cpu`: 全 CPU 基线
- `all_enclave`: 全 Enclave（input 除外）
- `pipeline_quarter`: 1/4 Enclave, 3/4 CPU
- `pipeline_half`: 1/2 Enclave, 1/2 CPU
- `pipeline_three_quarter`: 3/4 Enclave, 1/4 CPU

---

### 方法 2: 手动定义策略

```python
from python.utils.basic_utils import ExecutionModeOptions

# 示例：仅将计算密集的卷积层放在 Enclave，其他在 CPU
custom_overrides = {
    "input": ExecutionModeOptions.CPU,  # 必须
    
    # 敏感的早期特征提取在 Enclave
    "conv1": ExecutionModeOptions.Enclave,
    "relu": ExecutionModeOptions.Enclave,
    "maxpool": ExecutionModeOptions.Enclave,
    
    # Layer1 的主卷积在 Enclave
    "layer1_block0_conv1": ExecutionModeOptions.Enclave,
    "layer1_block0_conv2": ExecutionModeOptions.Enclave,
    # ... 其他层根据需要指定
    
    # 分类器在 CPU（非敏感）
    "avgpool": ExecutionModeOptions.CPU,
    "flatten": ExecutionModeOptions.CPU,
    "fc": ExecutionModeOptions.CPU,
    "output": ExecutionModeOptions.CPU,
}

result = run_distributed_inference(layer_mode_overrides=custom_overrides)
```

---

### 方法 3: 程序化生成策略

```python
def create_block_level_split(enclave_blocks: list):
    """
    按 block 级别分割
    
    Args:
        enclave_blocks: 需要在 Enclave 运行的 block 列表
                       例如: [(1,0), (1,1), (2,0)] 表示 layer1_block0/1 和 layer2_block0
    """
    overrides = {"input": ExecutionModeOptions.CPU}
    
    # 默认所有在 CPU
    for li in range(1, 5):
        for bi in range(2):
            for suffix in ["conv1", "relu1", "conv2", "skip", "downsample", "add", "relu2"]:
                overrides[f"layer{li}_block{bi}_{suffix}"] = ExecutionModeOptions.CPU
    
    # 指定的 block 在 Enclave
    for li, bi in enclave_blocks:
        for suffix in ["conv1", "relu1", "conv2", "skip", "downsample", "add", "relu2"]:
            overrides[f"layer{li}_block{bi}_{suffix}"] = ExecutionModeOptions.Enclave
    
    return overrides

# 使用：将 Layer1 和 Layer2 的所有 block 放在 Enclave
overrides = create_block_level_split([
    (1, 0), (1, 1),  # Layer1 全部
    (2, 0), (2, 1),  # Layer2 全部
])
```

---

## 性能调优建议

### 1. 选择合适的分割点

**经验法则：**
- 目标：**两个分区负载相对均衡**
- 测量：运行 benchmark，观察哪个线程先结束
- 调整：将工作从空闲分区移到繁忙分区

### 2. 最小化通信开销

**切边越少越好：**
```python
# 好：仅在 layer group 之间切割（少量切边）
pipeline_half: 3 条切边

# 差：每个 block 内部都切割（大量切边）
residual_split: 16+ 条切边
```

### 3. 考虑安全需求

**策略选择：**
- 高安全：敏感的早期特征在 Enclave，分类器在 CPU
- 平衡：中间层在 Enclave（特征既重要又不太大）
- 高性能：仅关键层在 Enclave，其余 CPU

---

## 调试技巧

### 查看拓扑分析结果

```python
# 在 run_distributed_inference 中会自动打印
[Topology] Found cut edge: LayerX->LayerY (Mode1 -> Mode2)
```

### 查看层执行时序

```python
# 每层执行都会打印时间戳
[Partition-Enclave] layer1_block0_conv1 start @ 18:04:24.897974
[Partition-Enclave] layer1_block0_conv1 end @ 18:04:24.899729 (+1.755 ms)
```

- 观察两个分区是否真正并行（时间戳重叠）
- 识别瓶颈层（耗时最长）

### 常见问题

**Q: "Tags must linked before tensor initialization" 错误**  
A: 确保使用了最新的 `distributed_resnet.py`，它使用共享模型实例避免此问题。

**Q: "Trying to create tensor with negative dimension" 错误**  
A: 输入尺寸太小。ResNet-18 需要至少 64×64（经过 5 次 stride=2 下采样）。

**Q: 程序卡住不动**  
A: 可能是死锁。检查：
1. 是否有未满足的依赖（某个 Worker 在等待永远不会到来的数据）
2. GlobalTensor 是否被正确初始化
3. 使用 Ctrl+C 中断并查看堆栈

**Q: LayerA 在 Enclave 模式报 0x1006 错误**  
A: 这是已知限制。始终将 `input` 层设为 CPU 模式。

---

## 性能分析工具

### 提取每层耗时

```python
result = run_distributed_inference(...)
timings = result["timings"]

# 找出最慢的 5 层
sorted_layers = sorted(timings.items(), key=lambda x: x[1], reverse=True)
print("Top 5 slowest layers:")
for layer_name, time_ms in sorted_layers[:5]:
    print(f"  {layer_name}: {time_ms:.3f} ms")
```

### 分析并行效率

```python
# 理想并行时间 = max(Enclave总耗时, CPU总耗时)
# 实际并行时间 = 测量的总延迟
# 并行效率 = 理想并行时间 / 实际并行时间

enclave_time = sum(t for name, t in timings.items() if "Enclave" in name)
cpu_time = sum(t for name, t in timings.items() if "Host" in name)
ideal = max(enclave_time, cpu_time)
actual = result["latency_ms"]
efficiency = ideal / actual
print(f"并行效率: {efficiency:.1%}")
```

---

## 扩展到其他模型

要将此框架应用到新模型（如 DenseNet, MobileNet）：

1. **创建模型定义** (`experiments/models/sgx_<model>.py`)
   ```python
   class SGX<Model>:
       def __init__(self, ..., layer_mode_overrides=None):
           self.layer_mode_overrides = layer_mode_overrides or {}
           self.layers = self._build_network()
       
       def _get_mode(self, layer_name):
           return self.layer_mode_overrides.get(layer_name, self.enclave_mode)
       
       def _build_network(self):
           # 使用 manually_register_prev/next=True
           # 使用 self._get_mode(layer_name) 设置每层模式
           ...
   ```

2. **创建分布式脚本** (`experiments/models/distributed_<model>.py`)
   ```python
   # 复制 distributed_resnet.py
   # 替换模型类导入
   from experiments.models.sgx_<model> import SGX<Model>
   
   # 在 run_distributed_inference 中实例化新模型
   shared_model = SGX<Model>(...)
   ```

3. **定义分割策略并测试**

---

## 总结

本框架实现了：
- ✅ **灵活性**：任意层级分割
- ✅ **通用性**：适用于各种网络拓扑
- ✅ **性能**：ResNet-18 上 35% 加速
- ✅ **易用性**：预定义策略 + 自定义支持

适用于需要在**安全性**（SGX Enclave）和**性能**（CPU/GPU）之间权衡的深度学习推理场景。

