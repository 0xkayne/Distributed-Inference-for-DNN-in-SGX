# BERT Per-Head Attention Profiling 使用指南

## 概述

本指南说明如何使用 per-head attention profiling 功能来获取每个注意力头的细粒度性能数据。

## 修复内容

### 1. 修复 `dependencies` 未定义错误

**问题**：在 `_profile_matmul_enclave` 函数中使用了 `dependencies` 变量但未定义。

**修复**：在第 956 行添加：
```python
# Infer dependencies
dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
```

### 2. 添加 Per-Head Profiling 支持

**新增功能**：
- `use_per_head_attention` 参数：控制是否启用 per-head 模式
- `_profile_encoder_block_per_head` 方法：对每个头进行独立测量
- `--per-head` 命令行参数：方便启用该功能

## 使用方法

### 标准批量模式（默认）

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_bert_enclave --model base
```

**输出示例**：
```
encoder0_attn_q_proj          2.34 ms
encoder0_attn_k_proj          2.31 ms
encoder0_attn_v_proj          2.35 ms
encoder0_attn_qk_matmul       5.23 ms  ← 所有12个头的总时间
encoder0_attn_softmax         1.45 ms
encoder0_attn_v_matmul        5.18 ms
encoder0_attn_out_proj        2.38 ms
...
```

**层数**：约 125 层（12 个 encoder + embedding + classifier）

### Per-Head 细粒度模式

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_bert_enclave --model base --per-head
```

**输出示例**：
```
encoder0_attn_q_proj                  2.34 ms
encoder0_attn_k_proj                  2.31 ms
encoder0_attn_v_proj                  2.35 ms
    --- Head 0 ---
encoder0_attn_head0_qk_matmul         0.45 ms  ← 单独测量 Head 0
encoder0_attn_head0_softmax           0.12 ms
encoder0_attn_head0_attn_v_matmul     0.43 ms
    --- Head 1 ---
encoder0_attn_head1_qk_matmul         0.46 ms  ← 单独测量 Head 1
encoder0_attn_head1_softmax           0.13 ms
encoder0_attn_head1_attn_v_matmul     0.44 ms
...
    --- Head 11 ---
encoder0_attn_head11_qk_matmul        0.45 ms  ← 单独测量 Head 11
encoder0_attn_head11_softmax          0.12 ms
encoder0_attn_head11_attn_v_matmul    0.43 ms
encoder0_attn_out_proj                2.38 ms
...
```

**层数**：约 1500+ 层（每个 encoder 的 12 个头都单独测量）

### 输出文件

**批量模式**：
- `experiments/data/bert_base_enclave_layers.csv`
- `experiments/data/bert_base_enclave_layers.json`

**Per-head 模式**：
- `experiments/data/bert_base_enclave_per_head_layers.csv`
- `experiments/data/bert_base_enclave_per_head_layers.json`

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型变体 (mini/base/large) | base |
| `--seq-len` | 序列长度 | 128 |
| `--iterations` | 测量迭代次数 | 10 |
| `--warmup` | 预热迭代次数 | 3 |
| `--output-dir` | 输出目录 | experiments/data |
| `--quiet` | 静默模式 | False |
| `--per-head` | **启用 per-head 模式** | False |

## 性能对比

### 批量模式

**优点**：
- ✅ 执行速度快
- ✅ 层数少（~125 层）
- ✅ 内存占用小
- ✅ 符合标准 Transformer 实现

**缺点**：
- ❌ 无法分析单个头的性能
- ❌ 无法发现头之间的性能差异

**适用场景**：
- 整体性能评估
- 与其他模型对比
- 标准 profiling

### Per-Head 模式

**优点**：
- ✅ 每个头的性能单独测量
- ✅ 可以发现头之间的性能差异
- ✅ 更详细的依赖关系图
- ✅ 支持 attention head pruning 分析

**缺点**：
- ❌ 执行时间更长（约 10-15x）
- ❌ 层数多（~1500+ 层）
- ❌ 输出文件更大
- ❌ 需要更频繁的 Enclave 重置

**适用场景**：
- 细粒度性能分析
- 优化特定注意力头
- 研究头之间的差异
- TEE 内存优化

## 实际例子

### BERT-base (12 层, 12 头)

#### 批量模式
```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_bert_enclave \
    --model base \
    --seq-len 128 \
    --iterations 10
```

- 总层数：~125 层
- 每个 encoder：~10 层
- Enclave 重置：每 4 个 encoder

#### Per-head 模式
```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_bert_enclave \
    --model base \
    --seq-len 128 \
    --iterations 10 \
    --per-head
```

- 总层数：~1500+ 层
- 每个 encoder：~40 层（Q/K/V + 12头×3操作 + Out + 2LayerNorm + FFN）
- Enclave 重置：**每 1 个 encoder**（因为层数多）

### BERT-mini (4 层, 4 头)

适合快速测试：

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_bert_enclave \
    --model mini \
    --seq-len 128 \
    --iterations 10 \
    --per-head
```

## 数据分析

### CSV 输出字段

Per-head 模式下，group 字段会包含头的信息：

```csv
name,type,group,execution_mode,enclave_time_mean,...
encoder0_attn_q_proj,Linear,Encoder0,Enclave,2.34,...
encoder0_attn_head0_qk_matmul,MatMul,Encoder0_Head0,Enclave,0.45,...
encoder0_attn_head0_softmax,Softmax,Encoder0_Head0,Enclave,0.12,...
encoder0_attn_head1_qk_matmul,MatMul,Encoder0_Head1,Enclave,0.46,...
...
```

### 分析头之间的性能差异

```python
import pandas as pd

# 读取 per-head profiling 结果
df = pd.read_csv('experiments/data/bert_base_enclave_per_head_layers.csv')

# 筛选特定 encoder 的所有头
encoder0_heads = df[df['group'].str.contains('Encoder0_Head')]

# 按头分组统计
head_stats = encoder0_heads.groupby(
    encoder0_heads['name'].str.extract(r'head(\d+)')[0]
)['enclave_time_mean'].sum()

print("各个头的总计算时间:")
print(head_stats)

# 分析
print(f"\n最快的头: Head {head_stats.idxmin()} ({head_stats.min():.2f} ms)")
print(f"最慢的头: Head {head_stats.idxmax()} ({head_stats.max():.2f} ms)")
print(f"性能差异: {(head_stats.max() / head_stats.min() - 1) * 100:.1f}%")
```

## 注意事项

### 1. Enclave 内存限制

Per-head 模式会创建更多层，需要更频繁的 Enclave 重置：
- 批量模式：每 4 个 encoder 重置
- Per-head 模式：**每 1 个 encoder 重置**

### 2. 执行时间

Per-head 模式的执行时间会显著增加：
- BERT-mini: 约 5-10 分钟
- BERT-base: 约 30-60 分钟
- BERT-large: 约 2-3 小时

### 3. 磁盘空间

CSV 文件大小：
- 批量模式：~50 KB
- Per-head 模式：~500 KB - 1 MB

## 故障排查

### 问题：Enclave 内存不足

**症状**：
```
Error: Enclave memory exhausted
```

**解决方案**：
1. 减少 `ENCLAVE_RESET_INTERVAL`（已自动调整为 1）
2. 使用更小的模型（mini 而非 base）
3. 减少 `seq_len`

### 问题：运行时间过长

**解决方案**：
1. 减少 `--iterations` 参数（如改为 5）
2. 减少 `--warmup` 参数（如改为 2）
3. 使用 `--model mini` 进行快速验证

### 问题：NameError: name 'dependencies' is not defined

**状态**：✅ 已修复

在所有 `_profile_*_enclave` 函数中添加了依赖推断：
```python
dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
```

## 验证清单

运行前检查：
- [ ] 已编译项目 (`make`)
- [ ] Enclave 库存在 (`App/bin/enclave_bridge.so`)
- [ ] LD_PRELOAD 已设置
- [ ] 在 taoism conda 环境中

运行后验证：
- [ ] 无 NameError 错误
- [ ] 所有层都成功测量
- [ ] CSV 文件生成
- [ ] JSON 文件生成
- [ ] Per-head 模式下有 head0-head11 的层

## 后续工作

1. **自动化分析脚本**
   ```python
   # 分析每个头的贡献
   python experiments/analysis/analyze_per_head_performance.py \
       --input experiments/data/bert_base_enclave_per_head_layers.csv
   ```

2. **可视化**
   ```python
   # 可视化头之间的性能差异
   python experiments/visualization/plot_head_performance.py
   ```

3. **优化建议**
   - 识别最慢的头进行优化
   - 分析是否可以剪枝某些头
   - 为不同的头选择不同的执行环境（CPU vs Enclave）

---

**最后更新**：2026-01-20
**作者**：TAOISM 项目团队
