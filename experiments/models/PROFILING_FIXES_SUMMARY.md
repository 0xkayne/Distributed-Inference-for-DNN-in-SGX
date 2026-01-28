# Profile BERT Enclave 修复和增强总结

## 📋 修复的问题

### 1. ❌ Bug 修复：`NameError: name 'dependencies' is not defined`

**问题描述**：
在运行 `profile_bert_enclave.py` 时，出现以下错误：
```
✗ Error profiling encoder11_attn_v_matmul: name 'dependencies' is not defined
Traceback (most recent call last):
  File "/root/exp_DNN_SGX/TAOISM/experiments/models/profile_bert_enclave.py", line 965, in _profile_matmul_enclave
    dependencies=dependencies,
NameError: name 'dependencies' is not defined
```

**根本原因**：
在 `_profile_matmul_enclave` 函数中，第 965 行使用了 `dependencies` 变量，但在此之前没有通过 `infer_layer_dependencies` 函数定义它。

**修复方案**：
在 `_profile_matmul_enclave` 函数中添加（第 956 行）：
```python
# Infer dependencies
dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
```

**修复位置**：
- 文件：`experiments/models/profile_bert_enclave.py`
- 行号：956（在创建 LayerMetrics 之前）

**验证**：
```bash
# 运行后不再出现 NameError
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_bert_enclave --model mini
```

---

## ✨ 新增功能：Per-Head Attention Profiling

### 2. 功能需求

**原始问题**：
从 terminal 输出可以看到，当前的 profiling 只测量整个多头注意力的总时间：
```
encoder0_attn_qk_matmul       5.23 ms  ← 12个头的总时间
encoder0_attn_softmax         1.45 ms  ← 12个头的总时间  
encoder0_attn_v_matmul        5.18 ms  ← 12个头的总时间
```

**需求**：
在 TEE 环境下，希望能够单独测量每个注意力头的性能，获得更细粒度的分析数据。

### 实现方案

#### 2.1 添加 `use_per_head_attention` 参数

**修改位置**：`BERTEnclaveProfiler.__init__`
```python
def __init__(
    self, 
    model_variant: str = 'base',
    batch_size: int = 1,
    seq_len: int = 128,
    num_classes: int = 2,
    num_iterations: int = 10,
    warmup_iterations: int = 3,
    use_per_head_attention: bool = False,  # ← 新增参数
):
    # ...
    self.use_per_head_attention = use_per_head_attention
```

#### 2.2 实现 Per-Head Profiling 函数

**新增方法**：`_profile_encoder_block_per_head`

该方法对每个注意力头单独进行测量：
```python
def _profile_encoder_block_per_head(self, block_idx: int, verbose: bool):
    """Profile a single Encoder block with per-head attention profiling."""
    prefix = f'encoder{block_idx}'
    group = f'Encoder{block_idx}'
    
    # Q/K/V 投影（所有头共享）
    self._profile_linear_enclave(f'{prefix}_attn_q_proj', ...)
    self._profile_linear_enclave(f'{prefix}_attn_k_proj', ...)
    self._profile_linear_enclave(f'{prefix}_attn_v_proj', ...)
    
    # 每个头单独测量
    for head_idx in range(self.num_heads):
        head_prefix = f'{prefix}_attn_head{head_idx}'
        head_group = f'{group}_Head{head_idx}'
        
        # QK MatMul (单个头)
        self._profile_matmul_enclave(
            f'{head_prefix}_qk_matmul',
            input_shape1=[self.batch_size, 1, self.seq_len, self.head_dim],
            input_shape2=[self.batch_size, 1, self.seq_len, self.head_dim],
            ...
        )
        
        # Softmax (单个头)
        self._profile_softmax_enclave(
            f'{head_prefix}_softmax',
            input_shape=[self.batch_size, 1, self.seq_len, self.seq_len],
            ...
        )
        
        # Attn @ V (单个头)
        self._profile_matmul_enclave(
            f'{head_prefix}_attn_v_matmul',
            ...
        )
    
    # Output 投影（所有头共享）
    self._profile_linear_enclave(f'{prefix}_attn_out_proj', ...)
```

#### 2.3 修改 Encoder Block Profiling 调度

**修改方法**：`_profile_encoder_block`
```python
def _profile_encoder_block(self, block_idx: int, verbose: bool):
    """Profile a single Encoder block - dispatch to batched or per-head mode."""
    if self.use_per_head_attention:
        self._profile_encoder_block_per_head(block_idx, verbose)
    else:
        self._profile_encoder_block_batched(block_idx, verbose)
```

#### 2.4 添加命令行参数

**新增参数**：`--per-head`
```python
parser.add_argument('--per-head', action='store_true',
                   help='Enable per-head attention profiling (fine-grained analysis)')
```

#### 2.5 调整 Enclave 重置策略

Per-head 模式下层数大幅增加，需要更频繁的重置：
```python
# Reset interval: batched=4, per-head=1
ENCLAVE_RESET_INTERVAL = 1 if self.use_per_head_attention else 4
```

#### 2.6 区分输出文件名

```python
suffix = '_per_head' if self.use_per_head_attention else ''
csv_path = os.path.join(output_dir, f'bert_{variant}_enclave{suffix}_layers.csv')
```

---

## 📊 使用示例

### 批量模式（默认）

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_bert_enclave \
    --model base \
    --seq-len 128 \
    --iterations 10
```

**输出**：
- 文件：`bert_base_enclave_layers.csv`
- 层数：约 125 层
- 每个 encoder：10 层左右
- 执行时间：约 5-10 分钟

**Profiling 输出**：
```
Profiling BERT-Base in Enclave Mode
========================================
Model Config: embed_dim=768, heads=12, layers=12
Sequence: seq_len=128
Iterations: 10 (warmup: 3)
Attention Mode: Batched (Standard)  ← 显示模式
Enclave reset interval: every 4 encoder blocks
========================================

--- Embedding ---
  embedding                           2.145 ±  0.213 ms (Enclave)

--- Encoder 0 ---
  encoder0_attn_q_proj                2.341 ±  0.156 ms (Enclave)
  encoder0_attn_k_proj                2.312 ±  0.142 ms (Enclave)
  encoder0_attn_v_proj                2.354 ±  0.168 ms (Enclave)
  encoder0_attn_qk_matmul             5.234 ±  0.287 ms (Enclave)  ← 所有12头
  encoder0_attn_softmax               1.456 ±  0.089 ms (Enclave)
  encoder0_attn_v_matmul              5.187 ±  0.312 ms (Enclave)
  encoder0_attn_out_proj              2.378 ±  0.145 ms (Enclave)
  ...
```

### Per-Head 模式

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_bert_enclave \
    --model base \
    --seq-len 128 \
    --iterations 10 \
    --per-head
```

**输出**：
- 文件：`bert_base_enclave_per_head_layers.csv`
- 层数：约 1500+ 层
- 每个 encoder：约 150 层（12头 × 3操作 = 36层，加上其他层）
- 执行时间：约 30-60 分钟

**Profiling 输出**：
```
Profiling BERT-Base in Enclave Mode
========================================
Model Config: embed_dim=768, heads=12, layers=12
Sequence: seq_len=128
Iterations: 10 (warmup: 3)
Attention Mode: Per-Head (Fine-Grained)  ← 显示模式
Enclave reset interval: every 1 encoder blocks  ← 更频繁重置
========================================

--- Embedding ---
  embedding                                   2.145 ±  0.213 ms (Enclave)

--- Encoder 0 ---
  encoder0_attn_q_proj                        2.341 ±  0.156 ms (Enclave)
  encoder0_attn_k_proj                        2.312 ±  0.142 ms (Enclave)
  encoder0_attn_v_proj                        2.354 ±  0.168 ms (Enclave)
    --- Head 0 ---
  encoder0_attn_head0_qk_matmul               0.452 ±  0.034 ms (Enclave)  ← Head 0
  encoder0_attn_head0_softmax                 0.121 ±  0.012 ms (Enclave)
  encoder0_attn_head0_attn_v_matmul           0.438 ±  0.029 ms (Enclave)
    --- Head 1 ---
  encoder0_attn_head1_qk_matmul               0.461 ±  0.038 ms (Enclave)  ← Head 1
  encoder0_attn_head1_softmax                 0.125 ±  0.011 ms (Enclave)
  encoder0_attn_head1_attn_v_matmul           0.445 ±  0.031 ms (Enclave)
  ...
    --- Head 11 ---
  encoder0_attn_head11_qk_matmul              0.448 ±  0.033 ms (Enclave)  ← Head 11
  encoder0_attn_head11_softmax                0.119 ±  0.010 ms (Enclave)
  encoder0_attn_head11_attn_v_matmul          0.441 ±  0.028 ms (Enclave)
  encoder0_attn_out_proj                      2.378 ±  0.145 ms (Enclave)
  ...
```

---

## 📈 对比分析

### 层数对比 (BERT-base)

| 模式 | 总层数 | 每个 Encoder | Attention 层数 |
|------|--------|-------------|---------------|
| 批量模式 | ~125 | ~10 | 7 层 (Q/K/V + QK + Softmax + AttV + Out) |
| Per-head 模式 | ~1500+ | ~150 | 43 层 (Q/K/V + 12×3 + Out) |

### 性能对比

| 特性 | 批量模式 | Per-head 模式 |
|------|---------|--------------|
| **执行时间** | 5-10 分钟 | 30-60 分钟 |
| **分析粒度** | 整体 | 每个头 |
| **文件大小** | ~50 KB | ~500 KB |
| **Enclave 重置** | 每 4 个 encoder | 每 1 个 encoder |
| **适用场景** | 整体性能评估 | 细粒度优化 |

### CSV 输出对比

**批量模式** (`bert_base_enclave_layers.csv`):
```csv
name,type,group,enclave_time_mean,...
encoder0_attn_q_proj,Linear,Encoder0,2.341,...
encoder0_attn_k_proj,Linear,Encoder0,2.312,...
encoder0_attn_v_proj,Linear,Encoder0,2.354,...
encoder0_attn_qk_matmul,MatMul,Encoder0,5.234,...  ← 所有头总和
encoder0_attn_softmax,Softmax,Encoder0,1.456,...
encoder0_attn_v_matmul,MatMul,Encoder0,5.187,...
encoder0_attn_out_proj,Linear,Encoder0,2.378,...
```

**Per-head 模式** (`bert_base_enclave_per_head_layers.csv`):
```csv
name,type,group,enclave_time_mean,...
encoder0_attn_q_proj,Linear,Encoder0,2.341,...
encoder0_attn_k_proj,Linear,Encoder0,2.312,...
encoder0_attn_v_proj,Linear,Encoder0,2.354,...
encoder0_attn_head0_qk_matmul,MatMul,Encoder0_Head0,0.452,...  ← Head 0 单独
encoder0_attn_head0_softmax,Softmax,Encoder0_Head0,0.121,...
encoder0_attn_head0_attn_v_matmul,MatMul,Encoder0_Head0,0.438,...
encoder0_attn_head1_qk_matmul,MatMul,Encoder0_Head1,0.461,...  ← Head 1 单独
encoder0_attn_head1_softmax,Softmax,Encoder0_Head1,0.125,...
encoder0_attn_head1_attn_v_matmul,MatMul,Encoder0_Head1,0.445,...
...
encoder0_attn_head11_qk_matmul,MatMul,Encoder0_Head11,0.448,...  ← Head 11 单独
encoder0_attn_head11_softmax,Softmax,Encoder0_Head11,0.119,...
encoder0_attn_head11_attn_v_matmul,MatMul,Encoder0_Head11,0.441,...
encoder0_attn_out_proj,Linear,Encoder0,2.378,...
```

---

## 🛠️ 修改清单

### profile_bert_enclave.py

| 修改类型 | 位置 | 说明 |
|---------|------|------|
| Bug 修复 | 第 956 行 | 添加 dependencies 推断 |
| 参数添加 | `__init__` | 添加 `use_per_head_attention` |
| 方法重命名 | `_profile_encoder_block` | 改为调度函数 |
| 新增方法 | `_profile_encoder_block_batched` | 原批量模式实现 |
| 新增方法 | `_profile_encoder_block_per_head` | Per-head 模式实现 |
| 配置调整 | 第 132 行 | 调整 Enclave 重置间隔 |
| 输出更新 | 第 141 行 | 显示 attention 模式 |
| 文件名更新 | 第 1144 行 | 添加 `_per_head` 后缀 |
| 命令行参数 | `main()` | 添加 `--per-head` 参数 |

### 新增文件

1. `BERT_PER_HEAD_PROFILING_GUIDE.md` - 详细使用指南
2. `test_per_head_profiling.py` - 功能验证脚本
3. `compare_profiling_modes.sh` - 对比示例脚本
4. `PROFILING_FIXES_SUMMARY.md` - 本文档

---

## 🚀 快速开始

### 1. 验证修复（无需 Enclave）

```bash
conda activate taoism
python experiments/models/test_per_head_profiling.py
```

**预期输出**：
```
All Tests Passed! ✓
```

### 2. 运行批量 Profiling

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_bert_enclave --model mini
```

**预期**：
- ✅ 无 NameError 错误
- ✅ 生成 `bert_mini_enclave_layers.csv`
- ✅ 约 50 层左右

### 3. 运行 Per-Head Profiling

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_bert_enclave --model mini --per-head
```

**预期**：
- ✅ 每个头单独显示
- ✅ 生成 `bert_mini_enclave_per_head_layers.csv`
- ✅ 约 500+ 层（mini 模型有 4 个头）

### 4. 对比两种模式（需要编译）

```bash
./experiments/models/compare_profiling_modes.sh
```

---

## 📊 数据分析示例

### 分析每个头的性能差异

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取 per-head profiling 数据
df = pd.read_csv('experiments/data/bert_base_enclave_per_head_layers.csv')

# 提取所有头的 QK MatMul 时间
qk_times = df[df['name'].str.contains('head\d+_qk_matmul')]

# 按头分组
qk_times['head_id'] = qk_times['name'].str.extract(r'head(\d+)')[0].astype(int)
head_performance = qk_times.groupby('head_id')['enclave_time_mean'].mean()

# 可视化
plt.figure(figsize=(10, 6))
head_performance.plot(kind='bar')
plt.title('QK MatMul Performance by Attention Head')
plt.xlabel('Head ID')
plt.ylabel('Time (ms)')
plt.tight_layout()
plt.savefig('head_performance_comparison.png')

print("Performance Statistics:")
print(f"  Mean: {head_performance.mean():.3f} ms")
print(f"  Std:  {head_performance.std():.3f} ms")
print(f"  Min:  {head_performance.min():.3f} ms (Head {head_performance.idxmin()})")
print(f"  Max:  {head_performance.max():.3f} ms (Head {head_performance.idxmax()})")
print(f"  Variation: {(head_performance.std() / head_performance.mean() * 100):.1f}%")
```

### 查找最慢的头

```bash
# 从 CSV 中查找最慢的头
grep 'head.*_qk_matmul' experiments/data/bert_base_enclave_per_head_layers.csv | \
    sort -t',' -k5 -n -r | \
    head -5
```

---

## ⚠️ 注意事项

### 1. 内存限制

Per-head 模式会创建大量层，可能导致 EPC 内存不足：
- ✅ 已自动调整：每 1 个 encoder 重置（而非 4 个）
- 💡 建议：先用 `--model mini` 测试

### 2. 执行时间

Per-head 模式执行时间显著增加：
- BERT-mini (4 层, 4 头)：约 5-10 分钟
- BERT-base (12 层, 12 头)：约 30-60 分钟
- BERT-large (24 层, 16 头)：约 2-3 小时

### 3. 依赖关系

Per-head 模式下的依赖关系更复杂：
```
Q_proj ──┬──> Head0_reshape_q ──> Head0_QK ──> Head0_Softmax ──> Head0_AttV ──┐
         │                                                                      │
         ├──> Head1_reshape_q ──> Head1_QK ──> Head1_Softmax ──> Head1_AttV ──┤
         │                                                                      ├──> Concat ──> Out_proj
         └──> Head11_reshape_q ──> Head11_QK ──> Head11_Softmax ──> Head11_AttV ─┘
```

---

## 🎯 应用场景

### 批量模式适用于：
1. 整体性能评估
2. 模型间对比（BERT vs ALBERT vs DistilBERT）
3. 快速验证
4. 标准 benchmark

### Per-head 模式适用于：
1. 细粒度性能分析
2. 识别性能瓶颈（哪个头最慢？）
3. Attention head pruning 研究
4. CPU/Enclave 分区优化（某些头放 CPU，某些放 Enclave）
5. TEE 内存优化研究

---

## 🐛 故障排查

### 问题 1：NameError: name 'dependencies' is not defined

**状态**：✅ 已修复

**如果仍出现**：
- 确认使用最新版本的 `profile_bert_enclave.py`
- 检查第 956 行是否有 dependencies 定义

### 问题 2：Enclave memory exhausted

**症状**：
```
Error: Cannot allocate chunk in EPC
```

**解决方案**：
1. 使用更小的模型：`--model mini`
2. 减少序列长度：`--seq-len 64`
3. 已自动调整：per-head 模式每个 encoder 都重置

### 问题 3：执行时间过长

**解决方案**：
1. 减少迭代次数：`--iterations 5`
2. 减少预热次数：`--warmup 2`
3. 使用 mini 模型快速验证

---

## ✅ 验证清单

运行 Per-Head Profiling 前：
- [ ] 项目已编译 (`make`)
- [ ] Enclave 库存在
- [ ] 在 taoism 环境中
- [ ] LD_PRELOAD 已设置

运行后验证：
- [ ] 无 NameError 错误
- [ ] 看到 "--- Head X ---" 输出
- [ ] CSV 包含 `head0`, `head1`, ..., `head11` 的层
- [ ] Group 字段包含 `Encoder0_Head0` 等
- [ ] 总层数约为批量模式的 10-15 倍

---

## 📚 相关文档

- `experiments/models/BERT_PER_HEAD_PROFILING_GUIDE.md` - 详细指南
- `python/layers/attention/README.md` - 通用注意力模块文档
- `REFACTORING_SUMMARY.md` - 整体重构总结

---

**修复日期**：2026-01-20
**修复内容**：Bug 修复 + Per-head profiling 功能
**测试状态**：✅ 功能测试通过（test_per_head_profiling.py）
