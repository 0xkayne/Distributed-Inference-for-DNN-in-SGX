# BERT Per-Head 依赖关系修复总结

## ✅ 已完成的修复

### 1. 修改的核心函数

所有 `_profile_*_enclave` 函数添加了 `dependencies` 参数：

| 函数 | 修改内容 |
|------|---------|
| `_profile_linear_enclave` | 添加 `dependencies: Optional[List[str]] = None` 参数 |
| `_profile_layernorm_enclave` | 添加 `dependencies: Optional[List[str]] = None` 参数 |
| `_profile_softmax_enclave` | 添加 `dependencies: Optional[List[str]] = None` 参数 |
| `_profile_gelu_enclave` | 添加 `dependencies: Optional[List[str]] = None` 参数 |
| `_profile_matmul_enclave` | 添加 `dependencies: Optional[List[str]] = None` 参数 |

### 2. 修复的依赖关系逻辑

#### A. Embedding 层
```python
self._profile_linear_enclave(
    'embedding',
    ...,
    dependencies=[]  # 第一层，无依赖
)
```

#### B. Per-Head 模式 - 并行注意力头

**Q/K/V 投影**（并行）：
```python
# 所有投影都依赖前一个 block 的输出
prev_block_output = 'embedding' if block_idx == 0 else f'encoder{block_idx-1}_norm2'

self._profile_linear_enclave(f'{prefix}_attn_q_proj', ..., dependencies=[prev_block_output])
self._profile_linear_enclave(f'{prefix}_attn_k_proj', ..., dependencies=[prev_block_output])
self._profile_linear_enclave(f'{prefix}_attn_v_proj', ..., dependencies=[prev_block_output])
```

**每个头的计算**（12个头并行）：
```python
for head_idx in range(num_heads):
    # QK MatMul: 依赖 Q 和 K 投影
    self._profile_matmul_enclave(
        f'{prefix}_attn_head{head_idx}_qk_matmul',
        ...,
        dependencies=[f'{prefix}_attn_q_proj', f'{prefix}_attn_k_proj']
    )
    
    # Softmax: 依赖本头的 QK matmul
    self._profile_softmax_enclave(
        f'{prefix}_attn_head{head_idx}_softmax',
        ...,
        dependencies=[f'{prefix}_attn_head{head_idx}_qk_matmul']
    )
    
    # Attn @ V: 依赖本头的 softmax 和 V 投影
    self._profile_matmul_enclave(
        f'{prefix}_attn_head{head_idx}_attn_v_matmul',
        ...,
        dependencies=[f'{prefix}_attn_head{head_idx}_softmax', f'{prefix}_attn_v_proj']
    )
```

**Output 投影**（汇聚所有头）：
```python
# 收集所有头的输出
all_head_outputs = [f'{prefix}_attn_head{i}_attn_v_matmul' for i in range(num_heads)]

# Out projection 依赖所有头
self._profile_linear_enclave(
    f'{prefix}_attn_out_proj',
    ...,
    dependencies=all_head_outputs  # 12 个依赖！
)
```

#### C. 批量模式 - 批量注意力计算

```python
# QK MatMul (所有头一起)
self._profile_matmul_enclave(
    f'{prefix}_attn_qk_matmul',
    ...,
    dependencies=[f'{prefix}_attn_q_proj', f'{prefix}_attn_k_proj']
)

# Softmax
self._profile_softmax_enclave(
    f'{prefix}_attn_softmax',
    ...,
    dependencies=[f'{prefix}_attn_qk_matmul']
)

# Attn @ V
self._profile_matmul_enclave(
    f'{prefix}_attn_v_matmul',
    ...,
    dependencies=[f'{prefix}_attn_softmax', f'{prefix}_attn_v_proj']
)

# Out projection
self._profile_linear_enclave(
    f'{prefix}_attn_out_proj',
    ...,
    dependencies=[f'{prefix}_attn_v_matmul']
)
```

#### D. FFN 部分（顺序）

```python
self._profile_layernorm_enclave(f'{prefix}_norm1', ..., dependencies=[f'{prefix}_attn_out_proj'])
self._profile_linear_enclave(f'{prefix}_ffn_fc1', ..., dependencies=[f'{prefix}_norm1'])
self._profile_gelu_enclave(f'{prefix}_ffn_gelu', ..., dependencies=[f'{prefix}_ffn_fc1'])
self._profile_linear_enclave(f'{prefix}_ffn_fc2', ..., dependencies=[f'{prefix}_ffn_gelu'])
self._profile_layernorm_enclave(f'{prefix}_norm2', ..., dependencies=[f'{prefix}_ffn_fc2'])
```

#### E. Classifier 头

```python
self._profile_linear_enclave('pooler', ..., dependencies=[f'encoder{num_layers-1}_norm2'])
self._profile_linear_enclave('classifier', ..., dependencies=['pooler'])
```

## 📊 依赖关系对比

### 错误的依赖（修复前）

```
encoder0_attn_q_proj       → embedding
encoder0_attn_k_proj       → encoder0_attn_q_proj  ❌
encoder0_attn_v_proj       → encoder0_attn_k_proj  ❌
encoder0_attn_head0_qk     → encoder0_attn_v_proj  ❌
encoder0_attn_head1_qk     → encoder0_attn_head0_attn_v  ❌ (串行!)
encoder0_attn_head2_qk     → encoder0_attn_head1_attn_v  ❌ (串行!)
...
```

### 正确的依赖（修复后）

```
encoder0_attn_q_proj       → embedding
encoder0_attn_k_proj       → embedding  ✓ (并行)
encoder0_attn_v_proj       → embedding  ✓ (并行)

encoder0_attn_head0_qk     → [q_proj, k_proj]  ✓
encoder0_attn_head1_qk     → [q_proj, k_proj]  ✓ (并行!)
encoder0_attn_head2_qk     → [q_proj, k_proj]  ✓ (并行!)
...
encoder0_attn_head11_qk    → [q_proj, k_proj]  ✓ (并行!)

encoder0_attn_out_proj     → [head0_v, head1_v, ..., head11_v]  ✓
```

## 🚀 重新运行 Profiling

修复代码后，需要重新运行 profiling 以生成正确的依赖关系：

```bash
# 清除旧的结果（可选）
rm -f experiments/data/bert_base_enclave_per_head_layers.csv

# 重新运行 per-head profiling
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_bert_enclave \
    --model base \
    --seq-len 128 \
    --iterations 10 \
    --per-head

# 验证依赖关系
python experiments/models/verify_dependencies.py
```

### 预期验证结果

```
======================================================================
Per-Head Attention Dependencies Verification
======================================================================

✓ Successes (180+):
  ✓ Embedding has no dependencies
  ✓ encoder0_attn_q_proj depends on embedding
  ✓ encoder0_attn_k_proj depends on embedding
  ✓ encoder0_attn_v_proj depends on embedding
  ✓ All head0-11 qk_matmul have correct dependencies [q_proj, k_proj]
  ✓ All head0-11 softmax depend on their own qk_matmul
  ✓ All head0-11 attn_v depend on softmax and v_proj
  ✓ out_proj depends on all 12 heads
  ✓ FFN chain is correct
  ✓ Pooler and classifier dependencies correct
  ...

======================================================================
Parallelism Verification
======================================================================

✓ All heads are independent (no inter-head dependencies)
  This confirms that all 12 heads can execute in parallel!

======================================================================
✅ All checks passed! Dependencies are correct.
======================================================================
```

## 📈 并行度分析

### 理论并行度（每个 Encoder Block）

| 阶段 | 层数 | 并行度 | 说明 |
|------|------|--------|------|
| Q/K/V 投影 | 3 | 3 | 3 个投影可并行 |
| **多头注意力** | 36 | **12** | **12 个头可并行** |
| - QK MatMul | 12 | 12 | 每个头独立 |
| - Softmax | 12 | 12 | 每个头独立 |
| - Attn@V MatMul | 12 | 12 | 每个头独立 |
| Output 投影 | 1 | 1 | 汇聚点 |
| LayerNorm | 1 | 1 | 顺序 |
| FFN | 3 | 1 | 顺序 |
| LayerNorm | 1 | 1 | 顺序 |

**总计每个 Encoder**：~44 层，最大并行度 12

### 关键路径长度

```
Critical Path:
embedding → q_proj → headX_qk → headX_softmax → headX_attn_v → out_proj → 
norm1 → fc1 → gelu → fc2 → norm2

Total: ~10 步（与头数无关！）
```

### 错误依赖的影响

**修复前**（串行）：
- 关键路径：embedding → q → k → v → head0(3步) → head1(3步) → ... → head11(3步) → out
- 总步数：~40 步
- **4x 更长！**

**修复后**（并行）：
- 关键路径：embedding → q/k/v → headX(3步) → out
- 总步数：~10 步
- ✓ 正确反映架构

## 🎯 DAG 可视化工具

### 生成依赖关系图

```python
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def visualize_dependencies(csv_path, output_path='dependency_graph.png'):
    """Visualize layer dependency DAG."""
    df = pd.read_csv(csv_path)
    
    # Build graph
    G = nx.DiGraph()
    
    for idx, row in df.iterrows():
        layer = row['name']
        deps = ast.literal_eval(row['dependencies'])
        
        G.add_node(layer)
        for dep in deps:
            G.add_edge(dep, layer)
    
    # Analyze parallelism
    print(f"Total layers: {len(G.nodes)}")
    print(f"Total edges: {len(G.edges)}")
    print(f"Max in-degree: {max(dict(G.in_degree()).values())}")
    print(f"Max out-degree: {max(dict(G.out_degree()).values())}")
    
    # Find layers with high out-degree (fan-out points)
    high_fanout = [(n, d) for n, d in G.out_degree() if d > 5]
    print(f"\nHigh fan-out layers (potential parallel execution):")
    for node, degree in high_fanout:
        print(f"  {node}: {degree} dependents")
    
    # Calculate critical path
    try:
        longest_path = nx.dag_longest_path(G)
        print(f"\nCritical path length: {len(longest_path)} layers")
        print(f"Path: {longest_path[0]} → ... → {longest_path[-1]}")
    except:
        print("\nCannot calculate critical path (graph may have cycles)")
```

## 🔧 修复清单

- [x] 添加 `dependencies` 参数到所有 profile 函数
- [x] 修复 embedding 依赖 → `[]`
- [x] 修复 Q/K/V 投影依赖 → `[prev_block_output]` (并行)
- [x] 修复每个头的依赖 → 正确的 Q/K/V 引用
- [x] 修复 out_proj 依赖 → 所有头的输出
- [x] 修复 FFN 依赖 → 顺序链
- [x] 修复 classifier 依赖 → 最后 encoder 的输出
- [x] 创建验证脚本
- [x] 创建依赖关系说明文档
- [ ] **重新运行 profiling** 生成正确的 CSV

## 📝 下一步操作

### 1. 重新运行 Per-Head Profiling

```bash
# 备份旧文件
mv experiments/data/bert_base_enclave_per_head_layers.csv \
   experiments/data/bert_base_enclave_per_head_layers.csv.old

# 重新运行
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_bert_enclave \
    --model base \
    --per-head \
    --iterations 10
```

### 2. 验证新的结果

```bash
python experiments/models/verify_dependencies.py
```

**预期输出**：
```
✅ All checks passed! Dependencies are correct.
```

### 3. 对比新旧依赖

```bash
# 对比脚本
python experiments/models/compare_dependencies.py \
    --old bert_base_enclave_per_head_layers.csv.old \
    --new bert_base_enclave_per_head_layers.csv
```

## 🎓 技术要点

### 1. 多头并行的关键

所有 `headX_qk_matmul` 层应该有**相同的依赖**：
```python
dependencies: ['encoderN_attn_q_proj', 'encoderN_attn_k_proj']
```

**不应该是**：
```python
head0_qk_matmul: ['encoderN_attn_v_proj']  # ❌
head1_qk_matmul: ['encoderN_attn_head0_attn_v_matmul']  # ❌ 串行!
```

### 2. Out Projection 的汇聚点

`out_proj` 层应该依赖**所有头的输出**：
```python
dependencies: [
    'head0_attn_v_matmul',
    'head1_attn_v_matmul',
    ...,
    'head11_attn_v_matmul'
]  # 12 个依赖
```

这个 12-路汇聚明确标记了：
- 所有头必须完成才能执行 out_proj
- 所有头可以并行计算
- Out_proj 是同步点

### 3. 调度器优化机会

正确的依赖关系使调度器可以：

**并行执行 12 个头**：
```
Time:  |----Q/K/V proj----|---Head0---|
                           |---Head1---|
                           |---Head2---|
                           |---...-----|
                           |---Head11--|
                                      |--Out proj--|
```

而不是：
```
Time:  |----Q/K/V proj----|---Head0---|---Head1---|---...|---Head11--|--Out proj--|
```

### 4. TEE 优化策略

根据正确的依赖关系，可以：

1. **并行 CPU/Enclave 执行**
   - Head 0-5 在 Enclave
   - Head 6-11 在 CPU
   - 并行执行后汇聚

2. **流水线优化**
   - Head 0 完成 → 立即开始 Head 1
   - 不需要等待所有头完成投影

3. **内存优化**
   - 可以逐头处理并释放内存
   - Out projection 等待时才需要保留所有头

## ⚠️ 注意事项

### 当前 CSV 文件状态

`experiments/data/bert_base_enclave_per_head_layers.csv` **包含错误的依赖关系**，因为它是用旧代码生成的。

### 必须重新运行

**不能**直接使用当前的 CSV 文件进行调度优化，必须重新运行 profiling。

### 验证方法

运行后执行：
```bash
# 快速检查
head -20 experiments/data/bert_base_enclave_per_head_layers.csv | \
    grep head0_qk_matmul | \
    cut -d',' -f19  # dependencies 列

# 应该看到：
"['encoder0_attn_q_proj', 'encoder0_attn_k_proj']"
```

---

**修复完成时间**：2026-01-20
**影响范围**：Per-head profiling 模式的依赖关系生成
**需要行动**：重新运行 profiling 生成正确的 CSV
**验证工具**：`experiments/models/verify_dependencies.py`
