# 统一多头注意力模块

## 概述

本模块为所有 Transformer 架构模型（BERT、ALBERT、DistilBERT、TinyBERT、ViT、Swin）提供统一的多头注意力实现。

## 设计理念

**核心思想**: 所有 Transformer 模型的多头注意力计算流程完全一致，差异仅在于输入形状和序列长度。

### 架构层次

```
python/layers/attention/
├── __init__.py                  # 模块导出
├── base_attention.py            # 基础抽象类
├── batched_attention.py         # 批量模式（高效）
├── per_head_attention.py        # Per-head 模式（细粒度分析）
├── attention_factory.py         # 工厂函数
└── README.md                    # 本文档
```

## 两种实现模式

### 1. BatchedMultiHeadAttention（批量模式）

**特点**：
- 所有头在单次矩阵运算中并行计算
- 高效，适合生产环境
- 较粗粒度的性能分析

**适用场景**：
- 标准模型训练和推理
- 对整体性能要求高的场景

### 2. PerHeadMultiHeadAttention（Per-head 模式）

**特点**：
- 每个头独立计算
- 可以单独测量每个头的性能
- 清晰的层间依赖关系
- 更好的内存管理（在 TEE 中可逐头处理）

**适用场景**：
- TEE 环境下的细粒度性能分析
- 需要追踪每个头计算时间的场景
- 优化单个头的计算

## 使用方法

### 基本用法

```python
from python.layers.attention import create_multi_head_attention

# BERT/ALBERT/DistilBERT/TinyBERT
attn = create_multi_head_attention(
    sid=0,
    name_prefix="encoder0_attn",
    enclave_mode=ExecutionModeOptions.Enclave,
    embed_dim=768,
    num_heads=12,
    batch_size=1,
    seq_len=128,
    per_head_mode=False,  # 使用批量模式
    layer_mode_overrides={}
)

# ViT
attn = create_multi_head_attention(
    sid=0,
    name_prefix="block0_attn",
    enclave_mode=ExecutionModeOptions.Enclave,
    embed_dim=768,
    num_heads=12,
    batch_size=1,
    seq_len=197,  # 196 patches + 1 CLS token
    per_head_mode=True,  # 使用 per-head 模式进行细粒度分析
    layer_mode_overrides={}
)

# Swin Transformer (Window Attention)
num_windows = 64  # (H/window_size) * (W/window_size)
window_tokens = 49  # 7x7 window
attn = create_multi_head_attention(
    sid=0,
    name_prefix="stage0_block0_attn",
    enclave_mode=ExecutionModeOptions.Enclave,
    embed_dim=96,
    num_heads=3,
    batch_size=num_windows,  # 每个窗口作为一个 batch
    seq_len=window_tokens,
    per_head_mode=True,
    layer_mode_overrides={}
)
```

### 连接到前一层

```python
# 创建注意力模块
attn = create_multi_head_attention(...)

# 连接到前一层
output_layer = attn.connect(prev_layer)

# 获取所有层
all_layers = attn.get_all_layers()
```

## 模型适配情况

### ✅ 已完成适配的模型

1. **BERT** (`experiments/models/sgx_bert_native.py`)
   - 支持批量和 per-head 模式
   - 通过 `use_per_head_attention` 参数控制

2. **ALBERT** (`experiments/models/sgx_albert_native.py`)
   - 支持批量和 per-head 模式
   - 参数共享特性保持不变

3. **DistilBERT** (`experiments/models/sgx_distilbert_native.py`)
   - 支持批量和 per-head 模式
   - Pre-norm 架构保持不变

4. **TinyBERT** (`experiments/models/sgx_tinybert_native.py`)
   - 支持批量和 per-head 模式
   - 轻量级配置保持不变

5. **ViT** (`experiments/models/sgx_vit_native.py`)
   - 支持批量和 per-head 模式
   - Patch embedding 和 CLS token 处理保持不变

6. **Swin Transformer** (`experiments/models/sgx_swin_native.py`)
   - 通过将每个窗口视为 batch 来适配
   - 局部注意力特性完美适配
   - Window attention 和 Shifted window attention 都支持

## 参数说明

### create_multi_head_attention 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `sid` | int | Session ID |
| `name_prefix` | str | 层名称前缀（如 "encoder0_attn"） |
| `enclave_mode` | ExecutionModeOptions | 默认执行模式（CPU/Enclave） |
| `embed_dim` | int | 嵌入维度（必须能被 num_heads 整除） |
| `num_heads` | int | 注意力头数量 |
| `batch_size` | int | 批次大小 |
| `seq_len` | int | 序列长度（token 数量） |
| `per_head_mode` | bool | 是否使用 per-head 模式（默认 False） |
| `use_shared_qkv` | bool | Per-head 模式下是否共享 QKV 投影（默认 True） |
| `layer_mode_overrides` | Dict | 每层的执行模式覆盖 |

## Per-Head 模式详细说明

### 计算流程

在 per-head 模式下，多头注意力被分解为：

```
Input
  ↓
共享 Q/K/V 投影 (可选)
  ↓
┌─────────────┬─────────────┬─────────────┐
│   Head 0    │   Head 1    │   Head N    │
│   ↓         │   ↓         │   ↓         │
│ Reshape Q   │ Reshape Q   │ Reshape Q   │
│ Reshape K   │ Reshape K   │ Reshape K   │
│ Reshape V   │ Reshape V   │ Reshape V   │
│   ↓         │   ↓         │   ↓         │
│ Q @ K^T     │ Q @ K^T     │ Q @ K^T     │
│   ↓         │   ↓         │   ↓         │
│ Softmax     │ Softmax     │ Softmax     │
│   ↓         │   ↓         │   ↓         │
│ Attn @ V    │ Attn @ V    │ Attn @ V    │
└─────────────┴─────────────┴─────────────┘
  ↓
Concatenate
  ↓
Output Projection
  ↓
Output
```

### 性能分析优势

使用 per-head 模式，可以在 profiling 时得到：

```csv
name,type,execution_mode,time_ms,...
encoder0_attn_q_proj,Linear,Enclave,2.34
encoder0_attn_k_proj,Linear,Enclave,2.31
encoder0_attn_v_proj,Linear,Enclave,2.35
encoder0_attn_head0_qk_matmul,MatMul,Enclave,0.45
encoder0_attn_head0_softmax,Softmax,Enclave,0.12
encoder0_attn_head0_attn_v_matmul,MatMul,Enclave,0.43
encoder0_attn_head1_qk_matmul,MatMul,Enclave,0.46
encoder0_attn_head1_softmax,Softmax,Enclave,0.13
encoder0_attn_head1_attn_v_matmul,MatMul,Enclave,0.44
...
encoder0_attn_out_proj,Linear,Enclave,2.38
```

## 性能对比

### 批量模式 vs Per-head 模式

| 特性 | 批量模式 | Per-head 模式 |
|------|---------|--------------|
| 计算效率 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 分析粒度 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 内存峰值 | 高 | 中（可逐头处理） |
| 层数量 | ~10 层/注意力 | ~40 层/注意力（12头） |
| 适用场景 | 生产推理 | TEE 性能分析 |

## 常见问题

### Q: 如何在模型中启用 per-head 模式？

A: 在创建模型时传递 `use_per_head_attention=True`：

```python
model = create_bert_base(
    num_classes=2,
    enclave_mode=ExecutionModeOptions.Enclave,
    use_per_head_attention=True  # 启用 per-head 模式
)
```

### Q: Per-head 模式会影响计算结果吗？

A: 不会。两种模式在数学上完全等价，仅计算顺序不同。

### Q: Swin Transformer 的 window attention 如何适配？

A: 将每个窗口视为一个独立的 batch：
- `batch_size = num_windows * original_batch_size`
- `seq_len = window_size * window_size`

### Q: 如何选择使用哪种模式？

A: 
- **批量模式**：标准训练和推理，追求性能
- **Per-head 模式**：TEE 环境性能分析，需要细粒度测量

## 示例：完整的 profiling 流程

```python
from experiments.models.sgx_bert_native import create_bert_base
from python.utils.basic_utils import ExecutionModeOptions

# 创建启用 per-head 分析的模型
model = create_bert_base(
    num_classes=2,
    batch_size=1,
    seq_len=128,
    enclave_mode=ExecutionModeOptions.Enclave,
    use_per_head_attention=True  # 关键参数
)

# 获取所有层（包括每个头的计算层）
all_layers = model.get_all_layers()

print(f"Total layers: {len(all_layers)}")
# 输出: Total layers: ~1500+ (包括每个头的独立层)

# 进行 profiling...
```

## 技术细节

### 内存布局

**批量模式**：
```
Q/K/V: (batch, seq_len, num_heads, head_dim)
       一次性计算所有头
```

**Per-head 模式**：
```
Head 0: (batch, 1, seq_len, head_dim)
Head 1: (batch, 1, seq_len, head_dim)
...
分别计算，然后拼接
```

### TEE 优势

在 TEE (Trusted Execution Environment) 中：

1. **细粒度性能分析**：每个头的 get/compute/store 时间单独测量
2. **内存管理**：可以逐头处理，降低内存峰值
3. **依赖追踪**：清晰的层间依赖图，便于优化 CPU/Enclave 分区

## 维护者

本模块由 TAOISM 项目团队开发和维护。

如有问题或建议，请提交 issue。
