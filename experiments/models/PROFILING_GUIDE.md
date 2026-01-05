# TAOISM 模型推理性能与内存分析指南

本文档介绍如何使用 TAOISM 项目中的 profiling 脚本来测量深度学习模型在 CPU 和 SGX Enclave 模式下的逐层推理时延和内存开销。

## 目录

- [概述](#概述)
- [环境准备](#环境准备)
- [支持的模型](#支持的模型)
- [命令行参数说明](#命令行参数说明)
- [运行示例](#运行示例)
- [输出文件说明](#输出文件说明)
- [内存分析字段详解](#内存分析字段详解)
- [常见问题](#常见问题)

---

## 概述

### 执行模式

TAOISM 支持两种执行模式进行性能分析：

| 模式 | 描述 | 脚本命名 |
|------|------|----------|
| **CPU (Native)** | 在标准 CPU 环境中执行，作为性能基准 | `profile_{model}_native.py` |
| **Enclave (SGX)** | 在 Intel SGX 安全飞地中执行，测量可信执行环境开销 | `profile_{model}_enclave.py` |

### 测量指标

每个脚本会测量并输出以下指标：

- **推理时延**：包括均值 (mean)、标准差 (std)、最小值 (min)、最大值 (max)、P95、P99
- **内存开销**：CPU 内存、TEE 内存、加密开销、权重/偏置/激活大小
- **数据流**：输入/输出形状和字节数、层间依赖关系

---

## 环境准备

### 1. 激活 Conda 环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate taoism
```

### 2. Enclave 模式额外配置

运行 Enclave 模式脚本时，需要设置 `LD_PRELOAD` 以解决 libstdc++ 版本兼容问题：

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

或者在每条命令前添加此前缀：

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python -m ...
```

### 3. 确保 SGX Enclave 已编译

```bash
cd /root/exp_DNN_SGX/TAOISM
make  # 编译 Enclave 和相关组件
```

---

## 支持的模型

| 模型 | 类型 | 变体 (--model) | 参考论文 |
|------|------|----------------|----------|
| **ViT** | Vision Transformer | tiny, small, base | An Image is Worth 16x16 Words (ICLR 2021) |
| **BERT** | NLP Transformer | base | BERT: Pre-training of Deep Bidirectional Transformers (NAACL 2019) |
| **DistilBERT** | 蒸馏 BERT | base | DistilBERT (NeurIPS 2019) |
| **TinyBERT** | 小型 BERT | 4layer_312hidden | TinyBERT (EMNLP 2020) |
| **ALBERT** | 轻量级 BERT | base | ALBERT (ICLR 2020) |
| **Swin Transformer** | 层次化 Vision Transformer | tiny, small, base | Swin Transformer (ICCV 2021) |

---

## 命令行参数说明

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | 因模型而异 | 模型变体，如 `base`、`tiny`、`small` |
| `--batch-size` | int | 1 | 批次大小 |
| `--iterations` | int | 30 (native) / 10 (enclave) | 测量迭代次数（不含预热） |
| `--warmup` | int | 5 (native) / 3 (enclave) | 预热迭代次数 |
| `--output-dir` | str | `experiments/data` | 输出目录 |
| `--quiet` | flag | - | 静默模式，减少输出 |

### 视觉模型专用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--img-size` / `--image-size` | int | 224 | 输入图像尺寸 |
| `--num-classes` | int | 1000 | 分类类别数 |
| `--patch-size` | int | 16 | Patch 大小 (ViT) |
| `--window-size` | int | 7 | 窗口大小 (Swin) |

### NLP 模型专用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--seq-len` | int | 128 | 输入序列长度 |
| `--vocab-size` | int | 30522 | 词表大小 |
| `--num-classes` | int | 2 | 分类类别数 |

---

## 运行示例

### ViT (Vision Transformer)

```bash
# CPU 模式 - tiny 变体
python -m experiments.models.profile_vit_native --model tiny --iterations 30

# CPU 模式 - base 变体
python -m experiments.models.profile_vit_native --model base --iterations 30

# Enclave 模式 - tiny 变体
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  python -m experiments.models.profile_vit_enclave --model tiny --iterations 10

# 自定义图像尺寸
python -m experiments.models.profile_vit_native --model base --img-size 384 --patch-size 16
```

### BERT

```bash
# CPU 模式
python -m experiments.models.profile_bert_native --model base --seq-len 128

# Enclave 模式
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  python -m experiments.models.profile_bert_enclave --model base --seq-len 128

# 长序列测试
python -m experiments.models.profile_bert_native --model base --seq-len 512 --iterations 20
```

### DistilBERT

```bash
# CPU 模式
python -m experiments.models.profile_distilbert_native --model base --seq-len 128

# Enclave 模式
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  python -m experiments.models.profile_distilbert_enclave --model base --seq-len 128
```

### TinyBERT

```bash
# CPU 模式 (4层, 312维隐藏层)
python -m experiments.models.profile_tinybert_native --model 4l --seq-len 128

# CPU 模式 (6层, 768维隐藏层)
python -m experiments.models.profile_tinybert_native --model 6l --seq-len 128

# Enclave 模式
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  python -m experiments.models.profile_tinybert_enclave --model 4l --seq-len 128
```

### ALBERT

```bash
# CPU 模式
python -m experiments.models.profile_albert_native --model base --seq-len 128

# Enclave 模式
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  python -m experiments.models.profile_albert_enclave --model base --seq-len 128
```

### Swin Transformer

```bash
# CPU 模式 - tiny 变体
python -m experiments.models.profile_swin_native --model tiny --image-size 224

# CPU 模式 - base 变体
python -m experiments.models.profile_swin_native --model base --image-size 224

# Enclave 模式
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  python -m experiments.models.profile_swin_enclave --model tiny --image-size 224
```

---

## 输出文件说明

### 输出目录结构

```
experiments/data/
├── vit_tiny_layers.csv           # ViT tiny CPU 模式
├── vit_tiny_enclave_layers.csv   # ViT tiny Enclave 模式
├── vit_base_layers.csv           # ViT base CPU 模式
├── bert_base_layers.csv          # BERT base CPU 模式
├── bert_base_enclave_layers.csv  # BERT base Enclave 模式
├── distilbert_base_layers.csv    # DistilBERT base CPU 模式
├── tinybert_4layer_312hidden_layers.csv  # TinyBERT CPU 模式
├── albert_base_layers.csv        # ALBERT base CPU 模式
├── swin_tiny_layers.csv          # Swin tiny CPU 模式
└── ...
```

### 文件命名规则

- **CPU 模式**：`{model}_{variant}_layers.csv`
- **Enclave 模式**：`{model}_{variant}_enclave_layers.csv`

### CSV 列说明

| 列名 | 类型 | 说明 |
|------|------|------|
| `name` | string | 层名称，如 `block0_attn_qkv_proj` |
| `type` | string | 层类型，如 `Linear`, `Conv2d`, `LayerNorm` |
| `group` | string | 分组名称，如 `Block0`, `PatchEmbed`, `ClassHead` |
| `execution_mode` | string | 执行模式，`CPU` 或 `Enclave` |

**时间统计列 (单位: 毫秒)**

| 列名 | 说明 |
|------|------|
| `cpu_time_mean` | CPU 模式平均时延 |
| `cpu_time_std` | CPU 模式标准差 |
| `cpu_time_min` | CPU 模式最小时延 |
| `cpu_time_max` | CPU 模式最大时延 |
| `enclave_time_mean` | Enclave 模式平均时延 |
| `enclave_time_std` | Enclave 模式标准差 |
| `enclave_time_min` | Enclave 模式最小时延 |
| `enclave_time_max` | Enclave 模式最大时延 |
| `enclave_time_p95` | Enclave 模式 P95 分位数 |
| `enclave_time_p99` | Enclave 模式 P99 分位数 |

**数据流列**

| 列名 | 类型 | 说明 |
|------|------|------|
| `input_bytes` | int | 输入张量字节数 |
| `output_bytes` | int | 输出张量字节数 |
| `input_shape` | string | 输入形状，如 `[1, 197, 192]` |
| `output_shape` | string | 输出形状，如 `[1, 197, 576]` |
| `dependencies` | string | 依赖层列表 |
| `num_iterations` | int | 测量迭代次数 |

---

## 内存分析字段详解

### 内存字段

| 列名 | 单位 | 说明 |
|------|------|------|
| `cpu_memory_bytes` | bytes | CPU 模式总内存 (激活 + 权重 + 偏置) |
| `tee_memory_bytes` | bytes | TEE 模式张量内存 (与 CPU 相同) |
| `tee_encryption_overhead` | bytes | AES-GCM 加密元数据开销 |
| `tee_total_memory_bytes` | bytes | TEE 总内存 = 张量 + 加密开销 |
| `weight_bytes` | bytes | 权重张量大小 |
| `bias_bytes` | bytes | 偏置张量大小 |
| `activation_bytes` | bytes | 激活张量大小 (输入 + 输出) |
| `num_chunks` | int | ChunkPool 块数 |
| `chunk_metadata_bytes` | bytes | 每块加密元数据大小 (约 60 字节) |

### TEE 加密开销计算

SGX 使用 AES-GCM 加密模式，每个 Chunk 的加密元数据包括：

```
CHUNK_ENCRYPTION_OVERHEAD = SGX_AESGCM_IV_SIZE (12) + 
                            SGX_AESGCM_MAC_SIZE (16) + 
                            SGX_AES_GCM_STRUCT_SIZE (32)
                          ≈ 60 bytes
```

公式：
```
tee_total_memory = tee_memory_bytes + (num_chunks × chunk_metadata_bytes)
```

### ChunkPool 共享开销

ChunkPool 是一个预分配的内存池，由所有层共享：

```
ChunkPool Overhead = THREAD_POOL_SIZE × 2 × STORE_CHUNK_ELEM × 4 bytes
                   = 4 × 2 × 4276896 × 4
                   ≈ 136 MB
```

这部分内存在模型加载时一次性分配，不计入单层开销。

### 各层类型权重/偏置估算

| 层类型 | 权重形状 | 偏置形状 |
|--------|----------|----------|
| Conv2d | `[out_ch, in_ch, kH, kW]` | `[out_ch]` |
| Linear | `[out_features, in_features]` | `[out_features]` |
| LayerNorm | `[normalized_dim]` (gamma) | `[normalized_dim]` (beta) |
| BatchNorm | `[num_features × 2]` | `[num_features × 2]` |
| Softmax, GELU, ReLU, MatMul | 无权重 | 无偏置 |

---

## 常见问题

### Q1: 运行 Enclave 脚本时报 `GLIBCXX_3.4.32 not found` 错误

**解决方案**：添加 `LD_PRELOAD` 环境变量：

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python -m experiments.models.profile_vit_enclave --model base
```

### Q2: 报 `ModuleNotFoundError: No module named 'torch'`

**解决方案**：确保已激活正确的 Conda 环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate taoism
```

### Q3: Enclave 模式报 `terminate called after throwing an instance of '_status_t'`

**可能原因**：
1. Enclave 未正确编译 - 运行 `make clean && make`
2. 内存不足 - 减少 `--batch-size` 或使用更小的模型变体

### Q4: 如何只测量特定层？

目前脚本会测量模型的所有层。如需测量特定层，可以修改脚本中的 `_profile_all_layers` 方法，或在运行后从 CSV 中筛选所需层。

### Q5: 如何增加测量精度？

增加 `--iterations` 参数值：

```bash
python -m experiments.models.profile_vit_native --model base --iterations 100 --warmup 10
```

### Q6: 输出 CSV 中 enclave_time 为 0 是什么原因？

在 Native (CPU) 模式脚本中，`enclave_time_*` 字段为 0 是正常的，因为该模式不涉及 Enclave 执行。反之亦然。

### Q7: 如何批量运行多个模型的测试？

可以创建批处理脚本：

```bash
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate taoism

# CPU 模式
for model in vit bert distilbert tinybert albert swin; do
    python -m experiments.models.profile_${model}_native --model base
done

# Enclave 模式
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
for model in vit bert distilbert tinybert albert swin; do
    python -m experiments.models.profile_${model}_enclave --model base
done
```

---

## 脚本源码位置

| 脚本类型 | 路径 |
|----------|------|
| Profiling 工具模块 | `experiments/models/profiler_utils.py` |
| ViT Native | `experiments/models/profile_vit_native.py` |
| ViT Enclave | `experiments/models/profile_vit_enclave.py` |
| BERT Native | `experiments/models/profile_bert_native.py` |
| BERT Enclave | `experiments/models/profile_bert_enclave.py` |
| DistilBERT Native | `experiments/models/profile_distilbert_native.py` |
| DistilBERT Enclave | `experiments/models/profile_distilbert_enclave.py` |
| TinyBERT Native | `experiments/models/profile_tinybert_native.py` |
| TinyBERT Enclave | `experiments/models/profile_tinybert_enclave.py` |
| ALBERT Native | `experiments/models/profile_albert_native.py` |
| ALBERT Enclave | `experiments/models/profile_albert_enclave.py` |
| Swin Native | `experiments/models/profile_swin_native.py` |
| Swin Enclave | `experiments/models/profile_swin_enclave.py` |

---

## 参考资料

- [Intel SGX Documentation](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html)
- [TAOISM Project](https://github.com/xxx/TAOISM)
- [ViT Paper](https://arxiv.org/abs/2010.11929)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
