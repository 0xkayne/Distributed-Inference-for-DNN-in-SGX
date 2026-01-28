# ALBERT/DistilBERT/TinyBERT Profiler 修复总结

## 修改内容

基于 BERT profiler 的修复，对 ALBERT、DistilBERT、TinyBERT 应用相同的修复。

### 1. ALBERT (`experiments/models/profile_albert_enclave.py`)

**状态**：✅ 已完成

**修改内容**：
- ✅ 添加 `use_per_head_attention` 参数到 `__init__`
- ✅ 调整 `ENCLAVE_RESET_INTERVAL`（per-head=1, batched=4）
- ✅ 添加 Attention Mode 显示
- ✅ `_profile_encoder_block` 改为调度函数
- ✅ 添加 `_profile_encoder_block_batched` 方法（带正确依赖）
- ✅ 添加 `_profile_encoder_block_per_head` 方法（带正确依赖）
- ✅ 所有 `_profile_*_enclave` 函数添加 `dependencies` 参数
- ✅ 修改输出文件名（添加 `_per_head` 后缀）
- ✅ 添加 `--per-head` 命令行参数
- ✅ 添加 `use_per_head_attention` 到 profiling_config

### 2. DistilBERT (`experiments/models/profile_distilbert_enclave.py`)

**状态**：⏳ 进行中

**需要的修改**（与 ALBERT 相同）：
- [ ] 添加 `use_per_head_attention` 参数
- [ ] 其他同 ALBERT

### 3. TinyBERT (`experiments/models/profile_tinybert_enclave.py`)

**状态**：⏳ 待处理

**需要的修改**（与 ALBERT 相同）

## 关键修改模式

### A. __init__ 参数

```python
def __init__(
    self,
    ...
    warmup_iterations: int = 3,
    use_per_head_attention: bool = False,  # 新增
):
    ...
    self.use_per_head_attention = use_per_head_attention
```

### B. ENCLAVE_RESET_INTERVAL

```python
ENCLAVE_RESET_INTERVAL = 1 if self.use_per_head_attention else 4
```

### C. _profile_encoder_block 调度

```python
def _profile_encoder_block(self, block_idx: int, verbose: bool):
    if self.use_per_head_attention:
        self._profile_encoder_block_per_head(block_idx, verbose)
    else:
        self._profile_encoder_block_batched(block_idx, verbose)
```

### D. 依赖关系（关键！）

#### 批量模式：
```python
prev_output = 'embedding' if block_idx == 0 else f'encoder{block_idx-1}_norm2'

self._profile_linear_enclave(..., dependencies=[prev_output])  # Q/K/V
self._profile_matmul_enclave(..., dependencies=[q_proj, k_proj])  # QK
self._profile_softmax_enclave(..., dependencies=[qk_matmul])
self._profile_matmul_enclave(..., dependencies=[softmax, v_proj])  # Attn@V
self._profile_linear_enclave(..., dependencies=[attn_v])  # Out
```

#### Per-head 模式：
```python
# Q/K/V 并行
dependencies=[prev_output] for all Q/K/V

# 每个头并行
for head in heads:
    head_qk: dependencies=[q_proj, k_proj]  # 所有头相同！
    head_softmax: dependencies=[head_qk]
    head_attn_v: dependencies=[head_softmax, v_proj]

# Out projection
dependencies=[all_head_outputs]  # 所有头的输出
```

### E. 命令行参数

```python
parser.add_argument('--per-head', action='store_true',
                   help='Enable per-head attention profiling')

profiler = Profiler(..., use_per_head_attention=args.per_head)
```

### F. 输出文件

```python
suffix = '_per_head' if self.use_per_head_attention else ''
csv_path = f'{model}_{variant}_enclave{suffix}_layers.csv'
json_path = f'{model}_{variant}_enclave{suffix}_layers.json'
```

## 验证

修改完成后，运行：

```bash
# ALBERT
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_albert_enclave --model base --per-head

# DistilBERT
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_distilbert_enclave --model base --per-head

# TinyBERT
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
python -m experiments.models.profile_tinybert_enclave --model 4l --per-head
```

验证依赖关系：

```bash
python experiments/models/verify_dependencies.py \
    --input experiments/data/albert_base_enclave_per_head_layers.csv \
    --heads 12 --layers 12
```

---

**状态**：ALBERT 完成，DistilBERT 和 TinyBERT 进行中
