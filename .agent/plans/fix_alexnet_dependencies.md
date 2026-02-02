# AlexNet Layer Dependencies Fix Plan

## Problem Analysis

### Current Issue
`profile_alexnet_enclave.py` 目前使用简单的顺序依赖跟踪方法（`self.last_layer_name`），导致输出的 CSV 文件中的依赖关系不准确。相比之下，`profile_bert_enclave.py` 使用显式的 `dependencies` 参数来精确跟踪层间依赖关系。

### Key Differences

#### BERT Profiler (Correct Approach)
- 每个 profiling 方法调用都显式传递 `dependencies` 参数
- 正确处理并行层（如 Q/K/V projections）的依赖关系
- 正确处理多输入层（如 MatMul 需要两个输入）的依赖关系
- 使用 `profiler_utils.py` 中的 `infer_layer_dependencies()` 作为后备方案

#### AlexNet Profiler (Current Issue)
- 仅使用 `self.last_layer_name` 跟踪上一层
- 所有层都简单地依赖于前一层，无法表达真实的数据流
- 依赖关系存储为：`deps = [self.last_layer_name] if self.last_layer_name else []`

### Root Cause
AlexNet 是一个序列模型，但仍需要正确的依赖关系表达：
1. **残差连接**: 虽然原始 AlexNet 没有残差，但依赖跟踪应该准确
2. **Memory Reset**: 在 FC 层前的 Enclave 重置导致依赖链断裂
3. **一致性**: 输出格式应与 BERT profiler 保持一致，便于下游分析工具使用

## Solution Design

### Approach 1: Explicit Dependencies (Recommended)
像 BERT profiler 一样，为每个层显式指定依赖关系。

**优点**:
- 精确控制依赖关系
- 与 BERT profiler 保持一致
- 便于未来扩展（如添加 skip connections）

**缺点**:
- 需要修改每个 profiling 调用

### Approach 2: Enhanced Sequential Tracking
改进 `self.last_layer_name` 机制，支持跨 Enclave reset 的依赖跟踪。

**优点**:
- 代码改动较小

**缺点**:
- 不够灵活
- 与 BERT profiler 不一致

## Implementation Plan (Approach 1)

### Step 1: Add Dependencies Parameter to Profiling Methods
修改所有 `_profile_*_enclave` 方法的签名，添加 `dependencies` 参数：

```python
def _profile_conv_enclave(
    self, name, input_shape, out_channels, k, s, p, group, verbose,
    dependencies: Optional[List[str]] = None  # NEW
):
```

同样修改：
- `_profile_linear_enclave`
- `_profile_relu_enclave`
- `_profile_maxpool_enclave`

### Step 2: Update LayerMetrics Creation
在创建 `LayerMetrics` 时使用传入的 `dependencies` 参数：

```python
# 使用提供的 dependencies 或作为后备使用 last_layer_name
if dependencies is None:
    dependencies = [self.last_layer_name] if self.last_layer_name else []

m = LayerMetrics(
    name, "Conv2d", group, "Enclave",
    input_shape=input_shape, output_shape=output_shape,
    input_bytes=_shape_to_bytes(input_shape),
    output_bytes=_shape_to_bytes(output_shape),
    dependencies=dependencies,  # 使用准确的依赖关系
    ...
)
```

### Step 3: Update profile_all Method
在 `profile_all()` 方法中，为每个层调用显式指定依赖关系：

```python
def profile_all(self, verbose: bool = True):
    # ...
    
    # Conv1 Block
    self._profile_conv_enclave(
        "conv1", [self.batch_size, 3, 224, 224],
        96, 11, 4, 2, "Feature", verbose,
        dependencies=[]  # First layer, no dependencies
    )
    self._profile_relu_enclave(
        "relu1", [self.batch_size, 96, 55, 55], "Feature", verbose,
        dependencies=["conv1"]  # Depends on conv1
    )
    self._profile_maxpool_enclave(
        "pool1", [self.batch_size, 96, 55, 55], 3, 2, 0, "Feature", verbose,
        dependencies=["relu1"]  # Depends on relu1
    )
    
    # Conv2 Block
    self._profile_conv_enclave(
        "conv2", [self.batch_size, 96, 27, 27],
        256, 5, 1, 2, "Feature", verbose,
        dependencies=["pool1"]  # Depends on pool1
    )
    # ... 继续为所有层指定依赖关系
```

### Step 4: Handle Enclave Resets
特别注意在 Enclave reset 后的层依赖关系仍然正确：

```python
# 在重置 Enclave 前记录最后一层
last_conv_layer = "pool3"

# Reset Enclave before HUGE FC layers
GlobalTensor.destroy()
GlobalTensor.init()

# FC1 仍然依赖于 pool3，即使 Enclave 已重置
self._profile_linear_enclave(
    "fc1", [self.batch_size, 9216], 4096, "Classifier", verbose,
    dependencies=[last_conv_layer]  # 跨 Enclave reset 的正确依赖
)
```

### Step 5: Remove or Keep last_layer_name
决定是否保留 `self.last_layer_name`：

**选项 A**: 完全移除（推荐）
- 强制显式指定依赖关系
- 与 BERT profiler 完全一致

**选项 B**: 保留作为后备
- 如果 `dependencies=None`，使用 `last_layer_name`
- 提供更多灵活性

## Expected Output Format

修复后，CSV 输出应该类似：

```csv
name,type,group,dependencies,...
conv1,Conv2d,Feature,[],...
relu1,ReLU,Feature,['conv1'],...
pool1,MaxPool,Feature,['relu1'],...
conv2,Conv2d,Feature,['pool1'],...
relu2,ReLU,Feature,['conv2'],...
...
fc1,Linear,Classifier,['pool3'],...
relu_fc1,ReLU,Classifier,['fc1'],...
fc2,Linear,Classifier,['relu_fc1'],...
...
```

## Testing Strategy

### 1. Unit Test Dependencies
验证每一层的依赖关系是否正确：
```python
assert metrics["conv1"].dependencies == []
assert metrics["relu1"].dependencies == ["conv1"]
assert metrics["pool1"].dependencies == ["relu1"]
assert metrics["conv2"].dependencies == ["pool1"]
# ...
assert metrics["fc1"].dependencies == ["pool3"]
```

### 2. Integration Test
运行完整的 profiling 并检查输出 CSV：
```bash
python -m experiments.models.profile_alexnet_enclave
```

### 3. Compare with BERT
确保输出格式与 BERT profiler 一致：
- CSV 列顺序一致
- dependencies 列格式一致（使用 Python list 的字符串表示）
- 所有必需的内存分析字段都存在

## Success Criteria

1. ✅ 每一层都有正确的 dependencies 列表
2. ✅ 跨 Enclave reset 的依赖关系正确
3. ✅ CSV 输出格式与 BERT profiler 一致
4. ✅ 下游分析工具（如算法代码）能正确解析依赖关系
5. ✅ 代码风格与 BERT profiler 保持一致

## Timeline

- **Step 1-2**: 修改方法签名和 LayerMetrics 创建逻辑（15 min）
- **Step 3**: 更新 profile_all 方法中的所有调用（20 min）
- **Step 4**: 处理 Enclave reset 的特殊情况（10 min）
- **Step 5**: 清理代码，移除 last_layer_name（5 min）
- **Testing**: 运行测试并验证输出（10 min）

**Total**: ~60 minutes

## Risk Assessment

- **Low Risk**: AlexNet 是纯序列模型，依赖关系简单
- **Impact**: 修复后将使下游分析工具能正确处理 AlexNet 数据
- **Rollback**: 保留当前版本作为备份

## Notes

- 这个修复也将为未来的其他 CNN 模型（如 ResNet、VGG）profiler 提供模板
- 考虑将通用的 profiling 逻辑提取到 `profiler_utils.py` 中
- 可以考虑为 CNN 模型添加专门的依赖推断函数（类似 Transformer 的 `infer_layer_dependencies`）
