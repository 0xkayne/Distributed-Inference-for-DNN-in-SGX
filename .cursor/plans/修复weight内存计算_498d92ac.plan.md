---
name: 修复Weight内存计算
overview: 修复 `_get_layer_weight_shape()` 函数，使其能正确识别 `SGXConvBase` 和 `SGXLinearBase` 层的权重形状属性。
todos:
  - id: fix-weight-shape
    content: 修改 _get_layer_weight_shape() 优先使用 pytorch_w_shape/bias_shape 属性
    status: completed
  - id: add-sgx-attrs
    content: 添加 n_output_channel/n_input_channel/filter_hw 属性检查
    status: completed
  - id: test-fix
    content: 重新运行 Stem-Part1 验证修复效果
    status: completed
---

# 修复 Weight/Bias 内存计算

## 问题根因

[`profile_inception.py`](experiments/models/profile_inception.py) 中的 `_get_layer_weight_shape()` 函数检查的属性名与实际层类不匹配：

```python
# 当前代码（错误）
if hasattr(layer, 'out_channels') and hasattr(layer, 'in_channels'):
    ...
```

[`SGXConvBase`](python/layers/sgx_conv_base.py) 实际使用：

```python
self.n_output_channel = n_output_channel
self.n_input_channel = n_input_channel
self.filter_hw = filter_hw
self.pytorch_w_shape = [n_output_channel, n_input_channel, filter_hw, filter_hw]
self.bias_shape = [n_output_channel]
```



## 修复方案

修改 [`experiments/models/profile_inception.py`](experiments/models/profile_inception.py) 中的 `_get_layer_weight_shape()` 函数：

### 优先使用直接形状属性

```python
def _get_layer_weight_shape(layer) -> Tuple[List[int], List[int]]:
    weight_shape = []
    bias_shape = []
    layer_type = type(layer).__name__
    
    # 方案1：直接使用层的形状属性（最可靠）
    if hasattr(layer, 'pytorch_w_shape') and layer.pytorch_w_shape:
        weight_shape = list(layer.pytorch_w_shape)
    if hasattr(layer, 'bias_shape') and layer.bias_shape:
        bias_shape = list(layer.bias_shape)
    
    # 方案2：从 SGXConvBase 特定属性推断
    if not weight_shape and ('Conv' in layer_type or 'SGXConv' in layer_type):
        if hasattr(layer, 'n_output_channel') and hasattr(layer, 'n_input_channel'):
            out_ch = layer.n_output_channel
            in_ch = layer.n_input_channel
            kh = getattr(layer, 'filter_hw', 3)
            weight_shape = [out_ch, in_ch, kh, kh]
            if not bias_shape:
                bias_shape = [out_ch]
    
    # 方案3：从标准 PyTorch 属性推断（fallback）
    ...
```



### 需要修改的位置

在 `_get_layer_weight_shape()` 函数中（约第 230-280 行），添加对以下属性的检查优先级：

1. `pytorch_w_shape` / `bias_shape` (直接形状)
2. `n_output_channel` / `n_input_channel` / `filter_hw` (SGXConvBase)
3. `out_features` / `in_features` (SGXLinearBase)
4. `num_features` (BatchNorm)