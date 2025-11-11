# 使用示例大全

## 🎯 快速开始示例

### 示例1：验证环境（必做）

```bash
cd /root/exp_DNN_SGX/TAOISM
python experiments/quick_test.py
```

**预期输出**：
```
✓ Imports: PASS
✓ Model Creation: PASS
✓ LayerProfiler: PASS
✓ DataCollector: PASS
```

---

## 📊 测量示例

### 示例2：测量单个模型的计算开销

```bash
# NiN模型，CPU模式，10次迭代（快速测试）
python experiments/measurement/measure_computation.py \
    --single-model NiN \
    --devices CPU \
    --batch-sizes 1 \
    --iterations 10
```

### 示例3：测量多个模型的计算开销

```bash
# 3个模型，CPU模式，100次迭代
python experiments/measurement/measure_computation.py \
    --models NiN ResNet18 AlexNet \
    --devices CPU \
    --batch-sizes 1 \
    --iterations 100
```

### 示例4：测量不同批大小的影响

```bash
# NiN模型，多个批大小
python experiments/measurement/measure_computation.py \
    --single-model NiN \
    --devices CPU \
    --batch-sizes 1 4 8 16 \
    --iterations 50
```

### 示例5：测量通信开销

```bash
# NiN模型，三种带宽条件
python experiments/measurement/measure_communication.py \
    --single-model NiN \
    --bandwidths 10 100 1000 \
    --iterations 100
```

### 示例6：测量安全开销（需要SGX）

```bash
# 确保SGX环境已配置
source /opt/intel/sgxsdk/environment
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH

# 运行测量
python experiments/measurement/measure_security.py \
    --models NiN ResNet18 \
    --batch-size 1 \
    --iterations 100
```

### 示例7：测量EPC换页开销

```bash
# NiN模型，多个内存压力级别
python experiments/measurement/measure_paging.py \
    --single-model NiN \
    --pressures 50 75 90 100 \
    --iterations 50
```

---

## 🔄 批量测试示例

### 示例8：快速批量测试

```bash
# 2个模型，减少迭代次数，约10分钟
python experiments/run_all_measurements.py --quick-test
```

### 示例9：完整批量测试

```bash
# 所有6个模型，标准迭代次数，约1-2小时
python experiments/run_all_measurements.py --models all
```

### 示例10：选择性批量测试

```bash
# 只测试3个模型
python experiments/run_all_measurements.py \
    --models NiN ResNet18 AlexNet
```

### 示例11：包含换页测量的批量测试

```bash
# 2个轻量模型，包含换页测量
python experiments/run_all_measurements.py \
    --models NiN ResNet18 \
    --include-paging
```

---

## 📈 数据分析示例

### 示例12：分析单个模型

```bash
# 分析NiN的所有测量结果
python experiments/analyze_results.py --model NiN --type all
```

### 示例13：分析特定类型

```bash
# 只分析计算开销
python experiments/analyze_results.py --model NiN --type computation

# 只分析通信开销
python experiments/analyze_results.py --model VGG16 --type communication

# 只分析安全开销
python experiments/analyze_results.py --model ResNet18 --type security
```

### 示例14：查看可用数据

```bash
python experiments/analyze_results.py --list
```

### 示例15：批量分析所有模型

```bash
# 为所有模型生成图表
for model in NiN ResNet18 AlexNet VGG16 InceptionV3 InceptionV4; do
    echo "Analyzing $model..."
    python experiments/analyze_results.py --model $model --type all
done

# 查看生成的图表
ls -lh experiments/figures/
```

---

## 🔧 高级用法示例

### 示例16：自定义迭代次数

```bash
# 快速测试（10次迭代）
python experiments/measurement/measure_computation.py \
    --single-model NiN --devices CPU --iterations 10

# 标准测试（100次迭代）
python experiments/measurement/measure_computation.py \
    --single-model NiN --devices CPU --iterations 100

# 高精度测试（1000次迭代）
python experiments/measurement/measure_computation.py \
    --single-model NiN --devices CPU --iterations 1000
```

### 示例17：只测量特定层

编辑Python代码，添加层过滤：

```python
# 在 layer_profiler.py 中
def profile_all_layers(self, batch_size=1, num_iterations=100, layer_indices=None):
    results = []
    for idx, layer in enumerate(self.model.layers):
        if layer_indices is None or idx in layer_indices:
            result = self.profile_single_layer(layer, idx, batch_size, num_iterations)
            if result is not None:
                results.append(result)
    return results
```

使用：
```python
# 只测量前5层
profiler.profile_all_layers(layer_indices=[0, 1, 2, 3, 4])
```

### 示例18：导出CSV格式

```python
# 将JSON数据转换为CSV
import json
import csv

# 读取JSON
with open('experiments/data/computation_cost_NiN_CPU.json') as f:
    data = json.load(f)

# 写入CSV
with open('nin_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['layer_name', 'mean_ms', 'param_count'])
    writer.writeheader()
    for layer in data.get('layers', []):
        writer.writerow({
            'layer_name': layer['name'],
            'mean_ms': layer['mean_ms'],
            'param_count': layer['param_count']
        })
```

---

## 🐛 故障排查示例

### 示例19：处理Import错误

```bash
# 问题：ModuleNotFoundError
# 解决：确保在正确目录并设置路径

cd /root/exp_DNN_SGX/TAOISM
export PYTHONPATH=/root/exp_DNN_SGX/TAOISM:$PYTHONPATH
python experiments/quick_test.py
```

### 示例20：处理Enclave初始化失败

```bash
# 检查SGX状态
bash scripts/check_sgx2_edmm.sh

# 如果SGX不可用，先用CPU模式
python experiments/measurement/measure_computation.py \
    --single-model NiN --devices CPU
```

### 示例21：处理内存不足

```bash
# 方案1：减小批大小
python experiments/measurement/measure_computation.py \
    --single-model VGG16 --batch-sizes 1

# 方案2：使用更小的模型
python experiments/measurement/measure_computation.py \
    --models NiN ResNet18  # 而不是VGG16

# 方案3：调整chunk配置并重新编译
# 编辑 Include/common_with_enclaves.h
# 减小 STORE_CHUNK_ELEM
# 然后: make clean && make
```

---

## 📊 数据使用示例

### 示例22：提取成本模型参数

```python
import json
import numpy as np
from sklearn.linear_model import LinearRegression

# 读取数据
with open('experiments/data/computation_cost_NiN_CPU.json') as f:
    data = json.load(f)

# 提取特征和标签
X = []  # [param_count, input_size]
y = []  # time_ms

for layer in data.get('layers', []):
    if layer['param_count'] > 0:  # 有参数的层
        X.append([layer['param_count'], layer.get('memory_mb', 0)])
        y.append(layer['mean_ms'])

X = np.array(X)
y = np.array(y)

# 拟合线性模型
model = LinearRegression()
model.fit(X, y)

print(f"计算成本模型: T = {model.coef_[0]:.6f} * params + {model.coef_[1]:.6f} * memory + {model.intercept_:.6f}")
print(f"R²: {model.score(X, y):.4f}")
```

### 示例23：生成论文表格

```python
import json
import pandas as pd

# 收集所有模型数据
models = ['NiN', 'ResNet18', 'AlexNet', 'VGG16']
table_data = []

for model in models:
    filename = f'experiments/data/computation_cost_{model}_aggregated.json'
    with open(filename) as f:
        data = json.load(f)
    
    cpu_data = data['devices']['CPU']['batch_1']
    summary = cpu_data['summary']
    
    table_data.append({
        'Model': model,
        'Layers': summary['total_layers'],
        'Params (M)': summary['total_params'] / 1e6,
        'Memory (MB)': summary['total_memory_mb'],
        'Time (ms)': summary['total_time_ms']
    })

# 创建表格
df = pd.DataFrame(table_data)
print(df.to_markdown(index=False))

# 保存为CSV
df.to_csv('model_comparison.csv', index=False)
```

### 示例24：绘制自定义图表

```python
import json
import matplotlib.pyplot as plt

# 读取多个模型的数据
models = ['NiN', 'ResNet18', 'AlexNet']
times = []
params = []

for model in models:
    with open(f'experiments/data/computation_cost_{model}_CPU.json') as f:
        data = json.load(f)
    
    summary = data.get('summary', {})
    times.append(summary['total_time_ms'])
    params.append(summary['total_params'] / 1e6)

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(params, times, s=100, alpha=0.6)

for i, model in enumerate(models):
    plt.annotate(model, (params[i], times[i]), 
                xytext=(5, 5), textcoords='offset points')

plt.xlabel('Parameters (Million)')
plt.ylabel('Inference Time (ms)')
plt.title('Model Size vs Inference Time')
plt.grid(True, alpha=0.3)
plt.savefig('custom_plot.png', dpi=300, bbox_inches='tight')
print("图表已保存: custom_plot.png")
```

---

## 🎓 论文写作示例

### 示例25：引用实验数据

```latex
% LaTeX论文示例

\section{实验评估}

\subsection{实验设置}

本文基于TAOISM框架实现了完整的测量系统，测试了6个代表性DNN模型：
NiN、ResNet18、AlexNet、VGG16、Inception V3和Inception V4。
每个模型进行100次推理测量，取平均值作为最终结果。

\subsection{计算开销}

表\ref{tab:computation}展示了各模型在CPU和SGX Enclave中的推理时间对比。

\begin{table}[h]
\centering
\caption{各模型计算开销对比}
\label{tab:computation}
\begin{tabular}{lrrr}
\hline
模型 & 参数量(M) & CPU(ms) & Enclave(ms) \\
\hline
NiN      & 1.0  & 45.2  & 58.7  \\
ResNet18 & 11.2 & 123.5 & 156.3 \\
AlexNet  & 60.0 & 234.1 & 298.4 \\
VGG16    & 138.4& 456.7 & 587.2 \\
\hline
\end{tabular}
\end{table}

% 数据来源: experiments/data/computation_cost_*_aggregated.json
```

### 示例26：引用图表

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/NiN_computation_layerwise.png}
\caption{NiN模型层级计算时间分布}
\label{fig:nin_computation}
\end{figure}

% 图表来源: python experiments/analyze_results.py --model NiN --type computation
```

---

## 🔬 研究分析示例

### 示例27：分析安全开销趋势

```python
import json
import numpy as np

models = ['NiN', 'ResNet18', 'AlexNet', 'VGG16']
overhead_percents = []

for model in models:
    with open(f'experiments/data/security_cost_{model}.json') as f:
        data = json.load(f)
    
    overhead = data['summary']['total_overhead_percent']
    overhead_percents.append(overhead)
    print(f"{model}: {overhead:.1f}% security overhead")

avg_overhead = np.mean(overhead_percents)
print(f"\n平均安全开销: {avg_overhead:.1f}%")
print(f"结论: TEE执行平均增加约{avg_overhead:.0f}%的时间开销")
```

### 示例28：分析带宽-延迟关系

```python
import json
import matplotlib.pyplot as plt

# 读取通信开销数据
with open('experiments/data/communication_cost_NiN.json') as f:
    data = json.load(f)

# 提取带宽和延迟
bandwidths = data['bandwidths_mbps']
total_costs = []

for bw in bandwidths:
    cost = data['summary']['total_comm_cost'][f'{bw}Mbps']
    total_costs.append(cost)

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(bandwidths, total_costs, 'o-', linewidth=2, markersize=8)
plt.xlabel('Bandwidth (Mbps)')
plt.ylabel('Communication Cost (ms)')
plt.title('Bandwidth vs Communication Latency')
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.savefig('bandwidth_latency.png', dpi=300)

# 拟合模型
# T_comm = a + b/BW
from scipy.optimize import curve_fit

def comm_model(bw, a, b):
    return a + b / bw

params, _ = curve_fit(comm_model, bandwidths, total_costs)
print(f"通信成本模型: T_comm = {params[0]:.2f} + {params[1]:.2f}/BW")
```

---

## 🛠️ 扩展开发示例

### 示例29：添加新的测量指标

```python
# 在 layer_profiler.py 中添加新指标

class LayerProfiler:
    def benchmark_layer(self, layer, input_tensor, num_iterations=100, warmup=10):
        times = []
        memory_usage = []  # 新增：内存使用记录
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            
            # 记录内存使用
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated()
            
            output = layer.forward(input_tensor)
            
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated()
                memory_usage.append(mem_after - mem_before)
            
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        return {
            'mean_ms': float(np.mean(times)),
            # ... 其他统计 ...
            'avg_memory_mb': float(np.mean(memory_usage)) / (1024*1024) if memory_usage else 0,
        }
```

### 示例30：添加新模型

```python
# experiments/models/my_custom_model.py

import sys
sys.path.insert(0, '.')

from python.layers.sgx_conv_base import SGXConvBase
from python.layers.sgx_linear_base import SGXLinearBase
# ... 其他imports

class SGXMyCustomModel:
    def __init__(self, sid=0, num_classes=10, 
                 enclave_mode=ExecutionModeOptions.Enclave):
        self.layers = self._build_network()
        self.model_name = 'MyCustomModel'
    
    def _build_network(self):
        layers = []
        # 定义您的模型结构
        # ...
        return layers

# 然后在 models/__init__.py 中注册
# from .my_custom_model import SGXMyCustomModel
# __all__.append('SGXMyCustomModel')

# 在测量脚本中添加
# MODEL_REGISTRY['MyCustomModel'] = SGXMyCustomModel
```

---

## 📝 论文数据准备示例

### 示例31：准备表格数据

```bash
# 收集所有模型的完整数据
python experiments/run_all_measurements.py --models all

# 提取为Excel友好格式
python << 'EOF'
import json
import pandas as pd

models = ['NiN', 'ResNet18', 'AlexNet', 'VGG16', 'InceptionV3', 'InceptionV4']
results = []

for model in models:
    # 计算开销
    with open(f'experiments/data/computation_cost_{model}_aggregated.json') as f:
        comp = json.load(f)
    cpu_time = comp['devices']['CPU']['batch_1']['summary']['total_time_ms']
    
    # 通信开销
    with open(f'experiments/data/communication_cost_{model}.json') as f:
        comm = json.load(f)
    comm_100mbps = comm['summary']['total_comm_cost']['100Mbps']
    
    results.append({
        'Model': model,
        'CPU Time (ms)': cpu_time,
        'Comm 100Mbps (ms)': comm_100mbps,
        'Total (ms)': cpu_time + comm_100mbps
    })

df = pd.DataFrame(results)
df.to_csv('paper_table_data.csv', index=False)
df.to_excel('paper_table_data.xlsx', index=False)
print("数据已导出到 paper_table_data.csv/xlsx")
EOF
```

### 示例32：准备图表数据

```bash
# 生成所有论文所需图表
for model in NiN ResNet18 AlexNet VGG16 InceptionV3 InceptionV4; do
    python experiments/analyze_results.py --model $model --type all
done

# 整理图表
mkdir -p paper_figures
cp experiments/figures/*.png paper_figures/

echo "图表已复制到 paper_figures/"
ls -lh paper_figures/
```

---

## 🎯 完整实验流程示例

### 示例33：从零到完成的完整流程

```bash
#!/bin/bash
# 完整实验流程

# 1. 环境验证
echo "Step 1: 验证环境"
python experiments/quick_test.py

# 2. 收集计算开销数据
echo "Step 2: 测量计算开销"
python experiments/measurement/measure_computation.py \
    --models NiN ResNet18 AlexNet \
    --devices CPU \
    --batch-sizes 1 4 8 \
    --iterations 100

# 3. 收集通信开销数据
echo "Step 3: 测量通信开销"
python experiments/measurement/measure_communication.py \
    --models NiN ResNet18 AlexNet \
    --bandwidths 10 100 1000 \
    --iterations 100

# 4. 收集安全开销数据
echo "Step 4: 测量安全开销"
python experiments/measurement/measure_security.py \
    --models NiN ResNet18 \
    --iterations 100

# 5. 生成图表
echo "Step 5: 生成图表"
for model in NiN ResNet18 AlexNet; do
    python experiments/analyze_results.py --model $model --type all
done

# 6. 整理结果
echo "Step 6: 整理结果"
mkdir -p final_results/{data,figures}
cp experiments/data/*.json final_results/data/
cp experiments/figures/*.png final_results/figures/

echo "完成！结果保存在 final_results/"
```

---

## 🎁 实用技巧

### 技巧1：并行测试多个模型

```bash
# 使用GNU parallel或后台任务
python experiments/measurement/measure_computation.py --single-model NiN &
python experiments/measurement/measure_computation.py --single-model ResNet18 &
wait
echo "两个模型测试完成"
```

### 技巧2：定时运行长时间测试

```bash
# 使用nohup在后台运行
nohup python experiments/run_all_measurements.py --models all > output.log 2>&1 &

# 查看进度
tail -f output.log
```

### 技巧3：快速比较两个模型

```bash
# 一行命令对比
python << 'EOF'
import json

models = ['NiN', 'ResNet18']
for m in models:
    with open(f'experiments/data/computation_cost_{m}_CPU.json') as f:
        data = json.load(f)
    time = sum(l['mean_ms'] for l in data['layers'])
    print(f"{m}: {time:.2f}ms")
EOF
```

---

**提示**：更多示例请参考各测量脚本的 `--help` 输出。

```bash
python experiments/measurement/measure_computation.py --help
python experiments/measurement/measure_communication.py --help
python experiments/measurement/measure_security.py --help
python experiments/measurement/measure_paging.py --help
python experiments/run_all_measurements.py --help
```

