# Distributed TEE Inference Simulation

本模块提供用于分析和模拟 DNN 模型在多个 TEE 节点上分布式推理的工具。

## 目录结构

```
experiments/distributed/
├── __init__.py          # 模块初始化
├── dag_model.py         # DAG 数据结构
├── cost_model.py        # 开销模型
├── simulator.py         # 分布式执行模拟器
├── scheduler.py         # 调度策略
├── run_experiments.py   # 实验运行脚本
├── visualize.py         # 可视化工具
└── README.md            # 本文件
```

## 快速开始

### 1. 使用样本数据运行实验

```bash
cd /root/exp_DNN_SGX/TAOISM

# 运行所有实验（使用样本DAG）
python experiments/distributed/run_experiments.py --use-sample

# 生成可视化图表
python experiments/distributed/visualize.py \
    --results-dir experiments/data/distributed \
    --output-dir experiments/figures/distributed
```

### 2. 使用实际测量数据

首先，运行 Inception V3 性能测量（需要 SGX 硬件）：

```bash
# 运行分组测量（需要 SGX 硬件）
python experiments/models/profile_inception.py \
    --iterations 30 \
    --warmup 5 \
    --output-dir experiments/data
```

然后使用测量数据运行分布式模拟：

```bash
# 使用测量数据运行实验
python experiments/distributed/run_experiments.py \
    --data-path experiments/data/inception_v3_layers.json
```

## 核心组件

### DAG 模型 (`dag_model.py`)

表示神经网络层依赖关系的有向无环图。

```python
from experiments.distributed import InceptionDAG

# 从测量数据加载
dag = InceptionDAG.build_from_json('experiments/data/inception_v3_layers.json')

# 分析并行性
dag.print_summary()
critical_path, cp_time = dag.compute_critical_path()
parallel_groups = dag.get_parallel_groups()
```

### 开销模型 (`cost_model.py`)

计算分布式执行的各项开销。

```python
from experiments.distributed import CostConfig, CostModel

config = CostConfig(
    network_bandwidth_mbps=1000,  # 1 Gbps
    network_latency_ms=0.1,
    enclave_init_ms=100,
)

model = CostModel(config)
comm_cost = config.get_communication_cost_ms(size_bytes=1024*1024)  # 1MB
```

### 模拟器 (`simulator.py`)

模拟多 TEE 节点并行执行。

```python
from experiments.distributed import DistributedSimulator, HEFTScheduler

sim = DistributedSimulator(dag, num_nodes=4, cost_config=config)
scheduler = HEFTScheduler(dag)
results = sim.simulate(scheduler.get_scheduler())

print(f"Speedup: {results['speedup']:.2f}x")
print(f"Makespan: {results['makespan_ms']:.2f} ms")
```

### 调度策略 (`scheduler.py`)

提供多种调度算法：

- `ASAPScheduler`: 贪心调度，尽快执行
- `HEFTScheduler`: 异构最早完成时间算法
- `CriticalPathFirstScheduler`: 关键路径优先
- `LoadBalancingScheduler`: 负载均衡
- `MinCommunicationScheduler`: 最小化通信

## 实验结果

运行实验后，结果保存在：

- `experiments/data/distributed/`: JSON 格式的实验数据
  - `node_scaling.json`: 节点数 vs 加速比
  - `bandwidth_scaling.json`: 带宽 vs 加速比
  - `scheduler_comparison.json`: 调度策略比较

- `experiments/figures/distributed/`: 可视化图表
  - `speedup_vs_nodes.png`: 加速比曲线
  - `utilization_heatmap.png`: 节点利用率热力图
  - `scheduler_comparison.png`: 调度策略对比

## API 参考

### InceptionDAG

```python
class InceptionDAG:
    def build_from_json(json_path: str) -> InceptionDAG
    def build_from_csv(csv_path: str) -> InceptionDAG
    def build_from_model(model) -> InceptionDAG
    
    def topological_sort() -> List[str]
    def compute_critical_path(use_enclave=True) -> Tuple[List[str], float]
    def get_parallel_groups() -> List[List[str]]
    def get_parallelism_stats() -> Dict
    def print_summary() -> None
```

### DistributedSimulator

```python
class DistributedSimulator:
    def __init__(dag, num_nodes, cost_config=None, use_enclave=True)
    def simulate(scheduler) -> Dict[str, Any]
    def get_gantt_data() -> List[Dict]
```

### 调度策略

```python
class SchedulingStrategy:
    def get_scheduler() -> Callable

# 可用策略
ASAPScheduler()
HEFTScheduler(dag)
CriticalPathFirstScheduler(dag)
LoadBalancingScheduler(dag)
MinCommunicationScheduler(dag)
```

## 依赖

```
numpy
matplotlib  # 可视化
networkx    # DAG结构可视化（可选）
```

安装：
```bash
pip install numpy matplotlib networkx
```

