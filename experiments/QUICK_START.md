# 快速开始指南

## 🎯 5分钟快速测试

### 步骤1：环境准备（30秒）

```bash
cd /root/exp_DNN_SGX/TAOISM
conda activate taoism
source /opt/intel/sgxsdk/environment
```

### 步骤2：验证安装（1分钟）

```bash
python experiments/quick_test.py
```

**预期输出**：所有测试显示 ✓ PASS

### 步骤3：运行第一个测量（3分钟）

```bash
# 测量NiN模型的计算开销（CPU模式，10次迭代）
python experiments/measurement/measure_computation.py \
    --single-model NiN \
    --devices CPU \
    --batch-sizes 1 \
    --iterations 10
```

**预期输出**：
- 显示每层的测量进度
- 生成`experiments/data/computation_cost_NiN_CPU.json`

### 步骤4：查看结果（30秒）

```bash
# 查看生成的数据文件
ls -lh experiments/data/

# 查看数据内容（前30行）
head -30 experiments/data/computation_cost_NiN_CPU.json
```

## 🚀 完整测试流程

### 选项A：快速模式（10-15分钟）

```bash
# 测试2个模型，减少迭代次数
python experiments/run_all_measurements.py --quick-test
```

这将运行：
- ✓ 计算开销测量（CPU模式）
- ✓ 通信开销测量
- ✓ 安全开销测量（需要SGX支持）

### 选项B：单项测试

```bash
# 1. 只测量计算开销
python experiments/measurement/measure_computation.py \
    --models NiN ResNet18 \
    --devices CPU \
    --batch-sizes 1 \
    --iterations 100

# 2. 只测量通信开销
python experiments/measurement/measure_communication.py \
    --models NiN \
    --bandwidths 10 100 1000 \
    --iterations 100

# 3. 只测量安全开销（需要SGX）
python experiments/measurement/measure_security.py \
    --models NiN \
    --batch-size 1 \
    --iterations 100
```

### 选项C：完整测试（30-60分钟）

```bash
# 测试所有可用模型，完整迭代次数
python experiments/run_all_measurements.py \
    --models NiN ResNet18 AlexNet
```

## 📊 分析结果

### 生成图表

```bash
# 分析NiN模型的所有测量结果
python experiments/analyze_results.py --model NiN --type all

# 只分析计算开销
python experiments/analyze_results.py --model NiN --type computation

# 查看可用的数据文件
python experiments/analyze_results.py --list
```

**输出位置**：`experiments/figures/`

## ⚠️ 常见问题速查

### 问题1：Import错误

```bash
# 解决方法：确保在正确的目录
cd /root/exp_DNN_SGX/TAOISM
python experiments/quick_test.py
```

### 问题2：Enclave初始化失败

```bash
# 检查SGX状态
bash scripts/check_sgx2_edmm.sh

# 如果SGX不可用，先用CPU模式测试
python experiments/measurement/measure_computation.py \
    --single-model NiN --devices CPU
```

### 问题3：matplotlib/seaborn未安装

```bash
# 安装可视化库
pip install matplotlib seaborn
```

## 📝 下一步

### 1. 收集更多模型数据

```bash
# 添加VGG16（需要先调整配置）
# 编辑 Include/common_with_enclaves.h
# 修改 STORE_CHUNK_ELEM 为 802816
# 然后：make clean && make

python experiments/measurement/measure_computation.py \
    --single-model VGG16 --devices CPU --iterations 10
```

### 2. 测试不同批大小

```bash
python experiments/measurement/measure_computation.py \
    --single-model NiN \
    --devices CPU \
    --batch-sizes 1 4 8 16 \
    --iterations 50
```

### 3. 测试Enclave模式

```bash
# 确保SGX可用
python experiments/measurement/measure_security.py \
    --models NiN \
    --iterations 100
```

### 4. 数据分析

```python
# Python脚本示例
import json

# 加载数据
with open('experiments/data/computation_cost_NiN_aggregated.json') as f:
    data = json.load(f)

# 提取信息
cpu_time = data['devices']['CPU']['batch_1']['summary']['total_time_ms']
print(f"NiN CPU inference time: {cpu_time:.2f}ms")
```

## 🎓 论文使用建议

### 阶段1：数据收集（1-2周）

```bash
# 收集4个模型的完整数据
for model in NiN ResNet18 AlexNet VGG16; do
    python experiments/measurement/measure_computation.py \
        --single-model $model --devices CPU --iterations 100
    
    python experiments/measurement/measure_communication.py \
        --single-model $model --iterations 100
done
```

### 阶段2：数据分析（1周）

1. 加载所有JSON数据
2. 提取层级信息
3. 建立成本模型
4. 生成论文图表

### 阶段3：论文撰写（1-2周）

使用收集的数据：
- 表格：模型对比
- 图表：开销分布
- 曲线：带宽vs通信时间
- 柱状图：安全开销对比

## 💡 优化技巧

### 加速测试

```bash
# 减少迭代次数
--iterations 10  # 而不是100

# 只测试关键层（修改profiler代码）
# 或使用更小的模型（NiN）
```

### 节省存储

```bash
# 测试完成后压缩数据
cd experiments/data
tar -czf measurements_backup.tar.gz *.json
rm *.json  # 保留备份
```

### 批量处理

```bash
# 创建批处理脚本
cat > run_batch.sh << 'EOF'
#!/bin/bash
for model in NiN ResNet18; do
    echo "Testing $model..."
    python experiments/measurement/measure_computation.py \
        --single-model $model --devices CPU --iterations 10
done
EOF

chmod +x run_batch.sh
./run_batch.sh
```

## 📞 获取帮助

- 详细文档：`experiments/README.md`
- 实现总结：`experiments/IMPLEMENTATION_SUMMARY.md`
- 主项目文档：`README.md`

---

**快速命令参考**：

```bash
# 测试环境
python experiments/quick_test.py

# 单模型测试
python experiments/measurement/measure_computation.py --single-model NiN --devices CPU --iterations 10

# 批量测试
python experiments/run_all_measurements.py --quick-test

# 分析结果
python experiments/analyze_results.py --model NiN --type all

# 查看数据
ls experiments/data/
ls experiments/figures/
```

