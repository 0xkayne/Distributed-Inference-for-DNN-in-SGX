# 🚀 从这里开始！

## 欢迎使用 TAOISM 实验框架

本框架为您的毕业论文提供完整的DNN模型测量工具。

---

## ⚡ 1分钟快速开始

### 设置环境并测试

```bash
# 进入目录
cd /root/exp_DNN_SGX/TAOISM

# 激活环境
conda activate taoism
source /opt/intel/sgxsdk/environment
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH

# 运行快速测试
python experiments/quick_test.py
```

**预期结果**：所有测试显示 ✓ PASS

---

## 📊 立即可用的功能

### ✅ 功能1：通信开销测量（已验证可用）

```bash
# 测量NiN模型的通信开销
python experiments/measurement/measure_communication.py \
    --single-model NiN \
    --bandwidths 10 100 1000 \
    --iterations 100
```

**输出**：
- `experiments/data/communication_cost_NiN.json`
- 包含层间数据传输开销

**用途**：
- 建立通信成本模型
- 分析带宽-延迟关系
- 论文图表数据

### ✅ 功能2：使用现有baseline（推荐用于计算/安全测量）

```bash
# ResNet CPU模式
cd teeslice
python -m sgx_resnet_cifar --arch resnet18 --mode CPU --batch_size 1 --num_repeat 100

# ResNet Enclave模式
python -m sgx_resnet_cifar --arch resnet18 --mode Enclave --batch_size 1 --num_repeat 100
```

**用途**：
- 获取计算开销数据
- 获取安全开销数据（CPU vs Enclave）
- 验证模型正确性

---

## 📋 完整文件导航

### 必读文档（按优先级）

1. **START_HERE.md** (本文档) - 立即开始
2. **STATUS_REPORT.md** - 当前状态说明
3. **TESTING_COMPLETE.md** - 测试结果
4. **QUICK_START.md** - 详细快速指南
5. **README.md** - 完整使用文档

### 参考文档

6. **USAGE_EXAMPLES.md** - 33个使用示例
7. **IMPLEMENTATION_SUMMARY.md** - 技术实现
8. **FILE_LIST.txt** - 文件清单

---

## 🎯 推荐工作流程

### 第1天：熟悉环境

```bash
# 1. 测试环境
python experiments/quick_test.py

# 2. 测试单个模型通信
python experiments/measurement/measure_communication.py --single-model NiN

# 3. 查看生成的数据
ls experiments/data/
cat experiments/data/communication_cost_NiN.json | python -m json.tool | head -30
```

### 第2-3天：收集通信数据

```bash
# 收集所有模型的通信开销数据
python experiments/measurement/measure_communication.py \
    --models NiN ResNet18 AlexNet \
    --bandwidths 10 100 1000 \
    --iterations 100
```

### 第4-5天：收集计算数据

```bash
# 使用teeslice baseline
cd teeslice

# ResNet18
python -m sgx_resnet_cifar --arch resnet18 --mode CPU --num_repeat 100 > resnet18_cpu.log
python -m sgx_resnet_cifar --arch resnet18 --mode Enclave --num_repeat 100 > resnet18_enclave.log

# ResNet50
python -m sgx_resnet_cifar --arch resnet50 --mode CPU --num_repeat 100 > resnet50_cpu.log
```

### 第2周：数据分析

```bash
# 分析通信数据
python experiments/analyze_results.py --model NiN --type communication

# 整合teeslice数据到experiments/data/
# 建立成本模型
```

---

## 📝 已验证工作的功能

### ✅ 核心功能

1. **环境配置** ✅
   - conda环境激活
   - SGX SDK加载
   - 库路径设置

2. **模型创建** ✅
   - 6个模型都可以创建
   - NiN已测试：24层

3. **通信测量** ✅
   - 数据已生成
   - 结果准确
   - JSON格式正确

4. **数据管理** ✅
   - 保存/加载正常
   - 文件组织清晰

### ⚠️ 需要适配的功能

5. **层级性能测量**
   - 代码已实现
   - 需要改为端到端方式
   - 建议：参考teeslice实现

6. **批量测试**
   - 框架已完成
   - 需要端到end支持
   - 建议：分步骤测试

---

## 🛠️ 故障排查

### 问题：Import错误

**解决**：
```bash
cd /root/exp_DNN_SGX/TAOISM
conda activate taoism
```

### 问题：libstdc++版本

**解决**：
```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH
```

### 问题：层无法forward

**原因**：层需要在网络中运行

**解决**：使用端到端测量或现有baseline

---

## 💡 推荐方案

### 对于论文实验数据收集

**推荐组合**：

1. **通信数据** → 使用 `experiments/measurement/measure_communication.py` ✅
2. **计算数据** → 使用 `teeslice/sgx_resnet_cifar.py` ✅
3. **安全数据** → 对比CPU和Enclave模式 ✅
4. **换页数据** → 分析不同内存配置 ⏸️

**原因**：
- 充分利用TAOISM现有功能
- 避免重复开发
- 数据质量有保证

---

## 📚 学习路径

1. **Day 1**：阅读 START_HERE.md（本文档）
2. **Day 2**：运行 quick_test.py 和通信测量
3. **Day 3**：学习 teeslice baseline 使用
4. **Day 4-5**：收集实验数据
5. **Week 2**：数据分析和成本建模
6. **Week 3+**：实现阶段二和三

---

## 🎁 已交付内容

- ✅ 6个DNN模型实现
- ✅ 4类测量脚本
- ✅ 完整工具链
- ✅ 8份详细文档
- ✅ 约6,000行代码

---

## 🎯 下一步

### 立即可做

```bash
# 收集NiN的通信数据
python experiments/measurement/measure_communication.py --single-model NiN

# 收集ResNet的计算数据
cd teeslice && python -m sgx_resnet_cifar --arch resnet18 --mode CPU --num_repeat 10
```

### 本周计划

- Day 1-2：收集通信数据（所有模型）
- Day 3-4：使用baseline收集计算数据
- Day 5：整理和分析数据

### 下周计划

- 开发端到端测量脚本
- 收集完整实验数据
- 建立成本模型

---

## 📞 获取帮助

**文档**：
- 状态说明：`STATUS_REPORT.md`
- 测试结果：`TESTING_COMPLETE.md`
- 使用示例：`USAGE_EXAMPLES.md`

**命令参考**：
```bash
# 环境设置
source experiments/setup_env.sh

# 快速测试
python experiments/quick_test.py

# 查看数据
ls experiments/data/
```

---

**祝您实验顺利！** 🚀

如有问题，请查阅相关文档或检查TEST_RESULTS.md中的故障排查部分。

