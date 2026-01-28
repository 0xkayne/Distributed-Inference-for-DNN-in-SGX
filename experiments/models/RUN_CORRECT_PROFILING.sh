#!/bin/bash
#
# 重新运行 BERT Profiling 以生成正确的依赖关系
#
# 本脚本将：
# 1. 备份旧的（错误依赖）profiling 结果
# 2. 重新运行 per-head profiling（使用修复后的代码）
# 3. 验证新的依赖关系是否正确
# 4. 对比新旧结果
#

set -e

echo "========================================================================"
echo "重新运行 BERT Per-Head Profiling（修复依赖关系）"
echo "========================================================================"
echo ""
echo "目的：生成正确的层间依赖关系"
echo "关键修复：多头注意力的头之间应该是并行的，而不是串行"
echo ""
echo "========================================================================"

# 检查环境
if [ ! -f "App/bin/enclave_bridge.so" ]; then
    echo "⚠ Enclave library not found. Run 'make' first."
    echo "Continuing anyway..."
fi

# 设置 LD_PRELOAD
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

echo ""
echo "步骤 1: 备份旧的 profiling 结果"
echo "-------------------------------------------"

if [ -f "experiments/data/bert_base_enclave_per_head_layers.csv" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mv experiments/data/bert_base_enclave_per_head_layers.csv \
       experiments/data/bert_base_enclave_per_head_layers_old_${TIMESTAMP}.csv
    echo "✓ 已备份旧文件（错误依赖）"
    echo "  → experiments/data/bert_base_enclave_per_head_layers_old_${TIMESTAMP}.csv"
else
    echo "  (无旧文件需要备份)"
fi

if [ -f "experiments/data/bert_base_enclave_per_head_layers.json" ]; then
    mv experiments/data/bert_base_enclave_per_head_layers.json \
       experiments/data/bert_base_enclave_per_head_layers_old_${TIMESTAMP}.json
fi

echo ""
echo "步骤 2: 重新运行 Per-Head Profiling（使用修复后的代码）"
echo "-------------------------------------------"
echo "模型：BERT-base (12 layers, 12 heads)"
echo "序列长度：128"
echo "迭代次数：10"
echo "模式：Per-Head (fine-grained)"
echo ""
echo "预计时间：30-60 分钟（BERT-base 很大）"
echo "如果时间紧张，可以先用 BERT-mini 测试（约 5-10 分钟）："
echo "  python -m experiments.models.profile_bert_enclave --model mini --per-head"
echo ""
read -p "继续运行 BERT-base 吗？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消。建议先运行 BERT-mini 快速验证："
    echo "  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \\"
    echo "  python -m experiments.models.profile_bert_enclave --model mini --per-head"
    exit 0
fi

echo "开始 profiling..."
python -m experiments.models.profile_bert_enclave \
    --model base \
    --seq-len 128 \
    --iterations 10 \
    --warmup 3 \
    --per-head

echo ""
echo "✓ Profiling 完成"
echo ""

echo "步骤 3: 验证新的依赖关系"
echo "-------------------------------------------"
python experiments/models/verify_dependencies.py

echo ""
echo "步骤 4: 对比新旧依赖关系（样本）"
echo "-------------------------------------------"

if [ -f "experiments/data/bert_base_enclave_per_head_layers_old_${TIMESTAMP}.csv" ]; then
    echo ""
    echo "旧依赖（错误）："
    grep 'encoder0_attn_head0_qk_matmul' \
        experiments/data/bert_base_enclave_per_head_layers_old_${TIMESTAMP}.csv | \
        cut -d',' -f1,19
    
    echo ""
    echo "新依赖（正确）："
    grep 'encoder0_attn_head0_qk_matmul' \
        experiments/data/bert_base_enclave_per_head_layers.csv | \
        cut -d',' -f1,19
    
    echo ""
    echo "旧依赖 head1_qk_matmul（错误 - 依赖 head0）："
    grep 'encoder0_attn_head1_qk_matmul' \
        experiments/data/bert_base_enclave_per_head_layers_old_${TIMESTAMP}.csv | \
        cut -d',' -f1,19
    
    echo ""
    echo "新依赖 head1_qk_matmul（正确 - 并行）："
    grep 'encoder0_attn_head1_qk_matmul' \
        experiments/data/bert_base_enclave_per_head_layers.csv | \
        cut -d',' -f1,19
fi

echo ""
echo "========================================================================"
echo "完成！"
echo "========================================================================"
echo ""
echo "新的 profiling 结果："
echo "  - experiments/data/bert_base_enclave_per_head_layers.csv"
echo "  - experiments/data/bert_base_enclave_per_head_layers.json"
echo ""
echo "关键改进："
echo "  ✓ 多头注意力的头之间并行（无依赖）"
echo "  ✓ Q/K/V 投影并行"
echo "  ✓ Out projection 依赖所有头"
echo "  ✓ 正确反映 Transformer 架构"
echo ""
echo "下一步："
echo "  - 使用新的依赖关系进行调度优化"
echo "  - 分析并行度和关键路径"
echo "  - 设计 CPU/Enclave 分区策略"
echo ""
echo "========================================================================"
