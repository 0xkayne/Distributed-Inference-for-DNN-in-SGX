#!/bin/bash
#
# TAOISM Experiments - Demo Script
# 演示如何使用实验框架
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║     TAOISM 毕业论文实验框架 - 演示脚本                    ║"
echo "║     Phase 1: 理论建模与基础测量                           ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

# 检查是否在正确的目录
if [ ! -d "experiments" ]; then
    echo -e "${RED}错误: 请在TAOISM根目录运行此脚本${NC}"
    echo "cd /root/exp_DNN_SGX/TAOISM"
    exit 1
fi

echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  步骤 1/4: 快速测试 (验证环境)${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}\n"

python experiments/quick_test.py

if [ $? -ne 0 ]; then
    echo -e "\n${RED}✗ 快速测试失败，请检查环境配置${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ 环境验证成功！${NC}"
read -p "按Enter继续..." dummy

echo -e "\n${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  步骤 2/4: 单模型测试 (NiN, CPU模式)${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}\n"

echo -e "${YELLOW}运行命令:${NC}"
echo "python experiments/measurement/measure_computation.py \\"
echo "    --single-model NiN --devices CPU --iterations 10"
echo ""

python experiments/measurement/measure_computation.py \
    --single-model NiN --devices CPU --iterations 10

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ 单模型测试成功！${NC}"
    echo -e "${GREEN}  数据已保存到: experiments/data/${NC}"
    ls -lh experiments/data/*.json 2>/dev/null | tail -3
else
    echo -e "\n${RED}✗ 单模型测试失败${NC}"
fi

read -p "按Enter继续..." dummy

echo -e "\n${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  步骤 3/4: 通信开销测试${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}\n"

echo -e "${YELLOW}运行命令:${NC}"
echo "python experiments/measurement/measure_communication.py \\"
echo "    --single-model NiN --bandwidths 100 --iterations 10"
echo ""

python experiments/measurement/measure_communication.py \
    --single-model NiN --bandwidths 100 --iterations 10

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ 通信测试成功！${NC}"
else
    echo -e "\n${YELLOW}⚠ 通信测试完成（可能有警告）${NC}"
fi

read -p "按Enter继续..." dummy

echo -e "\n${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  步骤 4/4: 查看生成的数据${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}\n"

echo -e "${YELLOW}生成的数据文件:${NC}"
ls -lh experiments/data/*.json 2>/dev/null || echo "  (暂无数据文件)"

echo ""
echo -e "${YELLOW}示例：查看NiN计算开销数据${NC}"
if [ -f "experiments/data/computation_cost_NiN_CPU.json" ]; then
    echo "前20行内容："
    head -20 experiments/data/computation_cost_NiN_CPU.json
else
    echo "  数据文件尚未生成"
fi

echo ""
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                      演示完成！                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${GREEN}✓ 所有演示步骤已完成${NC}\n"

echo "下一步建议："
echo ""
echo "1. 运行批量测试:"
echo -e "   ${CYAN}python experiments/run_all_measurements.py --quick-test${NC}"
echo ""
echo "2. 测试所有6个模型:"
echo -e "   ${CYAN}python experiments/run_all_measurements.py --models all${NC}"
echo ""
echo "3. 分析结果生成图表:"
echo -e "   ${CYAN}python experiments/analyze_results.py --model NiN --type all${NC}"
echo ""
echo "4. 查看详细文档:"
echo -e "   ${CYAN}cat experiments/README.md${NC}"
echo -e "   ${CYAN}cat experiments/QUICK_START.md${NC}"
echo ""

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  完整文件列表${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

find experiments -type f \( -name "*.py" -o -name "*.md" \) | sort | sed 's/^/  /'

echo ""
echo -e "${GREEN}实验框架已就绪，祝您科研顺利！${NC}"
echo ""

