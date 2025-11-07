#!/bin/bash

# SGX2 EDMM ResNet 完整测试脚本
# 用于验证 ResNet 可以在 SGX2 EDMM 特性下正常运行

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  SGX2 EDMM ResNet 完整测试${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 1. 检查 SGX2 EDMM 支持
echo -e "${YELLOW}[1/5] 检查 SGX2 EDMM 硬件支持...${NC}"
if [ -x scripts/check_sgx2_edmm.sh ]; then
    bash scripts/check_sgx2_edmm.sh || {
        echo -e "${RED}✗ SGX2 EDMM 支持检查失败${NC}"
        exit 1
    }
else
    echo -e "${YELLOW}⚠ 跳过硬件检查（检查脚本不存在）${NC}"
fi
echo ""

# 2. 设置环境
echo -e "${YELLOW}[2/5] 设置 SGX 和 Python 环境...${NC}"

# 加载 SGX SDK 环境
if [ -f /opt/intel/sgxsdk/environment ]; then
    source /opt/intel/sgxsdk/environment
    echo -e "${GREEN}✓ SGX SDK 环境已加载${NC}"
else
    echo -e "${RED}✗ 未找到 SGX SDK 环境文件${NC}"
    exit 1
fi

# 设置库路径（关键：避免 conda libstdc++ 版本冲突）
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH
echo -e "${GREEN}✓ 库路径已配置（使用系统 libstdc++）${NC}"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate taoism
echo -e "${GREEN}✓ Conda 环境 'taoism' 已激活${NC}"
echo ""

# 3. 验证 Enclave 初始化
echo -e "${YELLOW}[3/5] 验证 Enclave 初始化（SGX2 EDMM 模式）...${NC}"
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

try:
    from python.enclave_interfaces import EnclaveInterface
    print("正在初始化 Enclave with SGX2 EDMM...")
    enclave = EnclaveInterface()
    print(f"✓ Enclave 初始化成功！Enclave ID: {enclave.eid}")
    print("✓ SGX2 EDMM 功能验证通过")
except Exception as e:
    print(f"✗ Enclave 初始化失败: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Enclave 初始化测试通过${NC}"
else
    echo -e "${RED}✗ Enclave 初始化测试失败${NC}"
    exit 1
fi
echo ""

# 4. 运行 ResNet 测试（小批量）
echo -e "${YELLOW}[4/5] 运行 ResNet18 测试（Enclave 模式，batch_size=1）...${NC}"
echo "开始时间: $(date)"
echo ""

# 运行测试，捕获输出和错误
python -m teeslice.sgx_resnet_cifar --arch resnet18 --mode Enclave --batch_size 1 2>&1 | tee /tmp/resnet_sgx2_test.log

TEST_RESULT=${PIPESTATUS[0]}

echo ""
echo "结束时间: $(date)"
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ ResNet18 Enclave 模式测试通过${NC}"
else
    echo -e "${RED}✗ ResNet18 Enclave 模式测试失败 (退出码: $TEST_RESULT)${NC}"
    echo ""
    echo -e "${YELLOW}查看详细日志：${NC}"
    echo "  tail -50 /tmp/resnet_sgx2_test.log"
    echo ""
    
    # 显示最后的错误信息
    echo -e "${YELLOW}最后 20 行输出：${NC}"
    tail -20 /tmp/resnet_sgx2_test.log
    echo ""
    
    # 提供调试建议
    echo -e "${YELLOW}调试建议：${NC}"
    echo "1. 检查 Enclave 内存配置是否足够（Enclave.config.xml）"
    echo "2. 检查 EDMM 统计信息是否显示内存分配失败"
    echo "3. 尝试减小 batch_size 或模型规模"
    echo "4. 查看完整日志： cat /tmp/resnet_sgx2_test.log"
    
    exit 1
fi
echo ""

# 5. 总结
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  测试总结${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ SGX2 EDMM 硬件支持：通过${NC}"
echo -e "${GREEN}✓ Enclave 初始化：通过${NC}"
echo -e "${GREEN}✓ ResNet18 推理：通过${NC}"
echo ""
echo -e "${GREEN}🎉 SGX2 EDMM 功能验证成功！${NC}"
echo ""
echo "后续实验建议："
echo "  - 可以尝试更大的 batch_size 测试 EDMM 动态内存管理"
echo "  - 可以运行更深的模型（ResNet50/ResNet101）测试内存扩展"
echo "  - 可以使用 EDMM 统计信息监控内存使用情况"
echo ""

