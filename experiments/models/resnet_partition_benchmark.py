"""
ResNet-18 分布式推理性能基准测试

本脚本测试多种分割策略，展示在不同执行环境（Enclave/CPU）下的性能差异。

测试策略：
1. all_cpu: 所有层在 CPU（基线）
2. all_enclave: 所有层在 Enclave（除input层必须CPU）
3. pipeline_quarter: 前1/4在Enclave，后3/4在CPU
4. pipeline_half: 前1/2在Enclave，后1/2在CPU
5. pipeline_three_quarter: 前3/4在Enclave，后1/4在CPU
6. residual_split: 残差主路径在Enclave，跳跃连接在CPU
"""

import sys
sys.path.insert(0, '.')

from typing import Dict
from python.utils.basic_utils import ExecutionModeOptions
from experiments.models.distributed_resnet import run_distributed_inference


def get_all_layer_names():
    """生成 ResNet-18 的所有层名称"""
    names = ["input", "conv1", "relu", "maxpool"]
    
    # 4 个 layer group，每个 2 个 block
    for layer_idx in range(1, 5):
        for block_idx in range(2):
            prefix = f"layer{layer_idx}_block{block_idx}"
            names.extend([
                f"{prefix}_conv1",
                f"{prefix}_relu1",
                f"{prefix}_conv2",
                f"{prefix}_skip" if (layer_idx == 1 or block_idx == 1) else f"{prefix}_downsample",
                f"{prefix}_add",
                f"{prefix}_relu2",
            ])
    
    names.extend(["avgpool", "flatten", "fc", "output"])
    return names


def strategy_all_cpu() -> Dict[str, ExecutionModeOptions]:
    """所有层在 CPU"""
    return {name: ExecutionModeOptions.CPU for name in get_all_layer_names()}


def strategy_all_enclave() -> Dict[str, ExecutionModeOptions]:
    """所有层在 Enclave（除input层）"""
    overrides = {}
    overrides["input"] = ExecutionModeOptions.CPU  # 强制
    return overrides


def strategy_pipeline_quarter() -> Dict[str, ExecutionModeOptions]:
    """前 1/4 在 Enclave（stem + layer1），后 3/4 在 CPU"""
    all_names = get_all_layer_names()
    split_point = len(all_names) // 4
    
    overrides = {"input": ExecutionModeOptions.CPU}
    
    # Layer2, Layer3, Layer4 及之后在 CPU
    for layer_idx in range(2, 5):
        for block_idx in range(2):
            prefix = f"layer{layer_idx}_block{block_idx}"
            for suffix in ["conv1", "relu1", "conv2", "skip", "downsample", "add", "relu2"]:
                overrides[f"{prefix}_{suffix}"] = ExecutionModeOptions.CPU
    
    for name in ["avgpool", "flatten", "fc", "output"]:
        overrides[name] = ExecutionModeOptions.CPU
    
    return overrides


def strategy_pipeline_half() -> Dict[str, ExecutionModeOptions]:
    """前 1/2 在 Enclave（stem + layer1 + layer2），后 1/2 在 CPU"""
    overrides = {"input": ExecutionModeOptions.CPU}
    
    # Layer3, Layer4 及之后在 CPU
    for layer_idx in range(3, 5):
        for block_idx in range(2):
            prefix = f"layer{layer_idx}_block{block_idx}"
            for suffix in ["conv1", "relu1", "conv2", "skip", "downsample", "add", "relu2"]:
                overrides[f"{prefix}_{suffix}"] = ExecutionModeOptions.CPU
    
    for name in ["avgpool", "flatten", "fc", "output"]:
        overrides[name] = ExecutionModeOptions.CPU
    
    return overrides


def strategy_pipeline_three_quarter() -> Dict[str, ExecutionModeOptions]:
    """前 3/4 在 Enclave，后 1/4 在 CPU（仅 layer4 + classifier 在 CPU）"""
    overrides = {"input": ExecutionModeOptions.CPU}
    
    # 仅 Layer4 及之后在 CPU
    for block_idx in range(2):
        prefix = f"layer4_block{block_idx}"
        for suffix in ["conv1", "relu1", "conv2", "skip", "downsample", "add", "relu2"]:
            overrides[f"{prefix}_{suffix}"] = ExecutionModeOptions.CPU
    
    for name in ["avgpool", "flatten", "fc", "output"]:
        overrides[name] = ExecutionModeOptions.CPU
    
    return overrides


def strategy_residual_split() -> Dict[str, ExecutionModeOptions]:
    """
    残差分离策略：每个 block 的主路径在 Enclave，跳跃连接在 CPU
    这展示了细粒度的并行机会
    """
    overrides = {"input": ExecutionModeOptions.CPU}
    
    # Stem 在 Enclave
    # 每个 block：conv 在 Enclave，skip 和 add 在 CPU，最后的 relu 在 CPU
    for layer_idx in range(1, 5):
        for block_idx in range(2):
            prefix = f"layer{layer_idx}_block{block_idx}"
            # Skip/downsample 在 CPU
            overrides[f"{prefix}_skip"] = ExecutionModeOptions.CPU
            overrides[f"{prefix}_downsample"] = ExecutionModeOptions.CPU
            # Add 和最后的 ReLU 在 CPU
            overrides[f"{prefix}_add"] = ExecutionModeOptions.CPU
            overrides[f"{prefix}_relu2"] = ExecutionModeOptions.CPU
    
    # Classifier 在 CPU
    for name in ["avgpool", "flatten", "fc", "output"]:
        overrides[name] = ExecutionModeOptions.CPU
    
    return overrides


def strategy_alternating_layers() -> Dict[str, ExecutionModeOptions]:
    """交替策略：Layer1,3 在 Enclave，Layer2,4 在 CPU"""
    overrides = {"input": ExecutionModeOptions.CPU}
    
    # Layer2, Layer4 在 CPU
    for layer_idx in [2, 4]:
        for block_idx in range(2):
            prefix = f"layer{layer_idx}_block{block_idx}"
            for suffix in ["conv1", "relu1", "conv2", "skip", "downsample", "add", "relu2"]:
                overrides[f"{prefix}_{suffix}"] = ExecutionModeOptions.CPU
    
    # Classifier 在 CPU
    for name in ["avgpool", "flatten", "fc", "output"]:
        overrides[name] = ExecutionModeOptions.CPU
    
    return overrides


def main():
    """运行所有分割策略的性能测试"""
    
    strategies = [
        ("all_cpu", "所有层在CPU（基线）", strategy_all_cpu()),
        ("pipeline_quarter", "前1/4 Enclave + 后3/4 CPU", strategy_pipeline_quarter()),
        ("pipeline_half", "前1/2 Enclave + 后1/2 CPU", strategy_pipeline_half()),
        ("pipeline_three_quarter", "前3/4 Enclave + 后1/4 CPU", strategy_pipeline_three_quarter()),
        # ("residual_split", "主路径Enclave + 跳跃CPU", strategy_residual_split()),
        # ("alternating_layers", "Layer1,3 Enclave + Layer2,4 CPU", strategy_alternating_layers()),
    ]
    
    print("\n" + "="*80)
    print("ResNet-18 分布式推理性能基准测试")
    print("="*80)
    print(f"配置: 输入 64x64, Batch Size 1, 类别数 10")
    print(f"测试策略数: {len(strategies)}")
    print("="*80 + "\n")
    
    results = {}
    
    for strategy_name, description, overrides in strategies:
        print(f"\n{'#'*80}")
        print(f"# 策略: {strategy_name}")
        print(f"# 描述: {description}")
        print(f"{'#'*80}\n")
        
        try:
            result = run_distributed_inference(
                batch_size=1,
                input_size=64,
                num_classes=10,
                layer_mode_overrides=overrides,
            )
            results[strategy_name] = {
                "latency_ms": result["latency_ms"],
                "description": description,
                "success": True,
            }
        except Exception as e:
            print(f"❌ 策略 '{strategy_name}' 失败: {e}")
            import traceback
            traceback.print_exc()
            results[strategy_name] = {
                "latency_ms": None,
                "description": description,
                "success": False,
                "error": str(e),
            }
    
    # 打印汇总报告
    print("\n" + "="*80)
    print("性能对比报告")
    print("="*80)
    print(f"{'策略':<25} {'延迟 (ms)':<15} {'vs基线':<15} {'描述':<30}")
    print("-"*80)
    
    baseline_latency = results.get("all_cpu", {}).get("latency_ms")
    
    for strategy_name, data in results.items():
        if data["success"] and data["latency_ms"]:
            latency = data["latency_ms"]
            if baseline_latency and baseline_latency > 0:
                speedup = baseline_latency / latency
                vs_baseline = f"{speedup:.2f}x"
            else:
                vs_baseline = "N/A"
            print(f"{strategy_name:<25} {latency:>12.3f}    {vs_baseline:<15} {data['description']:<30}")
        else:
            print(f"{strategy_name:<25} {'FAILED':<15} {'N/A':<15} {data['description']:<30}")
    
    print("="*80)
    
    # 分析报告
    print("\n" + "="*80)
    print("分析总结")
    print("="*80)
    
    if baseline_latency:
        print(f"✓ CPU 基线延迟: {baseline_latency:.3f} ms")
        
        # 找出最快的策略
        best_strategy = None
        best_latency = float('inf')
        for name, data in results.items():
            if data["success"] and data["latency_ms"] and data["latency_ms"] < best_latency:
                best_latency = data["latency_ms"]
                best_strategy = name
        
        if best_strategy and best_strategy != "all_cpu":
            speedup = baseline_latency / best_latency
            improvement = (baseline_latency - best_latency) / baseline_latency * 100
            print(f"✓ 最佳策略: {best_strategy}")
            print(f"  - 延迟: {best_latency:.3f} ms")
            print(f"  - 加速比: {speedup:.2f}x")
            print(f"  - 改进: {improvement:.1f}%")
        elif best_strategy == "all_cpu":
            print("⚠ 所有分布式策略均未超过 CPU 基线")
            print("  可能原因：")
            print("  1. 模型规模太小，通信开销超过并行收益")
            print("  2. SGX 初始化/数据传输成本高")
            print("  3. 当前硬件配置下 CPU 执行更快")
    
    print("\n观察：")
    print("- 并行效果取决于模型规模、分割点、硬件性能")
    print("- ResNet 的残差结构为细粒度并行提供了机会")
    print("- 实际应用中需根据安全需求和性能权衡选择分割策略")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

