"""
EDMM DNN Inference Example — 在 TAOISM 框架下使用 SGX2 EDMM 特性进行 DNN 推理

本示例展示如何在启用 EDMM 的 SGX Enclave 中逐层执行 ResNet-18 前几层推理，
并通过 Enclave 内置 profiling 接口观测每层的计算/存取耗时。

EDMM 的核心价值：
  - 堆从 HeapMinSize (64KB) 按需增长到 HeapMaxSize (2GB)，由内核 EAUG 注入 EPC 页
  - 每个 DNN 层分配张量时触发 EAUG → EACCEPT，后续访问无额外开销
  - 推理结束后未使用的 EPC 页可被回收，降低常驻内存压力

运行方式 (从项目根目录):
    source /opt/intel/sgxsdk/environment
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SGX_SDK/lib64:$LD_LIBRARY_PATH
    python -m experiments.models.edmm_inference_demo

前置条件:
    1. make clean && make SGX_MODE=HW all  (使用更新后的 Enclave.config.xml)
    2. Enclave.config.xml 中包含:
       - <UserRegionSize>0x100000</UserRegionSize>
       - <HeapMinSize>0x10000</HeapMinSize>
       - <MiscMask>0xFFFFFFFF</MiscMask>
"""

import sys
import os
import time

sys.path.insert(0, ".")

import torch
import numpy as np

from python.enclave_interfaces import GlobalTensor
from python.layers.input import SecretInputLayer
from python.layers.sgx_conv_base import SGXConvBase
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.relu import SecretReLULayer
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.layers.batch_norm_2d import SecretBatchNorm2dLayer
from python.utils.basic_utils import ExecutionModeOptions


# ─── 工具函数 ─────────────────────────────────────────────────────

def fmt_shape(shape):
    return "x".join(str(s) for s in shape)


def fmt_bytes(n):
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def tensor_bytes(shape):
    return int(np.prod(shape)) * 4  # float32 = 4 bytes


def setup_layers(layers):
    """三阶段初始化: set_eid → init_shape/link → init"""
    eid = GlobalTensor.get_eid()
    for layer in layers:
        layer.set_eid(eid)
        layer.init_shape()
        layer.link_tensors()
    for layer in layers:
        layer.init(start_enclave=False)


def profile_layer_forward(layer, input_layer, input_tensor, warmup=1, repeats=3):
    """
    对单层执行 forward 并收集 profiling 数据。

    返回:
        dict: {python_ms, compute_ms, get_ms, store_ms}
    """
    times_python = []
    times_compute = []
    times_get = []
    times_store = []

    for i in range(warmup + repeats):
        input_layer.set_input(input_tensor)

        t0 = time.perf_counter()
        layer.forward()
        t1 = time.perf_counter()

        if i >= warmup:
            times_python.append((t1 - t0) * 1000)
            try:
                stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(
                    layer.LayerName
                )
                times_compute.append(stats["compute_ms"])
                times_get.append(stats["get_ms"])
                times_store.append(stats["store_ms"])
            except Exception:
                pass

    result = {"python_ms": np.mean(times_python) if times_python else 0.0}
    if times_compute:
        result["compute_ms"] = np.mean(times_compute)
        result["get_ms"] = np.mean(times_get)
        result["store_ms"] = np.mean(times_store)
    return result


# ─── 模型定义: ResNet-18 前端 (conv1 → bn1 → relu → maxpool) ──────

def build_resnet18_stem(sid=0):
    """
    构建 ResNet-18 的 stem 部分:
        Input(1,3,224,224) → Conv(64,7,s2,p3) → BN → ReLU → MaxPool(3,s2,p1)
    以及第一个残差块的第一个卷积:
        → Conv(64,3,s1,p1) → BN → ReLU

    返回 (layers_list, layer_info_list)
    """
    input_shape = [1, 3, 224, 224]
    mode = ExecutionModeOptions.Enclave

    # ── 各层定义 ──
    input_layer = SecretInputLayer(sid, "input", input_shape, mode)

    conv1 = SGXConvBase(
        sid, "conv1", mode,
        n_output_channel=64, filter_hw=7, stride=2, padding=3,
        batch_size=1, n_input_channel=3, img_hw=224,
    )
    conv1.register_prev_layer(input_layer)

    bn1 = SecretBatchNorm2dLayer(sid, "bn1", mode)
    bn1.register_prev_layer(conv1)

    relu1 = SecretReLULayer(sid, "relu1", mode)
    relu1.register_prev_layer(bn1)

    # MaxPool 在 TAOISM 中通常运行在 CPU 上
    maxpool = SecretMaxpool2dLayer(
        sid, "maxpool", ExecutionModeOptions.CPU, filter_hw=3, stride=2, padding=1
    )
    maxpool.register_prev_layer(relu1)

    # 第一个残差块的 conv
    conv2 = SGXConvBase(
        sid, "layer1_conv1", mode,
        n_output_channel=64, filter_hw=3, stride=1, padding=1,
        batch_size=1, n_input_channel=64, img_hw=56,
    )
    conv2.register_prev_layer(maxpool)

    bn2 = SecretBatchNorm2dLayer(sid, "layer1_bn1", mode)
    bn2.register_prev_layer(conv2)

    relu2 = SecretReLULayer(sid, "layer1_relu1", mode)
    relu2.register_prev_layer(bn2)

    layers = [input_layer, conv1, bn1, relu1, maxpool, conv2, bn2, relu2]

    info = [
        ("input",        "Input",     input_shape,        input_shape),
        ("conv1",        "Conv2d",    input_shape,        [1, 64, 112, 112]),
        ("bn1",          "BatchNorm", [1, 64, 112, 112],  [1, 64, 112, 112]),
        ("relu1",        "ReLU",      [1, 64, 112, 112],  [1, 64, 112, 112]),
        ("maxpool",      "MaxPool",   [1, 64, 112, 112],  [1, 64, 56, 56]),
        ("layer1_conv1", "Conv2d",    [1, 64, 56, 56],    [1, 64, 56, 56]),
        ("layer1_bn1",   "BatchNorm", [1, 64, 56, 56],    [1, 64, 56, 56]),
        ("layer1_relu1", "ReLU",      [1, 64, 56, 56],    [1, 64, 56, 56]),
    ]

    return layers, info


# ─── 主流程 ───────────────────────────────────────────────────────

def main():
    print()
    print("=" * 70)
    print("  TAOISM EDMM DNN 推理示例")
    print("  ResNet-18 Stem + Layer1.Block0 — SGX2 EDMM 动态内存按需分配")
    print("=" * 70)

    # ── 第一阶段: 读取 Enclave 配置并打印 EDMM 状态 ──
    print()
    print("[1/4] 读取 Enclave 配置 (Enclave/Enclave.config.xml)")
    print("-" * 50)

    config_path = "Enclave/Enclave.config.xml"
    edmm_config = {"UserRegionSize": "未设置", "HeapMinSize": "未设置",
                   "HeapMaxSize": "未设置", "MiscMask": "未设置"}
    if os.path.exists(config_path):
        with open(config_path) as f:
            content = f.read()
        import re
        for key in edmm_config:
            m = re.search(rf"<{key}>(.*?)</{key}>", content)
            if m:
                edmm_config[key] = m.group(1)

    heap_min = int(edmm_config["HeapMinSize"], 16) if edmm_config["HeapMinSize"] != "未设置" else 0
    heap_max = int(edmm_config["HeapMaxSize"], 16) if edmm_config["HeapMaxSize"] != "未设置" else 0
    user_region = int(edmm_config["UserRegionSize"], 16) if edmm_config["UserRegionSize"] != "未设置" else 0

    print(f"  HeapMinSize    = {edmm_config['HeapMinSize']:>14}  ({fmt_bytes(heap_min)}，EDMM 初始提交)")
    print(f"  HeapMaxSize    = {edmm_config['HeapMaxSize']:>14}  ({fmt_bytes(heap_max)}，按需增长上限)")
    print(f"  UserRegionSize = {edmm_config['UserRegionSize']:>14}  ({fmt_bytes(user_region)}，sgx_mm_alloc 专用)")
    print(f"  MiscMask       = {edmm_config['MiscMask']:>14}  (bit0=1 → 强制 EXINFO)")

    if user_region == 0:
        print()
        print("  !! 警告: UserRegionSize 未设置，sgx_mm_alloc() 将不可用")
        print("     请参考 docs/EDMM_FIX_RECORD.md 修改配置后重新编译")

    # ── 第二阶段: 初始化 Enclave ──
    print()
    print("[2/4] 初始化 SGX Enclave")
    print("-" * 50)

    t_init_start = time.perf_counter()
    GlobalTensor.init()
    t_init_end = time.perf_counter()

    print(f"  Enclave 初始化完成: {(t_init_end - t_init_start) * 1000:.1f} ms")
    print(f"  EDMM 模式下堆从 {fmt_bytes(heap_min)} 起步，随张量分配自动增长")

    # ── 第三阶段: 构建模型并逐层推理 ──
    print()
    print("[3/4] 构建 ResNet-18 Stem + Layer1 并逐层推理")
    print("-" * 50)

    layers, layer_info = build_resnet18_stem()

    t_setup_start = time.perf_counter()
    setup_layers(layers)
    t_setup_end = time.perf_counter()
    print(f"  模型构建 + 权重传入 Enclave: {(t_setup_end - t_setup_start) * 1000:.1f} ms")

    # 计算各层触发的内存分配量 (估算)
    cumulative_bytes = 0

    print()
    print(f"  {'层名':16} {'类型':10} {'输入形状':18} {'输出形状':18} {'分配量':>10} {'累计':>10} {'推理(ms)':>10}")
    print(f"  {'─' * 16} {'─' * 10} {'─' * 18} {'─' * 18} {'─' * 10} {'─' * 10} {'─' * 10}")

    input_tensor = torch.randn(1, 3, 224, 224)
    input_layer = layers[0]

    results = []

    for i, (name, ltype, in_shape, out_shape) in enumerate(layer_info):
        layer = layers[i]

        # 估算该层需要的 Enclave 内存 (input + output tensor)
        layer_alloc = tensor_bytes(in_shape) + tensor_bytes(out_shape)
        if ltype in ("Conv2d",):
            # 加上权重和偏置
            if hasattr(layer, "sgx_w_shape") and layer.sgx_w_shape:
                layer_alloc += tensor_bytes(layer.sgx_w_shape)
            if hasattr(layer, "bias_shape") and layer.bias_shape:
                layer_alloc += tensor_bytes(layer.bias_shape)
        if ltype == "BatchNorm":
            # gamma, beta, running_mean, running_var
            layer_alloc += tensor_bytes([in_shape[1]]) * 4

        cumulative_bytes += layer_alloc

        # 推理计时
        if name == "input":
            # Input 层不执行计算
            results.append({
                "name": name, "type": ltype,
                "in_shape": in_shape, "out_shape": out_shape,
                "alloc": layer_alloc, "cumulative": cumulative_bytes,
                "python_ms": 0.0,
            })
            continue

        stats = profile_layer_forward(layer, input_layer, input_tensor, warmup=1, repeats=3)

        results.append({
            "name": name, "type": ltype,
            "in_shape": in_shape, "out_shape": out_shape,
            "alloc": layer_alloc, "cumulative": cumulative_bytes,
            **stats,
        })

    # 打印结果表
    for r in results:
        python_ms = r.get("python_ms", 0.0)
        ms_str = f"{python_ms:.2f}" if python_ms > 0 else "—"
        print(
            f"  {r['name']:16} {r['type']:10} {fmt_shape(r['in_shape']):18} "
            f"{fmt_shape(r['out_shape']):18} {fmt_bytes(r['alloc']):>10} "
            f"{fmt_bytes(r['cumulative']):>10} {ms_str:>10}"
        )

    # ── 第四阶段: 汇总 ──
    print()
    print("[4/4] EDMM 推理汇总")
    print("-" * 50)

    total_python = sum(r.get("python_ms", 0) for r in results)
    total_compute = sum(r.get("compute_ms", 0) for r in results)

    print(f"  总层数          : {len(results) - 1} (不含 input)")
    print(f"  Python 端总耗时 : {total_python:.2f} ms")
    if total_compute > 0:
        print(f"  Enclave 计算耗时: {total_compute:.2f} ms")
    print(f"  Enclave 内存估算: {fmt_bytes(cumulative_bytes)} (张量 + 权重)")
    print()
    print("  EDMM 内存增长路径:")
    print(f"    Enclave 初始化时  → {fmt_bytes(heap_min)} (HeapMinSize)")
    print(f"    模型加载后       → ~{fmt_bytes(cumulative_bytes)} (按需 EAUG)")
    print(f"    可用上限          → {fmt_bytes(heap_max)} (HeapMaxSize)")
    print()
    print("  与传统 SGX1 模式对比:")
    print(f"    SGX1: 启动即提交 {fmt_bytes(heap_max)}，无论是否使用")
    print(f"    SGX2 EDMM: 从 {fmt_bytes(heap_min)} 起步，仅提交实际使用的页")
    print(f"    节省初始 EPC 占用: ~{fmt_bytes(heap_max - cumulative_bytes)}")
    print()

    # 清理
    GlobalTensor.destroy()
    print("  Enclave 已销毁。")
    print()


if __name__ == "__main__":
    main()
