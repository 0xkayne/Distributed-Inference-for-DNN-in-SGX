"""
并行推理示例网络，使用真实的卷积/池化/激活算子构成一个极简 CNN。

结构概要：
    LayerA (输入) -> LayerA_Pool (2x2 最大池化, stride=1)
    LayerB (3x3 卷积) -> LayerB_ReLU -> LayerC (3x3 卷积)
    LayerB 同时馈送 LayerD（CPU 分支），LayerC 与 LayerD 在 LayerE 处相加
    LayerE -> LayerF (输出占位层)
"""

import importlib
import sys
import time
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Set

sys.path.insert(0, '.')

from python.enclave_interfaces import GlobalTensor
from python.layers.add import SecretAddLayer
from python.layers.identity import SecretIdentityLayer
from python.layers.input import SecretInputLayer
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.layers.output import SecretOutputLayer
from python.layers.relu import SecretReLULayer
from python.layers.sgx_conv_base import SGXConvBase
from python.sgx_net import SecretNeuralNetwork
from python.utils.basic_utils import ExecutionModeOptions


class SecretParallelToyNet:
    """
    Minimal six-layer network with an explicit parallel branch between B and E.

    The concrete operations of each layer are intentionally simple; the purpose
    is to exercise the connection pattern inside the SGX layer framework.

    Args:
        layer_mode_overrides: Optional mapping from layer name to a forced
            ExecutionModeOptions. Useful for pinning specific layers to CPU
            while the rest of the network runs in enclave mode.
        active_layers: Optional iterable of layer names that should be included
            in the constructed network. When provided, every selected layer must
            have all of its predecessors included as well.
    """

    def __init__(
        self,
        sid=0,
        enclave_mode=ExecutionModeOptions.CPU,
        batch_size=1,
        channels=3,
        height=4,
        width=4,
        *,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
        active_layers: Optional[Sequence[str]] = None,
    ):
        self.sid = sid
        self.enclave_mode = enclave_mode
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        if height != width:
            raise ValueError("ParallelToyNet 目前仅支持方形输入，以便使用对称池化。")
        if height < 2:
            raise ValueError("ParallelToyNet 需要至少 2x2 的输入特征图以执行池化。")
        self.input_shape = [batch_size, channels, height, width]
        self.hidden_channels = max(8, channels * 2)

        self.layer_mode_overrides: Dict[str, ExecutionModeOptions] = (
            layer_mode_overrides.copy() if layer_mode_overrides else {}
        )
        self.active_layers: Optional[Set[str]] = (
            set(active_layers) if active_layers is not None else None
        )

        self.layers = self._build_layers()
        self.model_name = "ParallelToyNet"

    def _build_layers(self):
        sid = self.sid
        mode = self.enclave_mode
        pool_mode = self._resolve_mode(
            "LayerA_Pool",
            ExecutionModeOptions.CPU if mode is ExecutionModeOptions.Enclave else mode,
        )

        # Layer A: input (source tensor provider)
        layer_a = SecretInputLayer(
            sid,
            "LayerA",
            self.input_shape,
            self._resolve_mode("LayerA", mode),
            manually_register_next=True,
        )

        # LayerA_Pool: lightweight spatial downsampling before entering enclave conv
        layer_a_pool = SecretMaxpool2dLayer(
            sid,
            "LayerA_Pool",
            pool_mode,
            filter_hw=2,
            stride=1,
            padding=0,
            manually_register_prev=True,
            manually_register_next=True,
        )
        layer_a_pool.register_prev_layer(layer_a)

        # Bridge layer to move CPU pool output back into enclave memory when needed
        bridge_mode = self._resolve_mode(
            "LayerA_Pool_Bridge",
            ExecutionModeOptions.Enclave if mode is ExecutionModeOptions.Enclave else mode,
        )
        layer_pool_bridge = SecretIdentityLayer(
            sid,
            "LayerA_Pool_Bridge",
            bridge_mode,
            manually_register_prev=True,
            manually_register_next=True,
        )
        layer_pool_bridge.register_prev_layer(layer_a_pool)

        # Layer B: first learnable feature extractor (3x3 conv)
        layer_b = SGXConvBase(
            sid,
            "LayerB",
            self._resolve_mode("LayerB", mode),
            n_output_channel=self.hidden_channels,
            filter_hw=3,
            stride=1,
            padding=1,
            manually_register_prev=True,
            manually_register_next=True,
        )
        layer_b.register_prev_layer(layer_pool_bridge)

        # Non-linear activation before feeding LayerC
        layer_b_relu = SecretReLULayer(
            sid,
            "LayerB_ReLU",
            self._resolve_mode("LayerB_ReLU", mode),
            manually_register_prev=True,
            manually_register_next=True,
        )
        layer_b_relu.register_prev_layer(layer_b)

        # Layer C: second convolution preserving spatial resolution
        layer_c = SGXConvBase(
            sid,
            "LayerC",
            self._resolve_mode("LayerC", mode),
            n_output_channel=self.hidden_channels,
            filter_hw=3,
            stride=1,
            padding=1,
            manually_register_prev=True,
            manually_register_next=True,
        )
        layer_c.register_prev_layer(layer_b_relu)

        # Layer D: host-side branch placeholder fed directly by LayerB output
        layer_d = SecretIdentityLayer(
            sid,
            "LayerD",
            self._resolve_mode("LayerD", mode),
            manually_register_prev=True,
            manually_register_next=True,
        )
        layer_d.register_prev_layer(layer_b)

        # Layer E: merge enclave (LayerC) and host (LayerD) branches
        layer_e = SecretAddLayer(
            sid,
            "LayerE",
            self._resolve_mode("LayerE", mode),
            manually_register_prev=True,
            manually_register_next=True,
        )
        layer_e.register_prev_layer(layer_c)
        layer_e.register_prev_layer(layer_d)

        # Layer F: output, inference mode to skip loss computation
        layer_f = SecretOutputLayer(
            sid,
            "LayerF",
            self._resolve_mode("LayerF", ExecutionModeOptions.CPU),
            inference=True,
            manually_register_prev=True,
        )
        layer_f.register_prev_layer(layer_e)

        # Optional: keep a simple adjacency list for inspection without
        # instantiating the heavy SecretNeuralNetwork object.
        self._connections = {
            "LayerA": ["LayerA_Pool"],
            "LayerA_Pool": ["LayerA_Pool_Bridge"],
            "LayerA_Pool_Bridge": ["LayerB"],
            "LayerB": ["LayerB_ReLU", "LayerD"],
            "LayerB_ReLU": ["LayerC"],
            "LayerC": ["LayerE"],
            "LayerD": ["LayerE"],
            "LayerE": ["LayerF"],
            "LayerF": [],
        }

        ordered_layers: List = [
            layer_a,
            layer_a_pool,
            layer_pool_bridge,
            layer_b,
            layer_b_relu,
            layer_c,
            layer_d,
            layer_e,
            layer_f,
        ]

        if self.active_layers is None:
            return ordered_layers

        missing = self.active_layers - {layer.LayerName for layer in ordered_layers}
        if missing:
            raise ValueError(f"Unknown layers requested: {', '.join(sorted(missing))}")

        filtered = [layer for layer in ordered_layers if layer.LayerName in self.active_layers]
        self._validate_layer_dependencies(filtered)
        return filtered

    def describe_topology(self):
        """Print the logical fan-out/fan-in relationships."""
        print("ParallelToyNet topology (A -> pool -> B -> (B_ReLU -> C, D) -> E -> F):")
        for layer_name, next_layers in self._connections.items():
            if next_layers:
                print(f"  {layer_name} -> {', '.join(next_layers)}")
            else:
                print(f"  {layer_name} -> <end>")

    def _resolve_mode(
        self, layer_name: str, default_mode: ExecutionModeOptions
    ) -> ExecutionModeOptions:
        return self.layer_mode_overrides.get(layer_name, default_mode)

    def _validate_layer_dependencies(self, layers: Iterable[SecretIdentityLayer]) -> None:
        included = {layer.LayerName for layer in layers}
        for layer in layers:
            prev = getattr(layer, "PrevLayer", None)
            if prev is None:
                continue
            parents: List[str]
            if isinstance(prev, list):
                parents = [p.LayerName for p in prev if p is not None]
            else:
                parents = [prev.LayerName]
            missing = [name for name in parents if name not in included]
            if missing:
                raise ValueError(
                    f"Layer '{layer.LayerName}' requires missing predecessors: {missing}"
                )


def _ensure_torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("运行 ParallelToyNet 示例需要安装 PyTorch") from exc


def _run_secret_inference(model, warmup_runs=1, measured_runs=3):
    torch = _ensure_torch()
    inference_times = []
    try:
        if not GlobalTensor.is_init_global_tensor:
            GlobalTensor.init()

        secret_nn = SecretNeuralNetwork(model.sid, model.model_name)
        secret_nn.set_eid(GlobalTensor.get_eid())
        secret_nn.set_layers(model.layers)

        total_runs = warmup_runs + measured_runs
        input_shape = tuple(model.input_shape)
        for run_idx in range(total_runs):
            test_input = torch.randn(*input_shape)
            model.layers[0].set_input(test_input)
            elapsed_ms = secret_nn.forward_with_time()
            if run_idx >= warmup_runs:
                inference_times.append(elapsed_ms)
                print(f"  Enclave run {run_idx - warmup_runs + 1}: {elapsed_ms:.3f} ms")
    finally:
        if GlobalTensor.is_init_global_tensor:
            GlobalTensor.destroy()

    return inference_times


def _run_plain_cpu_inference(model, warmup_runs=1, measured_runs=3):
    torch = _ensure_torch()

    class PlainParallelTorchNet(torch.nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv1 = torch.nn.Conv2d(
                in_channels, hidden_channels, kernel_size=3, stride=1, padding=1
            )
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1
            )

        def forward(self, x):
            pooled = self.pool(x)
            b = self.conv1(pooled)
            c = self.conv2(self.relu(b))
            e = c + b  # Element-wise merge at LayerE
            return 0.8 * e  # LayerF scaling (keeps parity with host branch)

    device = torch.device("cpu")
    plain_model = PlainParallelTorchNet(model.channels, model.hidden_channels).to(device).eval()

    inference_times = []
    total_runs = warmup_runs + measured_runs
    input_shape = tuple(model.input_shape)

    with torch.no_grad():
        for run_idx in range(total_runs):
            test_input = torch.randn(*input_shape, device=device)
            start = time.perf_counter()
            _ = plain_model(test_input)
            elapsed_ms = (time.perf_counter() - start) * 1e3
            if run_idx >= warmup_runs:
                inference_times.append(elapsed_ms)
                print(f"  CPU run {run_idx - warmup_runs + 1}: {elapsed_ms:.3f} ms")

    return inference_times


def _run_inference(model, warmup_runs=1, measured_runs=3):
    if model.enclave_mode is ExecutionModeOptions.Enclave:
        return _run_secret_inference(model, warmup_runs, measured_runs)
    if model.enclave_mode is ExecutionModeOptions.CPU:
        return _run_plain_cpu_inference(model, warmup_runs, measured_runs)
    raise ValueError(f"Unsupported execution mode: {model.enclave_mode}")


def test_parallel():
    """
    Instantiate the toy topology and benchmark CPU / Enclave inference paths.

    Demonstrates how individual layers can be forced into specific execution
    modes so that only the desired subset runs inside SGX.
    """

    batch_size = 1
    channels = 3
    height = 4
    width = 4
    warmup_runs = 1
    measured_runs = 3

    mode_configs = [
        # {
        #     "label": "Full CPU baseline",
        #     "mode": ExecutionModeOptions.CPU,
        #     "overrides": None,
        # },
        # To exercise the enclave branch, append an entry such as:
        {
            "label": "SGX core + CPU host branch",
            "mode": ExecutionModeOptions.Enclave,
            "overrides": {
                "LayerD": ExecutionModeOptions.CPU,
                "LayerE": ExecutionModeOptions.CPU,
                "LayerF": ExecutionModeOptions.CPU,
            },
        },
    ]

    for cfg in mode_configs:
        mode = cfg["mode"]
        print(f"\nTesting ParallelToyNet in mode: {cfg['label']} ({mode})")
        model = SecretParallelToyNet(
            sid=0,
            enclave_mode=mode,
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            layer_mode_overrides=cfg["overrides"],
        )
        print(f"Successfully created {len(model.layers)} layers.")
        model.describe_topology()

        inference_times = _run_inference(
            model,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )

        if inference_times:
            avg_time = mean(inference_times)
            min_time = min(inference_times)
            max_time = max(inference_times)
            print(
                f"{mode.name} 平均推理时间: {avg_time:.3f} ms "
                f"(最小 {min_time:.3f} ms, 最大 {max_time:.3f} ms)"
            )


if __name__ == "__main__":
    test_parallel()

