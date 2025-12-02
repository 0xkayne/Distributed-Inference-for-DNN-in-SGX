"""
Distributed inference demo where Partition-1 runs inside an SGX enclave.

Partition 1 (thread #1 / enclave):
    Executes the real `SecretParallelToyNet` layers up through LayerC
    (including LayerA_Pool / LayerB_ReLU) inside SGX and streams B_out/C_out.

Partition 2 (thread #2 / host):
    Waits for B_out to continue with host-side LayerD,
    waits for C_out before computing E, and finally finishes with F.
    A sequential reference path is also available to quantify the parallel speedup.
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import torch as _torch_mod  # type: ignore[import]
except ModuleNotFoundError:
    _torch_mod = None

import sys

sys.path.insert(0, '.')

from experiments.models.test_parallel import SecretParallelToyNet
from python.enclave_interfaces import GlobalTensor
from python.sgx_net import SecretNeuralNetwork
from python.utils.basic_utils import ExecutionModeOptions


# LayerA is forced onto CPU to avoid SGX SetTen failures during enclave input upload.
_ENCLAVE_LAYER_OVERRIDES = {
    "LayerA": ExecutionModeOptions.CPU,
}


def _format_timestamp(ts: float) -> str:
    """
    Convert epoch timestamp to human-readable HH:MM:SS.microseconds string.
    """
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")


def _ensure_torch():
    if _torch_mod is None:
        raise ModuleNotFoundError(
            "torch is required for enclave partition execution. "
            "Please install PyTorch to continue."
        )
    return _torch_mod


def _time_layer(
    partition_label: str,
    layer_name: str,
    fn: Callable[[], Any],
    timings: Optional[Dict[str, float]] = None,
) -> Any:
    """
    Execute `fn` while logging wall-clock start/end timestamps for a layer.
    """
    start_ts = time.time()
    print(f"{partition_label} {layer_name} start @ {_format_timestamp(start_ts)}")
    result = fn()
    end_ts = time.time()
    duration_ms = (end_ts - start_ts) * 1000
    print(
        f"{partition_label} {layer_name} end @ {_format_timestamp(end_ts)} "
        f"(+{duration_ms:.3f} ms)"
    )
    if timings is not None:
        timings[layer_name] = duration_ms
    return result


def _compute_layer_d(torch_mod, b_out):
    """
    LayerD is modeled as an identity bridge feeding LayerE.
    """
    return b_out.detach().clone()


def _compute_layer_e(torch_mod, c_out, d_out):
    """
    LayerE performs the residual merge between the enclave and host branches.
    """
    return c_out + d_out


def _compute_layer_f(torch_mod, e_out):
    """
    LayerF scales the merged activations to keep parity with PlainParallelTorchNet.
    """
    return 0.8 * e_out


def _run_head_partition(
    partition_label: str,
    input_tensor: Any,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    publish: Optional[Callable[[str, Any], None]] = None,
) -> Dict[str, Any]:
    """
    Execute the SecretParallelToyNet layers up to LayerC and optionally publish
    intermediate tensors (LayerB / LayerC) as soon as they are available.
    """

    _ensure_torch()
    model = SecretParallelToyNet(
        sid=0,
        enclave_mode=ExecutionModeOptions.Enclave,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        layer_mode_overrides=_ENCLAVE_LAYER_OVERRIDES,
    )

    published: Dict[str, Any] = {}

    try:
        if not GlobalTensor.is_init_global_tensor:
            GlobalTensor.init()

        secret_nn = SecretNeuralNetwork(model.sid, model.model_name)
        secret_nn.set_eid(GlobalTensor.get_eid())
        secret_nn.set_layers(model.layers)

        input_layer = model.layers[0]
        input_layer.set_input(input_tensor)

        for layer in model.layers:
            layer_name = layer.LayerName
            _time_layer(partition_label, layer_name, layer.forward)

            if layer_name == "LayerB":
                layer.transfer_enclave_to_cpu("output")
                b_out = layer.get_cpu("output").detach().clone()
                published[layer_name] = b_out
                if publish is not None:
                    publish(layer_name, b_out)

            if layer_name == "LayerC":
                layer.transfer_enclave_to_cpu("output")
                c_out = layer.get_cpu("output").detach().clone()
                published[layer_name] = c_out
                if publish is not None:
                    publish(layer_name, c_out)
                break
    finally:
        if GlobalTensor.is_init_global_tensor:
            GlobalTensor.destroy()

    return published


class EnclavePartitionWorker(threading.Thread):
    """
    Executes layers A -> B -> C of SecretParallelToyNet inside SGX.
    """

    def __init__(
        self,
        input_tensor: Any,
        b_queue: "queue.Queue[Any]",
        c_queue: "queue.Queue[Any]",
        batch_size: int,
        channels: int,
        height: int,
        width: int,
    ):
        super().__init__(name="Partition-1-Enclave")
        self.input_tensor = input_tensor
        self.b_queue = b_queue
        self.c_queue = c_queue
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.partition_label = "[Partition-1]"

    def run(self) -> None:
        _run_head_partition(
            self.partition_label,
            self.input_tensor,
            self.batch_size,
            self.channels,
            self.height,
            self.width,
            publish=self._publish_output,
        )

    def _publish_output(self, layer_name: str, tensor: Any) -> None:
        if layer_name == "LayerB":
            self.b_queue.put(tensor)
            print(f"{self.partition_label} published B_out")
        elif layer_name == "LayerC":
            self.c_queue.put(tensor)
            print(f"{self.partition_label} published C_out")


class HostPartitionWorker(threading.Thread):
    """
    Host-side worker consuming B_out/C_out and executing D -> E -> F on CPU.
    """

    def __init__(
        self,
        b_queue: "queue.Queue[Any]",
        c_queue: "queue.Queue[Any]",
        result_queue: "queue.Queue[Any]",
    ):
        super().__init__(name="Partition-2-Host")
        self.b_queue = b_queue
        self.c_queue = c_queue
        self.result_queue = result_queue
        self.partition_label = "[Partition-2]"

    def run(self) -> None:
        torch = _ensure_torch()

        b_out = self.b_queue.get()
        print(f"{self.partition_label} received B_out")
        b_out_snapshot = b_out.detach().clone()

        d_out = _time_layer(
            self.partition_label,
            "LayerD",
            lambda: _compute_layer_d(torch, b_out),
        )

        c_out = self.c_queue.get()
        print(f"{self.partition_label} received C_out")
        c_out_snapshot = c_out.detach().clone()

        e_out = _time_layer(
            self.partition_label,
            "LayerE",
            lambda: _compute_layer_e(torch, c_out, d_out),
        )

        f_out = _time_layer(
            self.partition_label,
            "LayerF",
            lambda: _compute_layer_f(torch, e_out),
        )

        self.result_queue.put(
            {
                "b_out": b_out_snapshot,
                "c_out": c_out_snapshot,
                "final": f_out,
            }
        )
        print(f"{self.partition_label} inference completed")


def _run_sequential_reference(
    input_tensor: Any,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
) -> Dict[str, Any]:
    """
    Run the pipeline sequentially (without threads) to establish a baseline latency.
    """

    start_ts = time.time()
    published = _run_head_partition(
        "[Sequential-Head]",
        input_tensor,
        batch_size,
        channels,
        height,
        width,
        publish=None,
    )

    torch = _ensure_torch()
    tail_timings: Dict[str, float] = {}

    b_out = published.get("LayerB")
    c_out = published.get("LayerC")
    if b_out is None or c_out is None:
        raise RuntimeError("Sequential reference run failed to produce LayerB/LayerC outputs.")

    d_out = _time_layer(
        "[Sequential-Tail]",
        "LayerD",
        lambda: _compute_layer_d(torch, b_out),
        tail_timings,
    )
    e_out = _time_layer(
        "[Sequential-Tail]",
        "LayerE",
        lambda: _compute_layer_e(torch, c_out, d_out),
        tail_timings,
    )
    final_out = _time_layer(
        "[Sequential-Tail]",
        "LayerF",
        lambda: _compute_layer_f(torch, e_out),
        tail_timings,
    )

    total_latency_ms = (time.time() - start_ts) * 1000
    print(
        "[Sequential] end-to-end latency "
        f"{total_latency_ms:.3f} ms (tail breakdown: {tail_timings})"
    )

    return {
        "b_out": b_out,
        "c_out": c_out,
        "final": final_out,
        "latency_ms": total_latency_ms,
    }


def _report_tensor_delta(label: str, reference: Any, candidate: Any, atol: float = 1e-4) -> None:
    """
    Compare tensors from sequential vs parallel executions and log the delta.
    """
    torch = _ensure_torch()
    if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
        print(f"[Benchmark] {label} comparison skipped (non-tensor inputs).")
        return

    diff = torch.max(torch.abs(reference - candidate)).item()
    if torch.allclose(reference, candidate, atol=atol, rtol=0.0):
        print(
            f"[Benchmark] {label} matches sequential reference "
            f"(max |diff|={diff:.3e}, atol={atol})."
        )
    else:
        print(
            f"[Benchmark] WARNING: {label} deviates from sequential reference "
            f"(max |diff|={diff:.3e}, atol={atol})."
        )


def run_distributed_inference(
    batch_size: int = 1,
    channels: int = 3,
    height: int = 4,
    width: int = 4,
    measure_baseline: bool = True,
) -> Tuple[Any, Any]:
    """
    Launch both partitions, wait for completion, and return intermediate/final tensors.

    Args:
        batch_size/channels/height/width: Input tensor shape parameters.
        measure_baseline: When True, run a sequential reference pass before the
            threaded execution to highlight the latency improvement.
    """

    torch = _ensure_torch()

    torch.manual_seed(0)
    input_tensor = torch.randn(batch_size, channels, height, width)
    print(f"Input tensor shape: {tuple(input_tensor.shape)}")

    sequential_stats: Optional[Dict[str, Any]] = None
    if measure_baseline:
        sequential_stats = _run_sequential_reference(
            input_tensor.clone(),
            batch_size,
            channels,
            height,
            width,
        )

    b_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)
    c_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)
    result_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)

    enclave_worker = EnclavePartitionWorker(
        input_tensor=input_tensor,
        b_queue=b_queue,
        c_queue=c_queue,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
    )
    host_worker = HostPartitionWorker(b_queue, c_queue, result_queue)

    parallel_start = time.time()
    enclave_worker.start()
    host_worker.start()

    enclave_worker.join()
    host_worker.join()

    parallel_latency_ms = (time.time() - parallel_start) * 1000
    results = result_queue.get()
    b_out = results["b_out"]
    c_out = results["c_out"]
    final_output = results["final"]

    print(f"[Parallel] end-to-end latency {parallel_latency_ms:.3f} ms")

    if sequential_stats is not None:
        speedup = sequential_stats["latency_ms"] / max(parallel_latency_ms, 1e-9)
        print(
            "[Benchmark] Sequential vs parallel latency: "
            f"{sequential_stats['latency_ms']:.3f} ms vs {parallel_latency_ms:.3f} ms "
            f"(speedup {speedup:.2f}x)"
        )
        _report_tensor_delta("B_out", sequential_stats["b_out"], b_out)
        _report_tensor_delta("C_out", sequential_stats["c_out"], c_out)
        _report_tensor_delta("Final output", sequential_stats["final"], final_output)

    return b_out, final_output


def main() -> None:
    try:
        torch = _ensure_torch()
        b_out, final_out = run_distributed_inference()
        print("B_out sample:", b_out.view(-1)[:5])
        print("Final output sample:", final_out.view(-1)[:5])
    except ModuleNotFoundError as exc:
        print(
            "Missing dependency while attempting to run enclave partition. "
            "Please ensure SGX runtime prerequisites (e.g., numpy/torch) are installed."
        )
        raise exc


if __name__ == "__main__":
    main()

