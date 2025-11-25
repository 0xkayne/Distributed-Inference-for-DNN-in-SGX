"""
Distributed inference demo where Partition-1 runs inside an SGX enclave.

Partition 1 (thread #1 / enclave):
    Uses the real `SecretParallelToyNet` layers A -> B -> C inside SGX.
    Publishes both B_out and C_out (copied to host memory) via queues.

Partition 2 (thread #2 / host):
    Waits for B_out to continue with host-side layer D,
    waits for C_out before computing E, and finally finishes with F.
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from typing import Any, Tuple

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

    def run(self) -> None:
        torch = _ensure_torch()

        model = SecretParallelToyNet(
            sid=0,
            enclave_mode=ExecutionModeOptions.Enclave,
            batch_size=self.batch_size,
            channels=self.channels,
            height=self.height,
            width=self.width,
        )

        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()

            secret_nn = SecretNeuralNetwork(model.sid, model.model_name)
            secret_nn.set_eid(GlobalTensor.get_eid())
            secret_nn.set_layers(model.layers)

            input_layer = model.layers[0]
            input_layer.set_input(self.input_tensor)

            for layer in model.layers:
                layer_name = layer.LayerName
                start_ts = time.time()
                print(f"[Partition-1] {layer_name} start @ {_format_timestamp(start_ts)}")
                layer.forward()
                end_ts = time.time()
                duration_ms = (end_ts - start_ts) * 1000
                print(
                    f"[Partition-1] {layer_name} end @ {_format_timestamp(end_ts)} "
                    f"(+{duration_ms:.3f} ms)"
                )

                if layer.LayerName == "LayerB":
                    layer.transfer_enclave_to_cpu("output")
                    b_out = layer.get_cpu("output").detach().clone()
                    self.b_queue.put(b_out)
                    print("[Partition-1] published B_out")

                if layer.LayerName == "LayerC":
                    layer.transfer_enclave_to_cpu("output")
                    c_out = layer.get_cpu("output").detach().clone()
                    self.c_queue.put(c_out)
                    print("[Partition-1] published C_out")
                    break
        finally:
            if GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()


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

    def run(self) -> None:
        torch = _ensure_torch()

        b_out = self.b_queue.get()
        print("[Partition-2] received B_out")
        # Preserve B_out so the main thread can inspect it even after queue consumption
        b_out_snapshot = b_out.detach().clone()
        start_ts = time.time()
        print(f"[Partition-2] LayerD start @ {_format_timestamp(start_ts)}")
        d_out = torch.relu(b_out + 0.1)
        end_ts = time.time()
        print(
            f"[Partition-2] LayerD end @ {_format_timestamp(end_ts)} "
            f"(+{(end_ts - start_ts) * 1000:.3f} ms)"
        )

        c_out = self.c_queue.get()
        print("[Partition-2] received C_out")

        start_ts = time.time()
        print(f"[Partition-2] LayerE start @ {_format_timestamp(start_ts)}")
        e_out = c_out + d_out
        end_ts = time.time()
        print(
            f"[Partition-2] LayerE end @ {_format_timestamp(end_ts)} "
            f"(+{(end_ts - start_ts) * 1000:.3f} ms)"
        )

        start_ts = time.time()
        print(f"[Partition-2] LayerF start @ {_format_timestamp(start_ts)}")
        f_out = 0.8 * e_out
        end_ts = time.time()
        print(
            f"[Partition-2] LayerF end @ {_format_timestamp(end_ts)} "
            f"(+{(end_ts - start_ts) * 1000:.3f} ms)"
        )
        self.result_queue.put((b_out_snapshot, f_out))
        print("[Partition-2] inference completed")


def run_distributed_inference(
    batch_size: int = 1,
    channels: int = 3,
    height: int = 4,
    width: int = 4,
) -> Tuple[Any, Any]:
    """
    Launch both partitions, wait for completion, and return intermediate/final tensors.
    """

    torch = _ensure_torch()

    torch.manual_seed(0)
    input_tensor = torch.randn(batch_size, channels, height, width)
    print(f"Input tensor shape: {tuple(input_tensor.shape)}")

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

    enclave_worker.start()
    host_worker.start()

    enclave_worker.join()
    host_worker.join()

    b_out, final_output = result_queue.get()
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

