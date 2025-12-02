"""
Distributed inference demo with flexible graph partitioning.

Allows arbitrary mapping of layers to Enclave/CPU/GPU modes.
The runtime automatically analyzes the topology, creates communication queues for cut edges,
and orchestrates two worker threads (one for Enclave, one for Host) to execute the split graph.
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")


def _ensure_torch():
    if _torch_mod is None:
        raise ModuleNotFoundError(
            "torch is required. Please install PyTorch."
        )
    return _torch_mod


def _time_layer(
    partition_label: str,
    layer_name: str,
    fn: Callable[[], Any],
    timings: Optional[Dict[str, float]] = None,
) -> Any:
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

# Global lock to synchronize access to GlobalTensor's shared state (static dicts)
# and Enclave bridge calls if necessary.
_GT_LOCK = threading.Lock()

class FlexibleGraphWorker(threading.Thread):
    """
    Generic worker that executes a subset of layers based on their assigned mode.
    Handles data fetching from queues for cut edges (incoming) and publishing to queues (outgoing).
    
    Crucially, both workers SHARE the same model instance to avoid GlobalTensor double-initialization issues.
    """

    def __init__(
        self,
        worker_id: str,
        target_mode: ExecutionModeOptions,
        shared_model: SecretParallelToyNet,
        queues: Dict[str, "queue.Queue[Any]"],
        timings: Dict[str, float],
        init_event: threading.Event,
    ):
        super().__init__(name=worker_id)
        self.worker_id = worker_id
        self.target_mode = target_mode
        self.model = shared_model
        self.queues = queues  # Key: "SrcLayerName->DstLayerName"
        self.timings = timings
        self.init_event = init_event
        self.daemon = True # Allow exiting if main thread fails

    def run(self) -> None:
        _ensure_torch()
        
        # Wait for the main thread to finish initializing the shared model
        # This ensures GlobalTensor is fully set up before any worker starts processing.
        self.init_event.wait()
        
        try:
            for layer in self.model.layers:
                # 1. Skip layers not assigned to this worker
                if layer.EnclaveMode != self.target_mode:
                    continue

                # 2. Resolve dependencies
                if layer.PrevLayer:
                    parents = layer.PrevLayer if isinstance(layer.PrevLayer, list) else [layer.PrevLayer]
                    for parent in parents:
                        if parent.EnclaveMode != self.target_mode:
                            queue_key = f"{parent.LayerName}->{layer.LayerName}"
                            print(f"{self.worker_id} waiting for dependency: {queue_key}")
                            
                            # Blocking get
                            data = self.queues[queue_key].get()
                            
                            # Inject data. Needs lock because it touches GlobalTensor cpu_tensor dict
                            with _GT_LOCK:
                                parent.set_cpu("output", data)
                                # Note: forward_tensor_transfer inside layer.forward() will handle 
                                # CPU -> Enclave transfer if needed, assuming set_cpu updated the source.
                                
                            print(f"{self.worker_id} resolved dependency: {queue_key}")

                # 3. Execute Layer
                # We wrap forward execution in a try-except block to catch errors.
                # Note: Since we share the model, 'layer' objects are shared.
                # But each layer is executed by ONLY one worker (determined by EnclaveMode).
                # So there is no race condition on executing the *same* layer.
                # Race conditions on *data* (set_cpu/get_cpu) are handled by _GT_LOCK.
                _time_layer(self.worker_id, layer.LayerName, layer.forward, self.timings)

                # 4. Publish Output
                # Scan for outgoing edges
                # We look for queues starting with our layer name
                out_data_cache = None
                for key, q in self.queues.items():
                    src, dst = key.split("->")
                    if src == layer.LayerName:
                        # This is an outgoing cut edge
                        if out_data_cache is None:
                            # Fetch output
                            with _GT_LOCK:
                                if layer.EnclaveMode == ExecutionModeOptions.Enclave:
                                    layer.transfer_enclave_to_cpu("output")
                                out_data_cache = layer.get_cpu("output").detach().clone()
                        
                        q.put(out_data_cache)
                        print(f"{self.worker_id} published data to: {key}")

        except Exception as e:
            print(f"{self.worker_id} FAILED with error: {e}")
            import traceback
            traceback.print_exc()


def _analyze_topology_and_create_queues(model: SecretParallelToyNet) -> Dict[str, "queue.Queue[Any]"]:
    """
    Analyze the model's layers and execution modes to find 'Cut Edges'.
    Returns a dictionary of queues for these edges.
    """
    queues = {}
    
    for layer in model.layers:
        parents = []
        if layer.PrevLayer:
            if isinstance(layer.PrevLayer, list):
                parents = layer.PrevLayer
            else:
                parents = [layer.PrevLayer]
        
        for parent in parents:
            if parent.EnclaveMode != layer.EnclaveMode:
                key = f"{parent.LayerName}->{layer.LayerName}"
                queues[key] = queue.Queue(maxsize=1)
                print(f"[Topology] Found cut edge: {key} ({parent.EnclaveMode} -> {layer.EnclaveMode})")
                
    return queues


def run_distributed_inference(
    batch_size: int = 1,
    channels: int = 3,
    height: int = 4,
    width: int = 4,
    layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
) -> Dict[str, Any]:
    """
    Run distributed inference with flexible partitioning.
    """
    torch = _ensure_torch()
    torch.manual_seed(0)
    input_tensor = torch.randn(batch_size, channels, height, width)
    print(f"Input tensor shape: {tuple(input_tensor.shape)}")

    if layer_mode_overrides is None:
        layer_mode_overrides = {"LayerA": ExecutionModeOptions.CPU}
        
    # 1. Instantiate ONE shared model
    # This avoids the 'Tags must linked before tensor initialization' error
    # because we only init GlobalTensor and link tags once.
    
    if not GlobalTensor.is_init_global_tensor:
        GlobalTensor.init()
        
    shared_model = SecretParallelToyNet(
        sid=0,
        enclave_mode=ExecutionModeOptions.Enclave,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        layer_mode_overrides=layer_mode_overrides,
    )
    
    # Initialize the SecretNeuralNetwork wrapper to set up layer links and allocate tensors
    secret_nn = SecretNeuralNetwork(shared_model.sid, shared_model.model_name)
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(shared_model.layers)
    
    # Analyze topology after model is fully built and configured
    queues = _analyze_topology_and_create_queues(shared_model)
    
    # Set input on the shared model
    # If InputLayer is CPU, set_input sets CPU tensor.
    # If InputLayer is Enclave, it sets CPU and transfers to Enclave.
    # Since we do this in main thread, no lock needed yet (workers not started).
    shared_model.layers[0].set_input(input_tensor)

    timings: Dict[str, float] = {}
    init_event = threading.Event()

    # 2. Create Workers sharing the SAME model instance
    enclave_worker = FlexibleGraphWorker(
        worker_id="[Partition-Enclave]",
        target_mode=ExecutionModeOptions.Enclave,
        shared_model=shared_model,
        queues=queues,
        timings=timings,
        init_event=init_event
    )
    
    host_worker = FlexibleGraphWorker(
        worker_id="[Partition-Host]",
        target_mode=ExecutionModeOptions.CPU,
        shared_model=shared_model,
        queues=queues,
        timings=timings,
        init_event=init_event
    )

    print("\nStarting distributed inference...")
    start_ts = time.time()
    
    enclave_worker.start()
    host_worker.start()
    
    # Signal workers that init is done and model is ready
    init_event.set()
    
    enclave_worker.join()
    host_worker.join()
    
    end_ts = time.time()
    
    # Cleanup
    if GlobalTensor.is_init_global_tensor:
        GlobalTensor.destroy()
        
    latency_ms = (end_ts - start_ts) * 1000
    print(f"\n[Distributed] Total Latency: {latency_ms:.3f} ms")
    
    return {"latency_ms": latency_ms, "timings": timings}

def main():
    # Layer A/B/C on SGX, D/E/F on CPU
    # LayerA must be CPU due to issue
    overrides = {
        "LayerA": ExecutionModeOptions.CPU,
        "LayerD": ExecutionModeOptions.CPU,
        "LayerE": ExecutionModeOptions.CPU,
        "LayerF": ExecutionModeOptions.CPU,
    }
    
    run_distributed_inference(layer_mode_overrides=overrides)

if __name__ == "__main__":
    main()
