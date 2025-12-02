"""
Distributed inference for InceptionV3 with support for complex DAG topologies.

This script demonstrates true pipeline parallelism by splitting parallel branches
of Inception modules across different execution environments.
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

from experiments.models.sgx_inception import SGXInceptionV3
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


_GT_LOCK = threading.Lock()


class FlexibleGraphWorker(threading.Thread):
    """
    Generic worker that supports multi-predecessor dependencies (DAG).
    """

    def __init__(
        self,
        worker_id: str,
        target_mode: ExecutionModeOptions,
        shared_model: SGXInceptionV3,
        queues: Dict[str, "queue.Queue[Any]"],
        timings: Dict[str, float],
        init_event: threading.Event,
    ):
        super().__init__(name=worker_id)
        self.worker_id = worker_id
        self.target_mode = target_mode
        self.model = shared_model
        self.queues = queues
        self.timings = timings
        self.init_event = init_event
        self.daemon = True

    def run(self) -> None:
        _ensure_torch()
        self.init_event.wait()
        
        try:
            for layer in self.model.layers:
                if layer.EnclaveMode != self.target_mode:
                    continue

                # Handle multiple predecessors (DAG support)
                if layer.PrevLayer:
                    parents = layer.PrevLayer if isinstance(layer.PrevLayer, list) else [layer.PrevLayer]
                    
                    for parent_idx, parent in enumerate(parents):
                        if parent.EnclaveMode != self.target_mode:
                            # This is a dependency from another partition
                            queue_key = f"{parent.LayerName}->{layer.LayerName}"
                            print(f"{self.worker_id} waiting for dependency: {queue_key}")
                            
                            data = self.queues[queue_key].get()
                            
                            with _GT_LOCK:
                                # For layers with multiple inputs (like Concat/Add), we need to set the specific input
                                # SecretConcatenateLayer uses 'input', 'input1', 'input2'...
                                # Base layers usually use 'output' of parent linked to 'input' of child
                                # But here we are injecting into the PARENT's output slot
                                
                                # Crucial fix: We must inject data into the parent layer object
                                # The child layer will pull from parent.get_output() during forward()
                                parent.set_cpu("output", data)
                                
                            print(f"{self.worker_id} resolved dependency: {queue_key}")

                # Execute Layer
                _time_layer(self.worker_id, layer.LayerName, layer.forward, self.timings)

                # Publish Output to all remote successors
                # We scan queues to see if any start with our name
                out_data_cache = None
                for key, q in self.queues.items():
                    src, dst = key.split("->")
                    if src == layer.LayerName:
                        if out_data_cache is None:
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


def _analyze_topology_and_create_queues(model: SGXInceptionV3) -> Dict[str, "queue.Queue[Any]"]:
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


def run_sequential_inference(
    batch_size: int,
    input_size: int,
    num_classes: int,
    layer_mode_overrides: Dict[str, ExecutionModeOptions],
) -> float:
    """Sequential baseline execution"""
    torch = _ensure_torch()
    input_tensor = torch.randn(batch_size, 3, input_size, input_size)
    
    if not GlobalTensor.is_init_global_tensor:
        GlobalTensor.init()
        
    model = SGXInceptionV3(sid=0, enclave_mode=ExecutionModeOptions.Enclave,
                           batch_size=batch_size, input_size=input_size, num_classes=num_classes,
                           layer_mode_overrides=layer_mode_overrides)
    
    secret_nn = SecretNeuralNetwork(model.sid, model.model_name)
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(model.layers)
    
    # Need to explicitly enable CPU fallback for unsupported Enclave layers (like Concat)
    # This is handled inside SecretConcatenateLayer init
    
    model.layers[0].set_input(input_tensor)
    
    print("\n[Sequential] Starting...")
    start = time.time()
    for layer in model.layers:
        layer.forward()
    end = time.time()
    
    if GlobalTensor.is_init_global_tensor:
        GlobalTensor.destroy()
        
    return (end - start) * 1000


def run_distributed_inference(
    batch_size: int = 1,
    input_size: int = 96,
    num_classes: int = 10,
    layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
) -> float:
    torch = _ensure_torch()
    torch.manual_seed(0)
    input_tensor = torch.randn(batch_size, 3, input_size, input_size)

    if layer_mode_overrides is None:
        layer_mode_overrides = {}
    layer_mode_overrides["input"] = ExecutionModeOptions.CPU
        
    if not GlobalTensor.is_init_global_tensor:
        GlobalTensor.init()
        
    shared_model = SGXInceptionV3(
        sid=0, enclave_mode=ExecutionModeOptions.Enclave,
        batch_size=batch_size, input_size=input_size, num_classes=num_classes,
        layer_mode_overrides=layer_mode_overrides,
    )
    
    secret_nn = SecretNeuralNetwork(shared_model.sid, shared_model.model_name)
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(shared_model.layers)
    
    queues = _analyze_topology_and_create_queues(shared_model)
    shared_model.layers[0].set_input(input_tensor)

    timings: Dict[str, float] = {}
    init_event = threading.Event()

    enclave_worker = FlexibleGraphWorker(
        "[Partition-Enclave]", ExecutionModeOptions.Enclave,
        shared_model, queues, timings, init_event
    )
    
    host_worker = FlexibleGraphWorker(
        "[Partition-Host]", ExecutionModeOptions.CPU,
        shared_model, queues, timings, init_event
    )

    print("\n[Distributed] Starting...")
    start_ts = time.time()
    
    enclave_worker.start()
    host_worker.start()
    init_event.set()
    
    enclave_worker.join()
    host_worker.join()
    
    end_ts = time.time()
    
    if GlobalTensor.is_init_global_tensor:
        GlobalTensor.destroy()
        
    return (end_ts - start_ts) * 1000


# Global wrappers for multiprocessing
def run_seq_wrapper(q, overrides, input_size):
    try:
        t = run_sequential_inference(1, input_size, 10, overrides)
        q.put(("seq", t))
    except Exception as e:
        import traceback
        traceback.print_exc()
        q.put(("seq", None))
        
def run_dist_wrapper(q, overrides, input_size):
    try:
        t = run_distributed_inference(1, input_size, 10, overrides)
        q.put(("dist", t))
    except Exception as e:
        import traceback
        traceback.print_exc()
        q.put(("dist", None))

def main():
    print("\n" + "="*80)
    print("InceptionV3 Distributed Inference Benchmark")
    print("="*80)
    
    # Strategy: Branch Parallelism
    # Branches 1 & 2 in Enclave, Branches 3 & 4 in CPU
    # This allows TRUE parallel execution of branches before Concatenation
    
    overrides = {"input": ExecutionModeOptions.CPU}
    
    # Inc1 Block
    # Branch 3 (5x5 sim) & 4 (Pool) -> CPU
    cpu_branches = ["b3_1x1", "b3_relu1", "b3_3x3_1", "b3_relu2", "b3_3x3_2", "b3_relu3",
                    "b4_pool", "b4_1x1", "b4_relu"]
    
    for layer in cpu_branches:
        overrides[f"inc1_{layer}"] = ExecutionModeOptions.CPU
        overrides[f"inc2_{layer}"] = ExecutionModeOptions.CPU
        
    # Concat must be CPU (currently)
    overrides["inc1_concat"] = ExecutionModeOptions.CPU
    overrides["inc2_concat"] = ExecutionModeOptions.CPU
    
    # Classifier on CPU
    for l in ["avgpool", "flatten", "fc", "output"]:
        overrides[l] = ExecutionModeOptions.CPU

    print("\nTesting Strategy: Branch Parallelism")
    print("Enclave: Branch 1 & 2 (Conv heavy)")
    print("CPU: Branch 3 & 4 + Concat + Classifier")
    
    # Use separate processes to ensure clean Enclave state
    import multiprocessing as mp
    
    ctx = mp.get_context('spawn')
    q = ctx.Queue()

    # Use smaller input size
    input_sz = 64

    p1 = ctx.Process(target=run_seq_wrapper, args=(q, overrides, input_sz))
    p1.start()
    p1.join()
    
    p2 = ctx.Process(target=run_dist_wrapper, args=(q, overrides, input_sz))
    p2.start()
    p2.join()
    
    results = {}
    while not q.empty():
        k, v = q.get()
        results[k] = v
        
    seq_time = results.get("seq")
    par_time = results.get("dist")
    
    if seq_time:
        print(f"Sequential Time: {seq_time:.3f} ms")
    if par_time:
        print(f"Distributed Time: {par_time:.3f} ms")
    
    if seq_time and par_time:
        speedup = seq_time / par_time
        print(f"\nSpeedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()

