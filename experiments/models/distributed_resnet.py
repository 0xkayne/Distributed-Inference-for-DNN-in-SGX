"""
Distributed inference for ResNet-18 with flexible graph partitioning.

This script demonstrates distributed inference on ResNet-18 where different layers
can be assigned to run in Enclave or on CPU, with automatic communication management
between partitions.

Example partitioning strategies:
1. Pipeline: Early layers in Enclave, later layers on CPU
2. Block-level: Alternate ResNet blocks between Enclave and CPU
3. Residual: Main path in Enclave, skip connections on CPU
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

from experiments.models.sgx_resnet import SGXResNet18
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


# Global lock for thread-safe GlobalTensor access
_GT_LOCK = threading.Lock()


class FlexibleGraphWorker(threading.Thread):
    """
    Generic worker that executes a subset of layers based on their assigned mode.
    Handles data fetching from queues for cut edges (incoming) and publishing to queues (outgoing).
    
    Both workers SHARE the same model instance to avoid GlobalTensor double-initialization issues.
    """

    def __init__(
        self,
        worker_id: str,
        target_mode: ExecutionModeOptions,
        shared_model: SGXResNet18,
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
        
        # Wait for initialization to complete
        self.init_event.wait()
        
        try:
            for layer in self.model.layers:
                # Skip layers not assigned to this worker
                if layer.EnclaveMode != self.target_mode:
                    continue

                # Resolve dependencies
                if layer.PrevLayer:
                    parents = layer.PrevLayer if isinstance(layer.PrevLayer, list) else [layer.PrevLayer]
                    for parent in parents:
                        if parent.EnclaveMode != self.target_mode:
                            queue_key = f"{parent.LayerName}->{layer.LayerName}"
                            print(f"{self.worker_id} waiting for dependency: {queue_key}")
                            
                            data = self.queues[queue_key].get()
                            
                            with _GT_LOCK:
                                parent.set_cpu("output", data)
                                
                            print(f"{self.worker_id} resolved dependency: {queue_key}")

                # Execute Layer
                _time_layer(self.worker_id, layer.LayerName, layer.forward, self.timings)

                # Publish Output
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


def _analyze_topology_and_create_queues(model: SGXResNet18) -> Dict[str, "queue.Queue[Any]"]:
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
    input_size: int = 224,
    num_classes: int = 1000,
    layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
) -> Dict[str, Any]:
    """
    Run distributed inference on ResNet-18 with flexible partitioning.
    
    Args:
        batch_size: Batch size
        input_size: Input image size
        num_classes: Number of output classes
        layer_mode_overrides: Dict mapping layer names to ExecutionModeOptions
    
    Returns:
        Dict containing latency and per-layer timings
    """
    torch = _ensure_torch()
    torch.manual_seed(0)
    input_tensor = torch.randn(batch_size, 3, input_size, input_size)
    print(f"Input tensor shape: {tuple(input_tensor.shape)}")

    # Force input layer to CPU (known SGX limitation)
    if layer_mode_overrides is None:
        layer_mode_overrides = {}
    layer_mode_overrides["input"] = ExecutionModeOptions.CPU
        
    # Initialize GlobalTensor
    if not GlobalTensor.is_init_global_tensor:
        GlobalTensor.init()
        
    # Create shared model
    shared_model = SGXResNet18(
        sid=0,
        enclave_mode=ExecutionModeOptions.Enclave,
        batch_size=batch_size,
        input_size=input_size,
        num_classes=num_classes,
        layer_mode_overrides=layer_mode_overrides,
    )
    
    # Print architecture
    shared_model.print_architecture()
    
    # Initialize the SecretNeuralNetwork wrapper
    secret_nn = SecretNeuralNetwork(shared_model.sid, shared_model.model_name)
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(shared_model.layers)
    
    # Analyze topology and create queues
    queues = _analyze_topology_and_create_queues(shared_model)
    
    # Set input
    shared_model.layers[0].set_input(input_tensor)

    timings: Dict[str, float] = {}
    init_event = threading.Event()

    # Create workers
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

    print("\n" + "="*60)
    print("Starting distributed inference...")
    print("="*60 + "\n")
    
    start_ts = time.time()
    
    enclave_worker.start()
    host_worker.start()
    
    # Signal that initialization is complete
    init_event.set()
    
    enclave_worker.join()
    host_worker.join()
    
    end_ts = time.time()
    
    # Cleanup
    if GlobalTensor.is_init_global_tensor:
        GlobalTensor.destroy()
        
    latency_ms = (end_ts - start_ts) * 1000
    
    print("\n" + "="*60)
    print(f"Distributed Inference Complete")
    print("="*60)
    print(f"Total Latency: {latency_ms:.3f} ms")
    print(f"Layers executed: {len(timings)}")
    print("="*60 + "\n")
    
    return {"latency_ms": latency_ms, "timings": timings}


def get_partition_strategy(strategy_name: str) -> Dict[str, ExecutionModeOptions]:
    """
    Get predefined partitioning strategies for ResNet-18.
    
    Args:
        strategy_name: Name of the strategy
    
    Returns:
        layer_mode_overrides dict
    """
    strategies = {
        "all_cpu": {
            # All layers on CPU (baseline)
        },
        
        "all_enclave": {
            # All layers in Enclave except input
            # (input is forced to CPU automatically)
        },
        
        "pipeline_half": {
            # First half (stem + layer1-2) in Enclave
            # Second half (layer3-4 + classifier) on CPU
            "input": ExecutionModeOptions.CPU,
            **{f"layer3_block{i}_{suffix}": ExecutionModeOptions.CPU
               for i in range(2)
               for suffix in ["conv1", "relu1", "conv2", "downsample", "skip", "add", "relu2"]},
            **{f"layer4_block{i}_{suffix}": ExecutionModeOptions.CPU
               for i in range(2)
               for suffix in ["conv1", "relu1", "conv2", "downsample", "skip", "add", "relu2"]},
            "avgpool": ExecutionModeOptions.CPU,
            "flatten": ExecutionModeOptions.CPU,
            "fc": ExecutionModeOptions.CPU,
            "output": ExecutionModeOptions.CPU,
        },
        
        "alternating_blocks": {
            # Alternate blocks between Enclave and CPU
            "input": ExecutionModeOptions.CPU,
            **{f"layer1_block1_{suffix}": ExecutionModeOptions.CPU
               for suffix in ["conv1", "relu1", "conv2", "skip", "add", "relu2"]},
            **{f"layer2_block1_{suffix}": ExecutionModeOptions.CPU
               for suffix in ["conv1", "relu1", "conv2", "downsample", "add", "relu2"]},
            **{f"layer3_block1_{suffix}": ExecutionModeOptions.CPU
               for suffix in ["conv1", "relu1", "conv2", "downsample", "add", "relu2"]},
            **{f"layer4_block1_{suffix}": ExecutionModeOptions.CPU
               for suffix in ["conv1", "relu1", "conv2", "downsample", "add", "relu2"]},
        },
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name]


def main():
    """
    Main entry point for distributed ResNet-18 inference.
    
    Tests multiple partitioning strategies and compares performance.
    """
    print("\n" + "="*70)
    print("ResNet-18 Distributed Inference Benchmark")
    print("="*70 + "\n")
    
    # Test different partitioning strategies
    strategies_to_test = [
        ("all_cpu", ExecutionModeOptions.CPU),
        ("pipeline_half", ExecutionModeOptions.Enclave),
    ]
    
    results = {}
    
    for strategy_name, base_mode in strategies_to_test:
        print(f"\n{'#'*70}")
        print(f"# Testing Strategy: {strategy_name}")
        print(f"{'#'*70}\n")
        
        # Reset GlobalTensor state between tests to avoid tag conflicts
        if GlobalTensor.is_init_global_tensor:
            GlobalTensor.destroy()
        
        try:
            overrides = get_partition_strategy(strategy_name)
            
            # For all_cpu, set base mode to CPU
            if strategy_name == "all_cpu":
                overrides = {
                    layer_name: ExecutionModeOptions.CPU
                    for layer_name in [
                        "input", "conv1", "relu", "maxpool",
                        *[f"layer{li}_block{bi}_{suffix}"
                          for li in range(1, 5)
                          for bi in range(2)
                          for suffix in ["conv1", "relu1", "conv2", "downsample", "skip", "add", "relu2"]],
                        "avgpool", "flatten", "fc", "output"
                    ]
                }
            
            result = run_distributed_inference(
                batch_size=1,
                input_size=64,  # Minimum size to avoid negative dimensions after downsampling
                num_classes=10,
                layer_mode_overrides=overrides,
            )
            
            results[strategy_name] = result
            
        except Exception as e:
            print(f"Strategy '{strategy_name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("Benchmark Summary")
    print("="*70)
    for strategy_name, result in results.items():
        latency = result.get("latency_ms", "N/A")
        print(f"{strategy_name:30s}: {latency:>10.3f} ms" if isinstance(latency, float) else f"{strategy_name:30s}: {latency}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

