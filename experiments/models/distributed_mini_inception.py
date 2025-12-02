"""
Distributed inference for MiniInception to verify parallel branching.
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import multiprocessing as mp

try:
    import torch as _torch_mod
except ModuleNotFoundError:
    _torch_mod = None

import sys
sys.path.insert(0, '.')

from experiments.models.sgx_mini_inception import SGXMiniInception
from python.enclave_interfaces import GlobalTensor
from python.sgx_net import SecretNeuralNetwork
from python.utils.basic_utils import ExecutionModeOptions


def _ensure_torch():
    if _torch_mod is None:
        raise ModuleNotFoundError("torch is required")
    return _torch_mod

def _time_layer(partition_label, layer_name, fn, timings=None):
    start_ts = time.time()
    print(f"{partition_label} {layer_name} start @ {datetime.fromtimestamp(start_ts).strftime('%H:%M:%S.%f')}")
    result = fn()
    end_ts = time.time()
    duration_ms = (end_ts - start_ts) * 1000
    print(f"{partition_label} {layer_name} end @ {datetime.fromtimestamp(end_ts).strftime('%H:%M:%S.%f')} (+{duration_ms:.3f} ms)")
    if timings is not None:
        timings[layer_name] = duration_ms
    return result

_GT_LOCK = threading.Lock()

class FlexibleGraphWorker(threading.Thread):
    def __init__(self, worker_id, target_mode, shared_model, queues, timings, init_event):
        super().__init__(name=worker_id)
        self.worker_id = worker_id
        self.target_mode = target_mode
        self.model = shared_model
        self.queues = queues
        self.timings = timings
        self.init_event = init_event
        self.daemon = True

    def run(self):
        _ensure_torch()
        self.init_event.wait()
        
        try:
            for layer in self.model.layers:
                if layer.EnclaveMode != self.target_mode:
                    continue

                # Dependencies
                if layer.PrevLayer:
                    parents = layer.PrevLayer if isinstance(layer.PrevLayer, list) else [layer.PrevLayer]
                    for parent in parents:
                        if parent.EnclaveMode != self.target_mode:
                            queue_key = f"{parent.LayerName}->{layer.LayerName}"
                            print(f"{self.worker_id} waiting for: {queue_key}")
                            data = self.queues[queue_key].get()
                            with _GT_LOCK:
                                parent.set_cpu("output", data)
                            print(f"{self.worker_id} got: {queue_key}")

                # Execute
                _time_layer(self.worker_id, layer.LayerName, layer.forward, self.timings)

                # Publish
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
                        print(f"{self.worker_id} sent to: {key}")

        except Exception as e:
            print(f"{self.worker_id} ERROR: {e}")
            import traceback
            traceback.print_exc()

def _analyze_topology(model):
    queues = {}
    for layer in model.layers:
        parents = layer.PrevLayer if isinstance(layer.PrevLayer, list) else [layer.PrevLayer]
        for parent in parents:
            if parent and parent.EnclaveMode != layer.EnclaveMode:
                key = f"{parent.LayerName}->{layer.LayerName}"
                queues[key] = queue.Queue(maxsize=1)
    return queues

def run_distributed(batch_size, input_size, overrides):
    torch = _ensure_torch()
    torch.manual_seed(0)
    input_tensor = torch.randn(batch_size, 3, input_size, input_size)
    
    if not GlobalTensor.is_init_global_tensor:
        GlobalTensor.init()
        
    model = SGXMiniInception(0, 10, ExecutionModeOptions.Enclave, batch_size, input_size, overrides)
    
    secret_nn = SecretNeuralNetwork(model.sid, model.model_name)
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(model.layers)
    
    queues = _analyze_topology(model)
    model.layers[0].set_input(input_tensor)
    
    timings = {}
    init_event = threading.Event()
    
    w1 = FlexibleGraphWorker("[Enclave]", ExecutionModeOptions.Enclave, model, queues, timings, init_event)
    w2 = FlexibleGraphWorker("[Host]", ExecutionModeOptions.CPU, model, queues, timings, init_event)
    
    start = time.time()
    w1.start(); w2.start()
    init_event.set()
    w1.join(); w2.join()
    end = time.time()
    
    if GlobalTensor.is_init_global_tensor:
        GlobalTensor.destroy()
        
    return (end - start) * 1000

def run_sequential(batch_size, input_size, overrides):
    torch = _ensure_torch()
    input_tensor = torch.randn(batch_size, 3, input_size, input_size)
    
    if not GlobalTensor.is_init_global_tensor:
        GlobalTensor.init()
        
    model = SGXMiniInception(0, 10, ExecutionModeOptions.Enclave, batch_size, input_size, overrides)
    secret_nn = SecretNeuralNetwork(model.sid, model.model_name)
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(model.layers)
    model.layers[0].set_input(input_tensor)
    
    start = time.time()
    for layer in model.layers:
        layer.forward()
    end = time.time()
    
    if GlobalTensor.is_init_global_tensor:
        GlobalTensor.destroy()
        
    return (end - start) * 1000

def wrapper_seq(q, overrides, input_size):
    try:
        t = run_sequential(1, input_size, overrides)
        q.put(("seq", t))
    except Exception as e:
        print(e)
        q.put(("seq", None))

def wrapper_dist(q, overrides, input_size):
    try:
        t = run_distributed(1, input_size, overrides)
        q.put(("dist", t))
    except Exception as e:
        print(e)
        q.put(("dist", None))

def main():
    print("Mini Inception Benchmark")
    
    # Config: Branch 1 in Enclave, Branch 2 in CPU
    overrides = {
        "input": ExecutionModeOptions.CPU,
        "branch2_conv1": ExecutionModeOptions.CPU,
        "branch2_conv2": ExecutionModeOptions.CPU,
        "concat": ExecutionModeOptions.CPU,
        "flatten": ExecutionModeOptions.CPU,
        "fc": ExecutionModeOptions.CPU,
        "output": ExecutionModeOptions.CPU
    }
    # Branch 1 (branch1_conv) defaults to Enclave
    
    input_sz = 32
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    
    p1 = ctx.Process(target=wrapper_seq, args=(q, overrides, input_sz))
    p1.start(); p1.join()
    
    p2 = ctx.Process(target=wrapper_dist, args=(q, overrides, input_sz))
    p2.start(); p2.join()
    
    res = {}
    while not q.empty():
        k, v = q.get()
        res[k] = v
        
    print(f"Sequential: {res.get('seq')} ms")
    print(f"Distributed: {res.get('dist')} ms")
    
    if res.get('seq') and res.get('dist'):
        print(f"Speedup: {res['seq'] / res['dist']:.2f}x")

if __name__ == "__main__":
    main()

