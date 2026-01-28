#!/usr/bin/env python
"""
AlexNet Enclave Performance Profiler.

This script measures AlexNet layer execution in Enclave mode.
It uses the 2012 paper parameters (96, 256, 384, 384, 256 filters).

Output: experiments/data/alexnet_enclave_layers.csv
"""

import sys
import time
import os
import csv
import json
import numpy as np
import argparse
from collections import OrderedDict
from typing import Dict, List, Optional, Any

sys.path.insert(0, '.')

from experiments.models.profiler_utils import (
    LayerMetrics,
    calc_layer_memory_from_shapes,
    shape_to_bytes,
    print_memory_summary,
    CSV_FIELDNAMES,
    metrics_to_csv_row
)
from experiments.models.alexnet import AlexNetConfig, SGXAlexNet
from python.utils.basic_utils import ExecutionModeOptions

def _shape_to_bytes(shape: List[int], dtype_size: int = 4) -> int:
    return int(np.prod(shape)) * dtype_size

class AlexNetEnclaveProfiler:
    def __init__(
        self,
        batch_size: int = 1,
        num_iterations: int = 10,
        warmup_iterations: int = 3,
    ):
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        self.config = AlexNetConfig(batch_size=batch_size)
        self.metrics: Dict[str, LayerMetrics] = OrderedDict()
        self.reuse_single_enclave: bool = True
        self._runtime_stats: Dict[str, Dict[str, List[float]]] = {}
        # Keep layer objects alive to prevent GC from destroying Enclave
        self._layer_pool: List[Any] = []

    def _init_runtime_bucket(self, name: str):
        self._runtime_stats[name] = {
            'get_ms': [], 'get2_ms': [], 'compute_ms': [], 'store_ms': []
        }

    def _append_runtime_stats(self, name: str, stats: Dict):
        for k in ['get_ms', 'get2_ms', 'compute_ms', 'store_ms']:
            self._runtime_stats[name][k].append(stats.get(k, 0.0))

    def profile_all(self, verbose: bool = True):
        from python.enclave_interfaces import GlobalTensor
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Profiling AlexNet (2012 Paper) in Enclave Mode")
            print(f"{'='*60}")
        
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            # Conv1 Block
            self._profile_conv_enclave("conv1", [self.batch_size, 3, 224, 224], 
                                      96, 11, 4, 2, "Feature", verbose)
            self._profile_relu_enclave("relu1", [self.batch_size, 96, 55, 55], "Feature", verbose)
            self._profile_maxpool_enclave("pool1", [self.batch_size, 96, 55, 55], 3, 2, 0, "Feature", verbose)
            
            # Conv2 Block
            self._profile_conv_enclave("conv2", [self.batch_size, 96, 27, 27], 
                                      256, 5, 1, 2, "Feature", verbose)
            self._profile_relu_enclave("relu2", [self.batch_size, 256, 27, 27], "Feature", verbose)
            self._profile_maxpool_enclave("pool2", [self.batch_size, 256, 27, 27], 3, 2, 0, "Feature", verbose)
            
            # Conv3 Block
            self._profile_conv_enclave("conv3", [self.batch_size, 256, 13, 13], 
                                      384, 3, 1, 1, "Feature", verbose)
            self._profile_relu_enclave("relu3", [self.batch_size, 384, 13, 13], "Feature", verbose)
            
            # Conv4 Block
            self._profile_conv_enclave("conv4", [self.batch_size, 384, 13, 13], 
                                      384, 3, 1, 1, "Feature", verbose)
            self._profile_relu_enclave("relu4", [self.batch_size, 384, 13, 13], "Feature", verbose)
            
            # Conv5 Block
            self._profile_conv_enclave("conv5", [self.batch_size, 384, 13, 13], 
                                      256, 3, 1, 1, "Feature", verbose)
            self._profile_relu_enclave("relu5", [self.batch_size, 256, 13, 13], "Feature", verbose)
            self._profile_maxpool_enclave("pool3", [self.batch_size, 256, 13, 13], 3, 2, 0, "Feature", verbose)
            
            # Reset Enclave before HUGE FC layers to ensure EPC space
            if verbose: print("\n[Resetting Enclave before FC layers...]")
            GlobalTensor.destroy()
            GlobalTensor.init()
            
            # FC1 (9216 -> 4096)
            self._profile_linear_enclave("fc1", [self.batch_size, 9216], 4096, "Classifier", verbose)
            self._profile_relu_enclave("relu_fc1", [self.batch_size, 4096], "Classifier", verbose)
            
            # Reset again before FC2
            if verbose: print("[Resetting Enclave for FC2...]")
            GlobalTensor.destroy()
            GlobalTensor.init()
            
            # FC2 (4096 -> 4096)
            self._profile_linear_enclave("fc2", [self.batch_size, 4096], 4096, "Classifier", verbose)
            self._profile_relu_enclave("relu_fc2", [self.batch_size, 4096], "Classifier", verbose)
            
            # FC3 (4096 -> 1000)
            self._profile_linear_enclave("fc3", [self.batch_size, 4096], 1000, "Classifier", verbose)
            
            return self.metrics
        finally:
            if GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()

    def _profile_conv_enclave(self, name, input_shape, out_channels, k, s, p, group, verbose):
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.sgx_conv_base import SGXConvBase
        from python.layers.input import SecretInputLayer
        
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            sid = 0
            self._init_runtime_bucket(name)
            
            # Calculate output shape
            h_out = (input_shape[2] + 2*p - k) // s + 1
            output_shape = [input_shape[0], out_channels, h_out, h_out]
            
            in_layer = SecretInputLayer(sid, f"{name}_in", input_shape, ExecutionModeOptions.Enclave)
            conv_layer = SGXConvBase(
                sid, name, ExecutionModeOptions.Enclave, 
                n_output_channel=out_channels,
                n_input_channel=input_shape[1],
                filter_hw=k,
                stride=s,
                padding=p,
                batch_size=input_shape[0],
                img_hw=input_shape[2]
            )
            
            conv_layer.register_prev_layer(in_layer)
            
            # Keep layers alive to prevent GC
            self._layer_pool.extend([in_layer, conv_layer])
            
            times = []
            for i in range(self.warmup_iterations + self.num_iterations):
                in_layer.set_input(torch.randn(*input_shape))
                start = time.perf_counter()
                in_layer.forward()
                conv_layer.forward()
                elapsed = (time.perf_counter() - start) * 1000
                
                stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(conv_layer.LayerName)
                if i >= self.warmup_iterations:
                    times.append(elapsed)
                    self._append_runtime_stats(name, stats)
            
            mem = calc_layer_memory_from_shapes("Conv2d", input_shape, output_shape, kernel_size=k)
            m = LayerMetrics(name, "Conv2d", group, "Enclave", 
                            input_shape=input_shape, output_shape=output_shape,
                            input_bytes=_shape_to_bytes(input_shape), output_bytes=_shape_to_bytes(output_shape),
                            enclave_times=times, num_iterations=self.num_iterations,
                            enclave_get_ms=self._runtime_stats[name]['get_ms'],
                            enclave_get2_ms=self._runtime_stats[name]['get2_ms'],
                            enclave_compute_ms=self._runtime_stats[name]['compute_ms'],
                            enclave_store_ms=self._runtime_stats[name]['store_ms'],
                            **mem)
            m.compute_statistics()
            self.metrics[name] = m
            if verbose: print(f"  {name:20} {m.enclave_time_mean:8.3f} ms")
        
        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()

    def _profile_linear_enclave(self, name, input_shape, out_features, group, verbose):
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.sgx_linear_base import SGXLinearBase
        from python.layers.input import SecretInputLayer
        
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            sid = 0
            self._init_runtime_bucket(name)
            output_shape = [input_shape[0], out_features]
            
            in_layer = SecretInputLayer(sid, f"{name}_in", input_shape, ExecutionModeOptions.Enclave)
            fc_layer = SGXLinearBase(sid, name, ExecutionModeOptions.Enclave, 
                                   batch_size=input_shape[0], n_output_features=out_features, 
                                   n_input_features=input_shape[1])
            
            fc_layer.register_prev_layer(in_layer)
            
            # Keep layers alive to prevent GC
            self._layer_pool.extend([in_layer, fc_layer])
            
            times = []
            for i in range(self.warmup_iterations + self.num_iterations):
                in_layer.set_input(torch.randn(*input_shape))
                start = time.perf_counter()
                in_layer.forward()
                fc_layer.forward()
                elapsed = (time.perf_counter() - start) * 1000
                
                stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(fc_layer.LayerName)
                if i >= self.warmup_iterations:
                    times.append(elapsed)
                    self._append_runtime_stats(name, stats)
            
            mem = calc_layer_memory_from_shapes("Linear", input_shape, output_shape)
            m = LayerMetrics(name, "Linear", group, "Enclave", 
                            input_shape=input_shape, output_shape=output_shape,
                            input_bytes=_shape_to_bytes(input_shape), output_bytes=_shape_to_bytes(output_shape),
                            enclave_times=times, num_iterations=self.num_iterations,
                            enclave_get_ms=self._runtime_stats[name]['get_ms'],
                            enclave_get2_ms=self._runtime_stats[name]['get2_ms'],
                            enclave_compute_ms=self._runtime_stats[name]['compute_ms'],
                            enclave_store_ms=self._runtime_stats[name]['store_ms'],
                            **mem)
            m.compute_statistics()
            self.metrics[name] = m
            if verbose: print(f"  {name:20} {m.enclave_time_mean:8.3f} ms")
        
        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()

    def _profile_relu_enclave(self, name, input_shape, group, verbose):
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.relu import SecretReLULayer
        from python.layers.input import SecretInputLayer
        
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            sid = 0
            self._init_runtime_bucket(name)
            in_layer = SecretInputLayer(sid, f"{name}_in", input_shape, ExecutionModeOptions.Enclave)
            relu_layer = SecretReLULayer(sid, name, ExecutionModeOptions.Enclave)
            relu_layer.register_prev_layer(in_layer)
            
            # Keep layers alive to prevent GC
            self._layer_pool.extend([in_layer, relu_layer])
            
            times = []
            for i in range(self.warmup_iterations + self.num_iterations):
                in_layer.set_input(torch.randn(*input_shape))
                start = time.perf_counter()
                in_layer.forward()
                relu_layer.forward()
                elapsed = (time.perf_counter() - start) * 1000
                
                stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(relu_layer.LayerName)
                if i >= self.warmup_iterations:
                    times.append(elapsed)
                    self._append_runtime_stats(name, stats)
            
            mem = calc_layer_memory_from_shapes("ReLU", input_shape, input_shape)
            m = LayerMetrics(name, "ReLU", group, "Enclave", 
                            input_shape=input_shape, output_shape=input_shape,
                            input_bytes=_shape_to_bytes(input_shape), output_bytes=_shape_to_bytes(input_shape),
                            enclave_times=times, num_iterations=self.num_iterations,
                            enclave_get_ms=self._runtime_stats[name]['get_ms'],
                            enclave_get2_ms=self._runtime_stats[name]['get2_ms'],
                            enclave_compute_ms=self._runtime_stats[name]['compute_ms'],
                            enclave_store_ms=self._runtime_stats[name]['store_ms'],
                            **mem)
            m.compute_statistics()
            self.metrics[name] = m
            if verbose: print(f"  {name:20} {m.enclave_time_mean:8.3f} ms")
        
        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()

    def _profile_maxpool_enclave(self, name, input_shape, k, s, p, group, verbose):
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.maxpool2d import SecretMaxpool2dLayer
        from python.layers.input import SecretInputLayer
        
        sid = 0
        self._init_runtime_bucket(name)
        h_out = (input_shape[2] + 2*p - k) // s + 1
        output_shape = [input_shape[0], input_shape[1], h_out, h_out]
        
        in_layer = SecretInputLayer(sid, f"{name}_in", input_shape, ExecutionModeOptions.CPU) 
        pool_layer = SecretMaxpool2dLayer(sid, name, ExecutionModeOptions.CPU, filter_hw=k, stride=s, padding=p)
        pool_layer.register_prev_layer(in_layer)
        
        # Keep layers alive to prevent GC (even for CPU layers)
        self._layer_pool.extend([in_layer, pool_layer])
        
        times = []
        for i in range(self.warmup_iterations + self.num_iterations):
            in_layer.set_input(torch.randn(*input_shape))
            start = time.perf_counter()
            in_layer.forward()
            pool_layer.forward()
            elapsed = (time.perf_counter() - start) * 1000
            if i >= self.warmup_iterations:
                times.append(elapsed)
        
        mem = calc_layer_memory_from_shapes("MaxPool", input_shape, output_shape)
        m = LayerMetrics(name, "MaxPool", group, "CPU", 
                        input_shape=input_shape, output_shape=output_shape,
                        input_bytes=_shape_to_bytes(input_shape), output_bytes=_shape_to_bytes(output_shape),
                        enclave_times=times, num_iterations=self.num_iterations,
                        **mem)
        m.compute_statistics()
        self.metrics[name] = m
        if verbose: print(f"  {name:20} {m.enclave_time_mean:8.3f} ms (CPU)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    profiler = AlexNetEnclaveProfiler(batch_size=args.batch_size, num_iterations=args.iterations)
    metrics = profiler.profile_all()

    # Save to CSV
    output_dir = "experiments/data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "alexnet_enclave_layers.csv")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for m in metrics.values():
            writer.writerow(metrics_to_csv_row(m))

    print(f"\nProfiling results saved to {output_file}")
    print_memory_summary(metrics, "AlexNet Enclave Memory Summary")

if __name__ == "__main__":
    main()
