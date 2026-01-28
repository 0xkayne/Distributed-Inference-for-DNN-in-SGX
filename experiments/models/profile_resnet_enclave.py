import sys
import os
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
import time
import csv
import os
from collections import defaultdict, OrderedDict
from typing import List, Dict, Any, Optional

from experiments.models.profiler_utils import (
    LayerMetrics,
    calc_layer_memory_from_shapes,
    shape_to_bytes,
    print_memory_summary,
    CSV_FIELDNAMES,
    metrics_to_csv_row
)
from python.layers.input import SecretInputLayer
from python.layers.sgx_conv_base import SGXConvBase
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.relu import SecretReLULayer
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.layers.batch_norm_2d import SecretBatchNorm2dLayer
from python.layers.add import SecretAddLayer
from python.layers.avgpool2d import SecretAvgpool2dLayer
from python.layers.flatten import SecretFlattenLayer
from python.enclave_interfaces import GlobalTensor
from python.utils.basic_utils import ExecutionModeOptions
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel

class ResNetEnclaveProfiler:
    def __init__(self, warmup_iterations: int = 2, num_iterations: int = 5):
        self.warmup_iterations = warmup_iterations
        self.num_iterations = num_iterations
        self.metrics: Dict[str, LayerMetrics] = OrderedDict()
        self._runtime_stats = defaultdict(lambda: defaultdict(list))
        self._layer_pool = []
        self.reuse_single_enclave = True
        self.last_layer_name = None

    def _init_runtime_bucket(self, name: str):
        if name not in self._runtime_stats:
            self._runtime_stats[name] = {
                'get_ms': [],
                'get2_ms': [],
                'compute_ms': [],
                'store_ms': [],
                'total_python_ms': []
            }

    def _append_runtime_stats(self, name: str, stats: Dict):
        for k in ['get_ms', 'get2_ms', 'compute_ms', 'store_ms']:
            self._runtime_stats[name][k].append(stats.get(k, 0.0))

    def _setup_layers(self, layers: List[Any]):
        """Initialize layers with EID and setup shapes/linking/enclave in correct order."""
        from python.enclave_interfaces import GlobalTensor
        eid = GlobalTensor.get_eid()
        
        # 1. Structure Phase: Set EID, Init Shapes, Link Tensors
        for layer in layers:
            layer.set_eid(eid)
            layer.init_shape()
            layer.link_tensors()
            
        # 2. Init Phase: Allocate Enclave Memory & Transfer Weights
        for layer in layers:
            layer.init(start_enclave=False)

    def _profile_conv_enclave(self, name, input_shape, out_channels, k, s, p, group="Feature", dependencies=None, verbose=True):
        self._init_runtime_bucket(name)
        sid = 0
        if verbose:
            print(f"Profiling {name:20} | In: {str(input_shape):16} | Out: {out_channels:3} | K: {k} S: {s} P: {p}")

        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()

            h_out = (input_shape[2] + 2*p - k) // s + 1
            output_shape = [input_shape[0], out_channels, h_out, h_out]

            in_layer = SecretInputLayer(sid, f"{name}_in", input_shape, ExecutionModeOptions.Enclave)
            conv_layer = SGXConvBase(
                sid, name, ExecutionModeOptions.Enclave,
                n_output_channel=out_channels, filter_hw=k, stride=s, padding=p,
                batch_size=input_shape[0], n_input_channel=input_shape[1], img_hw=input_shape[2]
            )
            conv_layer.register_prev_layer(in_layer)
            
            self._layer_pool.extend([in_layer, conv_layer])
            self._setup_layers([in_layer, conv_layer])
            
            times = []
            for i in range(self.warmup_iterations + self.num_iterations):
                in_layer.set_input(torch.randn(*input_shape))
                
                start = time.perf_counter()
                conv_layer.forward()
                end = time.perf_counter()
                
                if i >= self.warmup_iterations:
                    times.append((end - start) * 1000)
                    stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
                    self._append_runtime_stats(name, stats)
            
            mem = calc_layer_memory_from_shapes("Conv2d", input_shape, output_shape, kernel_size=k)
            if dependencies is None:
                dependencies = [self.last_layer_name] if self.last_layer_name else []
            
            m = LayerMetrics(name, "Conv2d", group, "Enclave", 
                            input_shape=input_shape, output_shape=output_shape,
                            input_bytes=shape_to_bytes(input_shape), output_bytes=shape_to_bytes(output_shape),
                            dependencies=dependencies,
                            enclave_times=times, num_iterations=self.num_iterations,
                            enclave_get_ms=self._runtime_stats[name]['get_ms'],
                            enclave_get2_ms=self._runtime_stats[name]['get2_ms'],
                            enclave_compute_ms=self._runtime_stats[name]['compute_ms'],
                            enclave_store_ms=self._runtime_stats[name]['store_ms'],
                            **mem)
            m.compute_statistics()
            self.metrics[name] = m
            self.last_layer_name = name

        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()

    def _profile_bn_enclave(self, name, input_shape, group="Feature", dependencies=None, verbose=True):
        self._init_runtime_bucket(name)
        sid = 0
        if verbose:
            print(f"Profiling {name:20} | In: {str(input_shape):16}")

        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()

            in_layer = SecretInputLayer(sid, f"{name}_in", input_shape, ExecutionModeOptions.Enclave)
            bn_layer = SecretBatchNorm2dLayer(sid, name, ExecutionModeOptions.Enclave)
            bn_layer.register_prev_layer(in_layer)
            
            self._layer_pool.extend([in_layer, bn_layer])
            self._setup_layers([in_layer, bn_layer])
            
            times = []
            for i in range(self.warmup_iterations + self.num_iterations):
                in_layer.set_input(torch.randn(*input_shape))
                
                start = time.perf_counter()
                bn_layer.forward()
                end = time.perf_counter()
                
                if i >= self.warmup_iterations:
                    times.append((end - start) * 1000)
                    stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
                    self._append_runtime_stats(name, stats)
            
            mem = calc_layer_memory_from_shapes("BatchNorm", input_shape, input_shape)
            if dependencies is None:
                dependencies = [self.last_layer_name] if self.last_layer_name else []

            m = LayerMetrics(name, "BatchNorm", group, "Enclave", 
                            input_shape=input_shape, output_shape=input_shape,
                            input_bytes=shape_to_bytes(input_shape), output_bytes=shape_to_bytes(input_shape),
                            dependencies=dependencies,
                            enclave_times=times, num_iterations=self.num_iterations,
                            enclave_get_ms=self._runtime_stats[name]['get_ms'],
                            enclave_get2_ms=self._runtime_stats[name]['get2_ms'],
                            enclave_compute_ms=self._runtime_stats[name]['compute_ms'],
                            enclave_store_ms=self._runtime_stats[name]['store_ms'],
                            **mem)
            m.compute_statistics()
            self.metrics[name] = m
            self.last_layer_name = name

        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()

    def _profile_relu_enclave(self, name, input_shape, group="Feature", dependencies=None, verbose=True):
        self._init_runtime_bucket(name)
        sid = 0
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()

            in_layer = SecretInputLayer(sid, f"{name}_in", input_shape, ExecutionModeOptions.Enclave)
            relu_layer = SecretReLULayer(sid, name, ExecutionModeOptions.Enclave)
            relu_layer.register_prev_layer(in_layer)
            
            self._layer_pool.extend([in_layer, relu_layer])
            self._setup_layers([in_layer, relu_layer])
            
            times = []
            for i in range(self.warmup_iterations + self.num_iterations):
                in_layer.set_input(torch.randn(*input_shape))
                start = time.perf_counter()
                relu_layer.forward()
                end = time.perf_counter()
                if i >= self.warmup_iterations:
                    times.append((end - start) * 1000)
                    stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
                    self._append_runtime_stats(name, stats)
            
            mem = calc_layer_memory_from_shapes("ReLU", input_shape, input_shape)
            if dependencies is None:
                dependencies = [self.last_layer_name] if self.last_layer_name else []

            m = LayerMetrics(name, "ReLU", group, "Enclave", 
                            input_shape=input_shape, output_shape=input_shape,
                            input_bytes=shape_to_bytes(input_shape), output_bytes=shape_to_bytes(input_shape),
                            dependencies=dependencies,
                            enclave_times=times, num_iterations=self.num_iterations,
                            enclave_get_ms=self._runtime_stats[name]['get_ms'],
                            enclave_get2_ms=self._runtime_stats[name]['get2_ms'],
                            enclave_compute_ms=self._runtime_stats[name]['compute_ms'],
                            enclave_store_ms=self._runtime_stats[name]['store_ms'],
                            **mem)
            m.compute_statistics()
            self.metrics[name] = m
            self.last_layer_name = name

        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()

    def _profile_maxpool_enclave(self, name, input_shape, k, s, p, group="Feature", dependencies=None, verbose=True):
        self._init_runtime_bucket(name)
        sid = 0
        
        h_out = (input_shape[2] + 2*p - k) // s + 1
        output_shape = [input_shape[0], input_shape[1], h_out, h_out]

        # MaxPool in this codebase typically runs on CPU (modeled as non-enclave)
        in_layer = SecretInputLayer(sid, f"{name}_in", input_shape, ExecutionModeOptions.CPU)
        pool_layer = SecretMaxpool2dLayer(sid, name, ExecutionModeOptions.CPU, filter_hw=k, stride=s, padding=p)
        pool_layer.register_prev_layer(in_layer)
        
        self._layer_pool.extend([in_layer, pool_layer])
        self._setup_layers([in_layer, pool_layer])
        
        times = []
        for i in range(self.warmup_iterations + self.num_iterations):
            in_layer.set_input(torch.randn(*input_shape))
            start = time.perf_counter()
            pool_layer.forward()
            end = time.perf_counter()
            if i >= self.warmup_iterations:
                times.append((end - start) * 1000)
        
        mem = calc_layer_memory_from_shapes("MaxPool", input_shape, output_shape)
        if dependencies is None:
            dependencies = [self.last_layer_name] if self.last_layer_name else []

        m = LayerMetrics(name, "MaxPool", group, "CPU", 
                        input_shape=input_shape, output_shape=output_shape,
                        input_bytes=shape_to_bytes(input_shape), output_bytes=shape_to_bytes(output_shape),
                        dependencies=dependencies,
                        enclave_times=times, num_iterations=self.num_iterations,
                        **mem)
        m.compute_statistics()
        self.metrics[name] = m
        self.last_layer_name = name

    def _profile_add_enclave(self, name, input_shape, group="Feature", dependencies=None, verbose=True):
        self._init_runtime_bucket(name)
        sid = 0
        if verbose:
            print(f"Profiling {name:20} | In: {str(input_shape):16}")

        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()

            in1 = SecretInputLayer(sid, f"{name}_in1", input_shape, ExecutionModeOptions.Enclave)
            in2 = SecretInputLayer(sid, f"{name}_in2", input_shape, ExecutionModeOptions.Enclave)
            add_layer = SecretAddLayer(sid, name, ExecutionModeOptions.Enclave)
            add_layer.register_prev_layer(in1)
            add_layer.register_prev_layer(in2)
            
            self._layer_pool.extend([in1, in2, add_layer])
            self._setup_layers([in1, in2, add_layer])
            
            times = []
            for i in range(self.warmup_iterations + self.num_iterations):
                in1.set_input(torch.randn(*input_shape))
                in2.set_input(torch.randn(*input_shape))
                
                start = time.perf_counter()
                add_layer.forward()
                end = time.perf_counter()
                
                if i >= self.warmup_iterations:
                    times.append((end - start) * 1000)
                    stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
                    self._append_runtime_stats(name, stats)
            
            mem = calc_layer_memory_from_shapes("Add", input_shape, input_shape)
            # Add normally has its own set of dependencies passed in for ResNet
            if dependencies is None:
                dependencies = [self.last_layer_name] if self.last_layer_name else []

            m = LayerMetrics(name, "Add", group, "Enclave", 
                            input_shape=input_shape, output_shape=input_shape,
                            input_bytes=shape_to_bytes(input_shape), output_bytes=shape_to_bytes(input_shape),
                            dependencies=dependencies,
                            enclave_times=times, num_iterations=self.num_iterations,
                            enclave_get_ms=self._runtime_stats[name]['get_ms'],
                            enclave_get2_ms=self._runtime_stats[name]['get2_ms'],
                            enclave_compute_ms=self._runtime_stats[name]['compute_ms'],
                            enclave_store_ms=self._runtime_stats[name]['store_ms'],
                            **mem)
            m.compute_statistics()
            self.metrics[name] = m
            self.last_layer_name = name

        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()

    def _profile_avgpool_enclave(self, name, input_shape, k, s, p, group="Feature", dependencies=None, verbose=True):
        self._init_runtime_bucket(name)
        sid = 0
        if verbose:
            print(f"Profiling {name:20} | In: {str(input_shape):16} | K: {k} S: {s} P: {p}")

        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()

            h_out = (input_shape[2] + 2*p - k) // s + 1
            output_shape = [input_shape[0], input_shape[1], h_out, h_out]

            in_layer = SecretInputLayer(sid, f"{name}_in", input_shape, ExecutionModeOptions.Enclave)
            avg_layer = SecretAvgpool2dLayer(sid, name, ExecutionModeOptions.Enclave, filter_hw=k, stride=s, padding=p)
            avg_layer.register_prev_layer(in_layer)
            
            self._layer_pool.extend([in_layer, avg_layer])
            self._setup_layers([in_layer, avg_layer])
            
            times = []
            for i in range(self.warmup_iterations + self.num_iterations):
                in_layer.set_input(torch.randn(*input_shape))
                
                start = time.perf_counter()
                avg_layer.forward()
                end = time.perf_counter()
                
                if i >= self.warmup_iterations:
                    times.append((end - start) * 1000)
                    stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
                    self._append_runtime_stats(name, stats)
            
            mem = calc_layer_memory_from_shapes("AvgPool", input_shape, output_shape)
            if dependencies is None:
                dependencies = [self.last_layer_name] if self.last_layer_name else []

            m = LayerMetrics(name, "AvgPool", group, "Enclave", 
                            input_shape=input_shape, output_shape=output_shape,
                            input_bytes=shape_to_bytes(input_shape), output_bytes=shape_to_bytes(output_shape),
                            dependencies=dependencies,
                            enclave_times=times, num_iterations=self.num_iterations,
                            enclave_get_ms=self._runtime_stats[name]['get_ms'],
                            enclave_get2_ms=self._runtime_stats[name]['get2_ms'],
                            enclave_compute_ms=self._runtime_stats[name]['compute_ms'],
                            enclave_store_ms=self._runtime_stats[name]['store_ms'],
                            **mem)
            m.compute_statistics()
            self.metrics[name] = m
            self.last_layer_name = name

        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()

    def _profile_linear_enclave(self, name, input_shape, out_features, group="Classifier", dependencies=None, verbose=True):
        self._init_runtime_bucket(name)
        sid = 0
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            output_shape = [input_shape[0], out_features]

            in_layer = SecretInputLayer(sid, f"{name}_in", input_shape, ExecutionModeOptions.Enclave)
            fc_layer = SGXLinearBase(
                sid, name, ExecutionModeOptions.Enclave,
                batch_size=input_shape[0], n_output_features=out_features, n_input_features=input_shape[1]
            )
            fc_layer.register_prev_layer(in_layer)
            
            self._layer_pool.extend([in_layer, fc_layer])
            self._setup_layers([in_layer, fc_layer])
            
            times = []
            for i in range(self.warmup_iterations + self.num_iterations):
                in_layer.set_input(torch.randn(*input_shape))
                start = time.perf_counter()
                fc_layer.forward()
                end = time.perf_counter()
                if i >= self.warmup_iterations:
                    times.append((end - start) * 1000)
                    stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
                    self._append_runtime_stats(name, stats)
            
            mem = calc_layer_memory_from_shapes("Linear", input_shape, output_shape)
            if dependencies is None:
                dependencies = [self.last_layer_name] if self.last_layer_name else []

            m = LayerMetrics(name, "Linear", group, "Enclave", 
                            input_shape=input_shape, output_shape=output_shape,
                            input_bytes=shape_to_bytes(input_shape), output_bytes=shape_to_bytes(output_shape),
                            dependencies=dependencies,
                            enclave_times=times, num_iterations=self.num_iterations,
                            enclave_get_ms=self._runtime_stats[name]['get_ms'],
                            enclave_get2_ms=self._runtime_stats[name]['get2_ms'],
                            enclave_compute_ms=self._runtime_stats[name]['compute_ms'],
                            enclave_store_ms=self._runtime_stats[name]['store_ms'],
                            **mem)
            m.compute_statistics()
            self.metrics[name] = m
            self.last_layer_name = name

        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()

    def profile_resnet34(self, verbose=True):
        print("\n" + "="*60)
        print("Profiling ResNet34 Unique Layers")
        print("="*60)
        self.last_layer_name = None
        
        # Initial layers
        self._profile_conv_enclave("R34_Conv1", [1, 3, 224, 224], 64, 7, 2, 3, verbose=verbose)
        self._profile_bn_enclave("R34_BN1", [1, 64, 112, 112], verbose=verbose)
        self._profile_relu_enclave("R34_ReLU1", [1, 64, 112, 112], verbose=verbose)
        self._profile_maxpool_enclave("R34_MaxPool", [1, 64, 112, 112], 3, 2, 1, verbose=verbose)

        # Basic Stage 1 (56x56x64) - Identity is R34_MaxPool
        identity = "R34_MaxPool"
        self._profile_conv_enclave("R34_S1_Conv3x3", [1, 64, 56, 56], 64, 3, 1, 1, group="S1", verbose=verbose)
        self._profile_bn_enclave("R34_S1_BN", [1, 64, 56, 56], group="S1", verbose=verbose)
        self._profile_add_enclave("R34_S1_Add", [1, 64, 56, 56], group="S1", dependencies=["R34_S1_BN", identity], verbose=verbose)

        # Basic Stage 2 (56->28x28x128) - Identity is R34_S1_Add
        identity = "R34_S1_Add"
        self._profile_conv_enclave("R34_S2_Conv3x3_S2", [1, 64, 56, 56], 128, 3, 2, 1, group="S2", verbose=verbose)
        # Downsample branch
        self._profile_conv_enclave("R34_S2_Conv1x1_S2", [1, 64, 56, 56], 128, 1, 2, 0, group="S2", dependencies=[identity], verbose=verbose)
        self._profile_bn_enclave("R34_S2_BN", [1, 128, 28, 28], group="S2", dependencies=["R34_S2_Conv3x3_S2"], verbose=verbose)
        self._profile_add_enclave("R34_S2_Add", [1, 128, 28, 28], group="S2", dependencies=["R34_S2_BN", "R34_S2_Conv1x1_S2"], verbose=verbose)

        # Final layers
        self._profile_avgpool_enclave("R34_AvgPool", [1, 512, 7, 7], 7, 1, 0, verbose=verbose)
        self._profile_linear_enclave("R34_FC", [1, 512], 1000, verbose=verbose)

    def profile_resnet50(self, verbose=True):
        print("\n" + "="*60)
        print("Profiling ResNet50 Unique Layers")
        print("="*60)
        self.last_layer_name = None
        
        # Initial layers (same as R34 for conv1)
        self._profile_conv_enclave("R50_Conv1", [1, 3, 224, 224], 64, 7, 2, 3, verbose=verbose)
        self._profile_maxpool_enclave("R50_MaxPool", [1, 64, 112, 112], 3, 2, 1, verbose=verbose)
        
        # Bottleneck Stage 1 (56x56) - Identity is R50_MaxPool
        identity = "R50_MaxPool"
        # 1x1, 64 -> 3x3, 64 -> 1x1, 256
        self._profile_conv_enclave("R50_S1_Conv1x1_In", [1, 64, 56, 56], 64, 1, 1, 0, group="S1", verbose=verbose)
        self._profile_conv_enclave("R50_S1_Conv3x3", [1, 64, 56, 56], 64, 3, 1, 1, group="S1", verbose=verbose)
        self._profile_conv_enclave("R50_S1_Conv1x1_Out", [1, 64, 56, 56], 256, 1, 1, 0, group="S1", verbose=verbose)
        self._profile_add_enclave("R50_S1_Add", [1, 256, 56, 56], group="S1", dependencies=["R50_S1_Conv1x1_Out", identity], verbose=verbose)

        # Bottleneck Stage 2 (56->28) - Identity is R50_S1_Add
        identity = "R50_S1_Add"
        self._profile_conv_enclave("R50_S2_Conv1x1_In", [1, 256, 56, 56], 128, 1, 1, 0, group="S2", verbose=verbose)
        self._profile_conv_enclave("R50_S2_Conv3x3_S2", [1, 128, 56, 56], 128, 3, 2, 1, group="S2", verbose=verbose)
        self._profile_conv_enclave("R50_S2_Conv1x1_Out", [1, 128, 28, 28], 512, 1, 1, 0, group="S2", verbose=verbose)
        self._profile_conv_enclave("R50_S2_DS_Conv1x1", [1, 256, 56, 56], 512, 1, 2, 0, group="S2", dependencies=[identity], verbose=verbose) # Downsample
        self._profile_add_enclave("R50_S2_Add", [1, 512, 28, 28], group="S2", dependencies=["R50_S2_Conv1x1_Out", "R50_S2_DS_Conv1x1"], verbose=verbose)

        # Final layers
        self._profile_avgpool_enclave("R50_AvgPool", [1, 2048, 7, 7], 7, 1, 0, verbose=verbose)
        self._profile_linear_enclave("R50_FC", [1, 2048], 1000, verbose=verbose)

    def print_summary(self):
        print_memory_summary(self.metrics, "ResNet Enclave Memory Summary")

    def save_results(self, output_file: str):
        """Save profiling results to CSV in aligned format."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for m in self.metrics.values():
                writer.writerow(metrics_to_csv_row(m))
        print(f"\nProfiling results saved to {output_file}")

def main():
    print("="*60)
    print("Profiling ResNet in Enclave Mode")
    print("="*60)
    
    try:
        # 1. Global Enclave Initialization
        GlobalTensor.init()
        print("Enclave initialized. Starting profiling...")
        
        # 2. Profile ResNet34
        profiler34 = ResNetEnclaveProfiler(warmup_iterations=2, num_iterations=5)
        profiler34.profile_resnet34()
        profiler34.save_results("experiments/data/resnet34_enclave_layers.csv")
        profiler34.print_summary()
        
        # 3. Profile ResNet50
        profiler50 = ResNetEnclaveProfiler(warmup_iterations=2, num_iterations=5)
        profiler50.profile_resnet50()
        profiler50.save_results("experiments/data/resnet50_enclave_layers.csv")
        profiler50.print_summary()
        
    finally:
        if GlobalTensor.is_init_global_tensor:
            GlobalTensor.destroy()
            print("Enclave destroyed.")

if __name__ == "__main__":
    main()
