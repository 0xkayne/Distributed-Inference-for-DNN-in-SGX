#!/usr/bin/env python
"""
VGG16 Enclave Performance Profiler.

VGG16 has 13 conv layers + 3 FC layers with 224x224 input.
Large feature maps and FC weights require SGX2 EDMM for dynamic heap growth;
without EDMM the full 2GB heap commit at startup can exhaust EPC and crash.

Output: experiments/data/vgg16_enclave_layers.csv

Usage:
    python -m experiments.models.profile_vgg16_enclave
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

import torch

from experiments.models.profiler_utils import (
    LayerMetrics,
    calc_layer_memory_from_shapes,
    shape_to_bytes,
    print_memory_summary,
    CSV_FIELDNAMES,
    metrics_to_csv_row
)
from python.enclave_interfaces import GlobalTensor
from python.layers.input import SecretInputLayer
from python.layers.sgx_conv_base import SGXConvBase
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.relu import SecretReLULayer
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.utils.basic_utils import ExecutionModeOptions


def _shape_to_bytes(shape: List[int], dtype_size: int = 4) -> int:
    return int(np.prod(shape)) * dtype_size


class VGG16EnclaveProfiler:
    """
    VGG16 layer-by-layer enclave profiler.

    Architecture (standard VGG-D):
      Block 1: Conv3-64, Conv3-64, MaxPool
      Block 2: Conv3-128, Conv3-128, MaxPool
      Block 3: Conv3-256, Conv3-256, Conv3-256, MaxPool
      Block 4: Conv3-512, Conv3-512, Conv3-512, MaxPool
      Block 5: Conv3-512, Conv3-512, Conv3-512, MaxPool
      Classifier: FC-4096, FC-4096, FC-1000
    """

    def __init__(self, batch_size=1, num_iterations=5, warmup_iterations=2):
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        self.metrics: Dict[str, LayerMetrics] = OrderedDict()
        self._runtime_stats: Dict[str, Dict[str, List[float]]] = {}
        self._layer_pool: List[Any] = []
        self.last_layer_name = None

    # ── helpers ────────────────────────────────────────────────

    def _init_runtime_bucket(self, name):
        self._runtime_stats[name] = {
            'get_ms': [], 'get2_ms': [], 'compute_ms': [], 'store_ms': []
        }

    def _append_runtime_stats(self, name, stats):
        for k in ['get_ms', 'get2_ms', 'compute_ms', 'store_ms']:
            self._runtime_stats[name][k].append(stats.get(k, 0.0))

    def _setup_layers(self, layers):
        eid = GlobalTensor.get_eid()
        for layer in layers:
            layer.set_eid(eid)
            layer.init_shape()
            layer.link_tensors()
        for layer in layers:
            layer.init(start_enclave=False)

    # ── per-type profiling ────────────────────────────────────

    def _profile_conv(self, name, input_shape, out_ch, k, s, p, group,
                      verbose, dependencies=None):
        sid = 0
        self._init_runtime_bucket(name)

        if not GlobalTensor.is_init_global_tensor:
            GlobalTensor.init()

        h_out = (input_shape[2] + 2 * p - k) // s + 1
        output_shape = [input_shape[0], out_ch, h_out, h_out]

        in_layer = SecretInputLayer(sid, f"{name}_in", input_shape,
                                    ExecutionModeOptions.Enclave)
        conv_layer = SGXConvBase(
            sid, name, ExecutionModeOptions.Enclave,
            n_output_channel=out_ch, n_input_channel=input_shape[1],
            filter_hw=k, stride=s, padding=p,
            batch_size=input_shape[0], img_hw=input_shape[2])
        conv_layer.register_prev_layer(in_layer)
        self._layer_pool.extend([in_layer, conv_layer])
        self._setup_layers([in_layer, conv_layer])

        times = []
        for i in range(self.warmup_iterations + self.num_iterations):
            in_layer.set_input(torch.randn(*input_shape))
            t0 = time.perf_counter()
            in_layer.forward()
            conv_layer.forward()
            elapsed = (time.perf_counter() - t0) * 1000
            stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
            if i >= self.warmup_iterations:
                times.append(elapsed)
                self._append_runtime_stats(name, stats)

        mem = calc_layer_memory_from_shapes("Conv2d", input_shape, output_shape,
                                            kernel_size=k)
        if dependencies is None:
            dependencies = [self.last_layer_name] if self.last_layer_name else []
        m = LayerMetrics(
            name, "Conv2d", group, "Enclave",
            input_shape=input_shape, output_shape=output_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(output_shape),
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
        if verbose:
            print(f"  {name:20} {m.enclave_time_mean:8.3f} ms  "
                  f"in={input_shape}  out={output_shape}")

    def _profile_relu(self, name, input_shape, group, verbose,
                      dependencies=None):
        sid = 0
        self._init_runtime_bucket(name)

        if not GlobalTensor.is_init_global_tensor:
            GlobalTensor.init()

        in_layer = SecretInputLayer(sid, f"{name}_in", input_shape,
                                    ExecutionModeOptions.Enclave)
        relu_layer = SecretReLULayer(sid, name, ExecutionModeOptions.Enclave)
        relu_layer.register_prev_layer(in_layer)
        self._layer_pool.extend([in_layer, relu_layer])
        self._setup_layers([in_layer, relu_layer])

        times = []
        for i in range(self.warmup_iterations + self.num_iterations):
            in_layer.set_input(torch.randn(*input_shape))
            t0 = time.perf_counter()
            in_layer.forward()
            relu_layer.forward()
            elapsed = (time.perf_counter() - t0) * 1000
            stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
            if i >= self.warmup_iterations:
                times.append(elapsed)
                self._append_runtime_stats(name, stats)

        mem = calc_layer_memory_from_shapes("ReLU", input_shape, input_shape)
        if dependencies is None:
            dependencies = [self.last_layer_name] if self.last_layer_name else []
        m = LayerMetrics(
            name, "ReLU", group, "Enclave",
            input_shape=input_shape, output_shape=input_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(input_shape),
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
        if verbose:
            print(f"  {name:20} {m.enclave_time_mean:8.3f} ms")

    def _profile_maxpool(self, name, input_shape, k, s, p, group, verbose,
                         dependencies=None):
        sid = 0
        self._init_runtime_bucket(name)
        h_out = (input_shape[2] + 2 * p - k) // s + 1
        output_shape = [input_shape[0], input_shape[1], h_out, h_out]

        in_layer = SecretInputLayer(sid, f"{name}_in", input_shape,
                                    ExecutionModeOptions.CPU)
        pool_layer = SecretMaxpool2dLayer(sid, name, ExecutionModeOptions.CPU,
                                         filter_hw=k, stride=s, padding=p)
        pool_layer.register_prev_layer(in_layer)
        self._layer_pool.extend([in_layer, pool_layer])
        self._setup_layers([in_layer, pool_layer])

        times = []
        for i in range(self.warmup_iterations + self.num_iterations):
            in_layer.set_input(torch.randn(*input_shape))
            t0 = time.perf_counter()
            in_layer.forward()
            pool_layer.forward()
            elapsed = (time.perf_counter() - t0) * 1000
            if i >= self.warmup_iterations:
                times.append(elapsed)

        mem = calc_layer_memory_from_shapes("MaxPool", input_shape, output_shape)
        if dependencies is None:
            dependencies = [self.last_layer_name] if self.last_layer_name else []
        m = LayerMetrics(
            name, "MaxPool", group, "CPU",
            input_shape=input_shape, output_shape=output_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(output_shape),
            dependencies=dependencies,
            enclave_times=times, num_iterations=self.num_iterations,
            **mem)
        m.compute_statistics()
        self.metrics[name] = m
        self.last_layer_name = name
        if verbose:
            print(f"  {name:20} {m.enclave_time_mean:8.3f} ms (CPU)")

    def _profile_linear(self, name, input_shape, out_features, group, verbose,
                        dependencies=None):
        sid = 0
        self._init_runtime_bucket(name)

        if not GlobalTensor.is_init_global_tensor:
            GlobalTensor.init()

        output_shape = [input_shape[0], out_features]

        in_layer = SecretInputLayer(sid, f"{name}_in", input_shape,
                                    ExecutionModeOptions.Enclave)
        fc_layer = SGXLinearBase(
            sid, name, ExecutionModeOptions.Enclave,
            batch_size=input_shape[0],
            n_output_features=out_features,
            n_input_features=input_shape[1])
        fc_layer.register_prev_layer(in_layer)
        self._layer_pool.extend([in_layer, fc_layer])
        self._setup_layers([in_layer, fc_layer])

        times = []
        for i in range(self.warmup_iterations + self.num_iterations):
            in_layer.set_input(torch.randn(*input_shape))
            t0 = time.perf_counter()
            in_layer.forward()
            fc_layer.forward()
            elapsed = (time.perf_counter() - t0) * 1000
            stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
            if i >= self.warmup_iterations:
                times.append(elapsed)
                self._append_runtime_stats(name, stats)

        mem = calc_layer_memory_from_shapes("Linear", input_shape, output_shape)
        if dependencies is None:
            dependencies = [self.last_layer_name] if self.last_layer_name else []
        m = LayerMetrics(
            name, "Linear", group, "Enclave",
            input_shape=input_shape, output_shape=output_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(output_shape),
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
        if verbose:
            print(f"  {name:20} {m.enclave_time_mean:8.3f} ms  "
                  f"in={input_shape}  out={output_shape}")

    # ── main profiling entry ──────────────────────────────────

    def profile_all(self, verbose=True):
        if verbose:
            print(f"\n{'='*70}")
            print("  VGG16 Enclave Layer Profiling  (requires SGX2 EDMM)")
            print(f"{'='*70}")

        B = self.batch_size

        try:
            GlobalTensor.init()

            # ── Block 1: 224 → 112 ──
            if verbose: print("\n  [Block 1]  3→64, 224x224")
            self._profile_conv("conv1_1", [B, 3, 224, 224],   64, 3, 1, 1, "Block1", verbose, dependencies=[])
            self._profile_relu("relu1_1", [B, 64, 224, 224],  "Block1", verbose)
            self._profile_conv("conv1_2", [B, 64, 224, 224],  64, 3, 1, 1, "Block1", verbose)
            self._profile_relu("relu1_2", [B, 64, 224, 224],  "Block1", verbose)
            self._profile_maxpool("pool1", [B, 64, 224, 224], 2, 2, 0, "Block1", verbose)

            # ── Block 2: 112 → 56 ──
            if verbose: print("\n  [Block 2]  64→128, 112x112")
            self._profile_conv("conv2_1", [B, 64, 112, 112],  128, 3, 1, 1, "Block2", verbose)
            self._profile_relu("relu2_1", [B, 128, 112, 112], "Block2", verbose)
            self._profile_conv("conv2_2", [B, 128, 112, 112], 128, 3, 1, 1, "Block2", verbose)
            self._profile_relu("relu2_2", [B, 128, 112, 112], "Block2", verbose)
            self._profile_maxpool("pool2", [B, 128, 112, 112], 2, 2, 0, "Block2", verbose)

            # ── Block 3: 56 → 28 ──
            if verbose: print("\n  [Block 3]  128→256, 56x56")
            self._profile_conv("conv3_1", [B, 128, 56, 56],  256, 3, 1, 1, "Block3", verbose)
            self._profile_relu("relu3_1", [B, 256, 56, 56],  "Block3", verbose)
            self._profile_conv("conv3_2", [B, 256, 56, 56],  256, 3, 1, 1, "Block3", verbose)
            self._profile_relu("relu3_2", [B, 256, 56, 56],  "Block3", verbose)
            self._profile_conv("conv3_3", [B, 256, 56, 56],  256, 3, 1, 1, "Block3", verbose)
            self._profile_relu("relu3_3", [B, 256, 56, 56],  "Block3", verbose)
            self._profile_maxpool("pool3", [B, 256, 56, 56],  2, 2, 0, "Block3", verbose)

            # ── Block 4: 28 → 14 ──
            if verbose: print("\n  [Block 4]  256→512, 28x28")
            self._profile_conv("conv4_1", [B, 256, 28, 28],  512, 3, 1, 1, "Block4", verbose)
            self._profile_relu("relu4_1", [B, 512, 28, 28],  "Block4", verbose)
            self._profile_conv("conv4_2", [B, 512, 28, 28],  512, 3, 1, 1, "Block4", verbose)
            self._profile_relu("relu4_2", [B, 512, 28, 28],  "Block4", verbose)
            self._profile_conv("conv4_3", [B, 512, 28, 28],  512, 3, 1, 1, "Block4", verbose)
            self._profile_relu("relu4_3", [B, 512, 28, 28],  "Block4", verbose)
            self._profile_maxpool("pool4", [B, 512, 28, 28],  2, 2, 0, "Block4", verbose)

            # ── Block 5: 14 → 7 ──
            if verbose: print("\n  [Block 5]  512→512, 14x14")
            self._profile_conv("conv5_1", [B, 512, 14, 14],  512, 3, 1, 1, "Block5", verbose)
            self._profile_relu("relu5_1", [B, 512, 14, 14],  "Block5", verbose)
            self._profile_conv("conv5_2", [B, 512, 14, 14],  512, 3, 1, 1, "Block5", verbose)
            self._profile_relu("relu5_2", [B, 512, 14, 14],  "Block5", verbose)
            self._profile_conv("conv5_3", [B, 512, 14, 14],  512, 3, 1, 1, "Block5", verbose)
            self._profile_relu("relu5_3", [B, 512, 14, 14],  "Block5", verbose)
            self._profile_maxpool("pool5", [B, 512, 14, 14],  2, 2, 0, "Block5", verbose)

            # ── Classifier ──
            # Reset enclave before large FC layers to reclaim EPC
            if verbose: print("\n  [Classifier]  Resetting enclave for FC layers...")
            GlobalTensor.destroy()
            GlobalTensor.init()

            # pool5 output: 512*7*7 = 25088
            self._profile_linear("fc1", [B, 25088], 4096, "Classifier", verbose,
                                 dependencies=["pool5"])
            self._profile_relu("relu_fc1", [B, 4096], "Classifier", verbose)

            GlobalTensor.destroy()
            GlobalTensor.init()

            self._profile_linear("fc2", [B, 4096], 4096, "Classifier", verbose,
                                 dependencies=["relu_fc1"])
            self._profile_relu("relu_fc2", [B, 4096], "Classifier", verbose)

            self._profile_linear("fc3", [B, 4096], 1000, "Classifier", verbose,
                                 dependencies=["relu_fc2"])

            return self.metrics

        finally:
            if GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()

    def save_results(self, csv_path, json_path=None):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for m in self.metrics.values():
                writer.writerow(metrics_to_csv_row(m))
        print(f"\nCSV saved to {csv_path}")

        if json_path is None:
            json_path = csv_path.replace('.csv', '.json')
        data = [m.to_dict() for m in self.metrics.values()]
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"JSON saved to {json_path}")

    def print_summary(self):
        print_memory_summary(self.metrics, "VGG16 Enclave Memory Summary")


def main():
    parser = argparse.ArgumentParser(description="VGG16 Enclave Profiler (SGX2 EDMM)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    profiler = VGG16EnclaveProfiler(
        batch_size=args.batch_size,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
    )
    profiler.profile_all(verbose=True)
    profiler.save_results("experiments/data/vgg16_enclave_layers.csv")
    profiler.print_summary()


if __name__ == "__main__":
    main()
