#!/usr/bin/env python
"""
ResNeXt-50 32×4d Enclave Performance Profiler.

ResNeXt-50 32×4d uses grouped convolution (groups=32) in its 3×3 bottleneck
layers. This requires the grouped-conv support added to the full SGX stack.

Architecture (per bottleneck):
  1×1 conv (reduce)  →  3×3 conv groups=32 (transform)  →  1×1 conv (expand)

Stages:
  - Conv1: 7×7, 64, stride 2
  - Stage 1: 3 bottleneck blocks (256-wide, group_width=4, 56×56)
  - Stage 2: 4 bottleneck blocks (512-wide, group_width=4, 28×28)
  - Stage 3: 6 bottleneck blocks (1024-wide, group_width=4, 14×14)
  - Stage 4: 3 bottleneck blocks (2048-wide, group_width=4, 7×7)
  - AvgPool + FC-1000

Output: experiments/data/resnext50_32x4d_enclave_layers.csv

Usage:
    python -m experiments.models.profile_resnext50_enclave
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
from python.layers.batch_norm_2d import SecretBatchNorm2dLayer
from python.layers.add import SecretAddLayer
from python.layers.avgpool2d import SecretAvgpool2dLayer
from python.utils.basic_utils import ExecutionModeOptions


def _sb(shape, dtype_size=4):
    return int(np.prod(shape)) * dtype_size


class ResNeXt50EnclaveProfiler:
    def __init__(self, batch_size=1, num_iterations=5, warmup_iterations=2):
        self.B = batch_size
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        self.metrics: Dict[str, LayerMetrics] = OrderedDict()
        self._rs: Dict[str, Dict[str, List[float]]] = {}
        self._pool: List[Any] = []
        self.last = None

    # ── helpers ────────────────────────────────────────────
    def _ib(self, n):
        self._rs[n] = {'get_ms': [], 'get2_ms': [], 'compute_ms': [], 'store_ms': []}

    def _ar(self, n, s):
        for k in ['get_ms', 'get2_ms', 'compute_ms', 'store_ms']:
            self._rs[n][k].append(s.get(k, 0.0))

    def _setup(self, layers):
        eid = GlobalTensor.get_eid()
        for l in layers:
            l.set_eid(eid); l.init_shape(); l.link_tensors()
        for l in layers:
            l.init(start_enclave=False)

    def _save(self, n, ltype, group, mode, in_s, out_s, times, deps, k=3):
        mem = calc_layer_memory_from_shapes(ltype, in_s, out_s, kernel_size=k)
        if deps is None:
            deps = [self.last] if self.last else []
        kw = {}
        if n in self._rs:
            kw = dict(enclave_get_ms=self._rs[n]['get_ms'],
                      enclave_get2_ms=self._rs[n]['get2_ms'],
                      enclave_compute_ms=self._rs[n]['compute_ms'],
                      enclave_store_ms=self._rs[n]['store_ms'])
        m = LayerMetrics(n, ltype, group, mode,
                         input_shape=in_s, output_shape=out_s,
                         input_bytes=_sb(in_s), output_bytes=_sb(out_s),
                         dependencies=deps, enclave_times=times,
                         num_iterations=self.num_iterations, **mem, **kw)
        m.compute_statistics()
        self.metrics[n] = m
        self.last = n
        return m

    # ── per-type profiling ────────────────────────────────
    def _conv(self, name, in_s, out_ch, k, s, p, group, verbose, deps=None, groups=1):
        sid = 0; self._ib(name)
        if not GlobalTensor.is_init_global_tensor: GlobalTensor.init()
        h_o = (in_s[2]+2*p-k)//s+1
        out_s = [in_s[0], out_ch, h_o, h_o]
        il = SecretInputLayer(sid, f"{name}_in", in_s, ExecutionModeOptions.Enclave)
        cl = SGXConvBase(sid, name, ExecutionModeOptions.Enclave,
                         n_output_channel=out_ch, n_input_channel=in_s[1],
                         filter_hw=k, stride=s, padding=p,
                         batch_size=in_s[0], img_hw=in_s[2], groups=groups)
        cl.register_prev_layer(il)
        self._pool.extend([il, cl]); self._setup([il, cl])
        times = []
        for i in range(self.warmup_iterations + self.num_iterations):
            il.set_input(torch.randn(*in_s))
            t0 = time.perf_counter(); il.forward(); cl.forward()
            e = (time.perf_counter()-t0)*1000
            st = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
            if i >= self.warmup_iterations: times.append(e); self._ar(name, st)
        m = self._save(name, "Conv2d", group, "Enclave", in_s, out_s, times, deps, k)
        g_str = f" g={groups}" if groups > 1 else ""
        if verbose: print(f"  {name:28} {m.enclave_time_mean:8.3f} ms  {in_s} → {out_s}{g_str}")

    def _bn(self, name, in_s, group, verbose, deps=None):
        sid = 0; self._ib(name)
        if not GlobalTensor.is_init_global_tensor: GlobalTensor.init()
        il = SecretInputLayer(sid, f"{name}_in", in_s, ExecutionModeOptions.Enclave)
        bl = SecretBatchNorm2dLayer(sid, name, ExecutionModeOptions.Enclave)
        bl.register_prev_layer(il); self._pool.extend([il, bl]); self._setup([il, bl])
        times = []
        for i in range(self.warmup_iterations + self.num_iterations):
            il.set_input(torch.randn(*in_s))
            t0 = time.perf_counter(); il.forward(); bl.forward()
            e = (time.perf_counter()-t0)*1000
            st = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
            if i >= self.warmup_iterations: times.append(e); self._ar(name, st)
        m = self._save(name, "BatchNorm", group, "Enclave", in_s, in_s, times, deps)
        if verbose: print(f"  {name:28} {m.enclave_time_mean:8.3f} ms")

    def _relu(self, name, in_s, group, verbose, deps=None):
        sid = 0; self._ib(name)
        if not GlobalTensor.is_init_global_tensor: GlobalTensor.init()
        il = SecretInputLayer(sid, f"{name}_in", in_s, ExecutionModeOptions.Enclave)
        rl = SecretReLULayer(sid, name, ExecutionModeOptions.Enclave)
        rl.register_prev_layer(il); self._pool.extend([il, rl]); self._setup([il, rl])
        times = []
        for i in range(self.warmup_iterations + self.num_iterations):
            il.set_input(torch.randn(*in_s))
            t0 = time.perf_counter(); il.forward(); rl.forward()
            e = (time.perf_counter()-t0)*1000
            st = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
            if i >= self.warmup_iterations: times.append(e); self._ar(name, st)
        m = self._save(name, "ReLU", group, "Enclave", in_s, in_s, times, deps)
        if verbose: print(f"  {name:28} {m.enclave_time_mean:8.3f} ms")

    def _maxpool(self, name, in_s, k, s, p, group, verbose, deps=None):
        sid = 0; self._ib(name)
        h_o = (in_s[2]+2*p-k)//s+1; out_s = [in_s[0], in_s[1], h_o, h_o]
        il = SecretInputLayer(sid, f"{name}_in", in_s, ExecutionModeOptions.CPU)
        pl = SecretMaxpool2dLayer(sid, name, ExecutionModeOptions.CPU, filter_hw=k, stride=s, padding=p)
        pl.register_prev_layer(il); self._pool.extend([il, pl]); self._setup([il, pl])
        times = []
        for i in range(self.warmup_iterations + self.num_iterations):
            il.set_input(torch.randn(*in_s))
            t0 = time.perf_counter(); il.forward(); pl.forward()
            e = (time.perf_counter()-t0)*1000
            if i >= self.warmup_iterations: times.append(e)
        m = self._save(name, "MaxPool", group, "CPU", in_s, out_s, times, deps)
        if verbose: print(f"  {name:28} {m.enclave_time_mean:8.3f} ms (CPU)")

    def _add(self, name, in_s, group, verbose, deps=None):
        sid = 0; self._ib(name)
        if not GlobalTensor.is_init_global_tensor: GlobalTensor.init()
        i1 = SecretInputLayer(sid, f"{name}_i1", in_s, ExecutionModeOptions.Enclave)
        i2 = SecretInputLayer(sid, f"{name}_i2", in_s, ExecutionModeOptions.Enclave)
        al = SecretAddLayer(sid, name, ExecutionModeOptions.Enclave)
        al.register_prev_layer(i1); al.register_prev_layer(i2)
        self._pool.extend([i1, i2, al]); self._setup([i1, i2, al])
        times = []
        for i in range(self.warmup_iterations + self.num_iterations):
            i1.set_input(torch.randn(*in_s)); i2.set_input(torch.randn(*in_s))
            t0 = time.perf_counter(); al.forward()
            e = (time.perf_counter()-t0)*1000
            st = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
            if i >= self.warmup_iterations: times.append(e); self._ar(name, st)
        m = self._save(name, "Add", group, "Enclave", in_s, in_s, times, deps)
        if verbose: print(f"  {name:28} {m.enclave_time_mean:8.3f} ms")

    def _avgpool(self, name, in_s, k, s, p, group, verbose, deps=None):
        sid = 0; self._ib(name)
        if not GlobalTensor.is_init_global_tensor: GlobalTensor.init()
        h_o = (in_s[2]+2*p-k)//s+1; out_s = [in_s[0], in_s[1], h_o, h_o]
        il = SecretInputLayer(sid, f"{name}_in", in_s, ExecutionModeOptions.Enclave)
        al = SecretAvgpool2dLayer(sid, name, ExecutionModeOptions.Enclave, filter_hw=k, stride=s, padding=p)
        al.register_prev_layer(il); self._pool.extend([il, al]); self._setup([il, al])
        times = []
        for i in range(self.warmup_iterations + self.num_iterations):
            il.set_input(torch.randn(*in_s))
            t0 = time.perf_counter(); il.forward(); al.forward()
            e = (time.perf_counter()-t0)*1000
            st = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
            if i >= self.warmup_iterations: times.append(e); self._ar(name, st)
        m = self._save(name, "AvgPool", group, "Enclave", in_s, out_s, times, deps)
        if verbose: print(f"  {name:28} {m.enclave_time_mean:8.3f} ms")

    def _linear(self, name, in_s, out_f, group, verbose, deps=None):
        sid = 0; self._ib(name)
        if not GlobalTensor.is_init_global_tensor: GlobalTensor.init()
        out_s = [in_s[0], out_f]
        il = SecretInputLayer(sid, f"{name}_in", in_s, ExecutionModeOptions.Enclave)
        fl = SGXLinearBase(sid, name, ExecutionModeOptions.Enclave,
                           batch_size=in_s[0], n_output_features=out_f, n_input_features=in_s[1])
        fl.register_prev_layer(il); self._pool.extend([il, fl]); self._setup([il, fl])
        times = []
        for i in range(self.warmup_iterations + self.num_iterations):
            il.set_input(torch.randn(*in_s))
            t0 = time.perf_counter(); il.forward(); fl.forward()
            e = (time.perf_counter()-t0)*1000
            st = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(name)
            if i >= self.warmup_iterations: times.append(e); self._ar(name, st)
        m = self._save(name, "Linear", group, "Enclave", in_s, out_s, times, deps)
        if verbose: print(f"  {name:28} {m.enclave_time_mean:8.3f} ms  {in_s} → {out_s}")

    # ── bottleneck block profiling ────────────────────────
    def _bottleneck(self, prefix, in_s, mid_ch, out_ch, s, grp, gw, group_label, verbose, identity_name, downsample):
        """
        Profile one ResNeXt bottleneck block.
        mid_ch = groups * group_width (e.g. 32*4=128)
        """
        B = self.B
        h_in = in_s[2]; c_in = in_s[1]
        h_out = h_in // s

        # 1×1 reduce
        self._conv(f"{prefix}_1x1_reduce", in_s, mid_ch, 1, 1, 0, group_label, verbose)
        self._bn(f"{prefix}_bn1", [B, mid_ch, h_in, h_in], group_label, verbose)
        self._relu(f"{prefix}_relu1", [B, mid_ch, h_in, h_in], group_label, verbose)

        # 3×3 grouped conv (the core ResNeXt operation)
        self._conv(f"{prefix}_3x3_grp", [B, mid_ch, h_in, h_in], mid_ch, 3, s, 1,
                   group_label, verbose, groups=grp)
        self._bn(f"{prefix}_bn2", [B, mid_ch, h_out, h_out], group_label, verbose)
        self._relu(f"{prefix}_relu2", [B, mid_ch, h_out, h_out], group_label, verbose)

        # 1×1 expand
        self._conv(f"{prefix}_1x1_expand", [B, mid_ch, h_out, h_out], out_ch, 1, 1, 0,
                   group_label, verbose)
        self._bn(f"{prefix}_bn3", [B, out_ch, h_out, h_out], group_label, verbose)

        # Downsample shortcut if needed
        add_deps = [f"{prefix}_bn3"]
        if downsample:
            self._conv(f"{prefix}_ds_1x1", in_s, out_ch, 1, s, 0, group_label, verbose,
                       deps=[identity_name])
            self._bn(f"{prefix}_ds_bn", [B, out_ch, h_out, h_out], group_label, verbose)
            add_deps.append(f"{prefix}_ds_bn")
        else:
            add_deps.append(identity_name)

        # Residual add + ReLU
        self._add(f"{prefix}_add", [B, out_ch, h_out, h_out], group_label, verbose,
                  deps=add_deps)
        self._relu(f"{prefix}_relu_out", [B, out_ch, h_out, h_out], group_label, verbose)

    # ── main entry ────────────────────────────────────────
    def profile_all(self, verbose=True):
        """
        Profile ResNeXt-50 32×4d representative layers.

        We profile the FIRST block of each stage (which includes downsample)
        to capture the unique layer shapes. Repeated blocks within a stage
        share the same shapes and can be extrapolated.
        """
        B = self.B
        GROUPS = 32
        GW = 4  # group_width

        if verbose:
            print(f"\n{'='*70}")
            print(f"  ResNeXt-50 32×4d Enclave Profiling  (groups={GROUPS}, group_width={GW})")
            print(f"{'='*70}")

        try:
            GlobalTensor.init()

            # ── Stem ──
            if verbose: print("\n  [Stem]")
            self._conv("conv1", [B, 3, 224, 224], 64, 7, 2, 3, "Stem", verbose, deps=[])
            self._bn("bn1", [B, 64, 112, 112], "Stem", verbose)
            self._relu("relu1", [B, 64, 112, 112], "Stem", verbose)
            self._maxpool("maxpool", [B, 64, 112, 112], 3, 2, 1, "Stem", verbose)

            # ── Stage 1: 56×56, mid=128, out=256 ──
            if verbose: print("\n  [Stage 1]  56×56, C=128→256, 3 blocks")
            self._bottleneck("S1_B0", [B, 64, 56, 56], GROUPS*GW, 256, 1,
                             GROUPS, GW, "S1", verbose, "maxpool", downsample=True)

            # ── Stage 2: 56→28, mid=256, out=512 ──
            if verbose: print("\n  [Stage 2]  28×28, C=256→512, 4 blocks")
            self._bottleneck("S2_B0", [B, 256, 56, 56], GROUPS*GW*2, 512, 2,
                             GROUPS, GW, "S2", verbose, "S1_B0_relu_out", downsample=True)

            # ── Stage 3: 28→14, mid=512, out=1024 ──
            if verbose: print("\n  [Stage 3]  14×14, C=512→1024, 6 blocks")
            self._bottleneck("S3_B0", [B, 512, 28, 28], GROUPS*GW*4, 1024, 2,
                             GROUPS, GW, "S3", verbose, "S2_B0_relu_out", downsample=True)

            # ── Stage 4: 14→7, mid=1024, out=2048 ──
            if verbose: print("\n  [Stage 4]  7×7, C=1024→2048, 3 blocks")
            self._bottleneck("S4_B0", [B, 1024, 14, 14], GROUPS*GW*8, 2048, 2,
                             GROUPS, GW, "S4", verbose, "S3_B0_relu_out", downsample=True)

            # ── Classifier ──
            if verbose: print("\n  [Classifier]")
            self._avgpool("avgpool", [B, 2048, 7, 7], 7, 1, 0, "Classifier", verbose)
            self._linear("fc", [B, 2048], 1000, "Classifier", verbose)

            return self.metrics

        finally:
            if GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()

    def save_results(self, csv_path, json_path=None):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            w.writeheader()
            for m in self.metrics.values():
                w.writerow(metrics_to_csv_row(m))
        print(f"\nCSV saved to {csv_path}")
        if json_path is None:
            json_path = csv_path.replace('.csv', '.json')
        with open(json_path, 'w') as f:
            json.dump([m.to_dict() for m in self.metrics.values()], f, indent=2, default=str)
        print(f"JSON saved to {json_path}")

    def print_summary(self):
        print_memory_summary(self.metrics, "ResNeXt-50 32×4d Enclave Memory Summary")


def main():
    parser = argparse.ArgumentParser(description="ResNeXt-50 32×4d Enclave Profiler")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    profiler = ResNeXt50EnclaveProfiler(
        batch_size=args.batch_size,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup)
    profiler.profile_all(verbose=True)
    profiler.save_results("experiments/data/resnext50_32x4d_enclave_layers.csv")
    profiler.print_summary()


if __name__ == "__main__":
    main()
