"""
TinyBERT Performance Profiler for Distributed Inference Modeling.

This script measures:
1. Execution time of each layer in CPU mode (with statistical analysis).
2. Input/Output tensor size of each layer (for communication cost modeling).
3. Layer dependencies (for DAG construction).

Output: experiments/data/tinybert_{4l|6l}_layers.csv

TinyBERT Architecture Summary:
- TinyBERT-4L-312D: 4 layers, 312 hidden, 1200 intermediate (7.5x smaller)
- TinyBERT-6L-768D: 6 layers, 768 hidden, 3072 intermediate

Each Encoder Block (post-norm style):
- Attention (Q/K/V proj, MatMul, Softmax, Out proj) -> Residual -> LayerNorm
- FFN (FC1, GELU, FC2) -> Residual -> LayerNorm

Reference: "TinyBERT: Distilling BERT for Natural Language Understanding"
           (Jiao et al., EMNLP 2020)
"""

import sys
import time
import csv
import json
import numpy as np
import os
from collections import OrderedDict

from typing import Dict, List, Optional, Any
import argparse

sys.path.insert(0, '.')

try:
    import torch
except ImportError:
    torch = None
    print("Warning: PyTorch not available")

from experiments.models.profiler_utils import (
    LayerMetrics,
    calc_layer_memory_from_shapes,
    shape_to_bytes,
    print_memory_summary,
    infer_layer_dependencies,
)
from python.utils.basic_utils import ExecutionModeOptions


# Default measurement parameters
DEFAULT_NUM_ITERATIONS = 30
DEFAULT_WARMUP_ITERATIONS = 5



def _shape_to_bytes(shape: List[int]) -> int:
    """Convert shape to size in bytes (assuming float32)."""
    return shape_to_bytes(shape)


# ============================================================================
# TinyBERT Model Configuration
# ============================================================================

TINYBERT_CONFIG = {
    '4l': {
        'embed_dim': 312,
        'num_heads': 12,
        'num_layers': 4,
        'intermediate_size': 1200,
    },
    '6l': {
        'embed_dim': 768,
        'num_heads': 12,
        'num_layers': 6,
        'intermediate_size': 3072,
    }
}


class TinyBERTPureProfiler:
    """
    Pure PyTorch profiler for TinyBERT models.
    
    Measures CPU execution time for each layer using PyTorch operations only.
    """
    
    def __init__(
        self,
        model_variant: str = '4l',
        batch_size: int = 1,
        seq_len: int = 128,
        num_classes: int = 2,
        num_iterations: int = DEFAULT_NUM_ITERATIONS,
        warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    ):
        self.model_variant = model_variant
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        
        config = TINYBERT_CONFIG.get(model_variant, TINYBERT_CONFIG['4l'])
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.intermediate_size = config['intermediate_size']
        self.head_dim = self.embed_dim // self.num_heads
        
        self.metrics: Dict[str, LayerMetrics] = OrderedDict()
    
    def _create_pytorch_layers(self):
        """Create PyTorch layer instances for profiling."""
        import torch.nn as nn
        
        layers = OrderedDict()
        
        # Embedding layer (simulated as Linear for profiling)
        layers['embedding'] = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Encoder layers
        for i in range(self.num_layers):
            prefix = f'encoder{i}'
            
            # Attention Q/K/V projections
            layers[f'{prefix}_attn_q_proj'] = nn.Linear(self.embed_dim, self.embed_dim)
            layers[f'{prefix}_attn_k_proj'] = nn.Linear(self.embed_dim, self.embed_dim)
            layers[f'{prefix}_attn_v_proj'] = nn.Linear(self.embed_dim, self.embed_dim)
            layers[f'{prefix}_attn_out_proj'] = nn.Linear(self.embed_dim, self.embed_dim)
            
            # LayerNorms (post-norm style)
            layers[f'{prefix}_norm1'] = nn.LayerNorm(self.embed_dim)
            layers[f'{prefix}_norm2'] = nn.LayerNorm(self.embed_dim)
            
            # FFN
            layers[f'{prefix}_ffn_fc1'] = nn.Linear(self.embed_dim, self.intermediate_size)
            layers[f'{prefix}_ffn_fc2'] = nn.Linear(self.intermediate_size, self.embed_dim)
        
        # Pooler (Linear + Tanh)
        layers['pooler'] = nn.Linear(self.embed_dim, self.embed_dim)
        
        # Classifier
        layers['classifier'] = nn.Linear(self.embed_dim, self.num_classes)
        
        return layers
    
    def profile_cpu(self, verbose: bool = True) -> Dict[str, LayerMetrics]:
        """Profile all layers on CPU."""
        import torch
        import torch.nn.functional as F
        
        variant_name = "4L-312D" if self.model_variant == '4l' else "6L-768D"
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Profiling TinyBERT-{variant_name} on CPU")
            print(f"Config: embed_dim={self.embed_dim}, heads={self.num_heads}, "
                  f"layers={self.num_layers}")
            print(f"Sequence length: {self.seq_len}")
            print(f"Iterations: {self.num_iterations} (warmup: {self.warmup_iterations})")
            print(f"{'='*60}\n")
        
        torch.set_num_threads(1)
        
        layers = self._create_pytorch_layers()
        
        for layer in layers.values():
            layer.eval()
        
        with torch.no_grad():
            # ===== Embedding (simulated) =====
            x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
            self._profile_layer(
                'embedding', layers['embedding'], x,
                input_shape=[self.batch_size, self.seq_len, self.embed_dim],
                output_shape=[self.batch_size, self.seq_len, self.embed_dim],
                group='Embedding',
                verbose=verbose
            )
            
            # ===== Encoder Blocks =====
            for i in range(self.num_layers):
                prefix = f'encoder{i}'
                group = f'Encoder{i}'
                
                if verbose:
                    print(f"\n--- {group} (Layer {i+1}/{self.num_layers}) ---")
                
                # Q Projection
                self._profile_layer(
                    f'{prefix}_attn_q_proj', layers[f'{prefix}_attn_q_proj'], x,
                    input_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    output_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    group=group, verbose=verbose
                )
                
                # K Projection
                self._profile_layer(
                    f'{prefix}_attn_k_proj', layers[f'{prefix}_attn_k_proj'], x,
                    input_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    output_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    group=group, verbose=verbose
                )
                
                # V Projection
                self._profile_layer(
                    f'{prefix}_attn_v_proj', layers[f'{prefix}_attn_v_proj'], x,
                    input_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    output_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    group=group, verbose=verbose
                )
                
                # QK MatMul
                q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
                k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
                self._profile_matmul(
                    f'{prefix}_attn_qk_matmul', q, k.transpose(-2, -1),
                    input_shape=[self.batch_size, self.num_heads, self.seq_len, self.head_dim],
                    output_shape=[self.batch_size, self.num_heads, self.seq_len, self.seq_len],
                    group=group, verbose=verbose
                )
                
                # Softmax
                attn_weights = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.seq_len)
                self._profile_softmax(
                    f'{prefix}_attn_softmax', attn_weights,
                    input_shape=[self.batch_size, self.num_heads, self.seq_len, self.seq_len],
                    output_shape=[self.batch_size, self.num_heads, self.seq_len, self.seq_len],
                    group=group, verbose=verbose
                )
                
                # Attention @ V MatMul
                v = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
                attn = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.seq_len)
                self._profile_matmul(
                    f'{prefix}_attn_v_matmul', attn, v,
                    input_shape=[self.batch_size, self.num_heads, self.seq_len, self.seq_len],
                    output_shape=[self.batch_size, self.num_heads, self.seq_len, self.head_dim],
                    group=group, verbose=verbose
                )
                
                # Output Projection
                attn_out = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
                self._profile_layer(
                    f'{prefix}_attn_out_proj', layers[f'{prefix}_attn_out_proj'], attn_out,
                    input_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    output_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    group=group, verbose=verbose
                )
                
                # LayerNorm 1 (post-norm after attention)
                self._profile_layer(
                    f'{prefix}_norm1', layers[f'{prefix}_norm1'], x,
                    input_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    output_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    group=group, verbose=verbose
                )
                
                # FFN FC1
                self._profile_layer(
                    f'{prefix}_ffn_fc1', layers[f'{prefix}_ffn_fc1'], x,
                    input_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    output_shape=[self.batch_size, self.seq_len, self.intermediate_size],
                    group=group, verbose=verbose
                )
                
                # GELU
                ffn_hidden = torch.randn(self.batch_size, self.seq_len, self.intermediate_size)
                self._profile_gelu(
                    f'{prefix}_ffn_gelu', ffn_hidden,
                    input_shape=[self.batch_size, self.seq_len, self.intermediate_size],
                    output_shape=[self.batch_size, self.seq_len, self.intermediate_size],
                    group=group, verbose=verbose
                )
                
                # FFN FC2
                self._profile_layer(
                    f'{prefix}_ffn_fc2', layers[f'{prefix}_ffn_fc2'], ffn_hidden,
                    input_shape=[self.batch_size, self.seq_len, self.intermediate_size],
                    output_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    group=group, verbose=verbose
                )
                
                # LayerNorm 2 (post-norm after FFN)
                self._profile_layer(
                    f'{prefix}_norm2', layers[f'{prefix}_norm2'], x,
                    input_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    output_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    group=group, verbose=verbose
                )
            
            # ===== Pooler and Classifier =====
            if verbose:
                print(f"\n--- Classification Head ---")
            
            # Pooler (Linear + Tanh)
            cls_token = torch.randn(self.batch_size, self.embed_dim)
            self._profile_layer(
                'pooler', layers['pooler'], cls_token,
                input_shape=[self.batch_size, self.embed_dim],
                output_shape=[self.batch_size, self.embed_dim],
                group='ClassHead', verbose=verbose
            )
            
            # Tanh
            self._profile_tanh(
                'pooler_tanh', cls_token,
                input_shape=[self.batch_size, self.embed_dim],
                output_shape=[self.batch_size, self.embed_dim],
                group='ClassHead', verbose=verbose
            )
            
            # Classifier
            self._profile_layer(
                'classifier', layers['classifier'], cls_token,
                input_shape=[self.batch_size, self.embed_dim],
                output_shape=[self.batch_size, self.num_classes],
                group='ClassHead', verbose=verbose
            )
        
        # Compute statistics
        for metrics in self.metrics.values():
            metrics.compute_statistics()
        
        return self.metrics
    
    def _profile_layer(self, name: str, layer, x, input_shape, output_shape, group: str, verbose: bool):
        """Profile a single PyTorch layer."""
        import torch
        
        times = []
        
        for _ in range(self.warmup_iterations):
            _ = layer(x)
        
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            _ = layer(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        # Calculate memory footprint
        layer_type = type(layer).__name__
        mem_info = calc_layer_memory_from_shapes(
            layer_type=layer_type,
            input_shape=input_shape,
            output_shape=output_shape,
        )
        
        # Infer dependencies
        dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
        
        metrics = LayerMetrics(
            name=name,
            layer_type=layer_type,
            group=group,
            input_shape=input_shape,
            output_shape=output_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(output_shape),
            dependencies=dependencies,
            cpu_times=times,
            num_iterations=self.num_iterations,
            # Memory analysis fields
            cpu_memory_bytes=mem_info['cpu_memory_bytes'],
            tee_memory_bytes=mem_info['tee_memory_bytes'],
            tee_encryption_overhead=mem_info['tee_encryption_overhead'],
            tee_total_memory_bytes=mem_info['tee_total_memory_bytes'],
            weight_bytes=mem_info['weight_bytes'],
            bias_bytes=mem_info['bias_bytes'],
            activation_bytes=mem_info['activation_bytes'],
            num_chunks=mem_info['num_chunks'],
            chunk_metadata_bytes=mem_info['chunk_metadata_bytes'],
        )
        self.metrics[name] = metrics
        
        if verbose:
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms")
    
    def _profile_matmul(self, name: str, a, b, input_shape, output_shape, group: str, verbose: bool):
        """Profile a matrix multiplication."""
        import torch
        
        times = []
        
        for _ in range(self.warmup_iterations):
            _ = torch.matmul(a, b)
        
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            _ = torch.matmul(a, b)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        # Calculate memory footprint
        mem_info = calc_layer_memory_from_shapes(
            layer_type='MatMul',
            input_shape=input_shape,
            output_shape=output_shape,
        )
        
        # Infer dependencies
        dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
        
        metrics = LayerMetrics(
            name=name,
            layer_type='MatMul',
            group=group,
            input_shape=input_shape,
            output_shape=output_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(output_shape),
            dependencies=dependencies,
            cpu_times=times,
            num_iterations=self.num_iterations,
            # Memory analysis fields
            cpu_memory_bytes=mem_info['cpu_memory_bytes'],
            tee_memory_bytes=mem_info['tee_memory_bytes'],
            tee_encryption_overhead=mem_info['tee_encryption_overhead'],
            tee_total_memory_bytes=mem_info['tee_total_memory_bytes'],
            weight_bytes=mem_info['weight_bytes'],
            bias_bytes=mem_info['bias_bytes'],
            activation_bytes=mem_info['activation_bytes'],
            num_chunks=mem_info['num_chunks'],
            chunk_metadata_bytes=mem_info['chunk_metadata_bytes'],
        )
        self.metrics[name] = metrics
        
        if verbose:
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms")
    
    def _profile_softmax(self, name: str, x, input_shape, output_shape, group: str, verbose: bool):
        """Profile softmax operation."""
        import torch
        import torch.nn.functional as F
        
        times = []
        
        for _ in range(self.warmup_iterations):
            _ = F.softmax(x, dim=-1)
        
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            _ = F.softmax(x, dim=-1)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        # Calculate memory footprint
        mem_info = calc_layer_memory_from_shapes(
            layer_type='Softmax',
            input_shape=input_shape,
            output_shape=output_shape,
        )
        
        # Infer dependencies
        dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
        
        metrics = LayerMetrics(
            name=name,
            layer_type='Softmax',
            group=group,
            input_shape=input_shape,
            output_shape=output_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(output_shape),
            dependencies=dependencies,
            cpu_times=times,
            num_iterations=self.num_iterations,
            # Memory analysis fields
            cpu_memory_bytes=mem_info['cpu_memory_bytes'],
            tee_memory_bytes=mem_info['tee_memory_bytes'],
            tee_encryption_overhead=mem_info['tee_encryption_overhead'],
            tee_total_memory_bytes=mem_info['tee_total_memory_bytes'],
            weight_bytes=mem_info['weight_bytes'],
            bias_bytes=mem_info['bias_bytes'],
            activation_bytes=mem_info['activation_bytes'],
            num_chunks=mem_info['num_chunks'],
            chunk_metadata_bytes=mem_info['chunk_metadata_bytes'],
        )
        self.metrics[name] = metrics
        
        if verbose:
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms")
    
    def _profile_gelu(self, name: str, x, input_shape, output_shape, group: str, verbose: bool):
        """Profile GELU activation."""
        import torch
        import torch.nn.functional as F
        
        times = []
        
        try:
            _ = F.gelu(x[:1], approximate='tanh')
            use_approx = True
        except TypeError:
            use_approx = False
        
        for _ in range(self.warmup_iterations):
            if use_approx:
                _ = F.gelu(x, approximate='tanh')
            else:
                _ = F.gelu(x)
        
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            if use_approx:
                _ = F.gelu(x, approximate='tanh')
            else:
                _ = F.gelu(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        # Calculate memory footprint
        mem_info = calc_layer_memory_from_shapes(
            layer_type='GELU',
            input_shape=input_shape,
            output_shape=output_shape,
        )
        
        # Infer dependencies
        dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
        
        metrics = LayerMetrics(
            name=name,
            layer_type='GELU',
            group=group,
            input_shape=input_shape,
            output_shape=output_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(output_shape),
            dependencies=dependencies,
            cpu_times=times,
            num_iterations=self.num_iterations,
            # Memory analysis fields
            cpu_memory_bytes=mem_info['cpu_memory_bytes'],
            tee_memory_bytes=mem_info['tee_memory_bytes'],
            tee_encryption_overhead=mem_info['tee_encryption_overhead'],
            tee_total_memory_bytes=mem_info['tee_total_memory_bytes'],
            weight_bytes=mem_info['weight_bytes'],
            bias_bytes=mem_info['bias_bytes'],
            activation_bytes=mem_info['activation_bytes'],
            num_chunks=mem_info['num_chunks'],
            chunk_metadata_bytes=mem_info['chunk_metadata_bytes'],
        )
        self.metrics[name] = metrics
        
        if verbose:
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms")
    
    def _profile_tanh(self, name: str, x, input_shape, output_shape, group: str, verbose: bool):
        """Profile Tanh activation (TinyBERT uses Tanh in pooler)."""
        import torch
        
        times = []
        
        for _ in range(self.warmup_iterations):
            _ = torch.tanh(x)
        
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            _ = torch.tanh(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        # Calculate memory footprint
        mem_info = calc_layer_memory_from_shapes(
            layer_type='Tanh',
            input_shape=input_shape,
            output_shape=output_shape,
        )
        
        # Infer dependencies
        dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
        
        metrics = LayerMetrics(
            name=name,
            layer_type='Tanh',
            group=group,
            input_shape=input_shape,
            output_shape=output_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(output_shape),
            dependencies=dependencies,
            cpu_times=times,
            num_iterations=self.num_iterations,
            # Memory analysis fields
            cpu_memory_bytes=mem_info['cpu_memory_bytes'],
            tee_memory_bytes=mem_info['tee_memory_bytes'],
            tee_encryption_overhead=mem_info['tee_encryption_overhead'],
            tee_total_memory_bytes=mem_info['tee_total_memory_bytes'],
            weight_bytes=mem_info['weight_bytes'],
            bias_bytes=mem_info['bias_bytes'],
            activation_bytes=mem_info['activation_bytes'],
            num_chunks=mem_info['num_chunks'],
            chunk_metadata_bytes=mem_info['chunk_metadata_bytes'],
        )
        self.metrics[name] = metrics
        
        if verbose:
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms")
    
    def save_results(self, output_dir: str = 'experiments/data'):
        """Save profiling results to CSV and JSON."""
        os.makedirs(output_dir, exist_ok=True)
        
        variant = self.model_variant
        csv_path = os.path.join(output_dir, f'tinybert_{variant}_layers.csv')
        json_path = os.path.join(output_dir, f'tinybert_{variant}_layers.json')
        
        fieldnames = [
            'name', 'type', 'group', 'execution_mode',
            'cpu_time_mean', 'cpu_time_std', 'cpu_time_min', 'cpu_time_max',
            'enclave_time_mean', 'enclave_time_std', 'enclave_time_min', 'enclave_time_max',
            'enclave_time_p95', 'enclave_time_p99',
            'input_bytes', 'output_bytes',
            'input_shape', 'output_shape',
            'dependencies', 'num_iterations',
            # Memory analysis columns
            'cpu_memory_bytes', 'tee_memory_bytes', 'tee_encryption_overhead',
            'tee_total_memory_bytes', 'weight_bytes', 'bias_bytes',
            'activation_bytes', 'num_chunks', 'chunk_metadata_bytes'
        ]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metrics in self.metrics.values():
                row = metrics.to_dict()
                row['input_shape'] = str(row['input_shape'])
                row['output_shape'] = str(row['output_shape'])
                row['dependencies'] = str(row['dependencies'])
                writer.writerow(row)
        
        with open(json_path, 'w') as f:
            json.dump({
                'model_config': {
                    'variant': variant,
                    'model_type': 'TinyBERT',
                    'embed_dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'num_layers': self.num_layers,
                    'intermediate_size': self.intermediate_size,
                    'batch_size': self.batch_size,
                    'seq_len': self.seq_len,
                    'num_classes': self.num_classes,
                    'note': f'TinyBERT-{variant.upper()}: 7.5x smaller, 9.4x faster than BERT',
                },
                'profiling_config': {
                    'num_iterations': self.num_iterations,
                    'warmup_iterations': self.warmup_iterations,
                },
                'layers': [m.to_dict() for m in self.metrics.values()],
            }, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")
        
        return csv_path, json_path
    
    def print_summary(self):
        """Print summary statistics."""
        variant_name = "4L-312D" if self.model_variant == '4l' else "6L-768D"
        
        print(f"\n{'='*60}")
        print(f"TinyBERT-{variant_name} Profiling Summary")
        print(f"  (7.5x smaller, 9.4x faster than BERT-base)")
        print(f"{'='*60}")
        
        group_times = {}
        for metrics in self.metrics.values():
            group = metrics.group
            if group not in group_times:
                group_times[group] = 0.0
            group_times[group] += metrics.cpu_time_mean
        
        total_time = sum(group_times.values())
        
        print(f"\nPer-Group CPU Time:")
        print(f"{'Group':<20} {'Time (ms)':>12} {'%':>8}")
        print(f"{'-'*40}")
        for group, time_ms in group_times.items():
            pct = time_ms / total_time * 100 if total_time > 0 else 0
            print(f"{group:<20} {time_ms:>12.3f} {pct:>8.1f}%")
        print(f"{'-'*40}")
        print(f"{'Total':<20} {total_time:>12.3f} {'100.0':>8}%")
        
        type_times = {}
        type_counts = {}
        for metrics in self.metrics.values():
            ltype = metrics.layer_type
            if ltype not in type_times:
                type_times[ltype] = 0.0
                type_counts[ltype] = 0
            type_times[ltype] += metrics.cpu_time_mean
            type_counts[ltype] += 1
        
        print(f"\nPer-Layer-Type CPU Time:")
        print(f"{'Type':<20} {'Count':>8} {'Time (ms)':>12} {'%':>8}")
        print(f"{'-'*48}")
        for ltype in sorted(type_times.keys(), key=lambda x: type_times[x], reverse=True):
            time_ms = type_times[ltype]
            count = type_counts[ltype]
            pct = time_ms / total_time * 100 if total_time > 0 else 0
            print(f"{ltype:<20} {count:>8} {time_ms:>12.3f} {pct:>8.1f}%")
        
        print(f"\n{'='*60}\n")
        
        # Print memory summary
        print_memory_summary(self.metrics, "TinyBERT Memory Analysis")


def main():
    parser = argparse.ArgumentParser(description='Profile TinyBERT model layers on CPU')
    parser.add_argument('--model', type=str, default='4l',
                       choices=['4l', '6l'],
                       help='TinyBERT model variant (4l=4-layer-312dim, 6l=6-layer-768dim)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--seq-len', type=int, default=128,
                       help='Sequence length')
    parser.add_argument('--num-classes', type=int, default=2,
                       help='Number of output classes')
    parser.add_argument('--iterations', type=int, default=DEFAULT_NUM_ITERATIONS,
                       help='Number of measurement iterations')
    parser.add_argument('--warmup', type=int, default=DEFAULT_WARMUP_ITERATIONS,
                       help='Number of warmup iterations')
    parser.add_argument('--output-dir', type=str, default='experiments/data',
                       help='Output directory for results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    profiler = TinyBERTPureProfiler(
        model_variant=args.model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
    )
    
    profiler.profile_cpu(verbose=not args.quiet)
    profiler.print_summary()
    profiler.save_results(args.output_dir)


if __name__ == '__main__':
    main()
