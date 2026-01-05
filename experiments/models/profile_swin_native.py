"""
Swin Transformer Performance Profiler for Distributed Inference Modeling.

This script measures:
1. Execution time of each layer in CPU mode (with statistical analysis).
2. Input/Output tensor size of each layer (for communication cost modeling).
3. Layer dependencies (for DAG construction).

Output: experiments/data/swin_{tiny|small|base}_layers.csv

Swin Transformer Architecture Summary:
- Hierarchical 4-stage structure with decreasing spatial resolution
- Shifted Window Attention: W-MSA and SW-MSA alternating
- Patch Merging for downsampling between stages
- Window size 7x7 (49 tokens per window) - LOCAL attention

Stage Structure:
- Stage 0: 56x56, C=96, 2 blocks (Swin-T)
- Stage 1: 28x28, C=192, 2 blocks
- Stage 2: 14x14, C=384, 6 blocks
- Stage 3: 7x7, C=768, 2 blocks

Key Insight for TEE: Window attention bounds memory to window_size^2 = 49
tokens regardless of image size, unlike ViT's global attention.

Reference: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
           (Liu et al., ICCV 2021)
"""

import sys
import time
import csv
import json
import numpy as np
import os
from collections import OrderedDict

from typing import Dict, List, Optional, Any, Tuple
import argparse

sys.path.insert(0, '.')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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
# Swin Transformer Model Configuration
# ============================================================================

SWIN_CONFIG = {
    'tiny': {
        'embed_dim': 96,
        'depths': (2, 2, 6, 2),
        'num_heads': (3, 6, 12, 24),
        'window_size': 7,
        'mlp_ratio': 4.0,
    },
    'small': {
        'embed_dim': 96,
        'depths': (2, 2, 18, 2),
        'num_heads': (3, 6, 12, 24),
        'window_size': 7,
        'mlp_ratio': 4.0,
    },
    'base': {
        'embed_dim': 128,
        'depths': (2, 2, 18, 2),
        'num_heads': (4, 8, 16, 32),
        'window_size': 7,
        'mlp_ratio': 4.0,
    }
}


class SwinPureProfiler:
    """
    Pure PyTorch profiler for Swin Transformer models.
    
    Measures CPU execution time for each layer using PyTorch operations.
    """
    
    def __init__(
        self,
        model_variant: str = 'tiny',
        batch_size: int = 1,
        image_size: int = 224,
        patch_size: int = 4,
        num_classes: int = 1000,
        num_iterations: int = DEFAULT_NUM_ITERATIONS,
        warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    ):
        self.model_variant = model_variant
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        
        config = SWIN_CONFIG.get(model_variant, SWIN_CONFIG['tiny'])
        self.embed_dim = config['embed_dim']
        self.depths = config['depths']
        self.num_heads = config['num_heads']
        self.window_size = config['window_size']
        self.mlp_ratio = config['mlp_ratio']
        
        self.metrics: Dict[str, LayerMetrics] = OrderedDict()
    
    def profile_cpu(self, verbose: bool = True) -> Dict[str, LayerMetrics]:
        """Profile all layers on CPU."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Profiling Swin-{self.model_variant.capitalize()} on CPU")
            print(f"Config: C={self.embed_dim}, depths={self.depths}, "
                  f"heads={self.num_heads}")
            print(f"Window size: {self.window_size} (LOCAL attention!)")
            print(f"Image size: {self.image_size}, Patch size: {self.patch_size}")
            print(f"Iterations: {self.num_iterations} (warmup: {self.warmup_iterations})")
            print(f"{'='*60}\n")
        
        torch.set_num_threads(1)
        
        # Initial spatial dimensions
        H = self.image_size // self.patch_size  # 56
        W = self.image_size // self.patch_size  # 56
        
        with torch.no_grad():
            # ===== Patch Embedding =====
            if verbose:
                print(f"--- Patch Embedding ---")
            
            patch_embed_conv = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
            patch_embed_norm = nn.LayerNorm(self.embed_dim)
            patch_embed_conv.eval()
            patch_embed_norm.eval()
            
            x_img = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
            
            self._profile_layer(
                'patch_embed_conv', patch_embed_conv, x_img,
                input_shape=[self.batch_size, 3, self.image_size, self.image_size],
                output_shape=[self.batch_size, self.embed_dim, H, W],
                group='PatchEmbed',
                verbose=verbose
            )
            
            x_flat = torch.randn(self.batch_size, H * W, self.embed_dim)
            self._profile_layer(
                'patch_embed_norm', patch_embed_norm, x_flat,
                input_shape=[self.batch_size, H * W, self.embed_dim],
                output_shape=[self.batch_size, H * W, self.embed_dim],
                group='PatchEmbed',
                verbose=verbose
            )
            
            # ===== Stages =====
            dim = self.embed_dim
            
            for i_stage in range(len(self.depths)):
                stage_name = f'stage{i_stage}'
                depth = self.depths[i_stage]
                heads = self.num_heads[i_stage]
                head_dim = dim // heads
                
                num_tokens = H * W
                num_windows = (H // self.window_size) * (W // self.window_size)
                window_tokens = self.window_size ** 2  # 49
                
                if verbose:
                    print(f"\n--- {stage_name.upper()} (H={H}, W={W}, C={dim}, heads={heads}) ---")
                
                # Profile each block in the stage
                for i_block in range(depth):
                    shift_size = 0 if (i_block % 2 == 0) else self.window_size // 2
                    block_type = "W-MSA" if shift_size == 0 else "SW-MSA"
                    block_prefix = f'{stage_name}_block{i_block}'
                    group = f'{stage_name}_block{i_block}'
                    
                    if verbose:
                        print(f"  Block {i_block} ({block_type}):")
                    
                    # LayerNorm 1
                    norm1 = nn.LayerNorm(dim)
                    norm1.eval()
                    x_block = torch.randn(self.batch_size, num_tokens, dim)
                    self._profile_layer(
                        f'{block_prefix}_norm1', norm1, x_block,
                        input_shape=[self.batch_size, num_tokens, dim],
                        output_shape=[self.batch_size, num_tokens, dim],
                        group=group, verbose=verbose
                    )
                    
                    # Window Attention QKV projection
                    qkv_proj = nn.Linear(dim, dim * 3)
                    qkv_proj.eval()
                    x_windows = torch.randn(self.batch_size * num_windows, window_tokens, dim)
                    self._profile_layer(
                        f'{block_prefix}_attn_qkv', qkv_proj, x_windows,
                        input_shape=[self.batch_size * num_windows, window_tokens, dim],
                        output_shape=[self.batch_size * num_windows, window_tokens, dim * 3],
                        group=group, verbose=verbose
                    )
                    
                    # QK MatMul
                    q = torch.randn(self.batch_size * num_windows, heads, window_tokens, head_dim)
                    k = torch.randn(self.batch_size * num_windows, heads, window_tokens, head_dim)
                    self._profile_matmul(
                        f'{block_prefix}_attn_qk_matmul', q, k.transpose(-2, -1),
                        input_shape=[self.batch_size * num_windows, heads, window_tokens, head_dim],
                        output_shape=[self.batch_size * num_windows, heads, window_tokens, window_tokens],
                        group=group, verbose=verbose
                    )
                    
                    # Softmax
                    attn_weights = torch.randn(self.batch_size * num_windows, heads, window_tokens, window_tokens)
                    self._profile_softmax(
                        f'{block_prefix}_attn_softmax', attn_weights,
                        input_shape=[self.batch_size * num_windows, heads, window_tokens, window_tokens],
                        output_shape=[self.batch_size * num_windows, heads, window_tokens, window_tokens],
                        group=group, verbose=verbose
                    )
                    
                    # Attention @ V MatMul
                    v = torch.randn(self.batch_size * num_windows, heads, window_tokens, head_dim)
                    attn = torch.randn(self.batch_size * num_windows, heads, window_tokens, window_tokens)
                    self._profile_matmul(
                        f'{block_prefix}_attn_v_matmul', attn, v,
                        input_shape=[self.batch_size * num_windows, heads, window_tokens, window_tokens],
                        output_shape=[self.batch_size * num_windows, heads, window_tokens, head_dim],
                        group=group, verbose=verbose
                    )
                    
                    # Output projection
                    out_proj = nn.Linear(dim, dim)
                    out_proj.eval()
                    attn_out = torch.randn(self.batch_size * num_windows, window_tokens, dim)
                    self._profile_layer(
                        f'{block_prefix}_attn_out_proj', out_proj, attn_out,
                        input_shape=[self.batch_size * num_windows, window_tokens, dim],
                        output_shape=[self.batch_size * num_windows, window_tokens, dim],
                        group=group, verbose=verbose
                    )
                    
                    # LayerNorm 2
                    norm2 = nn.LayerNorm(dim)
                    norm2.eval()
                    self._profile_layer(
                        f'{block_prefix}_norm2', norm2, x_block,
                        input_shape=[self.batch_size, num_tokens, dim],
                        output_shape=[self.batch_size, num_tokens, dim],
                        group=group, verbose=verbose
                    )
                    
                    # MLP FC1
                    mlp_hidden = int(dim * self.mlp_ratio)
                    fc1 = nn.Linear(dim, mlp_hidden)
                    fc1.eval()
                    self._profile_layer(
                        f'{block_prefix}_mlp_fc1', fc1, x_block,
                        input_shape=[self.batch_size, num_tokens, dim],
                        output_shape=[self.batch_size, num_tokens, mlp_hidden],
                        group=group, verbose=verbose
                    )
                    
                    # GELU
                    x_mlp = torch.randn(self.batch_size, num_tokens, mlp_hidden)
                    self._profile_gelu(
                        f'{block_prefix}_mlp_gelu', x_mlp,
                        input_shape=[self.batch_size, num_tokens, mlp_hidden],
                        output_shape=[self.batch_size, num_tokens, mlp_hidden],
                        group=group, verbose=verbose
                    )
                    
                    # MLP FC2
                    fc2 = nn.Linear(mlp_hidden, dim)
                    fc2.eval()
                    self._profile_layer(
                        f'{block_prefix}_mlp_fc2', fc2, x_mlp,
                        input_shape=[self.batch_size, num_tokens, mlp_hidden],
                        output_shape=[self.batch_size, num_tokens, dim],
                        group=group, verbose=verbose
                    )
                
                # Patch Merging (except last stage)
                if i_stage < len(self.depths) - 1:
                    if verbose:
                        print(f"  Patch Merging ({H}x{W} -> {H//2}x{W//2}):")
                    
                    merge_norm = nn.LayerNorm(4 * dim)
                    merge_norm.eval()
                    x_merge = torch.randn(self.batch_size, (H // 2) * (W // 2), 4 * dim)
                    self._profile_layer(
                        f'{stage_name}_merge_norm', merge_norm, x_merge,
                        input_shape=[self.batch_size, (H // 2) * (W // 2), 4 * dim],
                        output_shape=[self.batch_size, (H // 2) * (W // 2), 4 * dim],
                        group=f'{stage_name}_merge', verbose=verbose
                    )
                    
                    merge_reduction = nn.Linear(4 * dim, 2 * dim)
                    merge_reduction.eval()
                    self._profile_layer(
                        f'{stage_name}_merge_reduction', merge_reduction, x_merge,
                        input_shape=[self.batch_size, (H // 2) * (W // 2), 4 * dim],
                        output_shape=[self.batch_size, (H // 2) * (W // 2), 2 * dim],
                        group=f'{stage_name}_merge', verbose=verbose
                    )
                    
                    # Update dimensions
                    H = H // 2
                    W = W // 2
                    dim = dim * 2
            
            # ===== Classification Head =====
            if verbose:
                print(f"\n--- Classification Head ---")
            
            final_norm = nn.LayerNorm(dim)
            final_norm.eval()
            x_final = torch.randn(self.batch_size, H * W, dim)
            self._profile_layer(
                'final_norm', final_norm, x_final,
                input_shape=[self.batch_size, H * W, dim],
                output_shape=[self.batch_size, H * W, dim],
                group='ClassHead', verbose=verbose
            )
            
            classifier = nn.Linear(dim, self.num_classes)
            classifier.eval()
            x_cls = torch.randn(self.batch_size, dim)
            self._profile_layer(
                'classifier', classifier, x_cls,
                input_shape=[self.batch_size, dim],
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
            print(f"    {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms")
    
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
            print(f"    {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms")
    
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
            print(f"    {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms")
    
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
            print(f"    {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms")
    
    def save_results(self, output_dir: str = 'experiments/data'):
        """Save profiling results to CSV and JSON."""
        os.makedirs(output_dir, exist_ok=True)
        
        variant = self.model_variant
        csv_path = os.path.join(output_dir, f'swin_{variant}_layers.csv')
        json_path = os.path.join(output_dir, f'swin_{variant}_layers.json')
        
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
                    'model_type': 'Swin Transformer',
                    'embed_dim': self.embed_dim,
                    'depths': self.depths,
                    'num_heads': self.num_heads,
                    'window_size': self.window_size,
                    'mlp_ratio': self.mlp_ratio,
                    'image_size': self.image_size,
                    'patch_size': self.patch_size,
                    'batch_size': self.batch_size,
                    'num_classes': self.num_classes,
                    'note': f'Swin Transformer uses LOCAL window attention ({self.window_size}x{self.window_size})',
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
        print(f"\n{'='*60}")
        print(f"Swin-{self.model_variant.capitalize()} Profiling Summary")
        print(f"  (Uses LOCAL window attention: {self.window_size}x{self.window_size} windows)")
        print(f"{'='*60}")
        
        # Group by stage
        stage_times = {}
        for metrics in self.metrics.values():
            group = metrics.group
            # Extract stage from group name
            if 'stage' in group:
                stage = group.split('_')[0]
            elif group == 'PatchEmbed':
                stage = 'PatchEmbed'
            elif group == 'ClassHead':
                stage = 'ClassHead'
            else:
                stage = group
            
            if stage not in stage_times:
                stage_times[stage] = 0.0
            stage_times[stage] += metrics.cpu_time_mean
        
        total_time = sum(stage_times.values())
        
        print(f"\nPer-Stage CPU Time:")
        print(f"{'Stage':<20} {'Time (ms)':>12} {'%':>8}")
        print(f"{'-'*40}")
        for stage, time_ms in stage_times.items():
            pct = time_ms / total_time * 100 if total_time > 0 else 0
            print(f"{stage:<20} {time_ms:>12.3f} {pct:>8.1f}%")
        print(f"{'-'*40}")
        print(f"{'Total':<20} {total_time:>12.3f} {'100.0':>8}%")
        
        # Per layer type
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
        print_memory_summary(self.metrics, "Swin Transformer Memory Analysis")


def main():
    parser = argparse.ArgumentParser(description='Profile Swin Transformer model layers on CPU')
    parser.add_argument('--model', type=str, default='tiny',
                       choices=['tiny', 'small', 'base'],
                       help='Swin Transformer model variant')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num-classes', type=int, default=1000,
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
    
    profiler = SwinPureProfiler(
        model_variant=args.model,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_classes=args.num_classes,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
    )
    
    profiler.profile_cpu(verbose=not args.quiet)
    profiler.print_summary()
    profiler.save_results(args.output_dir)


if __name__ == '__main__':
    main()
