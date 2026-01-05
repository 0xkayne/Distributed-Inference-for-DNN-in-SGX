"""
Vision Transformer (ViT) Performance Profiler for Distributed Inference Modeling.

This script measures:
1. Execution time of each layer in Enclave mode (with statistical analysis).
2. Execution time of each layer in CPU mode.
3. Input/Output tensor size of each layer (for communication cost modeling).
4. Layer dependencies (for DAG construction).

Output: experiments/data/vit_base_layers.csv

Key Differences from Inception Profiler:
- Transformer blocks have more uniform structure
- New layer types: LayerNorm, Softmax, GELU, MatMul, Scale, Reshape
- Sequence dimension instead of spatial dimensions
- No pooling layers with special STORE_CHUNK_ELEM constraints

ViT-Base Architecture Summary (12 blocks):
- Patch Embedding: Conv2d(3, 768, 16x16, stride=16)
- Each Block: LayerNorm -> MHSA -> Add -> LayerNorm -> FFN -> Add
- Classification: LayerNorm -> Linear
"""

import sys
import time
import csv
import json
import numpy as np
import os
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any
import argparse

sys.path.insert(0, '.')

try:
    import torch
except ImportError:
    torch = None
    print("Warning: PyTorch not available")

from python.utils.basic_utils import ExecutionModeOptions
from experiments.models.profiler_utils import (
    LayerMetrics,
    calc_layer_memory_from_shapes,
    shape_to_bytes,
    print_memory_summary,
    infer_layer_dependencies,
    CSV_FIELDNAMES,
)


# Default measurement parameters
DEFAULT_NUM_ITERATIONS = 30
DEFAULT_WARMUP_ITERATIONS = 5


# =============================================================================
# ViT Group Configuration
# =============================================================================
# Unlike Inception, ViT has a more uniform structure:
# - Patch Embedding (Conv-based)
# - 12 Transformer Blocks (each identical structure)
# - Classification Head
#
# For STORE_CHUNK_ELEM, ViT's constraints are simpler:
# - Linear layers: output_dim divisible
# - No pooling layers with complex spatial constraints
# =============================================================================

def generate_vit_group_configs(
    batch_size: int = 1,
    img_size: int = 224,
    patch_size: int = 16,
    embed_dim: int = 768,
    num_heads: int = 12,
    num_layers: int = 12,
    num_classes: int = 1000
) -> Dict:
    """
    Generate group configurations for ViT model.
    
    Each Transformer block is its own group for fine-grained profiling.
    """
    num_patches = (img_size // patch_size) ** 2  # 196
    seq_len = num_patches + 1  # 197 (with CLS token)
    head_dim = embed_dim // num_heads
    mlp_hidden = embed_dim * 4
    
    configs = OrderedDict()
    
    # Group 0: Input + Patch Embedding
    configs['PatchEmbed'] = {
        'input_shape': [batch_size, 3, img_size, img_size],
        'output_shape': [batch_size, num_patches, embed_dim],
        'description': f'Input + Patch Embedding ({img_size}x{img_size} -> {num_patches}x{embed_dim})',
        'layer_prefixes': ['input', 'patch_embed', 'flatten_patches'],
    }
    
    # Groups 1-12: Transformer Blocks
    for i in range(num_layers):
        configs[f'Block{i}'] = {
            'input_shape': [batch_size, seq_len, embed_dim],
            'output_shape': [batch_size, seq_len, embed_dim],
            'description': f'Transformer Block {i} ({seq_len}x{embed_dim})',
            'layer_prefixes': [f'block{i}_'],
            'components': {
                'norm1': {'type': 'LayerNorm', 'shape': [embed_dim]},
                'attn': {
                    'qkv_proj': {'type': 'Linear', 'in': embed_dim, 'out': 3*embed_dim},
                    'qk_matmul': {'type': 'MatMul', 'shape': [batch_size, num_heads, seq_len, seq_len]},
                    'softmax': {'type': 'Softmax', 'dim': -1},
                    'attn_v_matmul': {'type': 'MatMul', 'shape': [batch_size, num_heads, seq_len, head_dim]},
                    'out_proj': {'type': 'Linear', 'in': embed_dim, 'out': embed_dim},
                },
                'residual1': {'type': 'Add'},
                'norm2': {'type': 'LayerNorm', 'shape': [embed_dim]},
                'ffn': {
                    'fc1': {'type': 'Linear', 'in': embed_dim, 'out': mlp_hidden},
                    'gelu': {'type': 'GELU'},
                    'fc2': {'type': 'Linear', 'in': mlp_hidden, 'out': embed_dim},
                },
                'residual2': {'type': 'Add'},
            }
        }
    
    # Group 13: Classification Head
    configs['ClassHead'] = {
        'input_shape': [batch_size, seq_len, embed_dim],
        'output_shape': [batch_size, num_classes],
        'description': f'Classification Head ({embed_dim} -> {num_classes})',
        'layer_prefixes': ['head_norm', 'classifier', 'output'],
    }
    
    return configs


def get_layer_group(layer_name: str, configs: Dict) -> Optional[str]:
    """Determine which group a layer belongs to."""
    for group_name, config in configs.items():
        for prefix in config.get('layer_prefixes', []):
            if layer_name.startswith(prefix):
                return group_name
    return None


def _get_layer_dependencies(layer) -> List[str]:
    """Extract dependency layer names from a layer."""
    deps = []
    if hasattr(layer, 'PrevLayer') and layer.PrevLayer is not None:
        if isinstance(layer.PrevLayer, list):
            for prev in layer.PrevLayer:
                if hasattr(prev, 'LayerName'):
                    deps.append(prev.LayerName)
        else:
            if hasattr(layer.PrevLayer, 'LayerName'):
                deps.append(layer.PrevLayer.LayerName)
    return deps


def _get_layer_input_shape(layer) -> List[int]:
    """Get input shape of a layer."""
    if hasattr(layer, 'InputShape') and layer.InputShape is not None:
        return list(layer.InputShape)
    return []


def _get_layer_output_shape(layer) -> List[int]:
    """Get output shape of a layer."""
    if hasattr(layer, 'get_output_shape'):
        shape = layer.get_output_shape()
        if shape is not None:
            return list(shape)
    if hasattr(layer, 'OutputShape') and layer.OutputShape is not None:
        return list(layer.OutputShape)
    return []


def _shape_to_bytes(shape: List[int]) -> int:
    """Convert shape to size in bytes (assuming float32)."""
    return shape_to_bytes(shape)


class ViTPureProfiler:
    """
    Pure PyTorch profiler for ViT models.
    
    Measures CPU execution time for each layer using PyTorch operations only.
    This is useful when SGX enclave is not available.
    """
    
    def __init__(
        self,
        model_variant: str = 'base',  # 'tiny', 'small', 'base', 'large'
        batch_size: int = 1,
        img_size: int = 224,
        num_classes: int = 1000,
        num_iterations: int = DEFAULT_NUM_ITERATIONS,
        warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    ):
        self.model_variant = model_variant
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        
        # Model configuration based on variant
        self.configs = {
            'tiny': {'embed_dim': 192, 'num_heads': 3, 'num_layers': 12},
            'small': {'embed_dim': 384, 'num_heads': 6, 'num_layers': 12},
            'base': {'embed_dim': 768, 'num_heads': 12, 'num_layers': 12},
            'large': {'embed_dim': 1024, 'num_heads': 16, 'num_layers': 24},
        }
        
        config = self.configs.get(model_variant, self.configs['base'])
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.head_dim = self.embed_dim // self.num_heads
        self.mlp_hidden = self.embed_dim * 4
        
        self.patch_size = 16
        self.num_patches = (img_size // self.patch_size) ** 2
        self.seq_len = self.num_patches + 1
        
        # Generate group configs
        self.group_configs = generate_vit_group_configs(
            batch_size=batch_size,
            img_size=img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            num_classes=num_classes
        )
        
        self.metrics: Dict[str, LayerMetrics] = OrderedDict()
    
    def _create_pytorch_vit_layers(self):
        """Create PyTorch layer instances for profiling."""
        import torch.nn as nn
        
        layers = OrderedDict()
        
        # Patch Embedding
        layers['patch_embed'] = nn.Conv2d(
            3, self.embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Transformer Blocks
        for i in range(self.num_layers):
            prefix = f'block{i}'
            
            # Pre-norm 1
            layers[f'{prefix}_norm1'] = nn.LayerNorm(self.embed_dim)
            
            # Attention
            layers[f'{prefix}_attn_qkv_proj'] = nn.Linear(self.embed_dim, 3 * self.embed_dim)
            layers[f'{prefix}_attn_out_proj'] = nn.Linear(self.embed_dim, self.embed_dim)
            
            # Pre-norm 2
            layers[f'{prefix}_norm2'] = nn.LayerNorm(self.embed_dim)
            
            # FFN
            layers[f'{prefix}_ffn_fc1'] = nn.Linear(self.embed_dim, self.mlp_hidden)
            layers[f'{prefix}_ffn_fc2'] = nn.Linear(self.mlp_hidden, self.embed_dim)
        
        # Classification Head
        layers['head_norm'] = nn.LayerNorm(self.embed_dim)
        layers['classifier'] = nn.Linear(self.embed_dim, self.num_classes)
        
        return layers
    
    def profile_cpu(self, verbose: bool = True) -> Dict[str, LayerMetrics]:
        """Profile all layers on CPU."""
        import torch
        import torch.nn.functional as F
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Profiling ViT-{self.model_variant.capitalize()} on CPU")
            print(f"Config: embed_dim={self.embed_dim}, heads={self.num_heads}, layers={self.num_layers}")
            print(f"Iterations: {self.num_iterations} (warmup: {self.warmup_iterations})")
            print(f"{'='*60}\n")
        
        torch.set_num_threads(1)  # Single thread for consistent timing
        
        layers = self._create_pytorch_vit_layers()
        
        # Put all layers in eval mode
        for layer in layers.values():
            layer.eval()
        
        # Profile each layer type
        with torch.no_grad():
            # ===== Patch Embedding =====
            x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
            self._profile_layer(
                'patch_embed', layers['patch_embed'], x,
                input_shape=[self.batch_size, 3, self.img_size, self.img_size],
                output_shape=[self.batch_size, self.embed_dim, self.img_size//16, self.img_size//16],
                group='PatchEmbed',
                verbose=verbose
            )
            
            # Flatten for transformer
            x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
            
            # ===== Transformer Blocks =====
            for i in range(self.num_layers):
                prefix = f'block{i}'
                group = f'Block{i}'
                
                # LayerNorm 1
                self._profile_layer(
                    f'{prefix}_norm1', layers[f'{prefix}_norm1'], x,
                    input_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    output_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    group=group, verbose=verbose
                )
                
                # QKV Projection
                self._profile_layer(
                    f'{prefix}_attn_qkv_proj', layers[f'{prefix}_attn_qkv_proj'], x,
                    input_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    output_shape=[self.batch_size, self.seq_len, 3*self.embed_dim],
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
                
                # LayerNorm 2
                self._profile_layer(
                    f'{prefix}_norm2', layers[f'{prefix}_norm2'], x,
                    input_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    output_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    group=group, verbose=verbose
                )
                
                # FFN FC1
                self._profile_layer(
                    f'{prefix}_ffn_fc1', layers[f'{prefix}_ffn_fc1'], x,
                    input_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    output_shape=[self.batch_size, self.seq_len, self.mlp_hidden],
                    group=group, verbose=verbose
                )
                
                # GELU
                ffn_hidden = torch.randn(self.batch_size, self.seq_len, self.mlp_hidden)
                self._profile_gelu(
                    f'{prefix}_ffn_gelu', ffn_hidden,
                    input_shape=[self.batch_size, self.seq_len, self.mlp_hidden],
                    output_shape=[self.batch_size, self.seq_len, self.mlp_hidden],
                    group=group, verbose=verbose
                )
                
                # FFN FC2
                self._profile_layer(
                    f'{prefix}_ffn_fc2', layers[f'{prefix}_ffn_fc2'], ffn_hidden,
                    input_shape=[self.batch_size, self.seq_len, self.mlp_hidden],
                    output_shape=[self.batch_size, self.seq_len, self.embed_dim],
                    group=group, verbose=verbose
                )
            
            # ===== Classification Head =====
            x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
            
            self._profile_layer(
                'head_norm', layers['head_norm'], x,
                input_shape=[self.batch_size, self.seq_len, self.embed_dim],
                output_shape=[self.batch_size, self.seq_len, self.embed_dim],
                group='ClassHead', verbose=verbose
            )
            
            # Use CLS token only
            cls_token = torch.randn(self.batch_size, self.embed_dim)
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
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = layer(x)
        
        # Measure
        for _ in range(self.num_iterations):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            _ = layer(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        # Calculate memory footprint
        layer_type = type(layer).__name__
        mem_info = calc_layer_memory_from_shapes(
            layer_type=layer_type,
            input_shape=input_shape,
            output_shape=output_shape,
        )
        
        # Infer dependencies from layer name
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
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = torch.matmul(a, b)
        
        # Measure
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            _ = torch.matmul(a, b)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        # Calculate memory footprint (MatMul has no weights)
        mem_info = calc_layer_memory_from_shapes(
            layer_type='MatMul',
            input_shape=input_shape,
            output_shape=output_shape,
        )
        
        # Infer dependencies from layer name
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
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = F.softmax(x, dim=-1)
        
        # Measure
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            _ = F.softmax(x, dim=-1)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        # Calculate memory footprint (Softmax has no weights)
        mem_info = calc_layer_memory_from_shapes(
            layer_type='Softmax',
            input_shape=input_shape,
            output_shape=output_shape,
        )
        
        # Infer dependencies from layer name
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
        
        # Check if approximate parameter is supported (PyTorch >= 1.12)
        try:
            _ = F.gelu(x[:1], approximate='tanh')
            use_approx = True
        except TypeError:
            use_approx = False
        
        # Warmup
        for _ in range(self.warmup_iterations):
            if use_approx:
                _ = F.gelu(x, approximate='tanh')
            else:
                _ = F.gelu(x)
        
        # Measure
        for _ in range(self.num_iterations):
            start = time.perf_counter()
            if use_approx:
                _ = F.gelu(x, approximate='tanh')
            else:
                _ = F.gelu(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        # Calculate memory footprint (GELU has no weights)
        mem_info = calc_layer_memory_from_shapes(
            layer_type='GELU',
            input_shape=input_shape,
            output_shape=output_shape,
        )
        
        # Infer dependencies from layer name
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
    
    def save_results(self, output_dir: str = 'experiments/data'):
        """Save profiling results to CSV and JSON."""
        os.makedirs(output_dir, exist_ok=True)
        
        variant = self.model_variant
        csv_path = os.path.join(output_dir, f'vit_{variant}_layers.csv')
        json_path = os.path.join(output_dir, f'vit_{variant}_layers.json')
        
        # Save CSV with memory analysis fields
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
        
        # Save JSON
        with open(json_path, 'w') as f:
            json.dump({
                'model_config': {
                    'variant': variant,
                    'embed_dim': self.embed_dim,
                    'num_heads': self.num_heads,
                    'num_layers': self.num_layers,
                    'batch_size': self.batch_size,
                    'img_size': self.img_size,
                    'num_classes': self.num_classes,
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
        print(f"ViT-{self.model_variant.capitalize()} Profiling Summary")
        print(f"{'='*60}")
        
        # Group by block
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
        
        # Layer type breakdown
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
        print_memory_summary(self.metrics, f"ViT-{self.model_variant.capitalize()} Memory Analysis")


def main():
    parser = argparse.ArgumentParser(description='Profile ViT model layers')
    parser.add_argument('--model', type=str, default='base',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='ViT model variant')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=224,
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
    
    profiler = ViTPureProfiler(
        model_variant=args.model,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_classes=args.num_classes,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
    )
    
    # Profile on CPU
    profiler.profile_cpu(verbose=not args.quiet)
    
    # Print summary
    profiler.print_summary()
    
    # Save results
    profiler.save_results(args.output_dir)


if __name__ == '__main__':
    main()

