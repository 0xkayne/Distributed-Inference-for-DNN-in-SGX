#!/usr/bin/env python
"""
Vision Transformer (ViT) Enclave Performance Profiler.

This script measures ViT layer execution in hybrid mode:
- Supported layers (Conv, Linear): Run in Enclave
- Unsupported layers (LayerNorm, Softmax, GELU, MatMul): Run on CPU

Usage:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python -m experiments.models.profile_vit_enclave

Current Enclave Support Status:
- SGXConvBase: ✓ Supported
- SGXLinearBase: ✓ Supported  
- LayerNorm: ✗ Not implemented (CPU fallback)
- Softmax: ✗ Not implemented (CPU fallback)
- GELU: ✗ Not implemented (CPU fallback)
- MatMul: ✗ Not implemented (CPU fallback)

Output: experiments/data/vit_{variant}_enclave_layers.csv
"""

import sys
import time
import os
import csv
import json
import numpy as np
import argparse
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

sys.path.insert(0, '.')


@dataclass
class LayerMetrics:
    """Data class to store layer profiling metrics (compatible with profile_vit_native.py)."""
    name: str
    layer_type: str
    group: str
    execution_mode: str = 'Enclave'  # 'Enclave' or 'CPU'
    
    # Enclave timing statistics (in ms)
    enclave_time_mean: float = 0.0
    enclave_time_std: float = 0.0
    enclave_time_min: float = 0.0
    enclave_time_max: float = 0.0
    enclave_time_p95: float = 0.0
    enclave_time_p99: float = 0.0
    
    # CPU timing statistics (in ms) - for comparison
    cpu_time_mean: float = 0.0
    cpu_time_std: float = 0.0
    cpu_time_min: float = 0.0
    cpu_time_max: float = 0.0
    
    # Data sizes (in bytes)
    input_bytes: int = 0
    output_bytes: int = 0
    
    # Input/output shapes
    input_shape: List[int] = field(default_factory=list)
    output_shape: List[int] = field(default_factory=list)
    
    # Dependencies (predecessor layer names)
    dependencies: List[str] = field(default_factory=list)
    
    # Raw timing data for detailed analysis
    enclave_times: List[float] = field(default_factory=list)
    cpu_times: List[float] = field(default_factory=list)
    
    # Number of iterations used
    num_iterations: int = 0

    def compute_statistics(self):
        """Compute statistics from raw timing data."""
        if self.enclave_times:
            times = np.array(self.enclave_times)
            self.enclave_time_mean = float(np.mean(times))
            self.enclave_time_std = float(np.std(times))
            self.enclave_time_min = float(np.min(times))
            self.enclave_time_max = float(np.max(times))
            self.enclave_time_p95 = float(np.percentile(times, 95))
            self.enclave_time_p99 = float(np.percentile(times, 99))
            
        if self.cpu_times:
            times = np.array(self.cpu_times)
            self.cpu_time_mean = float(np.mean(times))
            self.cpu_time_std = float(np.std(times))
            self.cpu_time_min = float(np.min(times))
            self.cpu_time_max = float(np.max(times))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'type': self.layer_type,
            'group': self.group,
            'execution_mode': self.execution_mode,
            'enclave_time_mean': self.enclave_time_mean,
            'enclave_time_std': self.enclave_time_std,
            'enclave_time_min': self.enclave_time_min,
            'enclave_time_max': self.enclave_time_max,
            'enclave_time_p95': self.enclave_time_p95,
            'enclave_time_p99': self.enclave_time_p99,
            'cpu_time_mean': self.cpu_time_mean,
            'cpu_time_std': self.cpu_time_std,
            'cpu_time_min': self.cpu_time_min,
            'cpu_time_max': self.cpu_time_max,
            'input_bytes': self.input_bytes,
            'output_bytes': self.output_bytes,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'dependencies': self.dependencies,
            'num_iterations': self.num_iterations,
        }


def _shape_to_bytes(shape: List[int]) -> int:
    """Convert shape to size in bytes (assuming float32)."""
    if not shape:
        return 0
    return int(np.prod(shape)) * 4


def check_environment():
    """Check if environment is correctly configured for SGX."""
    ld_preload = os.environ.get('LD_PRELOAD', '')
    if 'libstdc++' not in ld_preload:
        print("=" * 60)
        print("WARNING: LD_PRELOAD not set!")
        print("Please run with:")
        print("  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python -m experiments.models.profile_vit_enclave")
        print("=" * 60)
        return False
    return True


class ViTEnclaveProfiler:
    """
    Enclave profiler for ViT models.
    
    Measures execution time for layers that can run in SGX Enclave.
    """
    
    def __init__(
        self,
        model_variant: str = 'base',
        batch_size: int = 1,
        img_size: int = 224,
        num_classes: int = 1000,
        num_iterations: int = 10,
        warmup_iterations: int = 3,
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
        
        self.metrics: Dict[str, LayerMetrics] = OrderedDict()
    
    def profile_all(self, verbose: bool = True) -> Dict[str, LayerMetrics]:
        """Profile all ViT layers (Enclave for supported, CPU for others)."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Profiling ViT-{self.model_variant.capitalize()} in Enclave Mode")
            print(f"Config: embed_dim={self.embed_dim}, heads={self.num_heads}, layers={self.num_layers}")
            print(f"Iterations: {self.num_iterations} (warmup: {self.warmup_iterations})")
            print(f"{'='*60}\n")
        
        # Profile Conv layer (Enclave)
        self._profile_patch_embed_enclave(verbose)
        
        # Profile all layers in each block
        for block_idx in range(self.num_layers):
            self._profile_block(block_idx, verbose)
        
        # Profile classifier (Enclave)
        self._profile_classifier_enclave(verbose)
        
        # Compute statistics for all metrics
        for metrics in self.metrics.values():
            metrics.compute_statistics()
        
        return self.metrics
    
    def _profile_patch_embed_enclave(self, verbose: bool):
        """Profile patch embedding Conv layer in Enclave."""
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.sgx_conv_base import SGXConvBase
        from python.layers.input import SecretInputLayer
        from python.layers.output import SecretOutputLayer
        from python.sgx_net import SecretNeuralNetwork
        from python.utils.basic_utils import ExecutionModeOptions
        
        if verbose:
            print("Profiling Patch Embedding (Conv) in Enclave...")
        
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            sid = 0
            layers = []
            
            input_shape = [self.batch_size, 3, self.img_size, self.img_size]
            output_hw = self.img_size // self.patch_size
            output_shape = [self.batch_size, self.embed_dim, output_hw, output_hw]
            
            input_layer = SecretInputLayer(
                sid, "input", input_shape,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer)
            
            conv_layer = SGXConvBase(
                sid, "patch_embed",
                ExecutionModeOptions.Enclave,
                n_output_channel=self.embed_dim,
                n_input_channel=3,
                filter_hw=self.patch_size,
                stride=self.patch_size,
                padding=0,
                batch_size=self.batch_size,
                img_hw=self.img_size,
                manually_register_prev=True,
                manually_register_next=True
            )
            conv_layer.register_prev_layer(input_layer)
            layers.append(conv_layer)
            
            output_layer = SecretOutputLayer(
                sid, "output",
                ExecutionModeOptions.CPU,
                manually_register_prev=True
            )
            output_layer.register_prev_layer(conv_layer)
            layers.append(output_layer)
            
            secret_nn = SecretNeuralNetwork(sid, "profile_patch_embed")
            secret_nn.set_eid(GlobalTensor.get_eid())
            secret_nn.set_layers(layers)
            
            times = []
            total_runs = self.warmup_iterations + self.num_iterations
            
            for run_idx in range(total_runs):
                test_input = torch.randn(*input_shape)
                layers[0].set_input(test_input)
                
                start_time = time.perf_counter()
                layers[0].forward()
                layers[1].forward()
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                if run_idx >= self.warmup_iterations:
                    times.append(elapsed_ms)
            
            metrics = LayerMetrics(
                name='patch_embed',
                layer_type='Conv2d',
                group='PatchEmbed',
                execution_mode='Enclave',
                input_shape=input_shape,
                output_shape=output_shape,
                input_bytes=_shape_to_bytes(input_shape),
                output_bytes=_shape_to_bytes(output_shape),
                enclave_times=times,
                num_iterations=self.num_iterations
            )
            self.metrics['patch_embed'] = metrics
            
            if verbose:
                mean_time = np.mean(times)
                std_time = np.std(times)
                print(f"  {'patch_embed':40} {mean_time:8.3f} ± {std_time:6.3f} ms (Enclave)")
            
        except Exception as e:
            print(f"✗ Error profiling patch_embed: {e}")
        
        finally:
            if GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()
    
    def _profile_block(self, block_idx: int, verbose: bool):
        """Profile a single Transformer block."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        prefix = f'block{block_idx}'
        group = f'Block{block_idx}'
        
        torch.set_num_threads(1)
        
        # ===== LayerNorm 1 (CPU) =====
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        ln1 = nn.LayerNorm(self.embed_dim)
        ln1.eval()
        self._profile_cpu_layer(
            f'{prefix}_norm1', ln1, x,
            input_shape=[self.batch_size, self.seq_len, self.embed_dim],
            output_shape=[self.batch_size, self.seq_len, self.embed_dim],
            layer_type='LayerNorm',
            group=group,
            verbose=verbose
        )
        
        # ===== QKV Projection (Enclave) =====
        self._profile_linear_enclave(
            f'{prefix}_attn_qkv_proj',
            input_shape=[self.batch_size, self.seq_len, self.embed_dim],
            output_features=3 * self.embed_dim,
            group=group,
            verbose=verbose
        )
        
        # ===== QK MatMul (CPU) =====
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        self._profile_matmul_cpu(
            f'{prefix}_attn_qk_matmul', q, k.transpose(-2, -1),
            input_shape=[self.batch_size, self.num_heads, self.seq_len, self.head_dim],
            output_shape=[self.batch_size, self.num_heads, self.seq_len, self.seq_len],
            group=group,
            verbose=verbose
        )
        
        # ===== Softmax (CPU) =====
        attn_weights = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        self._profile_softmax_cpu(
            f'{prefix}_attn_softmax', attn_weights,
            input_shape=[self.batch_size, self.num_heads, self.seq_len, self.seq_len],
            output_shape=[self.batch_size, self.num_heads, self.seq_len, self.seq_len],
            group=group,
            verbose=verbose
        )
        
        # ===== Attention @ V MatMul (CPU) =====
        v = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        attn = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        self._profile_matmul_cpu(
            f'{prefix}_attn_v_matmul', attn, v,
            input_shape=[self.batch_size, self.num_heads, self.seq_len, self.seq_len],
            output_shape=[self.batch_size, self.num_heads, self.seq_len, self.head_dim],
            group=group,
            verbose=verbose
        )
        
        # ===== Output Projection (Enclave) =====
        self._profile_linear_enclave(
            f'{prefix}_attn_out_proj',
            input_shape=[self.batch_size, self.seq_len, self.embed_dim],
            output_features=self.embed_dim,
            group=group,
            verbose=verbose
        )
        
        # ===== LayerNorm 2 (CPU) =====
        ln2 = nn.LayerNorm(self.embed_dim)
        ln2.eval()
        self._profile_cpu_layer(
            f'{prefix}_norm2', ln2, x,
            input_shape=[self.batch_size, self.seq_len, self.embed_dim],
            output_shape=[self.batch_size, self.seq_len, self.embed_dim],
            layer_type='LayerNorm',
            group=group,
            verbose=verbose
        )
        
        # ===== FFN FC1 (Enclave) =====
        self._profile_linear_enclave(
            f'{prefix}_ffn_fc1',
            input_shape=[self.batch_size, self.seq_len, self.embed_dim],
            output_features=self.mlp_hidden,
            group=group,
            verbose=verbose
        )
        
        # ===== GELU (CPU) =====
        ffn_hidden = torch.randn(self.batch_size, self.seq_len, self.mlp_hidden)
        self._profile_gelu_cpu(
            f'{prefix}_ffn_gelu', ffn_hidden,
            input_shape=[self.batch_size, self.seq_len, self.mlp_hidden],
            output_shape=[self.batch_size, self.seq_len, self.mlp_hidden],
            group=group,
            verbose=verbose
        )
        
        # ===== FFN FC2 (Enclave) =====
        self._profile_linear_enclave(
            f'{prefix}_ffn_fc2',
            input_shape=[self.batch_size, self.seq_len, self.mlp_hidden],
            output_features=self.embed_dim,
            group=group,
            verbose=verbose
        )
    
    def _profile_classifier_enclave(self, verbose: bool):
        """Profile classifier Linear layer in Enclave."""
        # Head LayerNorm (CPU)
        import torch
        import torch.nn as nn
        
        torch.set_num_threads(1)
        
        x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        head_norm = nn.LayerNorm(self.embed_dim)
        head_norm.eval()
        self._profile_cpu_layer(
            'head_norm', head_norm, x,
            input_shape=[self.batch_size, self.seq_len, self.embed_dim],
            output_shape=[self.batch_size, self.seq_len, self.embed_dim],
            layer_type='LayerNorm',
            group='ClassHead',
            verbose=verbose
        )
        
        # Classifier (Enclave)
        self._profile_linear_enclave(
            'classifier',
            input_shape=[self.batch_size, self.embed_dim],
            output_features=self.num_classes,
            group='ClassHead',
            verbose=verbose
        )
    
    def _profile_linear_enclave(
        self, name: str, input_shape: List[int], 
        output_features: int, group: str, verbose: bool
    ):
        """Profile a Linear layer in Enclave mode."""
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.sgx_linear_base import SGXLinearBase
        from python.layers.input import SecretInputLayer
        from python.layers.output import SecretOutputLayer
        from python.sgx_net import SecretNeuralNetwork
        from python.utils.basic_utils import ExecutionModeOptions
        
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            # Flatten input for Linear layer
            if len(input_shape) == 3:
                flat_input_shape = [input_shape[0] * input_shape[1], input_shape[2]]
                input_features = input_shape[2]
                output_shape = [input_shape[0], input_shape[1], output_features]
            else:
                flat_input_shape = input_shape
                input_features = input_shape[1]
                output_shape = [input_shape[0], output_features]
            
            sid = 0
            layers = []
            
            input_layer = SecretInputLayer(
                sid, "input", flat_input_shape,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer)
            
            linear_layer = SGXLinearBase(
                sid, "linear",
                ExecutionModeOptions.Enclave,
                batch_size=flat_input_shape[0],
                n_output_features=output_features,
                n_input_features=input_features,
                manually_register_prev=True,
                manually_register_next=True
            )
            linear_layer.register_prev_layer(input_layer)
            layers.append(linear_layer)
            
            output_layer = SecretOutputLayer(
                sid, "output",
                ExecutionModeOptions.CPU,
                manually_register_prev=True
            )
            output_layer.register_prev_layer(linear_layer)
            layers.append(output_layer)
            
            secret_nn = SecretNeuralNetwork(sid, f"profile_{name}")
            secret_nn.set_eid(GlobalTensor.get_eid())
            secret_nn.set_layers(layers)
            
            times = []
            total_runs = self.warmup_iterations + self.num_iterations
            
            for run_idx in range(total_runs):
                test_input = torch.randn(*flat_input_shape)
                layers[0].set_input(test_input)
                
                start_time = time.perf_counter()
                layers[0].forward()
                layers[1].forward()
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                if run_idx >= self.warmup_iterations:
                    times.append(elapsed_ms)
            
            metrics = LayerMetrics(
                name=name,
                layer_type='Linear',
                group=group,
                execution_mode='Enclave',
                input_shape=input_shape,
                output_shape=output_shape,
                input_bytes=_shape_to_bytes(input_shape),
                output_bytes=_shape_to_bytes(output_shape),
                enclave_times=times,
                num_iterations=self.num_iterations
            )
            self.metrics[name] = metrics
            
            if verbose:
                mean_time = np.mean(times)
                std_time = np.std(times)
                print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (Enclave)")
            
        except Exception as e:
            print(f"✗ Error profiling {name}: {e}")
        
        finally:
            if GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()
    
    def _profile_cpu_layer(
        self, name: str, layer, x, 
        input_shape: List[int], output_shape: List[int],
        layer_type: str, group: str, verbose: bool
    ):
        """Profile a PyTorch layer on CPU."""
        import torch
        
        times = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(self.warmup_iterations):
                _ = layer(x)
            
            # Measure
            for _ in range(self.num_iterations):
                start = time.perf_counter()
                _ = layer(x)
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        metrics = LayerMetrics(
            name=name,
            layer_type=layer_type,
            group=group,
            execution_mode='CPU',
            input_shape=input_shape,
            output_shape=output_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(output_shape),
            cpu_times=times,
            num_iterations=self.num_iterations
        )
        self.metrics[name] = metrics
        
        if verbose:
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (CPU)")
    
    def _profile_matmul_cpu(
        self, name: str, a, b,
        input_shape: List[int], output_shape: List[int],
        group: str, verbose: bool
    ):
        """Profile matrix multiplication on CPU."""
        import torch
        
        times = []
        
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = torch.matmul(a, b)
            
            for _ in range(self.num_iterations):
                start = time.perf_counter()
                _ = torch.matmul(a, b)
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        metrics = LayerMetrics(
            name=name,
            layer_type='MatMul',
            group=group,
            execution_mode='CPU',
            input_shape=input_shape,
            output_shape=output_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(output_shape),
            cpu_times=times,
            num_iterations=self.num_iterations
        )
        self.metrics[name] = metrics
        
        if verbose:
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (CPU)")
    
    def _profile_softmax_cpu(
        self, name: str, x,
        input_shape: List[int], output_shape: List[int],
        group: str, verbose: bool
    ):
        """Profile softmax on CPU."""
        import torch
        import torch.nn.functional as F
        
        times = []
        
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = F.softmax(x, dim=-1)
            
            for _ in range(self.num_iterations):
                start = time.perf_counter()
                _ = F.softmax(x, dim=-1)
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        metrics = LayerMetrics(
            name=name,
            layer_type='Softmax',
            group=group,
            execution_mode='CPU',
            input_shape=input_shape,
            output_shape=output_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(output_shape),
            cpu_times=times,
            num_iterations=self.num_iterations
        )
        self.metrics[name] = metrics
        
        if verbose:
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (CPU)")
    
    def _profile_gelu_cpu(
        self, name: str, x,
        input_shape: List[int], output_shape: List[int],
        group: str, verbose: bool
    ):
        """Profile GELU on CPU."""
        import torch
        import torch.nn.functional as F
        
        times = []
        
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = F.gelu(x)
            
            for _ in range(self.num_iterations):
                start = time.perf_counter()
                _ = F.gelu(x)
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        metrics = LayerMetrics(
            name=name,
            layer_type='GELU',
            group=group,
            execution_mode='CPU',
            input_shape=input_shape,
            output_shape=output_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(output_shape),
            cpu_times=times,
            num_iterations=self.num_iterations
        )
        self.metrics[name] = metrics
        
        if verbose:
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (CPU)")
    
    def save_results(self, output_dir: str = 'experiments/data'):
        """Save profiling results to CSV and JSON."""
        os.makedirs(output_dir, exist_ok=True)
        
        variant = self.model_variant
        csv_path = os.path.join(output_dir, f'vit_{variant}_enclave_layers.csv')
        json_path = os.path.join(output_dir, f'vit_{variant}_enclave_layers.json')
        
        # CSV fieldnames
        fieldnames = [
            'name', 'type', 'group', 'execution_mode',
            'enclave_time_mean', 'enclave_time_std', 'enclave_time_min', 'enclave_time_max',
            'enclave_time_p95', 'enclave_time_p99',
            'cpu_time_mean', 'cpu_time_std', 'cpu_time_min', 'cpu_time_max',
            'input_bytes', 'output_bytes',
            'input_shape', 'output_shape',
            'dependencies', 'num_iterations'
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
        print(f"ViT-{self.model_variant.capitalize()} Enclave Profiling Summary")
        print(f"{'='*60}")
        
        # Separate by execution mode
        enclave_layers = {k: v for k, v in self.metrics.items() if v.execution_mode == 'Enclave'}
        cpu_layers = {k: v for k, v in self.metrics.items() if v.execution_mode == 'CPU'}
        
        # Calculate totals
        enclave_total = sum(m.enclave_time_mean for m in enclave_layers.values())
        cpu_total = sum(m.cpu_time_mean for m in cpu_layers.values())
        total_time = enclave_total + cpu_total
        
        print(f"\nExecution Mode Breakdown:")
        print(f"{'Mode':<15} {'Layers':>8} {'Time (ms)':>12} {'%':>8}")
        print(f"{'-'*45}")
        print(f"{'Enclave':<15} {len(enclave_layers):>8} {enclave_total:>12.3f} {enclave_total/total_time*100:>8.1f}%")
        print(f"{'CPU':<15} {len(cpu_layers):>8} {cpu_total:>12.3f} {cpu_total/total_time*100:>8.1f}%")
        print(f"{'-'*45}")
        print(f"{'Total':<15} {len(self.metrics):>8} {total_time:>12.3f} {'100.0':>8}%")
        
        # Layer type breakdown
        type_times = {}
        type_counts = {}
        for metrics in self.metrics.values():
            ltype = metrics.layer_type
            time_val = metrics.enclave_time_mean if metrics.execution_mode == 'Enclave' else metrics.cpu_time_mean
            if ltype not in type_times:
                type_times[ltype] = 0.0
                type_counts[ltype] = 0
            type_times[ltype] += time_val
            type_counts[ltype] += 1
        
        print(f"\nPer-Layer-Type Time:")
        print(f"{'Type':<15} {'Count':>8} {'Time (ms)':>12} {'%':>8}")
        print(f"{'-'*45}")
        for ltype in sorted(type_times.keys(), key=lambda x: type_times[x], reverse=True):
            time_ms = type_times[ltype]
            count = type_counts[ltype]
            pct = time_ms / total_time * 100 if total_time > 0 else 0
            print(f"{ltype:<15} {count:>8} {time_ms:>12.3f} {pct:>8.1f}%")
        
        # Group breakdown
        group_times = {}
        for metrics in self.metrics.values():
            group = metrics.group
            time_val = metrics.enclave_time_mean if metrics.execution_mode == 'Enclave' else metrics.cpu_time_mean
            if group not in group_times:
                group_times[group] = 0.0
            group_times[group] += time_val
        
        print(f"\nPer-Group Time:")
        print(f"{'Group':<20} {'Time (ms)':>12} {'%':>8}")
        print(f"{'-'*42}")
        for group, time_ms in group_times.items():
            pct = time_ms / total_time * 100 if total_time > 0 else 0
            print(f"{group:<20} {time_ms:>12.3f} {pct:>8.1f}%")
        print(f"{'-'*42}")
        print(f"{'Total':<20} {total_time:>12.3f} {'100.0':>8}%")
        
        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Profile ViT layers in Enclave')
    parser.add_argument('--model', type=str, default='base',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='ViT model variant')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of measurement iterations')
    parser.add_argument('--warmup', type=int, default=3,
                       help='Number of warmup iterations')
    parser.add_argument('--output-dir', type=str, default='experiments/data',
                       help='Output directory for results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        print("\nContinuing anyway (may fail)...\n")
    
    profiler = ViTEnclaveProfiler(
        model_variant=args.model,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
    )
    
    # Profile all layers
    profiler.profile_all(verbose=not args.quiet)
    
    # Print summary
    profiler.print_summary()
    
    # Save results
    profiler.save_results(args.output_dir)


if __name__ == '__main__':
    main()
