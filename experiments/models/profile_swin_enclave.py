#!/usr/bin/env python
"""
Swin Transformer Enclave Performance Profiler.

This script measures Swin Transformer layer execution in Enclave mode.

Usage:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python -m experiments.models.profile_swin_enclave --model tiny

Swin Transformer Key Features:
- Hierarchical structure with 4 stages
- Shifted Window Attention: W-MSA and SW-MSA alternating
- Window size: 7x7 (LOCAL attention, memory bounded!)
- Patch Merging for downsampling

Key Insight for TEE: Window attention bounds memory to window_size^2 = 49
tokens per window. This is much more memory-efficient than ViT's global
attention where attention matrix scales with sequence length.

Current Enclave Support Status:
- SGXLinearBase: ✓ Supported
- SGXConvBase: ✓ Supported
- LayerNorm: ✓ Supported
- Softmax: ✓ Supported
- GELU: ✓ Supported
- MatMul: ✓ Supported

Output: experiments/data/swin_{tiny|small|base}_enclave_layers.csv
"""

import sys
import time
import os
import csv
import json
import numpy as np
import argparse
from collections import OrderedDict

from typing import Dict, List, Optional, Any, Tuple

sys.path.insert(0, '.')

from experiments.models.profiler_utils import (
    LayerMetrics,
    calc_layer_memory_from_shapes,
    shape_to_bytes,
    print_memory_summary,
    infer_layer_dependencies,
)



def _shape_to_bytes(shape: List[int], dtype_size: int = 4) -> int:
    return int(np.prod(shape)) * dtype_size


def check_environment():
    """Check if the environment is properly set up."""
    print("Checking environment...")
    
    lib_path = "App/bin/enclave_bridge.so"
    if not os.path.exists(lib_path):
        print(f"✗ Enclave library not found at {lib_path}")
        print("  Run 'make' to build the project first")
        return False
    print(f"✓ Enclave library found")
    
    ld_preload = os.environ.get('LD_PRELOAD', '')
    if 'libstdc++.so.6' not in ld_preload:
        print("⚠ LD_PRELOAD may need to be set for libstdc++ compatibility")
        print("  Try: LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    else:
        print("✓ LD_PRELOAD set correctly")
    
    return True


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


class SwinEnclaveProfiler:
    """Profiler for Swin Transformer layers in Enclave mode."""
    
    def __init__(
        self, 
        model_variant: str = 'tiny',
        batch_size: int = 1,
        image_size: int = 224,
        patch_size: int = 4,
        num_classes: int = 1000,
        num_iterations: int = 10,
        warmup_iterations: int = 3,
    ):
        self.model_variant = model_variant
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
            
        config = SWIN_CONFIG[model_variant]
        self.embed_dim = config['embed_dim']
        self.depths = config['depths']
        self.num_heads = config['num_heads']
        self.window_size = config['window_size']
        self.mlp_ratio = config['mlp_ratio']
            
        self.metrics: Dict[str, LayerMetrics] = OrderedDict()
        self.reuse_single_enclave: bool = True
        self._runtime_stats: Dict[str, Dict[str, List[float]]] = {}
            
        # Reset interval for memory management
        self.ENCLAVE_RESET_INTERVAL = 3  # Reset every N blocks
    
    def profile_all(self, verbose: bool = True):
        """Profile all layers."""
        from python.enclave_interfaces import GlobalTensor
            
        if verbose:
            print(f"\n{'='*60}")
            print(f"Profiling Swin-{self.model_variant.capitalize()} in Enclave Mode")
            print(f"{'='*60}")
            print(f"Model Config: C={self.embed_dim}, depths={self.depths}, "
                  f"heads={self.num_heads}")
            print(f"Window size: {self.window_size} (LOCAL attention)")
            print(f"Image: {self.image_size}x{self.image_size}")
            print(f"Iterations: {self.num_iterations} (warmup: {self.warmup_iterations})")
            print(f"Enclave reset interval: every {self.ENCLAVE_RESET_INTERVAL} blocks")
            print(f"{'='*60}\n")
            
        H = self.image_size // self.patch_size  # 56
        W = self.image_size // self.patch_size  # 56
            
        if self.reuse_single_enclave:
            try:
                if not GlobalTensor.is_init_global_tensor:
                    GlobalTensor.init()
                
                # Patch Embedding
                if verbose:
                    print(f"\n--- Patch Embedding ---")
                self._profile_conv_enclave(
                    'patch_embed_conv',
                    in_channels=3, out_channels=self.embed_dim,
                    kernel_size=self.patch_size, stride=self.patch_size,
                    image_size=self.image_size,
                    group='PatchEmbed',
                    verbose=verbose
                )
                
                self._profile_layernorm_enclave(
                    'patch_embed_norm',
                    input_shape=[self.batch_size, H * W, self.embed_dim],
                    group='PatchEmbed',
                    verbose=verbose
                )
                
                # Stages
                dim = self.embed_dim
                block_counter = 0
                
                for i_stage in range(len(self.depths)):
                    stage_name = f'stage{i_stage}'
                    depth = self.depths[i_stage]
                    heads = self.num_heads[i_stage]
                    head_dim = dim // heads
                    
                    num_tokens = H * W
                    num_windows = (H // self.window_size) * (W // self.window_size)
                    window_tokens = self.window_size ** 2
                    
                    if verbose:
                        print(f"\n--- {stage_name.upper()} (H={H}, W={W}, C={dim}) ---")
                    
                    # Profile each block
                    for i_block in range(depth):
                        block_prefix = f'{stage_name}_block{i_block}'
                        group = f'{stage_name}_block{i_block}'
                        
                        # Reset Enclave periodically
                        if block_counter > 0 and block_counter % self.ENCLAVE_RESET_INTERVAL == 0:
                            if verbose:
                                print(f"  [Resetting Enclave (block {block_counter})...]")
                            if GlobalTensor.is_init_global_tensor:
                                GlobalTensor.destroy()
                            GlobalTensor.init()
                        block_counter += 1
                        
                        shift_size = 0 if (i_block % 2 == 0) else self.window_size // 2
                        block_type = "W-MSA" if shift_size == 0 else "SW-MSA"
                        
                        if verbose:
                            print(f"  Block {i_block} ({block_type}):")
                        
                        self._profile_swin_block_enclave(
                            block_prefix, group, dim, heads, head_dim,
                            num_tokens, num_windows, window_tokens,
                            verbose
                        )
                    
                    # Patch Merging
                    if i_stage < len(self.depths) - 1:
                        # Reset before merge
                        if verbose:
                            print(f"  [Resetting Enclave before merge...]")
                        if GlobalTensor.is_init_global_tensor:
                            GlobalTensor.destroy()
                        GlobalTensor.init()
                        
                        if verbose:
                            print(f"  Patch Merging:")
                        
                        self._profile_layernorm_enclave(
                            f'{stage_name}_merge_norm',
                            input_shape=[self.batch_size, (H // 2) * (W // 2), 4 * dim],
                            group=f'{stage_name}_merge',
                            verbose=verbose
                        )
                        
                        self._profile_linear_enclave(
                            f'{stage_name}_merge_reduction',
                            input_shape=[self.batch_size, (H // 2) * (W // 2), 4 * dim],
                            output_features=2 * dim,
                            input_features=4 * dim,
                            group=f'{stage_name}_merge',
                            verbose=verbose
                        )
                        
                        H = H // 2
                        W = W // 2
                        dim = dim * 2
                
                # Classification Head
                if verbose:
                    print(f"\n[Resetting Enclave before classifier...]")
                if GlobalTensor.is_init_global_tensor:
                    GlobalTensor.destroy()
                GlobalTensor.init()
                
                if verbose:
                    print(f"\n--- Classification Head ---")
                
                self._profile_layernorm_enclave(
                    'final_norm',
                    input_shape=[self.batch_size, H * W, dim],
                    group='ClassHead',
                    verbose=verbose
                )
                
                self._profile_linear_enclave(
                    'classifier',
                    input_shape=[self.batch_size, dim],
                    output_features=self.num_classes,
                    input_features=dim,
                    group='ClassHead',
                    verbose=verbose
                )
                
                return self.metrics
                
            finally:
                if GlobalTensor.is_init_global_tensor:
                    GlobalTensor.destroy()
        else:
            # Non-reuse mode
            pass
            
        return self.metrics
    
    def _profile_swin_block_enclave(
        self, block_prefix: str, group: str, dim: int, heads: int, head_dim: int,
        num_tokens: int, num_windows: int, window_tokens: int, verbose: bool
    ):
        """Profile a single Swin Transformer block in Enclave."""
        # LayerNorm 1
        self._profile_layernorm_enclave(
            f'{block_prefix}_norm1',
            input_shape=[self.batch_size, num_tokens, dim],
            group=group, verbose=verbose
            )
            
        # QKV projection
        self._profile_linear_enclave(
            f'{block_prefix}_attn_qkv',
            input_shape=[self.batch_size * num_windows, window_tokens, dim],
            output_features=dim * 3,
            group=group, verbose=verbose
            )
            
        # QK MatMul
        self._profile_matmul_enclave(
            f'{block_prefix}_attn_qk_matmul',
            input_shape1=[self.batch_size * num_windows, heads, window_tokens, head_dim],
            input_shape2=[self.batch_size * num_windows, heads, window_tokens, head_dim],
            transpose_b=True,
            scale=1.0 / float(np.sqrt(head_dim)),
            group=group, verbose=verbose
            )
            
        # Softmax
        self._profile_softmax_enclave(
            f'{block_prefix}_attn_softmax',
            input_shape=[self.batch_size * num_windows, heads, window_tokens, window_tokens],
            group=group, verbose=verbose
            )
            
        # Attention @ V MatMul
        self._profile_matmul_enclave(
            f'{block_prefix}_attn_v_matmul',
            input_shape1=[self.batch_size * num_windows, heads, window_tokens, window_tokens],
            input_shape2=[self.batch_size * num_windows, heads, window_tokens, head_dim],
            transpose_b=False,
            scale=None,
            group=group, verbose=verbose
            )
            
        # Output projection
        self._profile_linear_enclave(
            f'{block_prefix}_attn_out_proj',
            input_shape=[self.batch_size * num_windows, window_tokens, dim],
            output_features=dim,
            group=group, verbose=verbose
            )
            
        # LayerNorm 2
        self._profile_layernorm_enclave(
            f'{block_prefix}_norm2',
            input_shape=[self.batch_size, num_tokens, dim],
            group=group, verbose=verbose
            )
            
        # MLP FC1
        mlp_hidden = int(dim * self.mlp_ratio)
        self._profile_linear_enclave(
            f'{block_prefix}_mlp_fc1',
            input_shape=[self.batch_size, num_tokens, dim],
            output_features=mlp_hidden,
            group=group, verbose=verbose
            )
            
        # GELU
        self._profile_gelu_enclave(
            f'{block_prefix}_mlp_gelu',
            input_shape=[self.batch_size, num_tokens, mlp_hidden],
            group=group, verbose=verbose
            )
            
        # MLP FC2
        self._profile_linear_enclave(
            f'{block_prefix}_mlp_fc2',
            input_shape=[self.batch_size, num_tokens, mlp_hidden],
            output_features=dim,
            group=group, verbose=verbose
            )
    
    def _profile_conv_enclave(
        self, name: str, in_channels: int, out_channels: int,
        kernel_size: int, stride: int, image_size: int,
        group: str, verbose: bool
    ):
        """Profile a Conv2d layer in Enclave mode."""
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.sgx_conv_base import SGXConvBase
        from python.layers.input import SecretInputLayer
        from python.layers.output import SecretOutputLayer
        from python.sgx_net import SecretNeuralNetwork
        from python.utils.basic_utils import ExecutionModeOptions
            
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            input_shape = [self.batch_size, in_channels, image_size, image_size]
            output_h = (image_size - kernel_size) // stride + 1
            output_shape = [self.batch_size, out_channels, output_h, output_h]
            
            sid = 0
            layers = []
            lname = lambda s: f"{name}::{s}"
            
            input_layer = SecretInputLayer(
                sid, lname("input"), input_shape,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer)
            
            conv_layer = SGXConvBase(
                sid, lname("conv"),
                ExecutionModeOptions.Enclave,
                batch_size=self.batch_size,
                n_input_channel=in_channels,
                n_output_channel=out_channels,
                filter_hw=kernel_size,
                img_hw=image_size,
                stride=stride,
                padding=0,
                manually_register_prev=True,
                manually_register_next=True
            )
            conv_layer.register_prev_layer(input_layer)
            layers.append(conv_layer)
            
            output_layer = SecretOutputLayer(
                sid, lname("output"),
                ExecutionModeOptions.CPU,
                manually_register_prev=True
            )
            output_layer.register_prev_layer(conv_layer)
            layers.append(output_layer)
            
            secret_nn = SecretNeuralNetwork(sid, f"profile_{name}")
            secret_nn.set_eid(GlobalTensor.get_eid())
            secret_nn.set_layers(layers)
            
            times = []
            self._init_runtime_bucket(name)
            total_runs = self.warmup_iterations + self.num_iterations
            
            for run_idx in range(total_runs):
                test_input = torch.randn(*input_shape)
                layers[0].set_input(test_input)
                
                start_time = time.perf_counter()
                layers[0].forward()
                layers[1].forward()
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(layers[1].LayerName)
                
                if run_idx >= self.warmup_iterations:
                    times.append(elapsed_ms)
                    self._append_runtime_stats(name, stats)
            
            # Calculate memory footprint
            mem_info = calc_layer_memory_from_shapes(
                layer_type='Conv2d',
                input_shape=input_shape,
                output_shape=output_shape,
            )
            
            # Infer dependencies
            dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
            
            metrics = LayerMetrics(
                name=name,
                layer_type='Conv2d',
                group=group,
                execution_mode='Enclave',
                input_shape=input_shape,
                output_shape=output_shape,
                input_bytes=_shape_to_bytes(input_shape),
                output_bytes=_shape_to_bytes(output_shape),
                dependencies=dependencies,
                enclave_times=times,
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
                mean_time = np.mean(times) if times else 0
                std_time = np.std(times) if times else 0
                print(f"    {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (Enclave)")
            
        except Exception as e:
            print(f"✗ Error profiling {name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _profile_linear_enclave(
        self, name: str, input_shape: List[int], 
        output_features: int, group: str, verbose: bool,
        input_features: Optional[int] = None
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
            
            if len(input_shape) == 3:
                flat_input_shape = [input_shape[0] * input_shape[1], input_shape[2]]
                in_features = input_features if input_features else input_shape[2]
                output_shape = [input_shape[0], input_shape[1], output_features]
            else:
                flat_input_shape = input_shape
                in_features = input_features if input_features else input_shape[-1]
                output_shape = [input_shape[0], output_features]
            
            sid = 0
            layers = []
            lname = lambda s: f"{name}::{s}"
            
            input_layer = SecretInputLayer(
                sid, lname("input"), flat_input_shape,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer)
            
            linear_layer = SGXLinearBase(
                sid, lname("linear"),
                ExecutionModeOptions.Enclave,
                batch_size=flat_input_shape[0],
                n_output_features=output_features,
                n_input_features=in_features,
                manually_register_prev=True,
                manually_register_next=True
            )
            linear_layer.register_prev_layer(input_layer)
            layers.append(linear_layer)
            
            output_layer = SecretOutputLayer(
                sid, lname("output"),
                ExecutionModeOptions.CPU,
                manually_register_prev=True
            )
            output_layer.register_prev_layer(linear_layer)
            layers.append(output_layer)
            
            secret_nn = SecretNeuralNetwork(sid, f"profile_{name}")
            secret_nn.set_eid(GlobalTensor.get_eid())
            secret_nn.set_layers(layers)
            
            times = []
            self._init_runtime_bucket(name)
            total_runs = self.warmup_iterations + self.num_iterations
            
            for run_idx in range(total_runs):
                test_input = torch.randn(*flat_input_shape)
                layers[0].set_input(test_input)
                
                start_time = time.perf_counter()
                layers[0].forward()
                layers[1].forward()
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(layers[1].LayerName)
                
                if run_idx >= self.warmup_iterations:
                    times.append(elapsed_ms)
                    self._append_runtime_stats(name, stats)
            
            # Calculate memory footprint
            mem_info = calc_layer_memory_from_shapes(
                layer_type='Linear',
                input_shape=input_shape,
                output_shape=output_shape,
            )
            
            # Infer dependencies
            dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
            
            metrics = LayerMetrics(
                name=name,
                layer_type='Linear',
                group=group,
                execution_mode='Enclave',
                input_shape=input_shape,
                output_shape=output_shape,
                input_bytes=_shape_to_bytes(input_shape),
                output_bytes=_shape_to_bytes(output_shape),
                dependencies=dependencies,
                enclave_times=times,
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
                mean_time = np.mean(times) if times else 0
                std_time = np.std(times) if times else 0
                print(f"    {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (Enclave)")
            
        except Exception as e:
            print(f"✗ Error profiling {name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _profile_layernorm_enclave(
        self, name: str, input_shape: List[int], 
        group: str, verbose: bool
    ):
        """Profile a LayerNorm layer in Enclave mode."""
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.layer_norm import SecretLayerNormLayer
        from python.layers.input import SecretInputLayer
        from python.layers.output import SecretOutputLayer
        from python.sgx_net import SecretNeuralNetwork
        from python.utils.basic_utils import ExecutionModeOptions
            
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            output_shape = input_shape
            sid = 0
            layers = []
            lname = lambda s: f"{name}::{s}"
            
            input_layer = SecretInputLayer(
                sid, lname("input"), input_shape,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer)
            
            normalized_shape = input_shape[-1]
            
            ln_layer = SecretLayerNormLayer(
                sid, lname("layernorm"),
                ExecutionModeOptions.Enclave,
                normalized_shape=normalized_shape,
                manually_register_prev=True,
                manually_register_next=True
            )
            ln_layer.register_prev_layer(input_layer)
            layers.append(ln_layer)
            
            output_layer = SecretOutputLayer(
                sid, lname("output"),
                ExecutionModeOptions.CPU,
                manually_register_prev=True
            )
            output_layer.register_prev_layer(ln_layer)
            layers.append(output_layer)
            
            secret_nn = SecretNeuralNetwork(sid, f"profile_{name}")
            secret_nn.set_eid(GlobalTensor.get_eid())
            secret_nn.set_layers(layers)
            
            times = []
            self._init_runtime_bucket(name)
            total_runs = self.warmup_iterations + self.num_iterations
            
            for run_idx in range(total_runs):
                test_input = torch.randn(*input_shape)
                layers[0].set_input(test_input)
                
                start_time = time.perf_counter()
                layers[0].forward()
                layers[1].forward()
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(layers[1].LayerName)
                
                if run_idx >= self.warmup_iterations:
                    times.append(elapsed_ms)
                    self._append_runtime_stats(name, stats)
            
            # Calculate memory footprint
            mem_info = calc_layer_memory_from_shapes(
                layer_type='LayerNorm',
                input_shape=input_shape,
                output_shape=output_shape,
            )
            
            # Infer dependencies
            dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
            
            metrics = LayerMetrics(
                name=name,
                layer_type='LayerNorm',
                group=group,
                execution_mode='Enclave',
                input_shape=input_shape,
                output_shape=output_shape,
                input_bytes=_shape_to_bytes(input_shape),
                output_bytes=_shape_to_bytes(output_shape),
                dependencies=dependencies,
                enclave_times=times,
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
                mean_time = np.mean(times) if times else 0
                std_time = np.std(times) if times else 0
                print(f"    {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (Enclave)")
            
        except Exception as e:
            print(f"✗ Error profiling {name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _profile_softmax_enclave(
        self, name: str, input_shape: List[int], 
        group: str, verbose: bool
    ):
        """Profile a Softmax layer in Enclave mode."""
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.softmax import SecretSoftmaxLayer
        from python.layers.input import SecretInputLayer
        from python.layers.output import SecretOutputLayer
        from python.sgx_net import SecretNeuralNetwork
        from python.utils.basic_utils import ExecutionModeOptions
            
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            output_shape = input_shape
            sid = 0
            layers = []
            lname = lambda s: f"{name}::{s}"
            
            input_layer = SecretInputLayer(
                sid, lname("input"), input_shape,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer)
            
            softmax_layer = SecretSoftmaxLayer(
                sid, lname("softmax"),
                ExecutionModeOptions.Enclave,
                dim=-1,
                manually_register_prev=True,
                manually_register_next=True
            )
            softmax_layer.register_prev_layer(input_layer)
            layers.append(softmax_layer)
            
            output_layer = SecretOutputLayer(
                sid, lname("output"),
                ExecutionModeOptions.CPU,
                manually_register_prev=True
            )
            output_layer.register_prev_layer(softmax_layer)
            layers.append(output_layer)
            
            secret_nn = SecretNeuralNetwork(sid, f"profile_{name}")
            secret_nn.set_eid(GlobalTensor.get_eid())
            secret_nn.set_layers(layers)
            
            times = []
            self._init_runtime_bucket(name)
            total_runs = self.warmup_iterations + self.num_iterations
            
            for run_idx in range(total_runs):
                test_input = torch.randn(*input_shape)
                layers[0].set_input(test_input)
                
                start_time = time.perf_counter()
                layers[0].forward()
                layers[1].forward()
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(layers[1].LayerName)
                
                if run_idx >= self.warmup_iterations:
                    times.append(elapsed_ms)
                    self._append_runtime_stats(name, stats)
            
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
                execution_mode='Enclave',
                input_shape=input_shape,
                output_shape=output_shape,
                input_bytes=_shape_to_bytes(input_shape),
                output_bytes=_shape_to_bytes(output_shape),
                dependencies=dependencies,
                enclave_times=times,
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
                mean_time = np.mean(times) if times else 0
                std_time = np.std(times) if times else 0
                print(f"    {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (Enclave)")
            
        except Exception as e:
            print(f"✗ Error profiling {name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _profile_gelu_enclave(
        self, name: str, input_shape: List[int], 
        group: str, verbose: bool
    ):
        """Profile a GELU layer in Enclave mode."""
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.gelu import SecretGELULayer
        from python.layers.input import SecretInputLayer
        from python.layers.output import SecretOutputLayer
        from python.sgx_net import SecretNeuralNetwork
        from python.utils.basic_utils import ExecutionModeOptions
            
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            output_shape = input_shape
            sid = 0
            layers = []
            lname = lambda s: f"{name}::{s}"
            
            input_layer = SecretInputLayer(
                sid, lname("input"), input_shape,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer)
            
            gelu_layer = SecretGELULayer(
                sid, lname("gelu"),
                ExecutionModeOptions.Enclave,
                approximate=True,
                manually_register_prev=True,
                manually_register_next=True
            )
            gelu_layer.register_prev_layer(input_layer)
            layers.append(gelu_layer)
            
            output_layer = SecretOutputLayer(
                sid, lname("output"),
                ExecutionModeOptions.CPU,
                manually_register_prev=True
            )
            output_layer.register_prev_layer(gelu_layer)
            layers.append(output_layer)
            
            secret_nn = SecretNeuralNetwork(sid, f"profile_{name}")
            secret_nn.set_eid(GlobalTensor.get_eid())
            secret_nn.set_layers(layers)
            
            times = []
            self._init_runtime_bucket(name)
            total_runs = self.warmup_iterations + self.num_iterations
            
            for run_idx in range(total_runs):
                test_input = torch.randn(*input_shape)
                layers[0].set_input(test_input)
                
                start_time = time.perf_counter()
                layers[0].forward()
                layers[1].forward()
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(layers[1].LayerName)
                
                if run_idx >= self.warmup_iterations:
                    times.append(elapsed_ms)
                    self._append_runtime_stats(name, stats)
            
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
                execution_mode='Enclave',
                input_shape=input_shape,
                output_shape=output_shape,
                input_bytes=_shape_to_bytes(input_shape),
                output_bytes=_shape_to_bytes(output_shape),
                dependencies=dependencies,
                enclave_times=times,
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
                mean_time = np.mean(times) if times else 0
                std_time = np.std(times) if times else 0
                print(f"    {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (Enclave)")
            
        except Exception as e:
            print(f"✗ Error profiling {name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _profile_matmul_enclave(
        self, name: str, 
        input_shape1: List[int], input_shape2: List[int],
        transpose_b: bool, scale: Optional[float],
        group: str, verbose: bool
    ):
        """Profile a MatMul layer in Enclave mode."""
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.matmul import SecretMatMulLayer
        from python.layers.input import SecretInputLayer
        from python.layers.output import SecretOutputLayer
        from python.sgx_net import SecretNeuralNetwork
        from python.utils.basic_utils import ExecutionModeOptions
            
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            if transpose_b:
                output_shape = input_shape1[:-2] + [input_shape1[-2], input_shape2[-2]]
            else:
                output_shape = input_shape1[:-2] + [input_shape1[-2], input_shape2[-1]]
            
            sid = 0
            layers = []
            lname = lambda s: f"{name}::{s}"
            
            input_layer1 = SecretInputLayer(
                sid, lname("input1"), input_shape1,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer1)
            
            input_layer2 = SecretInputLayer(
                sid, lname("input2"), input_shape2,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer2)
            
            matmul_layer = SecretMatMulLayer(
                sid, lname("matmul"),
                ExecutionModeOptions.Enclave,
                transpose_b=transpose_b,
                scale=scale,
                manually_register_prev=True,
                manually_register_next=True
            )
            matmul_layer.register_prev_layer(input_layer1)
            matmul_layer.register_prev_layer(input_layer2)
            layers.append(matmul_layer)
            
            output_layer = SecretOutputLayer(
                sid, lname("output"),
                ExecutionModeOptions.CPU,
                manually_register_prev=True
            )
            output_layer.register_prev_layer(matmul_layer)
            layers.append(output_layer)
            
            secret_nn = SecretNeuralNetwork(sid, f"profile_{name}")
            secret_nn.set_eid(GlobalTensor.get_eid())
            secret_nn.set_layers(layers)
            
            times = []
            self._init_runtime_bucket(name)
            total_runs = self.warmup_iterations + self.num_iterations
            
            for run_idx in range(total_runs):
                test_input1 = torch.randn(*input_shape1)
                test_input2 = torch.randn(*input_shape2)
                layers[0].set_input(test_input1)
                layers[1].set_input(test_input2)
                
                start_time = time.perf_counter()
                layers[0].forward()
                layers[1].forward()
                layers[2].forward()
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(layers[2].LayerName)
                
                if run_idx >= self.warmup_iterations:
                    times.append(elapsed_ms)
                    self._append_runtime_stats(name, stats)
            
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
                execution_mode='Enclave',
                input_shape=input_shape1,
                output_shape=output_shape,
                input_bytes=_shape_to_bytes(input_shape1) + _shape_to_bytes(input_shape2),
                output_bytes=_shape_to_bytes(output_shape),
                dependencies=dependencies,
                enclave_times=times,
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
                mean_time = np.mean(times) if times else 0
                std_time = np.std(times) if times else 0
                print(f"    {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (Enclave)")
            
        except Exception as e:
            print(f"✗ Error profiling {name}: {e}")
            import traceback
            traceback.print_exc()
    
    def save_results(self, output_dir: str = 'experiments/data'):
        """Save profiling results to CSV and JSON."""
        for metrics in self.metrics.values():
            metrics.compute_statistics()
            
        os.makedirs(output_dir, exist_ok=True)
            
        variant = self.model_variant
        csv_path = os.path.join(output_dir, f'swin_{variant}_enclave_layers.csv')
        json_path = os.path.join(output_dir, f'swin_{variant}_enclave_layers.json')
            
        fieldnames = [
            'name', 'type', 'group', 'execution_mode',
            'enclave_time_mean', 'enclave_time_std', 'enclave_time_min', 'enclave_time_max',
            'enclave_time_p95', 'enclave_time_p99',
            'cpu_time_mean', 'cpu_time_std', 'cpu_time_min', 'cpu_time_max',
            'input_bytes', 'output_bytes',
            'input_shape', 'output_shape',
            'dependencies', 'num_iterations',
            'xfer_edges_json', 'xfer_total_mean_ms', 'compute_mean_ms',
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
                    'note': f'Swin uses LOCAL window attention ({self.window_size}x{self.window_size})',
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
        for metrics in self.metrics.values():
            metrics.compute_statistics()
            
        print(f"\n{'='*60}")
        print(f"Swin-{self.model_variant.capitalize()} Enclave Profiling Summary")
        print(f"  (Uses LOCAL window attention: {self.window_size}x{self.window_size})")
        print(f"{'='*60}")
            
        total_time = sum(m.enclave_time_mean for m in self.metrics.values())
            
        # Per stage
        stage_times = {}
        for metrics in self.metrics.values():
            group = metrics.group
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
            stage_times[stage] += metrics.enclave_time_mean
            
        print(f"\nPer-Stage Enclave Time:")
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
            type_times[ltype] += metrics.enclave_time_mean
            type_counts[ltype] += 1
            
        print(f"\nPer-Layer-Type Enclave Time:")
        print(f"{'Type':<15} {'Count':>8} {'Time (ms)':>12} {'%':>8}")
        print(f"{'-'*45}")
        for ltype in sorted(type_times.keys(), key=lambda x: type_times[x], reverse=True):
            time_ms = type_times[ltype]
            count = type_counts[ltype]
            pct = time_ms / total_time * 100 if total_time > 0 else 0
            print(f"{ltype:<15} {count:>8} {time_ms:>12.3f} {pct:>8.1f}%")
            
        print(f"\n{'='*60}\n")
            
        # Print memory summary
        print_memory_summary(self.metrics, "Swin Transformer Enclave Memory Analysis")
    
    def _init_runtime_bucket(self, name: str):
        self._runtime_stats[name] = {
            "get_ms": [],
            "get2_ms": [],
            "compute_ms": [],
            "store_ms": [],
        }
    
    def _append_runtime_stats(self, name: str, stats: Dict[str, float]):
        if name not in self._runtime_stats:
            self._init_runtime_bucket(name)
        self._runtime_stats[name]["get_ms"].append(float(stats.get("get_ms", 0.0)))
        self._runtime_stats[name]["get2_ms"].append(float(stats.get("get2_ms", 0.0)))
        self._runtime_stats[name]["compute_ms"].append(float(stats.get("compute_ms", 0.0)))
        self._runtime_stats[name]["store_ms"].append(float(stats.get("store_ms", 0.0)))


def main():
    parser = argparse.ArgumentParser(description='Profile Swin Transformer layers in Enclave')
    parser.add_argument('--model', type=str, default='tiny',
                       choices=['tiny', 'small', 'base'],
                       help='Swin Transformer model variant')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of measurement iterations')
    parser.add_argument('--warmup', type=int, default=3,
                       help='Number of warmup iterations')
    parser.add_argument('--output-dir', type=str, default='experiments/data',
                       help='Output directory for results')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    if not check_environment():
        print("\nContinuing anyway (may fail)...\n")
    
    profiler = SwinEnclaveProfiler(
        model_variant=args.model,
        image_size=args.image_size,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
    )
    
    profiler.profile_all(verbose=not args.quiet)
    profiler.print_summary()
    profiler.save_results(args.output_dir)


if __name__ == '__main__':
    main()
