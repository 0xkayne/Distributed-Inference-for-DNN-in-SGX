#!/usr/bin/env python
"""
Vision Transformer (ViT) Enclave Performance Profiler.

This script measures ViT layer execution in Enclave mode:
- All layers (Conv, Linear, LayerNorm, Softmax, GELU, MatMul) run in Enclave

Usage:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python -m experiments.models.profile_vit_enclave

Current Enclave Support Status:
- SGXConvBase: ✓ Supported
- SGXLinearBase: ✓ Supported  
- LayerNorm: ✓ Supported
- Softmax: ✓ Supported
- GELU: ✓ Supported
- MatMul: ✓ Supported

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
from typing import Dict, List, Optional, Any

sys.path.insert(0, '.')

from experiments.models.profiler_utils import (
    LayerMetrics,
    calc_layer_memory_from_shapes,
    shape_to_bytes,
    print_memory_summary,
    infer_layer_dependencies,
)


def _shape_to_bytes(shape: List[int], dtype_size: int = 4) -> int:
    """Convert shape to bytes (float32 = 4 bytes)."""
    return shape_to_bytes(shape, dtype_size)


def check_environment():
    """Check if the environment is properly set up."""
    import subprocess
    
    print("Checking environment...")
    
    # Check if enclave library exists
    lib_path = "App/bin/enclave_bridge.so"
    if not os.path.exists(lib_path):
        print(f"✗ Enclave library not found at {lib_path}")
        print("  Run 'make' to build the project first")
        return False
    print(f"✓ Enclave library found")
    
    # Check LD_PRELOAD
    ld_preload = os.environ.get('LD_PRELOAD', '')
    if 'libstdc++.so.6' not in ld_preload:
        print("⚠ LD_PRELOAD may need to be set for libstdc++ compatibility")
        print("  Try: LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    else:
        print("✓ LD_PRELOAD set correctly")
    
    return True


# ============================================================================
# ViT Model Configurations
# ============================================================================

VIT_CONFIGS = {
    'tiny': {
        'embed_dim': 192,
        'num_heads': 3,
        'num_layers': 12,
        'mlp_ratio': 4,
    },
    'small': {
        'embed_dim': 384,
        'num_heads': 6,
        'num_layers': 12,
        'mlp_ratio': 4,
    },
    'base': {
        'embed_dim': 768,
        'num_heads': 12,
        'num_layers': 12,
        'mlp_ratio': 4,
    },
    'large': {
        'embed_dim': 1024,
        'num_heads': 16,
        'num_layers': 24,
        'mlp_ratio': 4,
    }
}


class ViTEnclaveProfiler:
    """Profiler for ViT layers in Enclave mode."""
    
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
        
        # Get config
        config = VIT_CONFIGS[model_variant]
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.mlp_ratio = config['mlp_ratio']
        self.mlp_hidden = self.embed_dim * self.mlp_ratio
        self.head_dim = self.embed_dim // self.num_heads
        
        # Patch embedding params
        self.patch_size = 16
        self.num_patches = (img_size // self.patch_size) ** 2
        self.seq_len = self.num_patches + 1  # +1 for CLS token
        
        # Store metrics
        self.metrics: Dict[str, LayerMetrics] = OrderedDict()

        # Reuse a single enclave for the whole profiling run.
        # This avoids repeated SGX init/destroy churn which can eventually crash (0x1006)
        # during long layer-by-layer profiling.
        self.reuse_single_enclave: bool = True
    
    def profile_all(self, verbose: bool = True):
        """Profile all layers."""
        from python.enclave_interfaces import GlobalTensor
        if verbose:
            print(f"\n{'='*60}")
            print(f"Profiling ViT-{self.model_variant.capitalize()} in Enclave Mode")
            print(f"{'='*60}")
            print(f"Model Config: embed_dim={self.embed_dim}, heads={self.num_heads}, "
                  f"layers={self.num_layers}")
            print(f"Sequence: patches={self.num_patches}, seq_len={self.seq_len}")
            print(f"Iterations: {self.num_iterations} (warmup: {self.warmup_iterations})")
            print(f"{'='*60}\n")

        # Reuse one enclave for the full run (recommended).
        # We namespace every layer name by its profile name to avoid tensor-tag collisions.
        if self.reuse_single_enclave:
            try:
                if not GlobalTensor.is_init_global_tensor:
                    GlobalTensor.init()

                # Profile Conv layer (Enclave)
                self._profile_patch_embed_enclave(verbose)

                # Profile all layers in each block
                for block_idx in range(self.num_layers):
                    if verbose:
                        print(f"\n--- Block {block_idx} ---")
                    self._profile_block(block_idx, verbose)

                # Profile classifier
                if verbose:
                    print(f"\n--- Classifier Head ---")
                self._profile_classifier_enclave(verbose)

                return self.metrics
            finally:
                if GlobalTensor.is_init_global_tensor:
                    GlobalTensor.destroy()
        else:
            # Legacy behavior: each sub-profile initializes/destroys enclave.
            self._profile_patch_embed_enclave(verbose)
            for block_idx in range(self.num_layers):
                if verbose:
                    print(f"\n--- Block {block_idx} ---")
                self._profile_block(block_idx, verbose)
            if verbose:
                print(f"\n--- Classifier Head ---")
            self._profile_classifier_enclave(verbose)
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
        
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            input_shape = [self.batch_size, 3, self.img_size, self.img_size]
            output_shape = [self.batch_size, self.embed_dim, 
                          self.img_size // self.patch_size, 
                          self.img_size // self.patch_size]
            
            sid = 0
            layers = []

            # Namespace layer names to avoid tensor-tag collisions when reusing one enclave
            lname = lambda s: f"patch_embed::{s}"
            
            input_layer = SecretInputLayer(
                sid, lname("input"), input_shape,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer)
            
            conv_layer = SGXConvBase(
                sid, lname("conv"),
                ExecutionModeOptions.Enclave,
                n_output_channel=self.embed_dim,
                n_input_channel=3,
                filter_hw=self.patch_size,
                stride=self.patch_size,
                padding=0,
                bias=True,
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

                # Fetch enclave runtime breakdown for conv layer
                stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(layers[1].LayerName)
                
                if run_idx >= self.warmup_iterations:
                    times.append(elapsed_ms)
                    # Map stats to breakdown lists
                    # Conv is single-input: get_ms/compute_ms/store_ms
                    self._append_runtime_stats('patch_embed', stats)
            
            # Calculate memory footprint
            mem_info = calc_layer_memory_from_shapes(
                layer_type='Conv2d',
                input_shape=input_shape,
                output_shape=output_shape,
                kernel_size=self.patch_size,
            )
            
            # Infer dependencies
            dependencies = infer_layer_dependencies('patch_embed', list(self.metrics.keys()) + ['patch_embed'])
            
            metrics = LayerMetrics(
                name='patch_embed',
                layer_type='Conv2d',
                group='PatchEmbed',
                execution_mode='Enclave',
                input_shape=input_shape,
                output_shape=output_shape,
                input_bytes=_shape_to_bytes(input_shape),
                output_bytes=_shape_to_bytes(output_shape),
                dependencies=dependencies,
                enclave_times=times,
                enclave_get_ms=self._runtime_stats['patch_embed']['get_ms'],
                enclave_get2_ms=self._runtime_stats['patch_embed']['get2_ms'],
                enclave_compute_ms=self._runtime_stats['patch_embed']['compute_ms'],
                enclave_store_ms=self._runtime_stats['patch_embed']['store_ms'],
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
            self.metrics['patch_embed'] = metrics
            
            if verbose:
                mean_time = np.mean(times)
                std_time = np.std(times)
                print(f"  {'patch_embed':40} {mean_time:8.3f} ± {std_time:6.3f} ms (Enclave)")
            
        except Exception as e:
            print(f"✗ Error profiling patch_embed: {e}")
        
        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()
    
    def _profile_block(self, block_idx: int, verbose: bool):
        """Profile a single Transformer block - ALL layers in Enclave."""
        import torch
        
        prefix = f'block{block_idx}'
        group = f'Block{block_idx}'
        
        torch.set_num_threads(1)
        
        # ===== LayerNorm 1 (Enclave) =====
        self._profile_layernorm_enclave(
            f'{prefix}_norm1',
            input_shape=[self.batch_size, self.seq_len, self.embed_dim],
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
        
        # ===== QK MatMul (Enclave) =====
        # Q @ K^T with scaling 1/sqrt(head_dim)
        self._profile_matmul_enclave(
            f'{prefix}_attn_qk_matmul',
            input_shape1=[self.batch_size, self.num_heads, self.seq_len, self.head_dim],  # Q
            input_shape2=[self.batch_size, self.num_heads, self.seq_len, self.head_dim],  # K
            transpose_b=True,
            scale=1.0 / float(np.sqrt(self.head_dim)),
            group=group,
            verbose=verbose
        )
        
        # ===== Softmax (Enclave) =====
        self._profile_softmax_enclave(
            f'{prefix}_attn_softmax',
            input_shape=[self.batch_size, self.num_heads, self.seq_len, self.seq_len],
            group=group,
            verbose=verbose
        )
        
        # ===== Attention @ V MatMul (Enclave) =====
        self._profile_matmul_enclave(
            f'{prefix}_attn_v_matmul',
            input_shape1=[self.batch_size, self.num_heads, self.seq_len, self.seq_len],  # Attn weights
            input_shape2=[self.batch_size, self.num_heads, self.seq_len, self.head_dim],  # V
            transpose_b=False,
            scale=None,
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
        
        # ===== LayerNorm 2 (Enclave) =====
        self._profile_layernorm_enclave(
            f'{prefix}_norm2',
            input_shape=[self.batch_size, self.seq_len, self.embed_dim],
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
        
        # ===== GELU (Enclave) =====
        self._profile_gelu_enclave(
            f'{prefix}_ffn_gelu',
            input_shape=[self.batch_size, self.seq_len, self.mlp_hidden],
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
        """Profile classifier head in Enclave."""
        # Head LayerNorm (Enclave)
        self._profile_layernorm_enclave(
            'head_norm',
            input_shape=[self.batch_size, self.embed_dim],
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
                n_input_features=input_features,
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
                enclave_get_ms=self._runtime_stats[name]['get_ms'],
                enclave_get2_ms=self._runtime_stats[name]['get2_ms'],
                enclave_compute_ms=self._runtime_stats[name]['compute_ms'],
                enclave_store_ms=self._runtime_stats[name]['store_ms'],
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
                print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (Enclave)")
            
        except Exception as e:
            print(f"✗ Error profiling {name}: {e}")
        
        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()
    
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
            
            # Normalized shape is the last dimension
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
                enclave_get_ms=self._runtime_stats[name]['get_ms'],
                enclave_get2_ms=self._runtime_stats[name]['get2_ms'],
                enclave_compute_ms=self._runtime_stats[name]['compute_ms'],
                enclave_store_ms=self._runtime_stats[name]['store_ms'],
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
                print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (Enclave)")
            
        except Exception as e:
            print(f"✗ Error profiling {name}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()
    
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
            
            # Calculate memory footprint (Softmax has no weights)
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
                enclave_get_ms=self._runtime_stats[name]['get_ms'],
                enclave_get2_ms=self._runtime_stats[name]['get2_ms'],
                enclave_compute_ms=self._runtime_stats[name]['compute_ms'],
                enclave_store_ms=self._runtime_stats[name]['store_ms'],
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
                print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (Enclave)")
            
        except Exception as e:
            print(f"✗ Error profiling {name}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()
    
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
            
            # Calculate memory footprint (GELU has no weights)
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
                enclave_get_ms=self._runtime_stats[name]['get_ms'],
                enclave_get2_ms=self._runtime_stats[name]['get2_ms'],
                enclave_compute_ms=self._runtime_stats[name]['compute_ms'],
                enclave_store_ms=self._runtime_stats[name]['store_ms'],
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
                print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (Enclave)")
            
        except Exception as e:
            print(f"✗ Error profiling {name}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()
    
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
            
            # Calculate output shape
            if transpose_b:
                output_shape = input_shape1[:-2] + [input_shape1[-2], input_shape2[-2]]
            else:
                output_shape = input_shape1[:-2] + [input_shape1[-2], input_shape2[-1]]
            
            sid = 0
            layers = []

            lname = lambda s: f"{name}::{s}"
            
            # Input layer 1
            input_layer1 = SecretInputLayer(
                sid, lname("input1"), input_shape1,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer1)
            
            # Input layer 2
            input_layer2 = SecretInputLayer(
                sid, lname("input2"), input_shape2,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer2)
            
            # MatMul layer
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
            
            # Calculate memory footprint (MatMul has no weights)
            # Use combined input shape for memory calculation
            combined_input_shape = input_shape1  # Primary input shape for memory calc
            mem_info = calc_layer_memory_from_shapes(
                layer_type='MatMul',
                input_shape=combined_input_shape,
                output_shape=output_shape,
            )
            # Add second input to activation bytes
            mem_info['activation_bytes'] += _shape_to_bytes(input_shape2)
            mem_info['cpu_memory_bytes'] += _shape_to_bytes(input_shape2)
            mem_info['tee_memory_bytes'] += _shape_to_bytes(input_shape2)
            mem_info['tee_total_memory_bytes'] += _shape_to_bytes(input_shape2)
            
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
                enclave_get_ms=self._runtime_stats[name]['get_ms'],
                enclave_get2_ms=self._runtime_stats[name]['get2_ms'],
                enclave_compute_ms=self._runtime_stats[name]['compute_ms'],
                enclave_store_ms=self._runtime_stats[name]['store_ms'],
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
                print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (Enclave)")
            
        except Exception as e:
            print(f"✗ Error profiling {name}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()
    
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
        
        # Calculate memory footprint (MatMul has no weights)
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
            execution_mode='CPU',
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
            print(f"  {name:40} {mean_time:8.3f} ± {std_time:6.3f} ms (CPU)")
    
    def save_results(self, output_dir: str = 'experiments/data'):
        """Save profiling results to CSV and JSON."""
        # Compute statistics for all metrics before saving
        for metrics in self.metrics.values():
            metrics.compute_statistics()
        
        os.makedirs(output_dir, exist_ok=True)
        
        variant = self.model_variant
        csv_path = os.path.join(output_dir, f'vit_{variant}_enclave_layers.csv')
        json_path = os.path.join(output_dir, f'vit_{variant}_enclave_layers.json')
        
        # CSV fieldnames (with memory analysis columns)
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
                # Build edge-level xfer JSON (mean values)
                xfer_edges_json, xfer_total_mean_ms, compute_mean_ms = self._build_xfer_edges_for_row(metrics)
                row['xfer_edges_json'] = xfer_edges_json
                row['xfer_total_mean_ms'] = xfer_total_mean_ms
                row['compute_mean_ms'] = compute_mean_ms
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
                'layers': [
                    dict(m.to_dict(), **{
                        'xfer_edges_json': self._build_xfer_edges_for_row(m)[0],
                        'xfer_total_mean_ms': self._build_xfer_edges_for_row(m)[1],
                        'compute_mean_ms': self._build_xfer_edges_for_row(m)[2],
                    })
                    for m in self.metrics.values()
                ],
            }, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")
        
        return csv_path, json_path
    
    def print_summary(self):
        """Print summary statistics."""
        # Compute statistics for all metrics before printing
        for metrics in self.metrics.values():
            metrics.compute_statistics()
        
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
        if total_time > 0:
            print(f"{'Enclave':<15} {len(enclave_layers):>8} {enclave_total:>12.3f} {enclave_total/total_time*100:>8.1f}%")
            print(f"{'CPU':<15} {len(cpu_layers):>8} {cpu_total:>12.3f} {cpu_total/total_time*100:>8.1f}%")
        else:
            print(f"{'Enclave':<15} {len(enclave_layers):>8} {enclave_total:>12.3f} {'N/A':>8}")
            print(f"{'CPU':<15} {len(cpu_layers):>8} {cpu_total:>12.3f} {'N/A':>8}")
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
        
        # Print memory summary
        print_memory_summary(self.metrics, f"ViT-{self.model_variant.capitalize()} Enclave Memory Analysis")

    # -------------------------------------------------------------------------
    # Runtime stats helpers (enclave get/compute/store)
    # -------------------------------------------------------------------------
    def _init_runtime_bucket(self, name: str):
        if not hasattr(self, "_runtime_stats"):
            self._runtime_stats = {}
        self._runtime_stats[name] = {
            "get_ms": [],
            "get2_ms": [],
            "compute_ms": [],
            "store_ms": [],
        }

    def _append_runtime_stats(self, name: str, stats: Dict[str, float]):
        if not hasattr(self, "_runtime_stats") or name not in self._runtime_stats:
            self._init_runtime_bucket(name)
        self._runtime_stats[name]["get_ms"].append(float(stats.get("get_ms", 0.0)))
        self._runtime_stats[name]["get2_ms"].append(float(stats.get("get2_ms", 0.0)))
        self._runtime_stats[name]["compute_ms"].append(float(stats.get("compute_ms", 0.0)))
        self._runtime_stats[name]["store_ms"].append(float(stats.get("store_ms", 0.0)))

    def _build_xfer_edges_for_row(self, m: LayerMetrics):
        """
        Build edge-level xfer JSON (mean) for CSV/JSON.
        This profiler uses small sub-graphs, so we use canonical internal node names:
          - single-input: input -> op, op -> output
          - matmul: input1 -> matmul, input2 -> matmul, matmul -> output
        """
        def mean(xs: List[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0

        get_mean = mean(m.enclave_get_ms)
        get2_mean = mean(m.enclave_get2_ms)
        store_mean = mean(m.enclave_store_ms)
        compute_mean = mean(m.enclave_compute_ms)

        edges = {}
        if m.layer_type == "MatMul":
            edges["input1->matmul"] = {"enclave_get_ms": get_mean}
            edges["input2->matmul"] = {"enclave_get_ms": get2_mean}
            edges["matmul->output"] = {"enclave_store_ms": store_mean}
            xfer_total = get_mean + get2_mean + store_mean
        else:
            edges["input->op"] = {"enclave_get_ms": get_mean}
            edges["op->output"] = {"enclave_store_ms": store_mean}
            xfer_total = get_mean + store_mean

        return json.dumps(edges), xfer_total, compute_mean


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
