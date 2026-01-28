#!/usr/bin/env python
"""
Video Swin Transformer 3D Enclave Performance Profiler.

Profiles Video Swin Transformer (3D) layer execution in Enclave mode with:
- 3D convolution for patch embedding
- 3D window partition and reverse
- Cyclic shift for SW-MSA
- Multi-head attention (with per-head profiling option)
- Relative position bias

Usage:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python -m experiments.models.profile_swin_enclave --model tiny

Output: experiments/data/swin_{model}_enclave_per_head_layers.csv
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


# Video Swin 3D Configurations
SWIN_3D_CONFIG = {
    'tiny': {
        'embed_dim': 96,
        'depths': (2, 2, 6, 2),
        'num_heads': (3, 6, 12, 24),
        'window_size': (2, 7, 7),  # 3D: (Wd, Wh, Ww)
        'mlp_ratio': 4.0,
        'patch_size': (2, 4, 4),  # 3D: (Pd, Ph, Pw)
    },
    'small': {
        'embed_dim': 96,
        'depths': (2, 2, 18, 2),
        'num_heads': (3, 6, 12, 24),
        'window_size': (2, 7, 7),
        'mlp_ratio': 4.0,
        'patch_size': (2, 4, 4),
    },
    'base': {
        'embed_dim': 128,
        'depths': (2, 2, 18, 2),
        'num_heads': (4, 8, 16, 32),
        'window_size': (2, 7, 7),
        'mlp_ratio': 4.0,
        'patch_size': (2, 4, 4),
    }
}


class VideoSwinEnclaveProfiler:
    """Profiler for Video Swin Transformer 3D layers in Enclave mode."""
    
    def __init__(
        self,
        model_variant: str = 'tiny',
        batch_size: int = 1,
        video_frames: int = 8,
        image_size: int = 224,
        num_classes: int = 400,
        num_iterations: int = 10,
        warmup_iterations: int = 3,
        use_per_head_attention: bool = False,
    ):
        self.model_variant = model_variant
        self.batch_size = batch_size
        self.video_frames = video_frames
        self.image_size = image_size
        self.num_classes = num_classes
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        self.use_per_head_attention = use_per_head_attention
        
        config = SWIN_3D_CONFIG[model_variant]
        self.embed_dim = config['embed_dim']
        self.depths = config['depths']
        self.num_heads = config['num_heads']
        self.window_size = config['window_size']
        self.mlp_ratio = config['mlp_ratio']
        self.patch_size = config['patch_size']
        
        # Calculate dimensions after patch embedding
        self.D = video_frames // self.patch_size[0]  # e.g., 8 // 2 = 4
        self.H = image_size // self.patch_size[1]     # e.g., 224 // 4 = 56
        self.W = image_size // self.patch_size[2]     # e.g., 224 // 4 = 56
        
        self.metrics: Dict[str, LayerMetrics] = OrderedDict()
        self.reuse_single_enclave: bool = True  # Keep enabled, use frequent resets instead
        self._runtime_stats: Dict[str, Dict[str, List[float]]] = {}
    
    def profile_all(self, verbose: bool = True):
        """Profile all layers."""
        from python.enclave_interfaces import GlobalTensor
        
        # Reset interval for enclave memory
        ENCLAVE_RESET_INTERVAL = 1 if self.use_per_head_attention else 2
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Profiling Video Swin Transformer 3D ({self.model_variant.capitalize()}) in Enclave Mode")
            print(f"{'='*70}")
            print(f"Model Config: embed_dim={self.embed_dim}, depths={self.depths}")
            print(f"Video: frames={self.video_frames}, size={self.image_size}x{self.image_size}")
            print(f"After patch: D={self.D}, H={self.H}, W={self.W}")
            print(f"Window size: {self.window_size} (3D)")
            print(f"Iterations: {self.num_iterations} (warmup: {self.warmup_iterations})")
            print(f"Attention Mode: {'Per-Head' if self.use_per_head_attention else 'Batched'}")
            print(f"Enclave reset interval: every {ENCLAVE_RESET_INTERVAL} stages")
            print(f"{'='*70}\n")
        
        if self.reuse_single_enclave:
            try:
                if not GlobalTensor.is_init_global_tensor:
                    GlobalTensor.init()
                
                # Profile patch embedding
                if verbose:
                    print(f"\n--- Patch Embedding (3D Conv) ---")
                self._profile_patch_embedding(verbose)
                
                # Profile stages
                D, H, W = self.D, self.H, self.W
                dim = self.embed_dim
                
                for i_stage in range(len(self.depths)):
                    if verbose:
                        print(f"\n--- Stage {i_stage} (dim={dim}, heads={self.num_heads[i_stage]}, {D}x{H}x{W}) ---")
                    
                    self._profile_stage(i_stage, dim, D, H, W, verbose)
                    
                    # Update dimensions for next stage
                    if i_stage < len(self.depths) - 1:
                        H = H // 2
                        W = W // 2
                        dim = dim * 2
                
                # Reset before classifier
                if verbose:
                    print(f"\n[Resetting Enclave before classifier...]")
                if GlobalTensor.is_init_global_tensor:
                    GlobalTensor.destroy()
                GlobalTensor.init()
                # Clear tensor name hash cache to avoid ID conflicts
                if hasattr(GlobalTensor.EnclaveInterface, 'deployed_name_seed'):
                    GlobalTensor.EnclaveInterface.deployed_name_seed.clear()
                
                # Profile classifier
                if verbose:
                    print(f"\n--- Classifier Head ---")
                self._profile_classifier(dim, D, H, W, verbose)
                
                return self.metrics
                
            finally:
                if GlobalTensor.is_init_global_tensor:
                    GlobalTensor.destroy()
        
        return self.metrics
    
    def _profile_patch_embedding(self, verbose: bool):
        """Profile 3D patch embedding layers."""
        # Conv3D
        self._profile_conv3d_enclave(
            name='patch_embed_conv3d',
            input_shape=[self.batch_size, 3, self.video_frames, self.image_size, self.image_size],
            output_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=(0, 0, 0),
            group='PatchEmbed',
            verbose=verbose,
            dependencies=[]
        )
        
        # LayerNorm
        num_tokens = self.batch_size * self.D * self.H * self.W
        self._profile_layernorm_enclave(
            'patch_embed_norm',
            input_shape=[num_tokens, self.embed_dim],
            group='PatchEmbed',
            verbose=verbose,
            dependencies=['patch_embed_conv3d']
        )
    
    def _profile_stage(self, i_stage: int, dim: int, D: int, H: int, W: int, verbose: bool):
        """Profile one stage with all blocks."""
        from python.enclave_interfaces import GlobalTensor
        
        depth = self.depths[i_stage]
        num_heads = self.num_heads[i_stage]
        
        # Block-level reset strategy (aligned with BERT/ViT)
        BLOCK_RESET_INTERVAL = 1 if self.use_per_head_attention else 2
        
        # Determine previous layer
        if i_stage == 0:
            prev_layer = 'patch_embed_norm'
        else:
            prev_stage_last_block = self.depths[i_stage - 1] - 1
            prev_layer = f'stage{i_stage-1}_block{prev_stage_last_block}_residual2'
        
        # Profile blocks
        for i_block in range(depth):
            if verbose:
                print(f"  Block {i_block} ({'W-MSA' if i_block % 2 == 0 else 'SW-MSA'})")
            
            if i_block == 0:
                block_prev = prev_layer
            else:
                block_prev = f'stage{i_stage}_block{i_block-1}_residual2'
            
            self._profile_swin_block(i_stage, i_block, dim, num_heads, D, H, W, block_prev, verbose)
            
            # Block-level Enclave reset to prevent memory exhaustion
            should_reset = (
                (self.use_per_head_attention and i_block < depth - 1) or
                (not self.use_per_head_attention and 
                 i_block > 0 and 
                 (i_block + 1) % BLOCK_RESET_INTERVAL == 0 and 
                 i_block < depth - 1)
            )
            
            if should_reset:
                if verbose:
                    print(f"    [Resetting Enclave after block {i_block} to free memory...]")
                if GlobalTensor.is_init_global_tensor:
                    GlobalTensor.destroy()
                GlobalTensor.init()
                # Clear tensor name hash cache to avoid ID conflicts
                if hasattr(GlobalTensor.EnclaveInterface, 'deployed_name_seed'):
                    GlobalTensor.EnclaveInterface.deployed_name_seed.clear()
        
        # Patch merging (except last stage)
        if i_stage < len(self.depths) - 1:
            last_block = depth - 1
            merge_prev = f'stage{i_stage}_block{last_block}_residual2'
            
            if verbose:
                print(f"  Patch Merging")
            self._profile_patch_merging(i_stage, dim, D, H, W, merge_prev, verbose)
    
    def _profile_swin_block(
        self, i_stage: int, i_block: int, dim: int, num_heads: int,
        D: int, H: int, W: int, prev_layer: str, verbose: bool
    ):
        """Profile a single Swin Transformer block."""
        from python.enclave_interfaces import GlobalTensor
        
        prefix = f'stage{i_stage}_block{i_block}'
        group = f'Stage{i_stage}_Block{i_block}'
        is_shifted = (i_block % 2 == 1)  # SW-MSA on odd blocks
        
        num_tokens = self.batch_size * D * H * W
        Wd, Wh, Ww = self.window_size
        num_windows = (D // Wd) * (H // Wh) * (W // Ww)
        window_tokens = Wd * Wh * Ww
        head_dim = dim // num_heads
        
        # LayerNorm 1
        self._profile_layernorm_enclave(
            f'{prefix}_norm1',
            input_shape=[num_tokens, dim],
            group=group,
            verbose=verbose,
            dependencies=[prev_layer]
        )
        
        # Cyclic shift (for SW-MSA)
        current_layer = f'{prefix}_norm1'
        if is_shifted:
            self._profile_cyclic_roll_enclave(
                name=f'{prefix}_cyclic_shift',
                input_shape=[self.batch_size, D, H, W, dim],
                shifts=tuple(-s//2 for s in self.window_size),
                group=group,
                verbose=verbose,
                dependencies=[current_layer]
            )
            current_layer = f'{prefix}_cyclic_shift'
        
        # Window partition
        self._profile_window_partition_enclave(
            name=f'{prefix}_win_part',
            input_shape=[self.batch_size, D, H, W, dim],
            window_size=self.window_size,
            group=group,
            verbose=verbose,
            dependencies=[current_layer]
        )
        
        # Window attention
        if self.use_per_head_attention:
            self._profile_window_attention_per_head(
                prefix, dim, num_heads, num_windows, window_tokens, group, verbose
            )
        else:
            self._profile_window_attention_batched(
                prefix, dim, num_heads, num_windows, window_tokens, group, verbose
            )
        
        # Window reverse
        self._profile_window_reverse_enclave(
            name=f'{prefix}_win_rev',
            input_shape=[num_windows * self.batch_size, window_tokens, dim],
            output_shape=[self.batch_size, D, H, W, dim],
            window_size=self.window_size,
            group=group,
            verbose=verbose,
            dependencies=[f'{prefix}_attn_proj' if not self.use_per_head_attention else f'{prefix}_attn_head{num_heads-1}_proj']
        )
        
        # Reverse cyclic shift (for SW-MSA)
        current_layer = f'{prefix}_win_rev'
        if is_shifted:
            self._profile_cyclic_roll_enclave(
                name=f'{prefix}_reverse_shift',
                input_shape=[self.batch_size, D, H, W, dim],
                shifts=tuple(s//2 for s in self.window_size),
                group=group,
                verbose=verbose,
                dependencies=[current_layer]
            )
            current_layer = f'{prefix}_reverse_shift'
        
        # Residual 1
        self._profile_add_enclave(
            name=f'{prefix}_residual1',
            input_shape=[num_tokens, dim],
            group=group,
            verbose=verbose,
            dependencies=[current_layer, prev_layer]
        )
        
        # CRITICAL: Reset Enclave after attention in per-head mode to prevent memory exhaustion
        # Attention creates many layers (3 heads × 4 layers/head = 12 layers + window ops)
        if self.use_per_head_attention:
            if verbose:
                print(f"    [Resetting Enclave after attention to free memory...]")
            if GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()
            GlobalTensor.init()
            # Clear tensor name hash cache to avoid ID conflicts
            if hasattr(GlobalTensor.EnclaveInterface, 'deployed_name_seed'):
                GlobalTensor.EnclaveInterface.deployed_name_seed.clear()
        
        # LayerNorm 2
        self._profile_layernorm_enclave(
            f'{prefix}_norm2',
            input_shape=[num_tokens, dim],
            group=group,
            verbose=verbose,
            dependencies=[f'{prefix}_residual1']
        )
        
        # Reset again before MLP in per-head mode (MLP layers are large)
        if self.use_per_head_attention:
            if verbose:
                print(f"    [Resetting Enclave before MLP to free memory...]")
            if GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()
            GlobalTensor.init()
            # Clear tensor name hash cache to avoid ID conflicts
            if hasattr(GlobalTensor.EnclaveInterface, 'deployed_name_seed'):
                GlobalTensor.EnclaveInterface.deployed_name_seed.clear()
        
        # MLP
        hidden_dim = int(dim * self.mlp_ratio)
        
        # FC1
        self._profile_linear_enclave(
            f'{prefix}_mlp_fc1',
            input_shape=[num_tokens, dim],
            output_features=hidden_dim,
            group=group,
            verbose=verbose,
            dependencies=[f'{prefix}_norm2']
        )
        
        # GELU
        self._profile_gelu_enclave(
            f'{prefix}_mlp_gelu',
            input_shape=[num_tokens, hidden_dim],
            group=group,
            verbose=verbose,
            dependencies=[f'{prefix}_mlp_fc1']
        )
        
        # FC2
        self._profile_linear_enclave(
            f'{prefix}_mlp_fc2',
            input_shape=[num_tokens, hidden_dim],
            output_features=dim,
            group=group,
            verbose=verbose,
            dependencies=[f'{prefix}_mlp_gelu']
        )
        
        # Residual 2
        self._profile_add_enclave(
            name=f'{prefix}_residual2',
            input_shape=[num_tokens, dim],
            group=group,
            verbose=verbose,
            dependencies=[f'{prefix}_mlp_fc2', f'{prefix}_residual1']
        )
    
    def _profile_window_attention_batched(
        self, prefix: str, dim: int, num_heads: int,
        num_windows: int, window_tokens: int, group: str, verbose: bool
    ):
        """Profile window attention in batched mode."""
        total_windows = num_windows * self.batch_size
        head_dim = dim // num_heads
        
        # QKV projection
        self._profile_linear_enclave(
            f'{prefix}_attn_qkv',
            input_shape=[total_windows * window_tokens, dim],
            output_features=3 * dim,
            group=group,
            verbose=verbose,
            dependencies=[f'{prefix}_win_part']
        )
        
        # Q @ K^T
        self._profile_matmul_enclave(
            f'{prefix}_attn_qk_matmul',
            input_shape1=[total_windows, num_heads, window_tokens, head_dim],
            input_shape2=[total_windows, num_heads, window_tokens, head_dim],
            transpose_b=True,
            scale=1.0 / float(np.sqrt(head_dim)),
            group=group,
            verbose=verbose,
            dependencies=[f'{prefix}_attn_qkv']
        )
        
        # Softmax
        self._profile_softmax_enclave(
            f'{prefix}_attn_softmax',
            input_shape=[total_windows, num_heads, window_tokens, window_tokens],
            group=group,
            verbose=verbose,
            dependencies=[f'{prefix}_attn_qk_matmul']
        )
        
        # Attn @ V
        self._profile_matmul_enclave(
            f'{prefix}_attn_v_matmul',
            input_shape1=[total_windows, num_heads, window_tokens, window_tokens],
            input_shape2=[total_windows, num_heads, window_tokens, head_dim],
            transpose_b=False,
            scale=None,
            group=group,
            verbose=verbose,
            dependencies=[f'{prefix}_attn_softmax', f'{prefix}_attn_qkv']
        )
        
        # Output projection
        self._profile_linear_enclave(
            f'{prefix}_attn_proj',
            input_shape=[total_windows * window_tokens, dim],
            output_features=dim,
            group=group,
            verbose=verbose,
            dependencies=[f'{prefix}_attn_v_matmul']
        )
    
    def _profile_window_attention_per_head(
        self, prefix: str, dim: int, num_heads: int,
        num_windows: int, window_tokens: int, group: str, verbose: bool
    ):
        """Profile window attention with per-head granularity."""
        total_windows = num_windows * self.batch_size
        head_dim = dim // num_heads
        
        # Shared QKV projection
        self._profile_linear_enclave(
            f'{prefix}_attn_qkv',
            input_shape=[total_windows * window_tokens, dim],
            output_features=3 * dim,
            group=group,
            verbose=verbose,
            dependencies=[f'{prefix}_win_part']
        )
        
        # Per-head attention
        for h in range(num_heads):
            head_prefix = f'{prefix}_attn_head{h}'
            
            # Q @ K^T for this head
            self._profile_matmul_enclave(
                f'{head_prefix}_qk',
                input_shape1=[total_windows, 1, window_tokens, head_dim],
                input_shape2=[total_windows, 1, window_tokens, head_dim],
                transpose_b=True,
                scale=1.0 / float(np.sqrt(head_dim)),
                group=group,
                verbose=verbose,
                dependencies=[f'{prefix}_attn_qkv']
            )
            
            # Softmax for this head
            self._profile_softmax_enclave(
                f'{head_prefix}_softmax',
                input_shape=[total_windows, 1, window_tokens, window_tokens],
                group=group,
                verbose=verbose,
                dependencies=[f'{head_prefix}_qk']
            )
            
            # Attn @ V for this head
            self._profile_matmul_enclave(
                f'{head_prefix}_v',
                input_shape1=[total_windows, 1, window_tokens, window_tokens],
                input_shape2=[total_windows, 1, window_tokens, head_dim],
                transpose_b=False,
                scale=None,
                group=group,
                verbose=verbose,
                dependencies=[f'{head_prefix}_softmax', f'{prefix}_attn_qkv']
            )
            
            # Per-head projection
            self._profile_linear_enclave(
                f'{head_prefix}_proj',
                input_shape=[total_windows * window_tokens, head_dim],
                output_features=head_dim,
                group=group,
                verbose=verbose,
                dependencies=[f'{head_prefix}_v']
            )
    
    def _profile_patch_merging(
        self, i_stage: int, dim: int, D: int, H: int, W: int, prev_layer: str, verbose: bool
    ):
        """Profile patch merging layer."""
        prefix = f'stage{i_stage}_merge'
        group = f'Stage{i_stage}_Merge'
        
        # After merging: D x (H/2) x (W/2) tokens with 4*dim -> 2*dim
        num_tokens_out = self.batch_size * D * (H // 2) * (W // 2)
        
        # LayerNorm (4*dim)
        num_tokens_in = self.batch_size * D * H * W
        self._profile_layernorm_enclave(
            f'{prefix}_norm',
            input_shape=[num_tokens_in, 4 * dim],
            group=group,
            verbose=verbose,
            dependencies=[prev_layer]
        )
        
        # Linear reduction: 4C -> 2C
        self._profile_linear_enclave(
            f'{prefix}_reduction',
            input_shape=[num_tokens_out, 4 * dim],
            output_features=2 * dim,
            group=group,
            verbose=verbose,
            dependencies=[f'{prefix}_norm']
        )
    
    def _profile_classifier(self, dim: int, D: int, H: int, W: int, verbose: bool):
        """Profile classifier head."""
        last_stage = len(self.depths) - 1
        last_block = self.depths[last_stage] - 1
        prev_layer = f'stage{last_stage}_block{last_block}_residual2'
        
        num_tokens = self.batch_size * D * H * W
        
        # Final LayerNorm
        self._profile_layernorm_enclave(
            'final_norm',
            input_shape=[num_tokens, dim],
            group='Classifier',
            verbose=verbose,
            dependencies=[prev_layer]
        )
        
        # Classifier (after global avg pooling)
        self._profile_linear_enclave(
            'classifier',
            input_shape=[self.batch_size, dim],
            output_features=self.num_classes,
            group='Classifier',
            verbose=verbose,
            dependencies=['final_norm']
        )
    
    # =========================================================================
    # Layer profiling methods - 3D operators for Video Swin Transformer
    # =========================================================================
    
    def _profile_conv3d_enclave(
        self, name: str, input_shape: List[int], 
        output_channels: int, kernel_size: Tuple[int, int, int], 
        stride: Tuple[int, int, int], padding: Tuple[int, int, int],
        group: str, verbose: bool,
        dependencies: Optional[List[str]] = None
    ):
        """Profile a Conv3D layer in Enclave mode."""
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.sgx_conv3d_base import SGXConv3DBase, calc_conv3d_output_shape_stride
        from python.layers.input import SecretInputLayer
        from python.layers.output import SecretOutputLayer
        from python.sgx_net import SecretNeuralNetwork
        from python.utils.basic_utils import ExecutionModeOptions
        
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            in_channels = input_shape[1]
            D, H, W = input_shape[2], input_shape[3], input_shape[4]
            weight_shape = [output_channels, in_channels] + list(kernel_size)
            output_shape = calc_conv3d_output_shape_stride(input_shape, weight_shape, padding, stride)
            
            sid = 0
            layers = []
            lname = lambda s: f"{name}::{s}"
            
            input_layer = SecretInputLayer(
                sid, lname("input"), input_shape,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer)
            
            conv_layer = SGXConv3DBase(
                sid, lname("conv3d"),
                ExecutionModeOptions.Enclave,
                n_output_channel=output_channels,
                n_input_channel=in_channels,
                filter_dhw=kernel_size,
                video_dhw=(D, H, W),
                stride=stride,
                padding=padding,
                batch_size=input_shape[0],
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
                layer_type='Conv3d',
                input_shape=input_shape,
                output_shape=output_shape,
                kernel_size=kernel_size[1],  # Use spatial kernel for estimation
            )
            
            # Use provided dependencies or infer them
            if dependencies is None:
                dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
            
            metrics = LayerMetrics(
                name=name,
                layer_type='Conv3d',
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
    
    def _profile_window_partition_enclave(
        self, name: str, input_shape: List[int], 
        window_size: Tuple[int, int, int],
        group: str, verbose: bool,
        dependencies: Optional[List[str]] = None
    ):
        """Profile a WindowPartition3D layer in Enclave mode."""
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.window_partition_3d import SecretWindowPartition3DLayer
        from python.layers.input import SecretInputLayer
        from python.layers.output import SecretOutputLayer
        from python.sgx_net import SecretNeuralNetwork
        from python.utils.basic_utils import ExecutionModeOptions
        
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            B, D, H, W, C = input_shape
            Wd, Wh, Ww = window_size
            num_windows = (D // Wd) * (H // Wh) * (W // Ww)
            window_tokens = Wd * Wh * Ww
            output_shape = [num_windows * B, window_tokens, C]
            
            sid = 0
            layers = []
            lname = lambda s: f"{name}::{s}"
            
            input_layer = SecretInputLayer(
                sid, lname("input"), input_shape,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer)
            
            partition_layer = SecretWindowPartition3DLayer(
                sid, lname("partition"),
                ExecutionModeOptions.Enclave,
                window_size=window_size,
                manually_register_prev=True,
                manually_register_next=True
            )
            partition_layer.register_prev_layer(input_layer)
            layers.append(partition_layer)
            
            output_layer = SecretOutputLayer(
                sid, lname("output"),
                ExecutionModeOptions.CPU,
                manually_register_prev=True
            )
            output_layer.register_prev_layer(partition_layer)
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
                layer_type='Reshape',
                input_shape=input_shape,
                output_shape=output_shape,
            )
            
            # Use provided dependencies or infer them
            if dependencies is None:
                dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
            
            metrics = LayerMetrics(
                name=name,
                layer_type='WindowPartition3D',
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
    
    def _profile_window_reverse_enclave(
        self, name: str, input_shape: List[int], output_shape: List[int],
        window_size: Tuple[int, int, int],
        group: str, verbose: bool,
        dependencies: Optional[List[str]] = None
    ):
        """Profile a WindowReverse3D layer in Enclave mode."""
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.window_reverse_3d import SecretWindowReverse3DLayer
        from python.layers.input import SecretInputLayer
        from python.layers.output import SecretOutputLayer
        from python.sgx_net import SecretNeuralNetwork
        from python.utils.basic_utils import ExecutionModeOptions
        
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            sid = 0
            layers = []
            lname = lambda s: f"{name}::{s}"
            
            input_layer = SecretInputLayer(
                sid, lname("input"), input_shape,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer)
            
            reverse_layer = SecretWindowReverse3DLayer(
                sid, lname("reverse"),
                ExecutionModeOptions.Enclave,
                window_size=window_size,
                output_shape=output_shape,
                manually_register_prev=True,
                manually_register_next=True
            )
            reverse_layer.register_prev_layer(input_layer)
            layers.append(reverse_layer)
            
            output_layer = SecretOutputLayer(
                sid, lname("output"),
                ExecutionModeOptions.CPU,
                manually_register_prev=True
            )
            output_layer.register_prev_layer(reverse_layer)
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
                layer_type='Reshape',
                input_shape=input_shape,
                output_shape=output_shape,
            )
            
            # Use provided dependencies or infer them
            if dependencies is None:
                dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
            
            metrics = LayerMetrics(
                name=name,
                layer_type='WindowReverse3D',
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
    
    def _profile_cyclic_roll_enclave(
        self, name: str, input_shape: List[int], 
        shifts: Tuple[int, int, int],
        group: str, verbose: bool,
        dependencies: Optional[List[str]] = None
    ):
        """Profile a CyclicRoll3D layer in Enclave mode."""
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.cyclic_roll_3d import SecretCyclicRoll3DLayer
        from python.layers.input import SecretInputLayer
        from python.layers.output import SecretOutputLayer
        from python.sgx_net import SecretNeuralNetwork
        from python.utils.basic_utils import ExecutionModeOptions
        
        try:
            if not GlobalTensor.is_init_global_tensor:
                GlobalTensor.init()
            
            output_shape = input_shape  # Shape unchanged
            sid = 0
            layers = []
            lname = lambda s: f"{name}::{s}"
            
            input_layer = SecretInputLayer(
                sid, lname("input"), input_shape,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer)
            
            roll_layer = SecretCyclicRoll3DLayer(
                sid, lname("roll"),
                ExecutionModeOptions.Enclave,
                shifts=shifts,
                dims=(1, 2, 3),
                manually_register_prev=True,
                manually_register_next=True
            )
            roll_layer.register_prev_layer(input_layer)
            layers.append(roll_layer)
            
            output_layer = SecretOutputLayer(
                sid, lname("output"),
                ExecutionModeOptions.CPU,
                manually_register_prev=True
            )
            output_layer.register_prev_layer(roll_layer)
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
                layer_type='Reshape',
                input_shape=input_shape,
                output_shape=output_shape,
            )
            
            # Use provided dependencies or infer them
            if dependencies is None:
                dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
            
            metrics = LayerMetrics(
                name=name,
                layer_type='CyclicRoll3D',
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
    
    # =========================================================================
    # Layer profiling methods - copied from BERT (verified implementation)
    # =========================================================================
    
    def _profile_linear_enclave(
        self, name: str, input_shape: List[int], 
        output_features: int, group: str, verbose: bool,
        dependencies: Optional[List[str]] = None
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
                input_features = input_shape[-1]
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
            
            # Use provided dependencies or infer them
            if dependencies is None:
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
            import traceback
            traceback.print_exc()
        
        finally:
            if (not self.reuse_single_enclave) and GlobalTensor.is_init_global_tensor:
                GlobalTensor.destroy()
    
    def _profile_layernorm_enclave(
        self, name: str, input_shape: List[int], 
        group: str, verbose: bool,
        dependencies: Optional[List[str]] = None
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
            
            # Use provided dependencies or infer them
            if dependencies is None:
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
        group: str, verbose: bool,
        dependencies: Optional[List[str]] = None
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
            
            # Use provided dependencies or infer them
            if dependencies is None:
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
        group: str, verbose: bool,
        dependencies: Optional[List[str]] = None
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
            
            # Use provided dependencies or infer them
            if dependencies is None:
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
        group: str, verbose: bool,
        dependencies: Optional[List[str]] = None
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
            
            # Calculate memory footprint (MatMul has no weights)
            mem_info = calc_layer_memory_from_shapes(
                layer_type='MatMul',
                input_shape=input_shape1,
                output_shape=output_shape,
            )
            # Add second input to activation bytes
            mem_info['activation_bytes'] += _shape_to_bytes(input_shape2)
            mem_info['cpu_memory_bytes'] += _shape_to_bytes(input_shape2)
            mem_info['tee_memory_bytes'] += _shape_to_bytes(input_shape2)
            mem_info['tee_total_memory_bytes'] += _shape_to_bytes(input_shape2)
            
            # Use provided dependencies or infer them
            if dependencies is None:
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
    
    def _profile_add_enclave(
        self, name: str, input_shape: List[int], 
        group: str, verbose: bool,
        dependencies: Optional[List[str]] = None
    ):
        """Profile an Add layer (residual connection) in Enclave mode."""
        import torch
        from python.enclave_interfaces import GlobalTensor
        from python.layers.add import SecretAddLayer
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
            
            # Create TWO input layers for residual connection (main path + shortcut)
            input_layer1 = SecretInputLayer(
                sid, lname("input1"), input_shape,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer1)
            
            input_layer2 = SecretInputLayer(
                sid, lname("input2"), input_shape,
                ExecutionModeOptions.Enclave,
                manually_register_next=True
            )
            layers.append(input_layer2)
            
            # Add layer requires TWO previous layers
            add_layer = SecretAddLayer(
                sid, lname("add"),
                ExecutionModeOptions.Enclave,
                manually_register_prev=True,
                manually_register_next=True
            )
            add_layer.register_prev_layer(input_layer1)  # First input
            add_layer.register_prev_layer(input_layer2)  # Second input (required!)
            layers.append(add_layer)
            
            output_layer = SecretOutputLayer(
                sid, lname("output"),
                ExecutionModeOptions.CPU,
                manually_register_prev=True
            )
            output_layer.register_prev_layer(add_layer)
            layers.append(output_layer)
            
            secret_nn = SecretNeuralNetwork(sid, f"profile_{name}")
            secret_nn.set_eid(GlobalTensor.get_eid())
            secret_nn.set_layers(layers)  # 4 layers: input1, input2, add, output
            
            times = []
            self._init_runtime_bucket(name)
            total_runs = self.warmup_iterations + self.num_iterations
            
            for run_idx in range(total_runs):
                test_input = torch.randn(*input_shape)
                # Set BOTH inputs (use same data for profiling)
                layers[0].set_input(test_input)
                layers[1].set_input(test_input)
                
                start_time = time.perf_counter()
                layers[0].forward()  # input1
                layers[1].forward()  # input2
                layers[2].forward()  # add layer (index 2, not 1!)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                stats = GlobalTensor.EnclaveInterface.get_layer_runtime_stats(layers[2].LayerName)
                
                if run_idx >= self.warmup_iterations:
                    times.append(elapsed_ms)
                    self._append_runtime_stats(name, stats)
            
            # Calculate memory footprint (Add has no weights)
            mem_info = calc_layer_memory_from_shapes(
                layer_type='Add',
                input_shape=input_shape,
                output_shape=output_shape,
            )
            
            # Use provided dependencies or infer them
            if dependencies is None:
                dependencies = infer_layer_dependencies(name, list(self.metrics.keys()) + [name])
            
            metrics = LayerMetrics(
                name=name,
                layer_type='Add',
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
    
    # =========================================================================
    # Utility methods for runtime statistics
    # =========================================================================
    
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
    
    def _build_xfer_edges_for_row(self, m: LayerMetrics):
        """Build edge-level xfer JSON (mean) for CSV/JSON."""
        def mean(xs: List[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0
        
        get_mean = mean(m.enclave_get_ms)
        get2_mean = mean(m.enclave_get2_ms)
        store_mean = mean(m.enclave_store_ms)
        compute_mean = mean(m.enclave_compute_ms)
        
        edges = []
        if get_mean > 0:
            edges.append({"from": "CPU", "to": "Enclave", "ms": get_mean, "type": "input"})
        if get2_mean > 0:
            edges.append({"from": "CPU", "to": "Enclave", "ms": get2_mean, "type": "input2"})
        if store_mean > 0:
            edges.append({"from": "Enclave", "to": "CPU", "ms": store_mean, "type": "output"})
        
        xfer_total = get_mean + get2_mean + store_mean
        return json.dumps(edges), xfer_total, compute_mean
    
    def save_results(self, output_dir: str = 'experiments/data'):
        """Save profiling results to CSV and JSON."""
        for metrics in self.metrics.values():
            metrics.compute_statistics()
        
        os.makedirs(output_dir, exist_ok=True)
        
        variant = self.model_variant
        suffix = '_per_head' if self.use_per_head_attention else ''
        csv_path = os.path.join(output_dir, f'swin_{variant}_enclave{suffix}_layers.csv')
        json_path = os.path.join(output_dir, f'swin_{variant}_enclave{suffix}_layers.json')
        
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
                xfer_edges_json, xfer_total_mean_ms, compute_mean_ms = self._build_xfer_edges_for_row(metrics)
                row['xfer_edges_json'] = xfer_edges_json
                row['xfer_total_mean_ms'] = xfer_total_mean_ms
                row['compute_mean_ms'] = compute_mean_ms
                row['input_shape'] = str(row['input_shape'])
                row['output_shape'] = str(row['output_shape'])
                row['dependencies'] = str(row['dependencies'])
                writer.writerow(row)
        
        with open(json_path, 'w') as f:
            json.dump({
                'model_config': {
                    'variant': variant,
                    'embed_dim': self.embed_dim,
                    'depths': list(self.depths),
                    'num_heads': list(self.num_heads),
                    'window_size': list(self.window_size),
                    'batch_size': self.batch_size,
                    'video_frames': self.video_frames,
                    'image_size': self.image_size,
                    'num_classes': self.num_classes,
                },
                'profiling_config': {
                    'num_iterations': self.num_iterations,
                    'warmup_iterations': self.warmup_iterations,
                    'use_per_head_attention': self.use_per_head_attention,
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


def main():
    parser = argparse.ArgumentParser(description='Profile Video Swin 3D in Enclave mode')
    parser.add_argument('--model', type=str, default='tiny', choices=['tiny', 'small', 'base'],
                       help='Model variant')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--video-frames', type=int, default=8, help='Video frames')
    parser.add_argument('--image-size', type=int, default=224, help='Image size')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=3, help='Warmup iterations')
    parser.add_argument('--per-head', action='store_true', help='Use per-head attention profiling')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file')
    
    args = parser.parse_args()
    
    if not check_environment():
        print("\n✗ Environment check failed")
        return 1
    
    # Determine output file
    if args.output is None:
        mode_suffix = 'per_head' if args.per_head else 'batched'
        args.output = f'experiments/data/swin_{args.model}_enclave_{mode_suffix}_layers.csv'
    
    # Create profiler
    profiler = VideoSwinEnclaveProfiler(
        model_variant=args.model,
        batch_size=args.batch_size,
        video_frames=args.video_frames,
        image_size=args.image_size,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup,
        use_per_head_attention=args.per_head
    )
    
    # Profile
    profiler.profile_all(verbose=True)
    
    # Save results
    csv_path, json_path = profiler.save_results('experiments/data')
    
    # Print summary
    print(f"\n{'='*70}")
    print("Profiling Summary")
    print(f"{'='*70}")
    
    # Compute statistics
    for metrics in profiler.metrics.values():
        metrics.compute_statistics()
    
    total_enclave_time = sum(
        np.mean(m.enclave_times) for m in profiler.metrics.values() if m.enclave_times
    )
    print(f"Total layers: {len(profiler.metrics)}")
    print(f"Total enclave time: {total_enclave_time:.2f} ms")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"{'='*70}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
