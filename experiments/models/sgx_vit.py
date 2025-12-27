"""
Vision Transformer (ViT) implementation for SGX distributed inference.

This module provides a modular ViT implementation that can be partitioned
across Enclave and CPU execution environments, following the same patterns
as sgx_resnet.py and sgx_inception.py.

Key Features:
1. Attention head parallelism: Different heads can run on different partitions
2. FFN tensor parallelism: Hidden dimension can be split across partitions
3. Layer-wise pipeline: Transformer blocks can be distributed across workers

Architecture (ViT-Small for experiments):
- Patch embedding: 16x16 patches from 224x224 image → 196 tokens
- 12 Transformer blocks (configurable)
- Each block: LayerNorm → MHSA → Residual → LayerNorm → FFN → Residual
- Classification head: LayerNorm → Linear

Memory Considerations for TEE:
- Attention matrix: O(N^2) where N = num_patches = (img_size/patch_size)^2
- For 224x224 with 16x16 patches: N=196, attention matrix = 196x196 = 38K floats
- FFN hidden: 4 * embed_dim = 4 * 384 = 1536 (ViT-Small)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class ViTConfig:
    """Configuration for Vision Transformer."""
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 384       # ViT-Small
    num_heads: int = 6
    num_layers: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    dropout: float = 0.0
    
    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2
    
    @property
    def mlp_hidden_dim(self) -> int:
        return int(self.embed_dim * self.mlp_ratio)
    
    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


@dataclass
class ViTTinyConfig(ViTConfig):
    """ViT-Tiny: Minimal model for fast experiments."""
    embed_dim: int = 192
    num_heads: int = 3
    num_layers: int = 12


@dataclass
class ViTSmallConfig(ViTConfig):
    """ViT-Small: Balanced model for research."""
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 12


@dataclass 
class ViTBaseConfig(ViTConfig):
    """ViT-Base: Standard model."""
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12


# ==============================================================================
# Parallel Execution Modes
# ==============================================================================

class ParallelMode(Enum):
    """Parallelism strategy for Transformer components."""
    SEQUENTIAL = "sequential"           # No parallelism
    HEAD_PARALLEL = "head_parallel"     # Split attention heads
    FFN_PARALLEL = "ffn_parallel"       # Split FFN hidden dimension
    LAYER_PIPELINE = "layer_pipeline"   # Pipeline across layers


@dataclass
class PartitionConfig:
    """Configuration for distributed execution."""
    mode: ParallelMode = ParallelMode.SEQUENTIAL
    num_partitions: int = 2
    
    # For HEAD_PARALLEL: which heads go to which partition
    head_assignment: Optional[Dict[int, List[int]]] = None
    
    # For LAYER_PIPELINE: which layers go to which partition
    layer_assignment: Optional[Dict[int, List[int]]] = None
    
    # For FFN_PARALLEL: split ratio (e.g., [0.5, 0.5] for equal split)
    ffn_split_ratio: Optional[List[float]] = None


# ==============================================================================
# Core Transformer Components (Partitionable)
# ==============================================================================

class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings.
    
    Input: (B, C, H, W)
    Output: (B, num_patches + 1, embed_dim)  # +1 for CLS token
    
    This layer is typically NOT parallelized as it's memory-efficient
    and must complete before any attention computation.
    """
    
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches
        
        # Convolutional projection (equivalent to splitting + linear)
        self.proj = nn.Conv2d(
            config.in_channels, 
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim)
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch projection: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        
        # Flatten: (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        return x
    
    def get_memory_footprint(self) -> Dict[str, int]:
        """Estimate memory usage in bytes (float32)."""
        B, C, H, W = 1, self.config.in_channels, self.config.img_size, self.config.img_size
        return {
            'input': B * C * H * W * 4,
            'output': B * (self.num_patches + 1) * self.config.embed_dim * 4,
            'parameters': sum(p.numel() for p in self.parameters()) * 4
        }


class PartitionableAttention(nn.Module):
    """
    Multi-Head Self-Attention with support for head-level partitioning.
    
    Parallelism Strategy:
    - Heads can be split across partitions (e.g., Enclave gets heads 0-2, CPU gets 3-5)
    - Each partition computes attention for its assigned heads
    - Results are concatenated at synchronization point
    
    Memory Analysis:
    - Q, K, V projections: 3 * embed_dim^2 parameters
    - Attention matrix per head: N^2 * head_dim (N = seq_length)
    - For ViT-Small (N=197, H=6, D=64): ~1.5MB per head for attention
    """
    
    def __init__(self, config: ViTConfig, partition_config: Optional[PartitionConfig] = None):
        super().__init__()
        self.config = config
        self.partition_config = partition_config or PartitionConfig()
        
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        
        # QKV projection (can be split for tensor parallelism)
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=True)
        
        # Output projection
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.proj_drop = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (all heads)."""
        B, N, D = x.shape
        
        # QKV projection and reshape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    def forward_heads_subset(
        self, 
        x: torch.Tensor, 
        head_indices: List[int]
    ) -> torch.Tensor:
        """
        Compute attention for a subset of heads.
        
        This is the key method for HEAD_PARALLEL mode:
        - Partition A calls forward_heads_subset(x, [0, 1, 2])
        - Partition B calls forward_heads_subset(x, [3, 4, 5])
        - Results are concatenated at synchronization point
        
        Args:
            x: Input tensor (B, N, D)
            head_indices: List of head indices to compute
            
        Returns:
            Partial output (B, N, len(head_indices) * head_dim)
        """
        B, N, D = x.shape
        num_heads_subset = len(head_indices)
        
        # Full QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Select only the specified heads
        q = q[:, head_indices, :, :]  # (B, H_subset, N, head_dim)
        k = k[:, head_indices, :, :]
        v = v[:, head_indices, :, :]
        
        # Attention for subset
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention
        out = (attn @ v)  # (B, H_subset, N, head_dim)
        out = out.transpose(1, 2)  # (B, N, H_subset, head_dim)
        out = out.reshape(B, N, num_heads_subset * self.head_dim)
        
        return out
    
    @staticmethod
    def merge_head_outputs(
        outputs: List[torch.Tensor],
        proj_weight: torch.Tensor,
        proj_bias: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge outputs from different head partitions and apply output projection.
        
        This is called after all partitions complete their forward_heads_subset.
        
        Args:
            outputs: List of partial outputs from each partition
            proj_weight: Output projection weight
            proj_bias: Output projection bias
            
        Returns:
            Final attention output (B, N, D)
        """
        # Concatenate along head dimension
        merged = torch.cat(outputs, dim=-1)  # (B, N, D)
        
        # Apply output projection
        return F.linear(merged, proj_weight, proj_bias)
    
    def get_memory_footprint(self, batch_size: int = 1) -> Dict[str, int]:
        """Estimate memory usage per head in bytes."""
        N = self.config.num_patches + 1
        return {
            'qkv_per_head': batch_size * N * self.head_dim * 3 * 4,
            'attention_matrix_per_head': batch_size * N * N * 4,
            'output_per_head': batch_size * N * self.head_dim * 4,
            'total_per_head': batch_size * (N * self.head_dim * 4 + N * N) * 4,
            'parameters': sum(p.numel() for p in self.parameters()) * 4
        }


class PartitionableFFN(nn.Module):
    """
    Feed-Forward Network with support for hidden dimension partitioning.
    
    Structure: Linear(D → 4D) → GELU → Linear(4D → D)
    
    Parallelism Strategy (FFN_PARALLEL):
    - Split the intermediate 4D dimension across partitions
    - Partition A: Linear(D → 2D) → GELU → Linear(2D → D)
    - Partition B: Linear(D → 2D) → GELU → Linear(2D → D)
    - Final output = sum of partition outputs (AllReduce)
    
    This is mathematically equivalent because:
    FFN(x) = W2 @ GELU(W1 @ x)
           = [W2_A | W2_B] @ GELU([W1_A; W1_B] @ x)
           = W2_A @ GELU(W1_A @ x) + W2_B @ GELU(W1_B @ x)
    """
    
    def __init__(self, config: ViTConfig, partition_config: Optional[PartitionConfig] = None):
        super().__init__()
        self.config = config
        self.partition_config = partition_config or PartitionConfig()
        
        self.fc1 = nn.Linear(config.embed_dim, config.mlp_hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.mlp_hidden_dim, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def forward_partition(
        self, 
        x: torch.Tensor, 
        partition_id: int,
        num_partitions: int = 2
    ) -> torch.Tensor:
        """
        Compute FFN for a partition of the hidden dimension.
        
        Args:
            x: Input tensor (B, N, D)
            partition_id: Which partition (0, 1, ...)
            num_partitions: Total number of partitions
            
        Returns:
            Partial output (B, N, D) - needs to be summed with other partitions
        """
        hidden_dim = self.config.mlp_hidden_dim
        hidden_per_partition = hidden_dim // num_partitions
        
        start_idx = partition_id * hidden_per_partition
        end_idx = start_idx + hidden_per_partition
        
        # Extract partition weights
        fc1_weight_part = self.fc1.weight[start_idx:end_idx, :]  # (hidden_part, D)
        fc1_bias_part = self.fc1.bias[start_idx:end_idx]
        
        fc2_weight_part = self.fc2.weight[:, start_idx:end_idx]  # (D, hidden_part)
        
        # Forward pass with partition
        hidden = F.linear(x, fc1_weight_part, fc1_bias_part)
        hidden = self.act(hidden)
        hidden = self.drop(hidden)
        
        # Note: fc2 bias is only added once (in partition 0)
        if partition_id == 0:
            out = F.linear(hidden, fc2_weight_part, self.fc2.bias)
        else:
            out = F.linear(hidden, fc2_weight_part, None)
        
        return out
    
    @staticmethod
    def merge_partition_outputs(outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge FFN outputs from different partitions (AllReduce sum).
        """
        return sum(outputs)
    
    def get_memory_footprint(self, batch_size: int = 1) -> Dict[str, int]:
        """Estimate memory usage in bytes."""
        N = self.config.num_patches + 1
        D = self.config.embed_dim
        H = self.config.mlp_hidden_dim
        
        return {
            'input': batch_size * N * D * 4,
            'hidden': batch_size * N * H * 4,
            'output': batch_size * N * D * 4,
            'parameters': sum(p.numel() for p in self.parameters()) * 4
        }


class TransformerBlock(nn.Module):
    """
    Single Transformer block with support for distributed execution.
    
    Structure:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    
    Execution Dependencies:
    - LayerNorm1 → Attention → Residual1 (sequential)
    - LayerNorm2 → FFN → Residual2 (sequential)
    - Attention heads within MHSA (parallel)
    - FFN partitions (parallel)
    """
    
    def __init__(
        self, 
        config: ViTConfig, 
        block_idx: int,
        partition_config: Optional[PartitionConfig] = None
    ):
        super().__init__()
        self.config = config
        self.block_idx = block_idx
        self.partition_config = partition_config or PartitionConfig()
        
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.attn = PartitionableAttention(config, partition_config)
        
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.ffn = PartitionableFFN(config, partition_config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    
    def forward_attention_only(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute only the attention part (for pipeline parallelism).
        Returns both the attention output and the normalized input for FFN.
        """
        norm1_out = self.norm1(x)
        attn_out = self.attn(norm1_out)
        x_after_attn = x + attn_out
        return x_after_attn, self.norm2(x_after_attn)
    
    def forward_ffn_only(self, x_after_attn: torch.Tensor, norm2_out: torch.Tensor) -> torch.Tensor:
        """
        Execute only the FFN part (for pipeline parallelism).
        """
        ffn_out = self.ffn(norm2_out)
        return x_after_attn + ffn_out
    
    def forward_with_head_partition(
        self,
        x: torch.Tensor,
        head_indices: List[int]
    ) -> torch.Tensor:
        """
        Execute block with only specified attention heads.
        Note: This returns partial results that need merging.
        """
        norm1_out = self.norm1(x)
        attn_partial = self.attn.forward_heads_subset(norm1_out, head_indices)
        return attn_partial, norm1_out, x  # Need these for merging


class SGXVisionTransformer(nn.Module):
    """
    Vision Transformer designed for SGX distributed inference.
    
    This class mirrors the structure of SGXResNet18 and SGXInceptionV3
    to integrate with the existing distributed inference framework.
    
    Features:
    1. Layer-wise profiling support
    2. Configurable execution modes per component
    3. DAG-based dependency tracking
    4. Memory footprint estimation
    """
    
    def __init__(
        self,
        config: Optional[ViTConfig] = None,
        partition_config: Optional[PartitionConfig] = None
    ):
        super().__init__()
        
        self.config = config or ViTSmallConfig()
        self.partition_config = partition_config or PartitionConfig()
        
        # Build model components
        self.patch_embed = PatchEmbedding(self.config)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(self.config, i, self.partition_config)
            for i in range(self.config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(self.config.embed_dim, eps=1e-6)
        self.head = nn.Linear(self.config.embed_dim, self.config.num_classes)
        
        # Layer registry for distributed inference (mirrors SGXResNet pattern)
        self.layers = []
        self._build_layer_registry()
    
    def _build_layer_registry(self):
        """
        Build a flat list of layers for the distributed inference framework.
        Each entry tracks: name, module, dependencies, execution mode.
        """
        self.layers = []
        
        # Patch embedding
        self.layers.append({
            'name': 'patch_embed',
            'module': self.patch_embed,
            'type': 'PatchEmbed',
            'dependencies': ['input'],
            'input_shape': (1, 3, self.config.img_size, self.config.img_size),
            'output_shape': (1, self.config.num_patches + 1, self.config.embed_dim)
        })
        
        # Transformer blocks
        prev_layer = 'patch_embed'
        for i, block in enumerate(self.blocks):
            block_name = f'block_{i}'
            
            # Option 1: Treat block as atomic unit
            self.layers.append({
                'name': block_name,
                'module': block,
                'type': 'TransformerBlock',
                'dependencies': [prev_layer],
                'input_shape': (1, self.config.num_patches + 1, self.config.embed_dim),
                'output_shape': (1, self.config.num_patches + 1, self.config.embed_dim)
            })
            prev_layer = block_name
            
            # Option 2: Decompose block into sub-layers (for finer-grained partitioning)
            # Uncomment below for attention-FFN level granularity
            """
            self.layers.append({
                'name': f'{block_name}_norm1',
                'module': block.norm1,
                'type': 'LayerNorm',
                'dependencies': [prev_layer],
            })
            self.layers.append({
                'name': f'{block_name}_attn',
                'module': block.attn,
                'type': 'Attention',
                'dependencies': [f'{block_name}_norm1'],
            })
            # ... etc
            """
        
        # Final norm and head
        self.layers.append({
            'name': 'norm',
            'module': self.norm,
            'type': 'LayerNorm',
            'dependencies': [prev_layer]
        })
        
        self.layers.append({
            'name': 'head',
            'module': self.head,
            'type': 'Linear',
            'dependencies': ['norm'],
            'output_shape': (1, self.config.num_classes)
        })
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # CLS token
        x = self.head(x)
        
        return x
    
    def forward_layer_by_layer(
        self,
        x: torch.Tensor,
        layer_timing_callback: Optional[Callable[[str, float], None]] = None
    ) -> torch.Tensor:
        """
        Execute model layer by layer with optional timing callback.
        Useful for profiling and distributed execution.
        """
        import time
        
        # Patch embedding
        start = time.perf_counter()
        x = self.patch_embed(x)
        if layer_timing_callback:
            layer_timing_callback('patch_embed', (time.perf_counter() - start) * 1000)
        
        # Transformer blocks
        for i, block in enumerate(self.blocks):
            start = time.perf_counter()
            x = block(x)
            if layer_timing_callback:
                layer_timing_callback(f'block_{i}', (time.perf_counter() - start) * 1000)
        
        # Final norm
        start = time.perf_counter()
        x = self.norm(x)
        if layer_timing_callback:
            layer_timing_callback('norm', (time.perf_counter() - start) * 1000)
        
        # Classification head
        start = time.perf_counter()
        x = x[:, 0]
        x = self.head(x)
        if layer_timing_callback:
            layer_timing_callback('head', (time.perf_counter() - start) * 1000)
        
        return x
    
    def get_layer_dependencies(self) -> Dict[str, List[str]]:
        """Return layer dependency graph for DAG-based scheduling."""
        return {layer['name']: layer['dependencies'] for layer in self.layers}
    
    def get_memory_footprint(self, batch_size: int = 1) -> Dict[str, Any]:
        """
        Estimate memory footprint for TEE execution planning.
        
        Returns detailed breakdown by component.
        """
        N = self.config.num_patches + 1
        D = self.config.embed_dim
        H = self.config.mlp_hidden_dim
        num_heads = self.config.num_heads
        head_dim = self.config.head_dim
        
        footprint = {
            'config': {
                'batch_size': batch_size,
                'seq_length': N,
                'embed_dim': D,
                'num_heads': num_heads,
                'num_layers': self.config.num_layers
            },
            'activations': {
                'patch_embed_output': batch_size * N * D * 4,
                'attention_qkv': batch_size * N * D * 3 * 4,
                'attention_matrix': batch_size * num_heads * N * N * 4,
                'ffn_hidden': batch_size * N * H * 4,
                'per_block': batch_size * N * D * 6 * 4,  # Rough estimate
            },
            'parameters': {
                'patch_embed': sum(p.numel() for p in self.patch_embed.parameters()) * 4,
                'per_block': sum(p.numel() for p in self.blocks[0].parameters()) * 4,
                'total_blocks': sum(p.numel() for p in self.blocks.parameters()) * 4,
                'head': sum(p.numel() for p in self.head.parameters()) * 4,
                'total': sum(p.numel() for p in self.parameters()) * 4
            }
        }
        
        footprint['summary'] = {
            'peak_activation_bytes': footprint['activations']['per_block'] * 2,  # ~2 blocks
            'total_parameter_bytes': footprint['parameters']['total'],
            'peak_total_bytes': (
                footprint['activations']['per_block'] * 2 + 
                footprint['parameters']['total']
            )
        }
        
        return footprint
    
    def suggest_partition_strategy(self) -> Dict[str, Any]:
        """
        Suggest partitioning strategy based on model architecture and TEE constraints.
        """
        mem = self.get_memory_footprint()
        
        strategies = []
        
        # Strategy 1: Layer-wise Pipeline (like ResNet)
        mid_layer = self.config.num_layers // 2
        strategies.append({
            'name': 'layer_pipeline_half',
            'description': f'Blocks 0-{mid_layer-1} in Enclave, {mid_layer}-{self.config.num_layers-1} in CPU',
            'enclave_layers': list(range(mid_layer)),
            'cpu_layers': list(range(mid_layer, self.config.num_layers)),
            'communication_points': 1,
            'estimated_benefit': 'Balanced workload, minimal communication'
        })
        
        # Strategy 2: Head Parallelism
        mid_head = self.config.num_heads // 2
        strategies.append({
            'name': 'head_parallel',
            'description': f'Heads 0-{mid_head-1} in Enclave, {mid_head}-{self.config.num_heads-1} in CPU',
            'enclave_heads': list(range(mid_head)),
            'cpu_heads': list(range(mid_head, self.config.num_heads)),
            'communication_points': self.config.num_layers,  # Sync after each block
            'estimated_benefit': 'Fine-grained parallelism, higher communication'
        })
        
        # Strategy 3: FFN Parallelism
        strategies.append({
            'name': 'ffn_parallel',
            'description': 'FFN hidden dimension split 50/50',
            'split_ratio': [0.5, 0.5],
            'communication_points': self.config.num_layers,  # AllReduce after each FFN
            'estimated_benefit': 'Good for memory-constrained TEE (smaller activations)'
        })
        
        return {
            'model_info': {
                'num_layers': self.config.num_layers,
                'num_heads': self.config.num_heads,
                'embed_dim': self.config.embed_dim,
                'memory_footprint': mem['summary']
            },
            'suggested_strategies': strategies,
            'recommendation': strategies[0]['name']  # Default to layer pipeline
        }


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_vit_tiny(num_classes: int = 1000) -> SGXVisionTransformer:
    """Create ViT-Tiny for fast experiments."""
    config = ViTTinyConfig(num_classes=num_classes)
    return SGXVisionTransformer(config)


def create_vit_small(num_classes: int = 1000) -> SGXVisionTransformer:
    """Create ViT-Small for balanced experiments."""
    config = ViTSmallConfig(num_classes=num_classes)
    return SGXVisionTransformer(config)


def create_vit_base(num_classes: int = 1000) -> SGXVisionTransformer:
    """Create ViT-Base for full-scale experiments."""
    config = ViTBaseConfig(num_classes=num_classes)
    return SGXVisionTransformer(config)


# ==============================================================================
# Quick Test
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SGX Vision Transformer - Architecture Analysis")
    print("=" * 70)
    
    # Create ViT-Small
    model = create_vit_small(num_classes=10)
    
    # Print model info
    print(f"\n[Model Configuration]")
    print(f"  - Image Size: {model.config.img_size}")
    print(f"  - Patch Size: {model.config.patch_size}")
    print(f"  - Num Patches: {model.config.num_patches}")
    print(f"  - Embed Dim: {model.config.embed_dim}")
    print(f"  - Num Heads: {model.config.num_heads}")
    print(f"  - Num Layers: {model.config.num_layers}")
    
    # Memory footprint
    print(f"\n[Memory Footprint]")
    mem = model.get_memory_footprint(batch_size=1)
    print(f"  - Total Parameters: {mem['parameters']['total'] / 1024 / 1024:.2f} MB")
    print(f"  - Peak Activations: {mem['summary']['peak_activation_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  - Attention Matrix/Head: {mem['activations']['attention_matrix'] / 1024:.2f} KB")
    
    # Layer dependencies
    print(f"\n[Layer Dependencies]")
    deps = model.get_layer_dependencies()
    for name, dep in list(deps.items())[:5]:
        print(f"  {name}: depends on {dep}")
    print(f"  ... ({len(deps)} total layers)")
    
    # Partitioning suggestions
    print(f"\n[Partitioning Strategies]")
    suggestions = model.suggest_partition_strategy()
    for strategy in suggestions['suggested_strategies']:
        print(f"  - {strategy['name']}: {strategy['description']}")
    
    # Test forward pass
    print(f"\n[Forward Pass Test]")
    x = torch.randn(1, 3, 224, 224)
    
    timings = {}
    def record_timing(name, ms):
        timings[name] = ms
    
    with torch.no_grad():
        output = model.forward_layer_by_layer(x, record_timing)
    
    print(f"  - Output shape: {output.shape}")
    print(f"  - Total time: {sum(timings.values()):.2f} ms")
    print(f"\n  Layer timings (top 5):")
    for name, ms in sorted(timings.items(), key=lambda x: -x[1])[:5]:
        print(f"    {name}: {ms:.2f} ms")
    
    print("\n" + "=" * 70)
    print("✓ SGX Vision Transformer ready for distributed inference experiments")
    print("=" * 70)

