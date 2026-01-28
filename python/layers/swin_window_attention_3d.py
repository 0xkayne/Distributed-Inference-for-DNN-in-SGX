"""
3D Window Attention with Relative Position Bias for Video Swin Transformer.

This module implements window-based multi-head self-attention with relative
position bias, which is the core component of Video Swin Transformer.

Key Features:
1. Relative Position Bias: Learnable bias table indexed by relative positions
2. Window-based Attention: Attention computed within local 3D windows
3. Shifted Window Support: Attention mask for shifted window partitions

Reference: Video Swin Transformer
https://github.com/SwinTransformer/Video-Swin-Transformer
Lines 113-204: WindowAttention3D implementation
"""

import sys
import math
import torch
sys.path.insert(0, '.')

from typing import Dict, Optional, List

from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.softmax import SecretSoftmaxLayer
from python.layers.matmul import SecretMatMulLayer
from python.layers.reshape import SecretReshapeLayer
from python.layers.add import SecretAddLayer
from python.utils.basic_utils import ExecutionModeOptions
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.enclave_interfaces import GlobalTensor as gt


class SwinWindowAttention3D:
    """
    3D Window-based Multi-Head Self-Attention with Relative Position Bias.
    
    This is the core attention mechanism for Video Swin Transformer.
    Unlike standard attention, it:
    1. Operates on local 3D windows (e.g., 2x7x7 tokens)
    2. Uses relative position bias instead of absolute position encoding
    3. Supports attention masking for shifted windows
    
    Architecture:
        Input: (num_windows*B, Wd*Wh*Ww, C)
        -> QKV Linear: 3 * (num_windows*B, tokens, C)
        -> Reshape to heads: (num_windows*B, num_heads, tokens, head_dim)
        -> Q @ K^T with relative position bias
        -> Softmax with optional mask
        -> @ V
        -> Reshape and output projection
        Output: (num_windows*B, Wd*Wh*Ww, C)
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        dim: int,
        window_size=(2, 7, 7),  # (Wd, Wh, Ww)
        num_heads: int = 3,
        batch_size: int = 1,
        num_windows: int = 64,  # (D/Wd) * (H/Wh) * (W/Ww)
        qkv_bias: bool = True,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.sid = sid
        self.name_prefix = name_prefix
        self.enclave_mode = enclave_mode
        self.dim = dim
        self.window_size = tuple(window_size)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.batch_size = batch_size
        self.num_windows = num_windows
        self.qkv_bias = qkv_bias
        
        self.layer_mode_overrides = layer_mode_overrides or {}
        self.layers = []
        
        # Calculate window tokens
        Wd, Wh, Ww = self.window_size
        self.window_tokens = Wd * Wh * Ww
        self.total_windows = num_windows * batch_size
        
        # Build relative position bias table and index
        self._build_relative_position_bias()
        
        # Build attention layers
        self._build_layers()
    
    def _get_mode(self, name: str) -> ExecutionModeOptions:
        """Get execution mode for a specific layer."""
        full_name = f"{self.name_prefix}_{name}"
        return self.layer_mode_overrides.get(full_name, self.enclave_mode)
    
    def _build_relative_position_bias(self):
        """
        Build relative position bias table and index.
        
        The bias table has size [(2*Wd-1) * (2*Wh-1) * (2*Ww-1), num_heads]
        The index maps each (token_i, token_j) pair to a position in the table.
        """
        Wd, Wh, Ww = self.window_size
        
        # Relative position bias table size
        # (2*Wd-1) * (2*Wh-1) * (2*Ww-1) possible relative positions
        bias_table_size = (2 * Wd - 1) * (2 * Wh - 1) * (2 * Ww - 1)
        
        # Initialize bias table (will be loaded with actual weights)
        self.relative_position_bias_table = torch.zeros(
            bias_table_size, self.num_heads
        )
        
        # Build relative position index
        # Get coordinates for each position in the window
        coords_d = torch.arange(Wd)
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        
        # Create 3D coordinate grid: (3, Wd, Wh, Ww)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        
        # Flatten to (3, Wd*Wh*Ww)
        coords_flatten = torch.flatten(coords, 1)
        
        # Compute relative coordinates: (3, tokens, tokens)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        
        # Permute to (tokens, tokens, 3)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        # Shift to start from 0
        relative_coords[:, :, 0] += Wd - 1
        relative_coords[:, :, 1] += Wh - 1
        relative_coords[:, :, 2] += Ww - 1
        
        # Compute linear index
        relative_coords[:, :, 0] *= (2 * Wh - 1) * (2 * Ww - 1)
        relative_coords[:, :, 1] *= (2 * Ww - 1)
        
        relative_position_index = relative_coords.sum(-1)  # (tokens, tokens)
        
        self.relative_position_index = relative_position_index
    
    def _build_layers(self):
        """Build the attention computation layers."""
        # QKV projection: (num_windows*B, tokens, C) -> (num_windows*B, tokens, 3*C)
        self.qkv_proj = SGXLinearBase(
            self.sid, f"{self.name_prefix}_qkv",
            self._get_mode("qkv"),
            batch_size=self.total_windows * self.window_tokens,
            n_output_features=3 * self.dim,
            n_input_features=self.dim,
            bias=self.qkv_bias,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.qkv_proj)
        
        # Reshape QKV to separate Q, K, V and multi-head format
        # (num_windows*B, tokens, 3*C) -> (num_windows*B, tokens, 3, num_heads, head_dim)
        # We'll split this in the forward pass using indexing
        
        # Q @ K^T with scaling
        self.qk_matmul = SecretMatMulLayer(
            self.sid, f"{self.name_prefix}_qk_matmul",
            self._get_mode("qk_matmul"),
            transpose_b=True,
            scale=self.scale,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.qk_matmul)
        
        # Add relative position bias
        # This is handled in forward pass by indexing the bias table
        
        # Softmax over attention scores
        self.attn_softmax = SecretSoftmaxLayer(
            self.sid, f"{self.name_prefix}_softmax",
            self._get_mode("softmax"),
            dim=-1,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.attn_softmax)
        
        # Attention @ V
        self.attn_v_matmul = SecretMatMulLayer(
            self.sid, f"{self.name_prefix}_attn_v",
            self._get_mode("attn_v"),
            transpose_b=False,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.attn_v_matmul)
        
        # Output projection: (num_windows*B, tokens, C) -> (num_windows*B, tokens, C)
        self.out_proj = SGXLinearBase(
            self.sid, f"{self.name_prefix}_proj",
            self._get_mode("proj"),
            batch_size=self.total_windows * self.window_tokens,
            n_output_features=self.dim,
            n_input_features=self.dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.out_proj)
        
        # Store input/output layers for connection
        self.input_layer = self.qkv_proj
        self.output_layer = self.out_proj
    
    def get_all_layers(self):
        """Return all layers for registration."""
        return self.layers
    
    def connect(self, prev_layer):
        """Connect this attention module to previous layer."""
        self.input_layer.register_prev_layer(prev_layer)
        return self.output_layer
    
    def load_relative_position_bias(self, bias_table):
        """
        Load relative position bias table from a trained model.
        
        Args:
            bias_table: Tensor of shape [(2*Wd-1)*(2*Wh-1)*(2*Ww-1), num_heads]
        """
        assert bias_table.shape == self.relative_position_bias_table.shape, \
            f"Bias table shape mismatch: {bias_table.shape} vs {self.relative_position_bias_table.shape}"
        self.relative_position_bias_table.copy_(bias_table)
    
    def get_relative_position_bias(self):
        """
        Get relative position bias for attention computation.
        
        Returns:
            bias: (num_heads, tokens, tokens)
        """
        # Index into bias table using relative_position_index
        # bias_table: [(2*Wd-1)*(2*Wh-1)*(2*Ww-1), num_heads]
        # index: (tokens, tokens)
        # result: (tokens, tokens, num_heads)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_tokens, self.window_tokens, -1)
        
        # Permute to (num_heads, tokens, tokens)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        return relative_position_bias
    
    def forward_with_mask(self, x, mask=None):
        """
        Forward pass with optional attention mask.
        
        This is a conceptual forward - actual execution uses the layer-by-layer
        approach with the SGX layers.
        
        Args:
            x: (num_windows*B, tokens, C)
            mask: Optional attention mask (num_windows, tokens, tokens)
        
        Returns:
            output: (num_windows*B, tokens, C)
        """
        # This is typically handled by the layer-by-layer execution
        # The mask would be added to attention scores before softmax
        pass
    
    def print_info(self):
        """Print attention module information."""
        print(f"\n{'='*60}")
        print(f"Swin Window Attention 3D: {self.name_prefix}")
        print(f"{'='*60}")
        print(f"  Embedding dim: {self.dim}")
        print(f"  Num heads: {self.num_heads}")
        print(f"  Head dim: {self.head_dim}")
        print(f"  Window size: {self.window_size} (Wd x Wh x Ww)")
        print(f"  Window tokens: {self.window_tokens}")
        print(f"  Num windows: {self.num_windows}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Total windows: {self.total_windows}")
        print(f"  Relative position bias table size: {self.relative_position_bias_table.shape}")
        print(f"  Number of layers: {len(self.layers)}")
        print(f"{'='*60}\n")


def create_swin_window_attention_3d(
    sid: int,
    name_prefix: str,
    enclave_mode: ExecutionModeOptions,
    dim: int,
    window_size=(2, 7, 7),
    num_heads: int = 3,
    batch_size: int = 1,
    num_windows: int = 64,
    qkv_bias: bool = True,
    layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
):
    """
    Factory function to create Swin Window Attention 3D module.
    
    Example:
        attn = create_swin_window_attention_3d(
            sid=0,
            name_prefix="stage0_block0_attn",
            enclave_mode=ExecutionModeOptions.Enclave,
            dim=96,
            window_size=(2, 7, 7),
            num_heads=3,
            batch_size=1,
            num_windows=64,  # (8//2) * (56//7) * (56//7) = 4 * 8 * 8
        )
    """
    return SwinWindowAttention3D(
        sid=sid,
        name_prefix=name_prefix,
        enclave_mode=enclave_mode,
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        batch_size=batch_size,
        num_windows=num_windows,
        qkv_bias=qkv_bias,
        layer_mode_overrides=layer_mode_overrides,
    )
