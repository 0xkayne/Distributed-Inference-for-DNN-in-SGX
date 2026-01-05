"""
Swin Transformer implementation for SGX distributed inference.

Swin Transformer is a hierarchical Vision Transformer using shifted windows.

Key Features:
1. Hierarchical structure: 4 stages with decreasing spatial resolution
2. Shifted Window Attention: Alternating W-MSA and SW-MSA for cross-window interaction
3. Patch Merging: Downsampling between stages (2x2 patches -> 1 patch, 4C channels)
4. Window size: Default 7x7 (local attention within windows)

Architecture (Swin-T):
- Input: 224x224x3
- Patch Partition: 56x56x96 (4x4 patches)
- Stage 1: 56x56, 2 blocks, C=96, heads=3
- Stage 2: 28x28, 2 blocks, C=192, heads=6 (after patch merging)
- Stage 3: 14x14, 6 blocks, C=384, heads=12 (after patch merging)
- Stage 4: 7x7, 2 blocks, C=768, heads=24 (after patch merging)
- Classification Head: Global Average Pool + Linear

Swin Block Structure:
- Block 1: LN -> W-MSA -> Residual -> LN -> MLP -> Residual
- Block 2: LN -> SW-MSA -> Residual -> LN -> MLP -> Residual

Reference: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
           (Liu et al., ICCV 2021)
GitHub: https://github.com/microsoft/Swin-Transformer
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any

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
class SwinConfig:
    """Configuration for Swin Transformer model."""
    image_size: int = 224           # Input image size
    patch_size: int = 4             # Initial patch size
    in_channels: int = 3            # Input channels
    embed_dim: int = 96             # Initial embedding dimension C
    depths: Tuple[int, ...] = (2, 2, 6, 2)  # Number of blocks per stage
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)  # Attention heads per stage
    window_size: int = 7            # Window size for local attention
    mlp_ratio: float = 4.0          # MLP hidden dimension ratio
    dropout: float = 0.0            # Dropout probability
    attention_dropout: float = 0.0  # Attention dropout
    drop_path_rate: float = 0.1     # Stochastic depth rate
    num_classes: int = 1000         # Number of output classes
    qkv_bias: bool = True           # Use bias in QKV projection
    
    @property
    def num_stages(self) -> int:
        return len(self.depths)
    
    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


@dataclass
class SwinTinyConfig(SwinConfig):
    """Swin-Tiny: C=96, depths=[2,2,6,2], heads=[3,6,12,24], ~29M params."""
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 6, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)


@dataclass  
class SwinSmallConfig(SwinConfig):
    """Swin-Small: C=96, depths=[2,2,18,2], heads=[3,6,12,24], ~50M params."""
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 18, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)


@dataclass
class SwinBaseConfig(SwinConfig):
    """Swin-Base: C=128, depths=[2,2,18,2], heads=[4,8,16,32], ~88M params."""
    embed_dim: int = 128
    depths: Tuple[int, ...] = (2, 2, 18, 2)
    num_heads: Tuple[int, ...] = (4, 8, 16, 32)


# ==============================================================================
# Core Components
# ==============================================================================

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.
    
    Args:
        x: (B, H, W, C)
        window_size: Window size M
    
    Returns:
        windows: (num_windows*B, M, M, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition.
    
    Args:
        windows: (num_windows*B, M, M, C)
        window_size: Window size M
        H, W: Feature map spatial dimensions
    
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding layer for Swin Transformer.
    
    Uses Conv2d for efficiency: 4x4 patches with stride 4.
    Output: (B, H/4 * W/4, embed_dim)
    """
    
    def __init__(self, config: SwinConfig):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.embed_dim = config.embed_dim
        
        self.proj = nn.Conv2d(
            config.in_channels, config.embed_dim,
            kernel_size=config.patch_size, stride=config.patch_size
        )
        self.norm = nn.LayerNorm(config.embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            x: (B, H*W, embed_dim)
            H, W: Spatial dimensions after patching
        """
        B, C, H, W = x.shape
        
        # Patch embedding via Conv2d
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        H_out, W_out = x.shape[2], x.shape[3]
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        x = self.norm(x)
        
        return x, H_out, W_out


class PatchMerging(nn.Module):
    """
    Patch Merging layer for downsampling.
    
    Merges 2x2 neighboring patches into 1 patch with 4*C channels,
    then projects to 2*C channels.
    
    Output resolution: H/2 x W/2
    Output channels: 2*dim
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, H*W, C)
            H, W: Spatial dimensions
        
        Returns:
            x: (B, H/2*W/2, 2*C)
            H_new, W_new: New spatial dimensions
        """
        B, L, C = x.shape
        assert L == H * W, f"Input size mismatch: {L} != {H}*{W}"
        
        x = x.view(B, H, W, C)
        
        # Merge 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)  # (B, H/2*W/2, 4*C)
        
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2*W/2, 2*C)
        
        return x, H // 2, W // 2


class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention (W-MSA / SW-MSA).
    
    Computes attention within local windows of size MxM.
    Supports both regular and shifted windows.
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Relative position bias table
        # (2*M-1) * (2*M-1) possible relative positions
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (num_windows*B, M*M, C)
            mask: Attention mask for shifted windows (num_windows, M*M, M*M)
        
        Returns:
            x: (num_windows*B, M*M, C)
        """
        B_, N, C = x.shape  # B_ = num_windows * B, N = M*M
        
        # QKV projection
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply attention mask for shifted windows
        if mask is not None:
            nW = mask.shape[0]  # number of windows
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SwinMLP(nn.Module):
    """MLP module for Swin Transformer block."""
    
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    
    Two types:
    - W-MSA: Regular window attention (shift_size=0)
    - SW-MSA: Shifted window attention (shift_size=window_size//2)
    
    Structure: LN -> W-MSA/SW-MSA -> Residual -> LN -> MLP -> Residual
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads, qkv_bias, attention_dropout, dropout
        )
        
        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMLP(dim, mlp_ratio, dropout)
    
    def forward(self, x: torch.Tensor, H: int, W: int, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C)
            H, W: Spatial dimensions
            attn_mask: Attention mask for shifted windows
        
        Returns:
            x: (B, H*W, C)
        """
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)  # (num_windows*B, M, M, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)
        
        # Residual connection
        x = shortcut + self.drop_path(x)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class SwinStage(nn.Module):
    """
    Swin Transformer Stage.
    
    Each stage contains:
    - Multiple Swin Transformer blocks (alternating W-MSA and SW-MSA)
    - Optional Patch Merging at the end (except last stage)
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: List[float] = None,
        downsample: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        
        # Build blocks (alternating W-MSA and SW-MSA)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=drop_path[i] if drop_path else 0.0,
            )
            for i in range(depth)
        ])
        
        # Patch merging (downsampling)
        if downsample:
            self.downsample = PatchMerging(dim)
        else:
            self.downsample = None
    
    def _create_attention_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        """Create attention mask for SW-MSA blocks."""
        if self.window_size <= 0:
            return None
        
        Hp = int(math.ceil(H / self.window_size)) * self.window_size
        Wp = int(math.ceil(W / self.window_size)) * self.window_size
        
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.window_size // 2),
            slice(-self.window_size // 2, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.window_size // 2),
            slice(-self.window_size // 2, None)
        )
        
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, H*W, C)
            H, W: Spatial dimensions
        
        Returns:
            x: Output tensor
            H_new, W_new: New spatial dimensions
        """
        attn_mask = self._create_attention_mask(H, W, x.device)
        
        for i, block in enumerate(self.blocks):
            # W-MSA uses no mask, SW-MSA uses attention mask
            mask = attn_mask if block.shift_size > 0 else None
            x = block(x, H, W, mask)
        
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        
        return x, H, W


class SGXSwinTransformer(nn.Module):
    """
    Swin Transformer model for SGX distributed inference.
    
    Architecture:
    - Patch Embedding: Conv2d (4x4 patches)
    - Stage 1: blocks with dim=C
    - Stage 2: Patch Merge + blocks with dim=2C
    - Stage 3: Patch Merge + blocks with dim=4C
    - Stage 4: Patch Merge + blocks with dim=8C
    - Classification Head: Global Average Pool + Linear
    """
    
    def __init__(self, config: Optional[SwinConfig] = None):
        super().__init__()
        self.config = config or SwinTinyConfig()
        
        # Patch embedding
        self.patch_embed = PatchEmbed(self.config)
        
        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, self.config.drop_path_rate, sum(self.config.depths))]
        
        # Build stages
        self.stages = nn.ModuleList()
        dim = self.config.embed_dim
        
        for i_stage in range(self.config.num_stages):
            stage = SwinStage(
                dim=dim,
                depth=self.config.depths[i_stage],
                num_heads=self.config.num_heads[i_stage],
                window_size=self.config.window_size,
                mlp_ratio=self.config.mlp_ratio,
                qkv_bias=self.config.qkv_bias,
                dropout=self.config.dropout,
                attention_dropout=self.config.attention_dropout,
                drop_path=dpr[sum(self.config.depths[:i_stage]):sum(self.config.depths[:i_stage+1])],
                downsample=(i_stage < self.config.num_stages - 1),  # No downsample for last stage
            )
            self.stages.append(stage)
            
            # Update dimension for next stage (after patch merging)
            if i_stage < self.config.num_stages - 1:
                dim = dim * 2
        
        self.final_dim = dim
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dim, self.config.num_classes)
        
        # Build layer registry
        self.layers = []
        self._build_layer_registry()
    
    def _build_layer_registry(self):
        """Build layer registry for profiling."""
        self.layers = []
        
        # Patch embedding
        self.layers.append({
            'name': 'patch_embed',
            'module': self.patch_embed,
            'type': 'PatchEmbed',
            'dependencies': ['input'],
        })
        
        # Stages
        for i_stage, stage in enumerate(self.stages):
            stage_name = f'stage{i_stage}'
            
            for i_block, block in enumerate(stage.blocks):
                block_type = 'W-MSA' if block.shift_size == 0 else 'SW-MSA'
                self.layers.append({
                    'name': f'{stage_name}_block{i_block}',
                    'module': block,
                    'type': f'SwinBlock_{block_type}',
                    'dependencies': [self.layers[-1]['name'] if self.layers else 'patch_embed'],
                })
            
            if stage.downsample is not None:
                self.layers.append({
                    'name': f'{stage_name}_downsample',
                    'module': stage.downsample,
                    'type': 'PatchMerging',
                    'dependencies': [self.layers[-1]['name']],
                })
        
        # Classification head
        self.layers.append({
            'name': 'norm',
            'module': self.norm,
            'type': 'LayerNorm',
            'dependencies': [self.layers[-1]['name']],
        })
        
        self.layers.append({
            'name': 'head',
            'module': self.head,
            'type': 'Linear',
            'dependencies': ['norm'],
        })
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input image
        
        Returns:
            logits: (B, num_classes)
        """
        # Patch embedding
        x, H, W = self.patch_embed(x)
        
        # Stages
        for stage in self.stages:
            x, H, W = stage(x, H, W)
        
        # Classification head
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).flatten(1)
        x = self.head(x)
        
        return x
    
    def forward_layer_by_layer(
        self,
        x: torch.Tensor,
        layer_timing_callback: Optional[Callable[[str, float], None]] = None
    ) -> torch.Tensor:
        """Execute model layer by layer with timing callback."""
        import time
        
        # Patch embedding
        start = time.perf_counter()
        x, H, W = self.patch_embed(x)
        if layer_timing_callback:
            layer_timing_callback('patch_embed', (time.perf_counter() - start) * 1000)
        
        # Stages
        for i_stage, stage in enumerate(self.stages):
            stage_name = f'stage{i_stage}'
            
            # Create attention mask once per stage
            attn_mask = stage._create_attention_mask(H, W, x.device)
            
            for i_block, block in enumerate(stage.blocks):
                mask = attn_mask if block.shift_size > 0 else None
                
                start = time.perf_counter()
                x = block(x, H, W, mask)
                if layer_timing_callback:
                    layer_timing_callback(f'{stage_name}_block{i_block}', (time.perf_counter() - start) * 1000)
            
            # Patch merging
            if stage.downsample is not None:
                start = time.perf_counter()
                x, H, W = stage.downsample(x, H, W)
                if layer_timing_callback:
                    layer_timing_callback(f'{stage_name}_downsample', (time.perf_counter() - start) * 1000)
        
        # Classification head
        start = time.perf_counter()
        x = self.norm(x)
        if layer_timing_callback:
            layer_timing_callback('norm', (time.perf_counter() - start) * 1000)
        
        start = time.perf_counter()
        x = self.avgpool(x.transpose(1, 2)).flatten(1)
        x = self.head(x)
        if layer_timing_callback:
            layer_timing_callback('head', (time.perf_counter() - start) * 1000)
        
        return x
    
    def get_memory_footprint(self, batch_size: int = 1) -> Dict[str, Any]:
        """Estimate memory footprint for TEE execution planning."""
        config = self.config
        H, W = config.image_size // config.patch_size, config.image_size // config.patch_size
        
        activations = {}
        dim = config.embed_dim
        
        for i_stage in range(config.num_stages):
            tokens = H * W
            activations[f'stage{i_stage}_hidden'] = batch_size * tokens * dim * 4
            activations[f'stage{i_stage}_attn'] = batch_size * config.num_heads[i_stage] * config.window_size ** 2 * config.window_size ** 2 * 4
            
            if i_stage < config.num_stages - 1:
                H, W = H // 2, W // 2
                dim = dim * 2
        
        parameters = sum(p.numel() for p in self.parameters()) * 4
        
        return {
            'config': {
                'embed_dim': config.embed_dim,
                'depths': config.depths,
                'num_heads': config.num_heads,
                'window_size': config.window_size,
                'image_size': config.image_size,
            },
            'activations': activations,
            'parameters_bytes': parameters,
        }


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_swin_tiny(num_classes: int = 1000) -> SGXSwinTransformer:
    """Create Swin-Tiny model."""
    config = SwinTinyConfig(num_classes=num_classes)
    return SGXSwinTransformer(config)


def create_swin_small(num_classes: int = 1000) -> SGXSwinTransformer:
    """Create Swin-Small model."""
    config = SwinSmallConfig(num_classes=num_classes)
    return SGXSwinTransformer(config)


def create_swin_base(num_classes: int = 1000) -> SGXSwinTransformer:
    """Create Swin-Base model."""
    config = SwinBaseConfig(num_classes=num_classes)
    return SGXSwinTransformer(config)


# ==============================================================================
# Quick Test
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SGX Swin Transformer Model - Architecture Analysis")
    print("=" * 70)
    
    # Create Swin-Tiny
    model = create_swin_tiny(num_classes=1000)
    
    print(f"\n[Swin-Tiny Configuration]")
    print(f"  - Image Size: {model.config.image_size}")
    print(f"  - Patch Size: {model.config.patch_size}")
    print(f"  - Embed Dim: {model.config.embed_dim}")
    print(f"  - Depths: {model.config.depths}")
    print(f"  - Num Heads: {model.config.num_heads}")
    print(f"  - Window Size: {model.config.window_size}")
    print(f"  - MLP Ratio: {model.config.mlp_ratio}")
    
    # Parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  - Total Parameters: {num_params / 1e6:.2f}M")
    
    # Memory footprint
    mem = model.get_memory_footprint(batch_size=1)
    print(f"  - Parameters Memory: {mem['parameters_bytes'] / 1024 / 1024:.2f} MB")
    
    # Test forward pass
    print(f"\n[Forward Pass Test]")
    batch_size = 1
    
    x = torch.randn(batch_size, 3, 224, 224)
    
    timings = {}
    def record_timing(name, ms):
        timings[name] = ms
    
    with torch.no_grad():
        output = model.forward_layer_by_layer(x, layer_timing_callback=record_timing)
    
    print(f"  - Output shape: {output.shape}")
    print(f"  - Total time: {sum(timings.values()):.2f} ms")
    print(f"\n  Layer timings (top 10):")
    for name, ms in sorted(timings.items(), key=lambda x: -x[1])[:10]:
        print(f"    {name}: {ms:.2f} ms")
    
    print("\n" + "=" * 70)
    print("✓ SGX Swin Transformer Model ready for profiling")
    print("  Key feature: Shifted Window attention for efficiency")
    print("=" * 70)
