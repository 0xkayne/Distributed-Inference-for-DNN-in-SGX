"""
Video Swin Transformer 3D implementation for SGX distributed inference.

Video Swin Transformer is a hierarchical Vision Transformer using shifted windows
in 3D space (temporal + spatial dimensions).

Key Features:
1. Hierarchical structure: 4 stages with decreasing spatial resolution
2. 3D Shifted Window Attention: Alternating W-MSA and SW-MSA for cross-window interaction
3. Patch Merging: Downsampling between stages (2x2 spatial patches -> 1 patch, 2C channels)
4. Window size: Default (2, 7, 7) - 2 temporal x 7x7 spatial (local attention within windows)

Architecture (Video Swin-Tiny):
- Input: (B, 3, D, H, W) = (1, 3, 8, 224, 224)
- Patch Partition: (B, 96, 4, 56, 56) via 3D conv (2x4x4 patches)
- Stage 1: 4x56x56, 2 blocks, C=96, heads=3
- Stage 2: 4x28x28, 2 blocks, C=192, heads=6 (after patch merging)
- Stage 3: 4x14x14, 6 blocks, C=384, heads=12 (after patch merging)
- Stage 4: 4x7x7, 2 blocks, C=768, heads=24 (after patch merging)
- Classification Head: Global Average Pool + Linear

Swin Block Structure:
- Block 1: LN -> W-MSA -> Residual -> LN -> MLP -> Residual
- Block 2: LN -> SW-MSA -> Residual -> LN -> MLP -> Residual

Reference: "Video Swin Transformer" (Liu et al., 2022)
           https://arxiv.org/abs/2106.13230
GitHub: https://github.com/SwinTransformer/Video-Swin-Transformer
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from functools import reduce
from operator import mul

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
class VideoSwinConfig:
    """Configuration for Video Swin Transformer model."""
    video_frames: int = 8           # Number of input frames
    image_size: int = 224           # Input image size (H, W)
    patch_size: Tuple[int, int, int] = (2, 4, 4)  # Patch size (D, H, W)
    in_channels: int = 3            # Input channels
    embed_dim: int = 96             # Initial embedding dimension C
    depths: Tuple[int, ...] = (2, 2, 6, 2)  # Number of blocks per stage
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)  # Attention heads per stage
    window_size: Tuple[int, int, int] = (2, 7, 7)  # Window size (Wd, Wh, Ww)
    mlp_ratio: float = 4.0          # MLP hidden dimension ratio
    dropout: float = 0.0            # Dropout probability
    attention_dropout: float = 0.0  # Attention dropout
    drop_path_rate: float = 0.1     # Stochastic depth rate
    num_classes: int = 400          # Number of output classes (Kinetics-400)
    qkv_bias: bool = True           # Use bias in QKV projection
    
    @property
    def num_stages(self) -> int:
        return len(self.depths)


@dataclass
class VideoSwinTinyConfig(VideoSwinConfig):
    """Video Swin-Tiny: C=96, depths=[2,2,6,2], heads=[3,6,12,24], ~28M params."""
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 6, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)


@dataclass  
class VideoSwinSmallConfig(VideoSwinConfig):
    """Video Swin-Small: C=96, depths=[2,2,18,2], heads=[3,6,12,24], ~49M params."""
    embed_dim: int = 96
    depths: Tuple[int, ...] = (2, 2, 18, 2)
    num_heads: Tuple[int, ...] = (3, 6, 12, 24)


@dataclass
class VideoSwinBaseConfig(VideoSwinConfig):
    """Video Swin-Base: C=128, depths=[2,2,18,2], heads=[4,8,16,32], ~88M params."""
    embed_dim: int = 128
    depths: Tuple[int, ...] = (2, 2, 18, 2)
    num_heads: Tuple[int, ...] = (4, 8, 16, 32)


# ==============================================================================
# Core Components
# ==============================================================================

def window_partition_3d(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Partition 3D feature map into non-overlapping windows.
    
    Args:
        x: (B, D, H, W, C)
        window_size: (Wd, Wh, Ww)
    
    Returns:
        windows: (num_windows*B, Wd*Wh*Ww, C)
    """
    B, D, H, W, C = x.shape
    Wd, Wh, Ww = window_size
    
    x = x.view(B, D // Wd, Wd, H // Wh, Wh, W // Ww, Ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, reduce(mul, window_size), C)
    
    return windows


def window_reverse_3d(
    windows: torch.Tensor,
    window_size: Tuple[int, int, int],
    B: int, D: int, H: int, W: int
) -> torch.Tensor:
    """
    Reverse window partition.
    
    Args:
        windows: (num_windows*B, Wd*Wh*Ww, C)
        window_size: (Wd, Wh, Ww)
        B, D, H, W: Original dimensions
    
    Returns:
        x: (B, D, H, W, C)
    """
    Wd, Wh, Ww = window_size
    x = windows.view(B, D // Wd, H // Wh, W // Ww, Wd, Wh, Ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, -1)
    
    return x


def compute_mask_3d(D, H, W, window_size, shift_size, device):
    """
    Compute attention mask for shifted window attention.
    
    Args:
        D, H, W: Padded dimensions
        window_size: (Wd, Wh, Ww)
        shift_size: (shift_d, shift_h, shift_w)
        device: torch device
    
    Returns:
        attn_mask: (num_windows, Wd*Wh*Ww, Wd*Wh*Ww)
    """
    Wd, Wh, Ww = window_size
    shift_d, shift_h, shift_w = shift_size
    
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    
    # Generate mask regions
    d_slices = (
        slice(0, -Wd),
        slice(-Wd, -shift_d) if shift_d > 0 else slice(-Wd, None),
        slice(-shift_d, None) if shift_d > 0 else slice(0, 0)
    )
    h_slices = (
        slice(0, -Wh),
        slice(-Wh, -shift_h),
        slice(-shift_h, None)
    )
    w_slices = (
        slice(0, -Ww),
        slice(-Ww, -shift_w),
        slice(-shift_w, None)
    )
    
    cnt = 0
    for d in d_slices:
        for h in h_slices:
            for w in w_slices:
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    
    # Partition mask into windows
    mask_windows = window_partition_3d(img_mask, window_size)  # (nW, Wd*Wh*Ww, 1)
    mask_windows = mask_windows.squeeze(-1)  # (nW, Wd*Wh*Ww)
    
    # Compute attention mask
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    
    return attn_mask


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding layer for Video Swin Transformer.
    
    Uses Conv3d for efficiency: (Pd, Ph, Pw) patches with stride (Pd, Ph, Pw).
    Output: (B, embed_dim, D/Pd, H/Ph, W/Pw)
    """
    
    def __init__(self, config: VideoSwinConfig):
        super().__init__()
        self.video_frames = config.video_frames
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.embed_dim = config.embed_dim
        
        self.proj = nn.Conv3d(
            config.in_channels, config.embed_dim,
            kernel_size=config.patch_size, stride=config.patch_size
        )
        self.norm = nn.LayerNorm(config.embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        """
        Args:
            x: (B, C, D, H, W)
        
        Returns:
            x: (B, D*H*W, embed_dim) - flattened
            D_out, H_out, W_out: Spatial dimensions after patching
        """
        B, C, D, H, W = x.shape
        
        # Patch embedding via Conv3d
        x = self.proj(x)  # (B, embed_dim, D/Pd, H/Ph, W/Pw)
        D_out, H_out, W_out = x.shape[2], x.shape[3], x.shape[4]
        
        # Flatten spatial dimensions: (B, embed_dim, D, H, W) -> (B, D*H*W, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x, D_out, H_out, W_out


class PatchMerging3D(nn.Module):
    """
    Patch Merging layer for spatial downsampling (temporal unchanged).
    
    Standard Video Swin: Merges 2x2 spatial patches (H, W), keeps temporal D.
    Concatenates 4 patches -> 4*C channels, then projects to 2*C channels.
    
    Output resolution: D x H/2 x W/2
    Output channels: 2*dim
    """
    
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(
        self, x: torch.Tensor, D: int, H: int, W: int
    ) -> Tuple[torch.Tensor, int, int, int]:
        """
        Args:
            x: (B, D*H*W, C)
            D, H, W: Spatial dimensions
        
        Returns:
            x: (B, D*H/2*W/2, 2*C)
            D_new, H_new, W_new: New dimensions
        """
        B, L, C = x.shape
        assert L == D * H * W, f"Input size mismatch: {L} != {D}*{H}*{W}"
        
        x = x.view(B, D, H, W, C)
        
        # Merge 2x2 spatial patches (keep D unchanged)
        x0 = x[:, :, 0::2, 0::2, :]  # (B, D, H/2, W/2, C)
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, D, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)  # (B, D*H/2*W/2, 4*C)
        
        x = self.norm(x)
        x = self.reduction(x)  # (B, D*H/2*W/2, 2*C)
        
        return x, D, H // 2, W // 2


class WindowAttention3D(nn.Module):
    """
    3D Window-based Multi-Head Self-Attention with Relative Position Bias.
    
    Computes attention within local 3D windows of size Wd x Wh x Ww.
    Uses relative position bias to encode spatial relationships.
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
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
        Wd, Wh, Ww = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wd - 1) * (2 * Wh - 1) * (2 * Ww - 1), num_heads)
        )
        
        # Relative position index
        coords_d = torch.arange(Wd)
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        relative_coords[:, :, 0] += Wd - 1
        relative_coords[:, :, 1] += Wh - 1
        relative_coords[:, :, 2] += Ww - 1
        relative_coords[:, :, 0] *= (2 * Wh - 1) * (2 * Ww - 1)
        relative_coords[:, :, 1] *= (2 * Ww - 1)
        
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
            x: (num_windows*B, Wd*Wh*Ww, C)
            mask: Attention mask for shifted windows (num_windows, Wd*Wh*Ww, Wd*Wh*Ww)
        
        Returns:
            x: (num_windows*B, Wd*Wh*Ww, C)
        """
        B_, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply attention mask for shifted windows
        if mask is not None:
            nW = mask.shape[0]
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
    """MLP module for Video Swin Transformer block."""
    
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


class SwinTransformerBlock3D(nn.Module):
    """
    Video Swin Transformer Block with 3D Shifted Window Attention.
    
    Two types:
    - W-MSA: Regular window attention (shift_size=(0,0,0))
    - SW-MSA: Shifted window attention (shift_size=window_size//2)
    
    Structure: LN -> W-MSA/SW-MSA -> Residual -> LN -> MLP -> Residual
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (2, 7, 7),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
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
        self.attn = WindowAttention3D(
            dim, window_size, num_heads, qkv_bias, attention_dropout, dropout
        )
        
        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout(drop_path)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMLP(dim, mlp_ratio, dropout)
    
    def forward(
        self, x: torch.Tensor, D: int, H: int, W: int,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, D*H*W, C)
            D, H, W: Spatial dimensions
            attn_mask: Attention mask for shifted windows
        
        Returns:
            x: (B, D*H*W, C)
        """
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)
        
        # Cyclic shift for SW-MSA
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(
                x,
                shifts=tuple(-s for s in self.shift_size),
                dims=(1, 2, 3)
            )
        else:
            shifted_x = x
        
        # Partition into windows
        x_windows = window_partition_3d(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, reduce(mul, self.window_size), C)
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, *self.window_size, C)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, B, D, H, W)
        
        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2, 3))
        else:
            x = shifted_x
        
        x = x.view(B, D * H * W, C)
        
        # Residual connection
        x = shortcut + self.drop_path(x)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class SwinStage3D(nn.Module):
    """
    Video Swin Transformer Stage.
    
    Each stage contains:
    - Multiple Swin Transformer blocks (alternating W-MSA and SW-MSA)
    - Optional Patch Merging at the end (except last stage)
    """
    
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: List[float] = None,
        downsample: bool = True,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        
        # Build blocks (alternating W-MSA and SW-MSA)
        Wd, Wh, Ww = window_size
        shift_size = (Wd // 2, Wh // 2, Ww // 2)
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else shift_size,
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
            self.downsample = PatchMerging3D(dim, norm_layer)
        else:
            self.downsample = None
    
    def forward(
        self, x: torch.Tensor, D: int, H: int, W: int
    ) -> Tuple[torch.Tensor, int, int, int]:
        """
        Args:
            x: (B, D*H*W, C)
            D, H, W: Spatial dimensions
        
        Returns:
            x: Output tensor
            D_new, H_new, W_new: New spatial dimensions
        """
        # Create attention mask for SW-MSA blocks
        Wd, Wh, Ww = self.window_size
        shift_size = (Wd // 2, Wh // 2, Ww // 2)
        
        Dp = int(math.ceil(D / Wd)) * Wd
        Hp = int(math.ceil(H / Wh)) * Wh
        Wp = int(math.ceil(W / Ww)) * Ww
        
        attn_mask = compute_mask_3d(Dp, Hp, Wp, self.window_size, shift_size, x.device)
        
        for i, block in enumerate(self.blocks):
            # W-MSA uses no mask, SW-MSA uses attention mask
            mask = attn_mask if (i % 2 == 1) else None
            x = block(x, D, H, W, mask)
        
        if self.downsample is not None:
            x, D, H, W = self.downsample(x, D, H, W)
        
        return x, D, H, W


class VideoSwinTransformer(nn.Module):
    """
    Video Swin Transformer model for video classification.
    
    Architecture:
    - Patch Embedding: Conv3d (Pd x Ph x Pw patches)
    - Stage 1: blocks with dim=C
    - Stage 2: Patch Merge + blocks with dim=2C
    - Stage 3: Patch Merge + blocks with dim=4C
    - Stage 4: Patch Merge + blocks with dim=8C
    - Classification Head: Global Average Pool + Linear
    """
    
    def __init__(self, config: Optional[VideoSwinConfig] = None):
        super().__init__()
        self.config = config or VideoSwinTinyConfig()
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(self.config)
        
        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, self.config.drop_path_rate, sum(self.config.depths))]
        
        # Build stages
        self.stages = nn.ModuleList()
        dim = self.config.embed_dim
        
        for i_stage in range(self.config.num_stages):
            stage = SwinStage3D(
                dim=dim,
                depth=self.config.depths[i_stage],
                num_heads=self.config.num_heads[i_stage],
                window_size=self.config.window_size,
                mlp_ratio=self.config.mlp_ratio,
                qkv_bias=self.config.qkv_bias,
                dropout=self.config.dropout,
                attention_dropout=self.config.attention_dropout,
                drop_path=dpr[sum(self.config.depths[:i_stage]):sum(self.config.depths[:i_stage+1])],
                downsample=(i_stage < self.config.num_stages - 1),
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) input video
        
        Returns:
            logits: (B, num_classes)
        """
        # Patch embedding
        x, D, H, W = self.patch_embed(x)
        
        # Stages
        for stage in self.stages:
            x, D, H, W = stage(x, D, H, W)
        
        # Classification head
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).flatten(1)
        x = self.head(x)
        
        return x


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_video_swin_tiny(num_classes: int = 400) -> VideoSwinTransformer:
    """Create Video Swin-Tiny model."""
    config = VideoSwinTinyConfig(num_classes=num_classes)
    return VideoSwinTransformer(config)


def create_video_swin_small(num_classes: int = 400) -> VideoSwinTransformer:
    """Create Video Swin-Small model."""
    config = VideoSwinSmallConfig(num_classes=num_classes)
    return VideoSwinTransformer(config)


def create_video_swin_base(num_classes: int = 400) -> VideoSwinTransformer:
    """Create Video Swin-Base model."""
    config = VideoSwinBaseConfig(num_classes=num_classes)
    return VideoSwinTransformer(config)


# ==============================================================================
# Quick Test
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Video Swin Transformer 3D - Architecture Analysis")
    print("=" * 70)
    
    # Create Video Swin-Tiny
    model = create_video_swin_tiny(num_classes=400)
    
    print(f"\n[Video Swin-Tiny Configuration]")
    print(f"  - Video frames: {model.config.video_frames}")
    print(f"  - Image Size: {model.config.image_size}")
    print(f"  - Patch Size: {model.config.patch_size} (3D)")
    print(f"  - Embed Dim: {model.config.embed_dim}")
    print(f"  - Depths: {model.config.depths}")
    print(f"  - Num Heads: {model.config.num_heads}")
    print(f"  - Window Size: {model.config.window_size} (3D: Temporal x Spatial)")
    print(f"  - MLP Ratio: {model.config.mlp_ratio}")
    
    # Parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n  - Total Parameters: {num_params / 1e6:.2f}M")
    
    # Test forward pass
    print(f"\n[Forward Pass Test]")
    batch_size = 1
    
    x = torch.randn(batch_size, 3, 8, 224, 224)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {output.shape}")
    
    print("\n" + "=" * 70)
    print("✓ Video Swin Transformer 3D model ready")
    print("  Key feature: 3D Shifted Window attention for efficient video modeling")
    print("  - Temporal + Spatial local attention")
    print("  - Relative position bias (3D)")
    print("  - Hierarchical feature extraction")
    print("=" * 70)
