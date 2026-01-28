"""
SGX Video Swin Transformer 3D - Native Implementation with Standard Architecture.

Video Swin Transformer is a hierarchical Vision Transformer for video understanding
that uses shifted window attention in 3D space (temporal + spatial).

Standard Architecture (Video Swin-Tiny):
- Input: (B, 3, D, H, W) - e.g., (1, 3, 8, 224, 224)
- Patch Embedding: 3D Conv (2x4x4) + LayerNorm
- Stage 0: 2 blocks, dim=96, heads=3, window=(2,7,7), resolution=4x56x56
- Stage 1: 2 blocks, dim=192, heads=6, window=(2,7,7), resolution=4x28x28
- Stage 2: 6 blocks, dim=384, heads=12, window=(2,7,7), resolution=4x14x14
- Stage 3: 2 blocks, dim=768, heads=24, window=(2,7,7), resolution=4x7x7

Key Features:
1. 3D Shifted Window Attention - LOCAL attention within 3D windows
2. Relative Position Bias (3D) - learnable bias for position relationships
3. Patch Merging - 2x2x2 downsampling between stages
4. Cyclic Shift - for cross-window connections in SW-MSA blocks

Reference: "Video Swin Transformer" (Liu et al., 2022)
GitHub: https://github.com/SwinTransformer/Video-Swin-Transformer
"""

import sys
import math
sys.path.insert(0, '.')

from typing import Dict, List, Optional, Tuple

from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.sgx_conv3d_base import SGXConv3DBase
from python.layers.layer_norm import SecretLayerNormLayer
from python.layers.gelu import SecretGELULayer
from python.layers.add import SecretAddLayer
from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer
from python.layers.window_partition_3d import SecretWindowPartition3DLayer
from python.layers.window_reverse_3d import SecretWindowReverse3DLayer
from python.layers.cyclic_roll_3d import SecretCyclicRoll3DLayer
from python.layers.swin_window_attention_3d import create_swin_window_attention_3d
from python.utils.basic_utils import ExecutionModeOptions


class SwinMLP3D:
    """
    MLP module for Video Swin Transformer block.
    
    Structure: Linear(dim, 4*dim) -> GELU -> Linear(4*dim, dim)
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        dim: int,
        mlp_ratio: float = 4.0,
        num_tokens: int = 6272,  # D*H*W tokens
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.layers = []
        hidden_dim = int(dim * mlp_ratio)
        overrides = layer_mode_overrides or {}
        
        def get_mode(name):
            return overrides.get(name, enclave_mode)
        
        # FC1
        self.fc1 = SGXLinearBase(
            sid, f"{name_prefix}_fc1", get_mode(f"{name_prefix}_fc1"),
            batch_size=num_tokens,
            n_output_features=hidden_dim,
            n_input_features=dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.fc1)
        
        # GELU
        self.gelu = SecretGELULayer(
            sid, f"{name_prefix}_gelu", get_mode(f"{name_prefix}_gelu"),
            approximate=True,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.gelu)
        
        # FC2
        self.fc2 = SGXLinearBase(
            sid, f"{name_prefix}_fc2", get_mode(f"{name_prefix}_fc2"),
            batch_size=num_tokens,
            n_output_features=dim,
            n_input_features=hidden_dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.fc2)
        
        self.input_layer = self.fc1
        self.output_layer = self.fc2


class SwinTransformerBlock3D:
    """
    Video Swin Transformer Block with 3D Shifted Window Attention.
    
    Standard Implementation Flow:
    1. x = input
    2. LN1(x)
    3. [Optional] Cyclic Shift (for SW-MSA)
    4. Window Partition (3D)
    5. Window Attention with Relative Position Bias
    6. Window Reverse
    7. [Optional] Reverse Cyclic Shift
    8. Residual Connection 1
    9. LN2
    10. MLP
    11. Residual Connection 2
    
    W-MSA: Regular window attention (shift_size=(0,0,0))
    SW-MSA: Shifted window attention (shift_size=(Wd//2, Wh//2, Ww//2))
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        dim: int,
        num_heads: int,
        window_size=(2, 7, 7),  # (Wd, Wh, Ww)
        shift_size=(0, 0, 0),   # (shift_d, shift_h, shift_w)
        mlp_ratio: float = 4.0,
        D: int = 4,  # Temporal dimension after patch embedding
        H: int = 56,
        W: int = 56,
        batch_size: int = 1,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.layers = []
        overrides = layer_mode_overrides or {}
        
        # Store dimensions
        self.D, self.H, self.W = D, H, W
        self.batch_size = batch_size
        self.window_size = window_size
        self.shift_size = shift_size
        
        # Calculate number of tokens and windows
        num_tokens = batch_size * D * H * W
        Wd, Wh, Ww = window_size
        num_windows = (D // Wd) * (H // Wh) * (W // Ww)
        
        def get_mode(name):
            return overrides.get(name, enclave_mode)
        
        # LayerNorm 1
        self.norm1 = SecretLayerNormLayer(
            sid, f"{name_prefix}_norm1", get_mode(f"{name_prefix}_norm1"),
            normalized_shape=[dim],
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.norm1)
        
        # Cyclic Shift (for SW-MSA only)
        if any(s > 0 for s in shift_size):
            self.cyclic_shift = SecretCyclicRoll3DLayer(
                sid, f"{name_prefix}_cyclic_shift",
                get_mode(f"{name_prefix}_cyclic_shift"),
                shifts=tuple(-s for s in shift_size),  # Negative for forward shift
                dims=(1, 2, 3),  # Shift D, H, W dimensions (for BDHWC format)
                manually_register_prev=True, manually_register_next=True
            )
            self.layers.append(self.cyclic_shift)
            self.has_shift = True
        else:
            self.has_shift = False
        
        # Window Partition
        self.window_partition = SecretWindowPartition3DLayer(
            sid, f"{name_prefix}_win_part",
            get_mode(f"{name_prefix}_win_part"),
            window_size=window_size,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.window_partition)
        
        # Window Attention with Relative Position Bias
        self.attn = create_swin_window_attention_3d(
            sid=sid,
            name_prefix=f"{name_prefix}_attn",
            enclave_mode=enclave_mode,
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            batch_size=batch_size,
            num_windows=num_windows,
            qkv_bias=True,
            layer_mode_overrides=overrides
        )
        self.layers.extend(self.attn.get_all_layers())
        
        # Window Reverse
        self.window_reverse = SecretWindowReverse3DLayer(
            sid, f"{name_prefix}_win_rev",
            get_mode(f"{name_prefix}_win_rev"),
            window_size=window_size,
            output_shape=[batch_size, D, H, W, dim],
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.window_reverse)
        
        # Reverse Cyclic Shift (for SW-MSA only)
        if self.has_shift:
            self.reverse_shift = SecretCyclicRoll3DLayer(
                sid, f"{name_prefix}_reverse_shift",
                get_mode(f"{name_prefix}_reverse_shift"),
                shifts=shift_size,  # Positive for reverse shift
                dims=(1, 2, 3),
                manually_register_prev=True, manually_register_next=True
            )
            self.layers.append(self.reverse_shift)
        
        # Residual 1
        self.residual1 = SecretAddLayer(
            sid, f"{name_prefix}_residual1",
            get_mode(f"{name_prefix}_residual1"),
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.residual1)
        
        # LayerNorm 2
        self.norm2 = SecretLayerNormLayer(
            sid, f"{name_prefix}_norm2",
            get_mode(f"{name_prefix}_norm2"),
            normalized_shape=[dim],
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.norm2)
        
        # MLP
        self.mlp = SwinMLP3D(
            sid, f"{name_prefix}_mlp", enclave_mode,
            dim=dim, mlp_ratio=mlp_ratio, num_tokens=num_tokens,
            layer_mode_overrides=overrides
        )
        self.layers.extend(self.mlp.layers)
        
        # Residual 2
        self.residual2 = SecretAddLayer(
            sid, f"{name_prefix}_residual2",
            get_mode(f"{name_prefix}_residual2"),
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.residual2)
        
        self.input_layer = self.norm1
        self.output_layer = self.residual2


class PatchMerging3D:
    """
    Patch Merging layer for 3D downsampling.
    
    Merges 2x2x2 neighboring patches into 1 patch:
    - Input: D*H*W patches with dim C
    - Output: (D/2)*(H/2)*(W/2) patches with dim 2C
    
    Implementation:
    - Downsample by taking every other element along D, H, W
    - Concatenate 8 downsampled versions -> 8C channels
    - LayerNorm
    - Linear projection 8C -> 2C
    
    Note: This simplification merges all 8 neighbors. Standard implementation
    may only merge spatial (2x2), keeping temporal unchanged.
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        dim: int,
        D: int,
        H: int,
        W: int,
        batch_size: int = 1,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.layers = []
        overrides = layer_mode_overrides or {}
        
        def get_mode(name):
            return overrides.get(name, enclave_mode)
        
        # After merging: (D/2)*(H/2)*(W/2) tokens with 8*dim channels
        # Note: Standard Video Swin often only merges spatial (4*dim), not temporal
        # For simplicity, we merge all dimensions (8*dim -> 2*dim)
        # Adjust as needed based on actual architecture requirements
        
        # For now, use standard spatial-only merging (4*dim -> 2*dim)
        num_tokens = batch_size * D * (H // 2) * (W // 2)
        
        # LayerNorm on concatenated features
        self.norm = SecretLayerNormLayer(
            sid, f"{name_prefix}_norm", get_mode(f"{name_prefix}_norm"),
            normalized_shape=[4 * dim],  # 4*dim for spatial 2x2 merging
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.norm)
        
        # Linear reduction: 4C -> 2C
        self.reduction = SGXLinearBase(
            sid, f"{name_prefix}_reduction", get_mode(f"{name_prefix}_reduction"),
            batch_size=num_tokens,
            n_output_features=2 * dim,
            n_input_features=4 * dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.reduction)
        
        self.input_layer = self.norm
        self.output_layer = self.reduction


class SGXVideoSwinTransformerTiny:
    """
    Video Swin Transformer Tiny for SGX inference with Standard 3D Architecture.
    
    Architecture (Video Swin-Tiny):
    - Input: (B, 3, D, H, W) = (1, 3, 8, 224, 224)
    - Patch Embed: Conv3D (2x4x4), stride (2x4x4) -> (B, 96, 4, 56, 56)
    - Stage 0: 2 blocks, C=96, heads=3, 4x56x56 (W-MSA, SW-MSA)
    - Patch Merge -> (B, 192, 4, 28, 28)
    - Stage 1: 2 blocks, C=192, heads=6, 4x28x28
    - Patch Merge -> (B, 384, 4, 14, 14)
    - Stage 2: 6 blocks, C=384, heads=12, 4x14x14
    - Patch Merge -> (B, 768, 4, 7, 7)
    - Stage 3: 2 blocks, C=768, heads=24, 4x7x7
    - Classifier: LayerNorm -> AvgPool -> Linear
    
    Key Features:
    - 3D Shifted Window Attention (2x7x7 windows)
    - Relative Position Bias (3D)
    - Hierarchical feature extraction
    - LOCAL attention (bounded by window size)
    """
    
    def __init__(
        self,
        sid: int = 0,
        num_classes: int = 400,  # Kinetics-400
        enclave_mode: ExecutionModeOptions = ExecutionModeOptions.Enclave,
        batch_size: int = 1,
        video_frames: int = 8,
        image_size: int = 224,
        patch_size=(2, 4, 4),  # (Pd, Ph, Pw)
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size=(2, 7, 7),  # (Wd, Wh, Ww)
        mlp_ratio: float = 4.0,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.sid = sid
        self.num_classes = num_classes
        self.enclave_mode = enclave_mode
        self.batch_size = batch_size
        self.video_frames = video_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.layer_mode_overrides = layer_mode_overrides or {}
        
        self.layers: List = []
        self.model_name = "Video-Swin-Tiny"
        
        self._build_network()
    
    def _get_mode(self, name: str) -> ExecutionModeOptions:
        return self.layer_mode_overrides.get(name, self.enclave_mode)
    
    def _build_network(self):
        """Build complete Video Swin Transformer network."""
        sid = self.sid
        
        # Initial dimensions after patch embedding
        Pd, Ph, Pw = self.patch_size
        D = self.video_frames // Pd  # e.g., 8 // 2 = 4
        H = self.image_size // Ph     # e.g., 224 // 4 = 56
        W = self.image_size // Pw     # e.g., 224 // 4 = 56
        
        # ========== Patch Embedding (3D) ==========
        patch_embed_conv = SGXConv3DBase(
            sid, "patch_embed_conv3d", self._get_mode("patch_embed_conv3d"),
            batch_size=self.batch_size,
            n_input_channel=3,
            n_output_channel=self.embed_dim,
            filter_dhw=self.patch_size,
            video_dhw=(self.video_frames, self.image_size, self.image_size),
            stride=self.patch_size,
            padding=(0, 0, 0),
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(patch_embed_conv)
        
        # LayerNorm after patch embedding
        patch_embed_norm = SecretLayerNormLayer(
            sid, "patch_embed_norm", self._get_mode("patch_embed_norm"),
            normalized_shape=[self.embed_dim],
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(patch_embed_norm)
        
        # ========== Stages ==========
        dim = self.embed_dim
        
        for i_stage in range(len(self.depths)):
            stage_name = f"stage{i_stage}"
            depth = self.depths[i_stage]
            heads = self.num_heads[i_stage]
            
            # Swin Transformer blocks (alternating W-MSA and SW-MSA)
            for i_block in range(depth):
                # Calculate shift size
                if i_block % 2 == 0:
                    shift_size = (0, 0, 0)  # W-MSA
                else:
                    # SW-MSA: shift by half window
                    Wd, Wh, Ww = self.window_size
                    shift_size = (Wd // 2, Wh // 2, Ww // 2)
                
                block = SwinTransformerBlock3D(
                    sid, f"{stage_name}_block{i_block}",
                    self.enclave_mode,
                    dim=dim,
                    num_heads=heads,
                    window_size=self.window_size,
                    shift_size=shift_size,
                    mlp_ratio=self.mlp_ratio,
                    D=D, H=H, W=W,
                    batch_size=self.batch_size,
                    layer_mode_overrides=self.layer_mode_overrides
                )
                self.layers.extend(block.layers)
            
            # Patch Merging (except for last stage)
            if i_stage < len(self.depths) - 1:
                merge = PatchMerging3D(
                    sid, f"{stage_name}_merge",
                    self.enclave_mode,
                    dim=dim,
                    D=D, H=H, W=W,
                    batch_size=self.batch_size,
                    layer_mode_overrides=self.layer_mode_overrides
                )
                self.layers.extend(merge.layers)
                
                # Update dimensions (spatial only for standard Video Swin)
                H = H // 2
                W = W // 2
                dim = dim * 2
                # Note: D typically stays the same in Video Swin
        
        # ========== Classification Head ==========
        # Final LayerNorm
        final_norm = SecretLayerNormLayer(
            sid, "final_norm", self._get_mode("final_norm"),
            normalized_shape=[dim],
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(final_norm)
        
        # Global average pooling happens in forward pass (average over D*H*W)
        # Then classifier
        classifier = SGXLinearBase(
            sid, "classifier", self._get_mode("classifier"),
            batch_size=self.batch_size,
            n_output_features=self.num_classes,
            n_input_features=dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(classifier)
        
        # Output layer
        output_layer = SecretOutputLayer(
            sid, "output", self._get_mode("output"),
            manually_register_prev=True
        )
        self.layers.append(output_layer)
    
    def get_all_layers(self) -> List:
        """Return all layers in the model."""
        return self.layers
    
    def get_layer_by_name(self, name: str):
        """Find a layer by its name."""
        for layer in self.layers:
            if hasattr(layer, 'LayerName') and layer.LayerName == name:
                return layer
        return None
    
    def get_model_info(self) -> Dict:
        """Return model configuration and statistics."""
        layer_counts = {}
        
        for layer in self.layers:
            layer_type = type(layer).__name__
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "video_frames": self.video_frames,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depths": self.depths,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "mlp_ratio": self.mlp_ratio,
            "total_layers": len(self.layers),
            "layer_type_counts": layer_counts,
        }
    
    def print_architecture(self):
        """Print the model architecture."""
        info = self.get_model_info()
        print(f"\n{'='*70}")
        print(f"Model: {info['model_name']}")
        print(f"{'='*70}")
        print(f"Configuration:")
        print(f"  - Video frames: {info['video_frames']}")
        print(f"  - Image size: {info['image_size']}")
        print(f"  - Patch size: {info['patch_size']} (3D)")
        print(f"  - Embed dim: {info['embed_dim']}")
        print(f"  - Depths: {info['depths']}")
        print(f"  - Num heads: {info['num_heads']}")
        print(f"  - Window size: {info['window_size']} (3D - Temporal + Spatial)")
        print(f"  - MLP ratio: {info['mlp_ratio']}")
        print(f"\nLayer counts:")
        for layer_type, count in sorted(info['layer_type_counts'].items()):
            print(f"  - {layer_type}: {count}")
        print(f"\nTotal layers: {info['total_layers']}")
        print(f"Note: Uses 3D Shifted Window Attention for efficient video modeling")
        print(f"{'='*70}\n")


def create_video_swin_tiny(num_classes=400, **kwargs):
    """Create Video Swin-Tiny model."""
    return SGXVideoSwinTransformerTiny(
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(2, 7, 7),
        **kwargs
    )


def create_video_swin_small(num_classes=400, **kwargs):
    """Create Video Swin-Small model."""
    return SGXVideoSwinTransformerTiny(
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(2, 7, 7),
        **kwargs
    )


if __name__ == "__main__":
    print("Creating Video Swin Transformer 3D model...")
    
    model = create_video_swin_tiny(
        num_classes=400,
        enclave_mode=ExecutionModeOptions.CPU,
        video_frames=8,
        batch_size=1
    )
    model.print_architecture()
    
    print("\n" + "="*70)
    print("✓ Video Swin Transformer 3D (Standard Architecture) Ready")
    print("  - 3D Convolution for patch embedding")
    print("  - 3D Window Partition and Reverse")
    print("  - Cyclic Shift for SW-MSA")
    print("  - Relative Position Bias (3D)")
    print("  - Hierarchical structure with Patch Merging")
    print("="*70)
