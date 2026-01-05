"""
SGX Swin Transformer Model - Native Implementation.

Swin Transformer Architecture (from ICCV 2021):
- Hierarchical structure with 4 stages
- Shifted Window Attention (W-MSA and SW-MSA alternating)
- Patch Merging for downsampling
- Window size: 7x7 (default)

Swin-Tiny Configuration:
- embed_dim: 96
- depths: [2, 2, 6, 2]
- num_heads: [3, 6, 12, 24]
- window_size: 7
- ~29M parameters

Swin Block Structure:
- LN -> W-MSA/SW-MSA -> Residual -> LN -> MLP -> Residual

Note: Swin Transformer uses LOCAL attention within windows, making it
more memory-efficient than ViT's global attention. This is beneficial
for TEE execution as attention matrix size is bounded by window_size^2.

Reference: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
           (Liu et al., ICCV 2021)
GitHub: https://github.com/microsoft/Swin-Transformer

This implementation provides layer-by-layer execution for profiling.
"""

import sys
import math
sys.path.insert(0, '.')

from typing import Dict, List, Optional, Tuple

from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.sgx_conv_base import SGXConvBase
from python.layers.layer_norm import SecretLayerNormLayer
from python.layers.gelu import SecretGELULayer
from python.layers.softmax import SecretSoftmaxLayer
from python.layers.matmul import SecretMatMulLayer
from python.layers.add import SecretAddLayer
from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer
from python.utils.basic_utils import ExecutionModeOptions


class WindowAttention:
    """
    Window-based Multi-Head Self-Attention module for Swin Transformer.
    
    Computes attention within local MxM windows.
    Window size is typically 7x7, so attention matrix is 49x49.
    
    Structure:
    - QKV projection: Linear(dim, 3*dim)
    - Scaled dot-product attention with relative position bias
    - Output projection: Linear(dim, dim)
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        batch_size: int = 1,
        num_windows: int = 64,  # (H/M) * (W/M) where M=window_size
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.layers = []
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.window_tokens = window_size * window_size  # M*M tokens per window
        self.num_windows = num_windows
        self.batch_size = batch_size
        overrides = layer_mode_overrides or {}
        
        def get_mode(name):
            return overrides.get(name, enclave_mode)
        
        # Total tokens = num_windows * batch_size * window_tokens
        total_window_batches = num_windows * batch_size
        
        # QKV projection (combined for efficiency)
        self.qkv_proj = SGXLinearBase(
            sid, f"{name_prefix}_qkv_proj", get_mode(f"{name_prefix}_qkv_proj"),
            batch_size=total_window_batches * self.window_tokens,
            n_output_features=dim * 3,
            n_input_features=dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.qkv_proj)
        
        # For profiling, we measure the attention computation
        # Q @ K^T with scaling
        self.qk_matmul = SecretMatMulLayer(
            sid, f"{name_prefix}_qk_matmul", get_mode(f"{name_prefix}_qk_matmul"),
            transpose_b=True,
            scale=1.0 / math.sqrt(self.head_dim),
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.qk_matmul)
        
        # Softmax
        self.attn_softmax = SecretSoftmaxLayer(
            sid, f"{name_prefix}_attn_softmax", get_mode(f"{name_prefix}_attn_softmax"),
            dim=-1,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.attn_softmax)
        
        # Attention @ V
        self.attn_v_matmul = SecretMatMulLayer(
            sid, f"{name_prefix}_attn_v_matmul", get_mode(f"{name_prefix}_attn_v_matmul"),
            transpose_b=False,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.attn_v_matmul)
        
        # Output projection
        self.out_proj = SGXLinearBase(
            sid, f"{name_prefix}_out_proj", get_mode(f"{name_prefix}_out_proj"),
            batch_size=total_window_batches * self.window_tokens,
            n_output_features=dim,
            n_input_features=dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.out_proj)
        
        self.input_layer = self.qkv_proj
        self.output_layer = self.out_proj


class SwinMLP:
    """
    MLP module for Swin Transformer block.
    
    Structure: Linear(dim, 4*dim) -> GELU -> Linear(4*dim, dim)
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        dim: int,
        mlp_ratio: float = 4.0,
        num_tokens: int = 3136,  # H*W tokens
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


class SwinTransformerBlock:
    """
    Single Swin Transformer block.
    
    Structure (pre-norm):
    - LN1 -> W-MSA/SW-MSA -> Residual
    - LN2 -> MLP -> Residual
    
    W-MSA: Regular window attention (shift_size=0)
    SW-MSA: Shifted window attention (shift_size=window_size//2)
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        H: int = 56,
        W: int = 56,
        batch_size: int = 1,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.layers = []
        overrides = layer_mode_overrides or {}
        num_tokens = batch_size * H * W
        num_windows = (H // window_size) * (W // window_size)
        
        def get_mode(name):
            return overrides.get(name, enclave_mode)
        
        # LayerNorm 1
        self.norm1 = SecretLayerNormLayer(
            sid, f"{name_prefix}_norm1", get_mode(f"{name_prefix}_norm1"),
            normalized_shape=[dim],
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.norm1)
        
        # Window Attention (W-MSA or SW-MSA)
        self.attn = WindowAttention(
            sid, f"{name_prefix}_attn", enclave_mode,
            dim=dim, num_heads=num_heads,
            window_size=window_size, batch_size=batch_size,
            num_windows=num_windows,
            layer_mode_overrides=overrides
        )
        self.layers.extend(self.attn.layers)
        
        # Residual 1
        self.residual1 = SecretAddLayer(
            sid, f"{name_prefix}_residual1", get_mode(f"{name_prefix}_residual1"),
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.residual1)
        
        # LayerNorm 2
        self.norm2 = SecretLayerNormLayer(
            sid, f"{name_prefix}_norm2", get_mode(f"{name_prefix}_norm2"),
            normalized_shape=[dim],
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.norm2)
        
        # MLP
        self.mlp = SwinMLP(
            sid, f"{name_prefix}_mlp", enclave_mode,
            dim=dim, mlp_ratio=mlp_ratio, num_tokens=num_tokens,
            layer_mode_overrides=overrides
        )
        self.layers.extend(self.mlp.layers)
        
        # Residual 2
        self.residual2 = SecretAddLayer(
            sid, f"{name_prefix}_residual2", get_mode(f"{name_prefix}_residual2"),
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.residual2)
        
        self.shift_size = shift_size
        self.input_layer = self.norm1
        self.output_layer = self.residual2


class PatchMerging:
    """
    Patch Merging layer for downsampling.
    
    Merges 2x2 neighboring patches into 1 patch:
    - Input: H*W patches with dim C
    - Output: (H/2)*(W/2) patches with dim 2C
    
    Implementation:
    - Reshape and concat 4 patches -> 4C channels
    - LayerNorm
    - Linear projection 4C -> 2C
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        dim: int,
        H: int,
        W: int,
        batch_size: int = 1,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.layers = []
        overrides = layer_mode_overrides or {}
        
        def get_mode(name):
            return overrides.get(name, enclave_mode)
        
        # After merging: (H/2)*(W/2) tokens with 4*dim channels
        num_tokens = batch_size * (H // 2) * (W // 2)
        
        # LayerNorm on concatenated features
        self.norm = SecretLayerNormLayer(
            sid, f"{name_prefix}_norm", get_mode(f"{name_prefix}_norm"),
            normalized_shape=[4 * dim],
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


class SGXSwinTransformerTiny:
    """
    Swin Transformer Tiny for SGX inference.
    
    Architecture (Swin-Tiny):
    - Patch Embed: Conv 4x4, stride 4 -> 56x56, C=96
    - Stage 0: 2 blocks, C=96, heads=3, 56x56
    - Stage 1: 2 blocks, C=192, heads=6, 28x28 (after Patch Merge)
    - Stage 2: 6 blocks, C=384, heads=12, 14x14 (after Patch Merge)
    - Stage 3: 2 blocks, C=768, heads=24, 7x7 (after Patch Merge)
    - Classifier: LayerNorm -> AvgPool -> Linear
    
    Key Insight for TEE:
    - Local attention within 7x7 windows (49 tokens)
    - Attention matrix bounded: 49x49 regardless of image size
    - Much more memory efficient than ViT's global attention
    """
    
    def __init__(
        self,
        sid: int = 0,
        num_classes: int = 1000,
        enclave_mode: ExecutionModeOptions = ExecutionModeOptions.Enclave,
        batch_size: int = 1,
        image_size: int = 224,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.sid = sid
        self.num_classes = num_classes
        self.enclave_mode = enclave_mode
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.layer_mode_overrides = layer_mode_overrides or {}
        
        self.layers: List = []
        self.model_name = "Swin-Tiny"
        
        self._build_network()
    
    def _get_mode(self, name: str) -> ExecutionModeOptions:
        return self.layer_mode_overrides.get(name, self.enclave_mode)
    
    def _build_network(self):
        """Build complete Swin Transformer network."""
        sid = self.sid
        
        # Initial spatial dimensions after patch embedding
        H = self.image_size // self.patch_size  # 56
        W = self.image_size // self.patch_size  # 56
        
        # ========== Patch Embedding ==========
        # Conv2d: 3 -> embed_dim, kernel=patch_size, stride=patch_size
        patch_embed_conv = SGXConvBase(
            sid, "patch_embed_conv", self._get_mode("patch_embed_conv"),
            batch_size=self.batch_size,
            n_input_channel=3,
            n_output_channel=self.embed_dim,
            filter_hw=self.patch_size,
            img_hw=self.image_size,
            stride=self.patch_size,
            padding=0,
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
                shift_size = 0 if (i_block % 2 == 0) else self.window_size // 2
                block_type = "W-MSA" if shift_size == 0 else "SW-MSA"
                
                block = SwinTransformerBlock(
                    sid, f"{stage_name}_block{i_block}",
                    self.enclave_mode,
                    dim=dim,
                    num_heads=heads,
                    window_size=self.window_size,
                    shift_size=shift_size,
                    mlp_ratio=self.mlp_ratio,
                    H=H, W=W,
                    batch_size=self.batch_size,
                    layer_mode_overrides=self.layer_mode_overrides
                )
                self.layers.extend(block.layers)
            
            # Patch Merging (except for last stage)
            if i_stage < len(self.depths) - 1:
                merge = PatchMerging(
                    sid, f"{stage_name}_merge",
                    self.enclave_mode,
                    dim=dim,
                    H=H, W=W,
                    batch_size=self.batch_size,
                    layer_mode_overrides=self.layer_mode_overrides
                )
                self.layers.extend(merge.layers)
                
                # Update dimensions
                H = H // 2
                W = W // 2
                dim = dim * 2
        
        # ========== Classification Head ==========
        # Final LayerNorm
        final_norm = SecretLayerNormLayer(
            sid, "final_norm", self._get_mode("final_norm"),
            normalized_shape=[dim],
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(final_norm)
        
        # Global average pooling is done in forward pass
        # Classifier: Linear(dim, num_classes)
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
        print(f"\n{'='*60}")
        print(f"Model: {info['model_name']}")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  - Image size: {info['image_size']}")
        print(f"  - Patch size: {info['patch_size']}")
        print(f"  - Embed dim: {info['embed_dim']}")
        print(f"  - Depths: {info['depths']}")
        print(f"  - Num heads: {info['num_heads']}")
        print(f"  - Window size: {info['window_size']} (LOCAL attention!)")
        print(f"  - MLP ratio: {info['mlp_ratio']}")
        print(f"\nLayer counts:")
        for layer_type, count in sorted(info['layer_type_counts'].items()):
            print(f"  - {layer_type}: {count}")
        print(f"\nTotal layers: {info['total_layers']}")
        print(f"Note: Swin uses shifted window attention (memory efficient)")
        print(f"{'='*60}\n")


def create_swin_tiny(num_classes=1000, **kwargs):
    """Create Swin-Tiny model."""
    return SGXSwinTransformerTiny(
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        **kwargs
    )


def create_swin_small(num_classes=1000, **kwargs):
    """Create Swin-Small model."""
    return SGXSwinTransformerTiny(
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        **kwargs
    )


if __name__ == "__main__":
    print("Creating Swin Transformer model...")
    
    model = create_swin_tiny(num_classes=1000, enclave_mode=ExecutionModeOptions.CPU)
    model.print_architecture()
