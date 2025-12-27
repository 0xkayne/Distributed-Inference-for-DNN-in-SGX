"""
SGX Vision Transformer (ViT) Model - Native Implementation.

ViT-Base Architecture:
- Patch Embedding: Conv2d (16x16 kernel, stride 16) 
- CLS Token + Position Embedding
- 12 Transformer Blocks, each containing:
  - LayerNorm -> Multi-Head Self-Attention -> Residual
  - LayerNorm -> FFN (MLP) -> Residual
- Classification Head: LayerNorm -> Linear

Reference: "An Image is Worth 16x16 Words" (Dosovitskiy et al., ICLR 2021)

This implementation provides layer-by-layer execution for profiling and
supports flexible CPU/Enclave partitioning for TEE optimization.
"""

import sys
import math
sys.path.insert(0, '.')

from typing import Dict, List, Optional, Tuple

from python.layers.sgx_conv_base import SGXConvBase
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.layer_norm import SecretLayerNormLayer
from python.layers.gelu import SecretGELULayer
from python.layers.softmax import SecretSoftmaxLayer
from python.layers.matmul import SecretMatMulLayer
from python.layers.scale import SecretScaleLayer
from python.layers.reshape import SecretReshapeLayer
from python.layers.add import SecretAddLayer
from python.layers.flatten import SecretFlattenLayer
from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer
from python.utils.basic_utils import ExecutionModeOptions


class MultiHeadSelfAttention:
    """
    Multi-Head Self-Attention module.
    
    Structure:
    - Input: (B, N, embed_dim) where N = num_patches + 1 (CLS token)
    - QKV projection: Linear(embed_dim, 3*embed_dim) -> reshape to 3 tensors
    - Attention: Q @ K^T / sqrt(d_k) -> Softmax -> @ V
    - Output projection: Linear(embed_dim, embed_dim)
    
    For SGX, we decompose this into individual operations for profiling.
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        embed_dim: int,
        num_heads: int,
        batch_size: int = 1,
        seq_len: int = 197,  # 196 patches + 1 CLS token
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.layers = []
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_size = batch_size
        self.seq_len = seq_len
        overrides = layer_mode_overrides or {}
        
        def get_mode(name):
            return overrides.get(name, enclave_mode)
        
        # QKV Projection: (B, N, embed_dim) -> (B, N, 3*embed_dim)
        self.qkv_proj = SGXLinearBase(
            sid, f"{name_prefix}_qkv_proj", get_mode(f"{name_prefix}_qkv_proj"),
            n_output_channel=3 * embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.qkv_proj)
        
        # Reshape QKV: (B, N, 3*embed_dim) -> (3, B, num_heads, N, head_dim)
        # We'll split this into Q, K, V reshapes for clarity
        self.reshape_q = SecretReshapeLayer(
            sid, f"{name_prefix}_reshape_q", get_mode(f"{name_prefix}_reshape_q"),
            target_shape=[batch_size, seq_len, num_heads, self.head_dim],
            permute_dims=[0, 2, 1, 3],  # (B, H, N, D)
            mode='view_permute',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.reshape_q)
        
        self.reshape_k = SecretReshapeLayer(
            sid, f"{name_prefix}_reshape_k", get_mode(f"{name_prefix}_reshape_k"),
            target_shape=[batch_size, seq_len, num_heads, self.head_dim],
            permute_dims=[0, 2, 1, 3],  # (B, H, N, D)
            mode='view_permute',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.reshape_k)
        
        self.reshape_v = SecretReshapeLayer(
            sid, f"{name_prefix}_reshape_v", get_mode(f"{name_prefix}_reshape_v"),
            target_shape=[batch_size, seq_len, num_heads, self.head_dim],
            permute_dims=[0, 2, 1, 3],  # (B, H, N, D)
            mode='view_permute',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.reshape_v)
        
        # Q @ K^T: (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
        self.qk_matmul = SecretMatMulLayer(
            sid, f"{name_prefix}_qk_matmul", get_mode(f"{name_prefix}_qk_matmul"),
            transpose_b=True,  # Transpose K
            scale=1.0 / math.sqrt(self.head_dim),  # Scale by 1/sqrt(d_k)
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.qk_matmul)
        
        # Softmax: (B, H, N, N) -> (B, H, N, N)
        self.attn_softmax = SecretSoftmaxLayer(
            sid, f"{name_prefix}_attn_softmax", get_mode(f"{name_prefix}_attn_softmax"),
            dim=-1,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.attn_softmax)
        
        # Attention @ V: (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)
        self.attn_v_matmul = SecretMatMulLayer(
            sid, f"{name_prefix}_attn_v_matmul", get_mode(f"{name_prefix}_attn_v_matmul"),
            transpose_b=False,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.attn_v_matmul)
        
        # Reshape back: (B, H, N, D) -> (B, N, embed_dim)
        self.reshape_concat = SecretReshapeLayer(
            sid, f"{name_prefix}_reshape_concat", get_mode(f"{name_prefix}_reshape_concat"),
            target_shape=[batch_size, num_heads, seq_len, self.head_dim],
            permute_dims=[0, 2, 1, 3],  # (B, N, H, D)
            mode='view_permute',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.reshape_concat)
        
        # Flatten heads: (B, N, H, D) -> (B, N, embed_dim)
        self.flatten_heads = SecretReshapeLayer(
            sid, f"{name_prefix}_flatten_heads", get_mode(f"{name_prefix}_flatten_heads"),
            target_shape=[batch_size, seq_len, embed_dim],
            mode='view',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.flatten_heads)
        
        # Output projection: (B, N, embed_dim) -> (B, N, embed_dim)
        self.out_proj = SGXLinearBase(
            sid, f"{name_prefix}_out_proj", get_mode(f"{name_prefix}_out_proj"),
            n_output_channel=embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.out_proj)
        
        # Store key layers for connection
        self.input_layer = self.qkv_proj
        self.output_layer = self.out_proj
    
    def connect(self, prev_layer):
        """Connect attention module to previous layer."""
        # QKV projection connects to input
        self.qkv_proj.register_prev_layer(prev_layer)
        
        # Note: In a real implementation, we'd split the QKV output into Q, K, V
        # For now, we simulate this with separate reshape layers
        # This is a simplified representation for profiling purposes
        
        self.reshape_q.register_prev_layer(self.qkv_proj)
        self.reshape_k.register_prev_layer(self.qkv_proj)  # Same input
        self.reshape_v.register_prev_layer(self.qkv_proj)  # Same input
        
        # Q @ K^T
        self.qk_matmul.register_prev_layer(self.reshape_q)
        self.qk_matmul.register_prev_layer(self.reshape_k)
        
        # Softmax
        self.attn_softmax.register_prev_layer(self.qk_matmul)
        
        # Attention @ V
        self.attn_v_matmul.register_prev_layer(self.attn_softmax)
        self.attn_v_matmul.register_prev_layer(self.reshape_v)
        
        # Reshape back
        self.reshape_concat.register_prev_layer(self.attn_v_matmul)
        self.flatten_heads.register_prev_layer(self.reshape_concat)
        
        # Output projection
        self.out_proj.register_prev_layer(self.flatten_heads)
        
        return self.out_proj


class FFN:
    """
    Feed-Forward Network (MLP) module.
    
    Structure:
    - Linear(embed_dim, 4*embed_dim)
    - GELU activation
    - Linear(4*embed_dim, embed_dim)
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.layers = []
        self.hidden_dim = int(embed_dim * mlp_ratio)
        overrides = layer_mode_overrides or {}
        
        def get_mode(name):
            return overrides.get(name, enclave_mode)
        
        # FC1: (B, N, embed_dim) -> (B, N, hidden_dim)
        self.fc1 = SGXLinearBase(
            sid, f"{name_prefix}_fc1", get_mode(f"{name_prefix}_fc1"),
            n_output_channel=self.hidden_dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.fc1)
        
        # GELU activation
        self.gelu = SecretGELULayer(
            sid, f"{name_prefix}_gelu", get_mode(f"{name_prefix}_gelu"),
            approximate=True,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.gelu)
        
        # FC2: (B, N, hidden_dim) -> (B, N, embed_dim)
        self.fc2 = SGXLinearBase(
            sid, f"{name_prefix}_fc2", get_mode(f"{name_prefix}_fc2"),
            n_output_channel=embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.fc2)
        
        self.input_layer = self.fc1
        self.output_layer = self.fc2
    
    def connect(self, prev_layer):
        """Connect FFN to previous layer."""
        self.fc1.register_prev_layer(prev_layer)
        self.gelu.register_prev_layer(self.fc1)
        self.fc2.register_prev_layer(self.gelu)
        return self.fc2


class TransformerBlock:
    """
    Single Transformer block (encoder layer).
    
    Structure:
    - norm1 -> attn -> residual1 (add with input)
    - norm2 -> ffn -> residual2 (add with residual1)
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        batch_size: int = 1,
        seq_len: int = 197,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.layers = []
        overrides = layer_mode_overrides or {}
        
        def get_mode(name):
            return overrides.get(name, enclave_mode)
        
        # Pre-norm 1 (before attention)
        self.norm1 = SecretLayerNormLayer(
            sid, f"{name_prefix}_norm1", get_mode(f"{name_prefix}_norm1"),
            normalized_shape=[embed_dim],
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.norm1)
        
        # Multi-Head Self-Attention
        self.attn = MultiHeadSelfAttention(
            sid, f"{name_prefix}_attn", enclave_mode,
            embed_dim=embed_dim, num_heads=num_heads,
            batch_size=batch_size, seq_len=seq_len,
            layer_mode_overrides=overrides
        )
        self.layers.extend(self.attn.layers)
        
        # Residual 1 (input + attn output)
        self.residual1 = SecretAddLayer(
            sid, f"{name_prefix}_residual1", get_mode(f"{name_prefix}_residual1"),
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.residual1)
        
        # Pre-norm 2 (before FFN)
        self.norm2 = SecretLayerNormLayer(
            sid, f"{name_prefix}_norm2", get_mode(f"{name_prefix}_norm2"),
            normalized_shape=[embed_dim],
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.norm2)
        
        # FFN
        self.ffn = FFN(
            sid, f"{name_prefix}_ffn", enclave_mode,
            embed_dim=embed_dim, mlp_ratio=mlp_ratio,
            layer_mode_overrides=overrides
        )
        self.layers.extend(self.ffn.layers)
        
        # Residual 2 (residual1 + ffn output)
        self.residual2 = SecretAddLayer(
            sid, f"{name_prefix}_residual2", get_mode(f"{name_prefix}_residual2"),
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.residual2)
        
        self.input_layer = self.norm1
        self.output_layer = self.residual2
    
    def connect(self, prev_layer):
        """Connect transformer block to previous layer."""
        # Norm1 -> Attention
        self.norm1.register_prev_layer(prev_layer)
        self.attn.connect(self.norm1)
        
        # Residual1: input + attention output
        self.residual1.register_prev_layer(prev_layer)  # Skip connection
        self.residual1.register_prev_layer(self.attn.output_layer)
        
        # Norm2 -> FFN
        self.norm2.register_prev_layer(self.residual1)
        self.ffn.connect(self.norm2)
        
        # Residual2: residual1 + ffn output
        self.residual2.register_prev_layer(self.residual1)  # Skip connection
        self.residual2.register_prev_layer(self.ffn.output_layer)
        
        return self.residual2


class SGXViTBase:
    """
    Vision Transformer Base model for SGX inference.
    
    Architecture (ViT-Base):
    - Patch Embedding: Conv2d(3, 768, kernel=16, stride=16)
    - Position Embedding: Added to patches (learnable)
    - CLS Token: Prepended to sequence
    - 12 Transformer Blocks
    - Classification Head: LayerNorm -> Linear
    
    Configurations:
    - ViT-Tiny: embed_dim=192, num_heads=3, num_layers=12
    - ViT-Small: embed_dim=384, num_heads=6, num_layers=12  
    - ViT-Base: embed_dim=768, num_heads=12, num_layers=12
    - ViT-Large: embed_dim=1024, num_heads=16, num_layers=24
    """
    
    def __init__(
        self,
        sid: int = 0,
        num_classes: int = 1000,
        enclave_mode: ExecutionModeOptions = ExecutionModeOptions.Enclave,
        batch_size: int = 1,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.sid = sid
        self.num_classes = num_classes
        self.enclave_mode = enclave_mode
        self.batch_size = batch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.layer_mode_overrides = layer_mode_overrides or {}
        
        # Computed values
        self.num_patches = (img_size // patch_size) ** 2  # 196 for 224/16
        self.seq_len = self.num_patches + 1  # +1 for CLS token
        
        self.layers: List = []
        self.model_name = f"ViT-{embed_dim}"
        
        self._build_network()
    
    def _get_mode(self, name: str) -> ExecutionModeOptions:
        return self.layer_mode_overrides.get(name, self.enclave_mode)
    
    def _build_network(self):
        """Build complete ViT network."""
        sid = self.sid
        
        # ========== Input Layer ==========
        input_layer = SecretInputLayer(
            sid, "input",
            [self.batch_size, 3, self.img_size, self.img_size],
            self._get_mode("input"),
            manually_register_next=True
        )
        self.layers.append(input_layer)
        
        # ========== Patch Embedding ==========
        # Conv2d: (B, 3, H, W) -> (B, embed_dim, H/16, W/16)
        patch_embed = SGXConvBase(
            sid, "patch_embed", self._get_mode("patch_embed"),
            n_output_channel=self.embed_dim,
            n_input_channel=3,
            filter_hw=self.patch_size,
            stride=self.patch_size,
            padding=0,
            batch_size=self.batch_size,
            img_hw=self.img_size,
            manually_register_prev=True, manually_register_next=True
        )
        patch_embed.register_prev_layer(input_layer)
        self.layers.append(patch_embed)
        
        # Flatten: (B, embed_dim, H/16, W/16) -> (B, num_patches, embed_dim)
        flatten_patches = SecretFlattenLayer(
            sid, "flatten_patches", self._get_mode("flatten_patches"),
            manually_register_prev=True, manually_register_next=True
        )
        flatten_patches.register_prev_layer(patch_embed)
        self.layers.append(flatten_patches)
        
        # Note: CLS token and position embedding are handled via parameter injection
        # For profiling, we treat them as part of the input preparation
        
        current_output = flatten_patches
        
        # ========== Transformer Blocks ==========
        for i in range(self.num_layers):
            block = TransformerBlock(
                sid, f"block{i}",
                self.enclave_mode,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                layer_mode_overrides=self.layer_mode_overrides
            )
            current_output = block.connect(current_output)
            self.layers.extend(block.layers)
        
        # ========== Classification Head ==========
        # Final LayerNorm
        head_norm = SecretLayerNormLayer(
            sid, "head_norm", self._get_mode("head_norm"),
            normalized_shape=[self.embed_dim],
            manually_register_prev=True, manually_register_next=True
        )
        head_norm.register_prev_layer(current_output)
        self.layers.append(head_norm)
        
        # Note: We only use the CLS token for classification
        # This would require a slice operation, simplified here
        
        # Classification Linear: (B, embed_dim) -> (B, num_classes)
        classifier = SGXLinearBase(
            sid, "classifier", self._get_mode("classifier"),
            n_output_channel=self.num_classes,
            manually_register_prev=True, manually_register_next=True
        )
        classifier.register_prev_layer(head_norm)
        self.layers.append(classifier)
        
        # Output layer
        output_layer = SecretOutputLayer(
            sid, "output", self._get_mode("output"),
            manually_register_prev=True
        )
        output_layer.register_prev_layer(classifier)
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
        total_params = 0
        layer_counts = {}
        
        for layer in self.layers:
            layer_type = type(layer).__name__
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "num_patches": self.num_patches,
            "seq_len": self.seq_len,
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
        print(f"  - Embedding dim: {info['embed_dim']}")
        print(f"  - Num heads: {info['num_heads']}")
        print(f"  - Num layers: {info['num_layers']}")
        print(f"  - Sequence length: {info['seq_len']}")
        print(f"  - MLP ratio: {info['mlp_ratio']}")
        print(f"\nLayer counts:")
        for layer_type, count in sorted(info['layer_type_counts'].items()):
            print(f"  - {layer_type}: {count}")
        print(f"\nTotal layers: {info['total_layers']}")
        print(f"{'='*60}\n")


def create_vit_tiny(num_classes=1000, **kwargs):
    """Create ViT-Tiny model."""
    return SGXViTBase(
        num_classes=num_classes,
        embed_dim=192,
        num_heads=3,
        num_layers=12,
        **kwargs
    )


def create_vit_small(num_classes=1000, **kwargs):
    """Create ViT-Small model."""
    return SGXViTBase(
        num_classes=num_classes,
        embed_dim=384,
        num_heads=6,
        num_layers=12,
        **kwargs
    )


def create_vit_base(num_classes=1000, **kwargs):
    """Create ViT-Base model."""
    return SGXViTBase(
        num_classes=num_classes,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        **kwargs
    )


def create_vit_large(num_classes=1000, **kwargs):
    """Create ViT-Large model."""
    return SGXViTBase(
        num_classes=num_classes,
        embed_dim=1024,
        num_heads=16,
        num_layers=24,
        **kwargs
    )


if __name__ == "__main__":
    # Test model creation
    print("Creating ViT models...")
    
    for name, factory in [
        ("ViT-Tiny", create_vit_tiny),
        ("ViT-Small", create_vit_small),
        ("ViT-Base", create_vit_base),
    ]:
        print(f"\n{name}:")
        model = factory(num_classes=10, enclave_mode=ExecutionModeOptions.CPU)
        model.print_architecture()


