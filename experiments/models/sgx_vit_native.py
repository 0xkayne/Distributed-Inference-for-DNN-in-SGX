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
from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer
from python.layers.cls_token import SecretCLSTokenLayer
from python.layers.position_embedding import SecretPositionEmbeddingLayer
from python.layers.slice import SecretSliceLayer
from python.utils.basic_utils import ExecutionModeOptions
from python.layers.attention import create_multi_head_attention


# Note: MultiHeadSelfAttention class has been replaced by the unified
# create_multi_head_attention factory from python.layers.attention


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
        batch_size: int,
        seq_len: int,
        mlp_ratio: float = 4.0,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.layers = []
        self.hidden_dim = int(embed_dim * mlp_ratio)
        self.tokens = batch_size * seq_len
        overrides = layer_mode_overrides or {}
        
        def get_mode(name):
            return overrides.get(name, enclave_mode)
        
        # FC1: (tokens, embed_dim) -> (tokens, hidden_dim)
        self.fc1 = SGXLinearBase(
            sid, f"{name_prefix}_fc1", get_mode(f"{name_prefix}_fc1"),
            batch_size=self.tokens,
            n_output_features=self.hidden_dim,
            n_input_features=embed_dim,
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
        
        # FC2: (tokens, hidden_dim) -> (tokens, embed_dim)
        self.fc2 = SGXLinearBase(
            sid, f"{name_prefix}_fc2", get_mode(f"{name_prefix}_fc2"),
            batch_size=self.tokens,
            n_output_features=embed_dim,
            n_input_features=self.hidden_dim,
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
        use_per_head_attention: bool = False,
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
        
        # Multi-Head Self-Attention (using unified factory)
        self.attn = create_multi_head_attention(
            sid=sid,
            name_prefix=f"{name_prefix}_attn",
            enclave_mode=enclave_mode,
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_size=batch_size,
            seq_len=seq_len,
            per_head_mode=use_per_head_attention,
            layer_mode_overrides=overrides
        )
        self.layers.extend(self.attn.get_all_layers())
        
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
            batch_size=batch_size, seq_len=seq_len,
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
        attn_output = self.attn.connect(self.norm1)
        
        # Residual1: input + attention output
        self.residual1.register_prev_layer(prev_layer)  # Skip connection
        self.residual1.register_prev_layer(attn_output)
        
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
        use_per_head_attention: bool = False,
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
        self.use_per_head_attention = use_per_head_attention
        
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

        # Tokenize patches:
        # conv output: (B, embed_dim, H', W') where H'=W'=img_size/patch_size
        # tokens:      (B, num_patches, embed_dim)
        num_patches_hw = self.img_size // self.patch_size
        num_patches = num_patches_hw * num_patches_hw

        reshape_tokens_3d = SecretReshapeLayer(
            sid, "reshape_tokens_3d", self._get_mode("reshape_tokens_3d"),
            target_shape=[self.batch_size, self.embed_dim, -1],
            permute_dims=[0, 2, 1],  # (B, num_patches, embed_dim)
            mode='view_permute',
            manually_register_prev=True, manually_register_next=True
        )
        reshape_tokens_3d.register_prev_layer(patch_embed)
        self.layers.append(reshape_tokens_3d)

        # ========== CLS Token ==========
        # Prepend learnable CLS token to patch tokens
        cls_token_layer = SecretCLSTokenLayer(
            sid, "cls_token", self._get_mode("cls_token"),
            embed_dim=self.embed_dim,
            batch_size=self.batch_size,
            manually_register_prev=True, manually_register_next=True
        )
        cls_token_layer.register_prev_layer(reshape_tokens_3d)
        self.layers.append(cls_token_layer)
        
        # Update seq_len to include CLS token
        self.seq_len = num_patches + 1
        
        # Reshape to 2D for subsequent layers (B, N+1, D) -> (B*(N+1), D)
        reshape_with_cls = SecretReshapeLayer(
            sid, "reshape_with_cls", self._get_mode("reshape_with_cls"),
            target_shape=[self.batch_size * self.seq_len, self.embed_dim],
            mode='view',
            manually_register_prev=True, manually_register_next=True
        )
        reshape_with_cls.register_prev_layer(cls_token_layer)
        self.layers.append(reshape_with_cls)
        
        # ========== Position Embedding ==========
        # Add learnable position embeddings
        # First reshape back to 3D for position embedding
        reshape_for_pos = SecretReshapeLayer(
            sid, "reshape_for_pos", self._get_mode("reshape_for_pos"),
            target_shape=[self.batch_size, self.seq_len, self.embed_dim],
            mode='view',
            manually_register_prev=True, manually_register_next=True
        )
        reshape_for_pos.register_prev_layer(reshape_with_cls)
        self.layers.append(reshape_for_pos)
        
        pos_embed_layer = SecretPositionEmbeddingLayer(
            sid, "pos_embed", self._get_mode("pos_embed"),
            seq_len=self.seq_len,
            embed_dim=self.embed_dim,
            batch_size=self.batch_size,
            manually_register_prev=True, manually_register_next=True
        )
        pos_embed_layer.register_prev_layer(reshape_for_pos)
        self.layers.append(pos_embed_layer)
        
        # Reshape back to 2D for transformer blocks
        reshape_tokens_2d = SecretReshapeLayer(
            sid, "reshape_tokens_2d", self._get_mode("reshape_tokens_2d"),
            target_shape=[self.batch_size * self.seq_len, self.embed_dim],
            mode='view',
            manually_register_prev=True, manually_register_next=True
        )
        reshape_tokens_2d.register_prev_layer(pos_embed_layer)
        self.layers.append(reshape_tokens_2d)

        current_output = reshape_tokens_2d
        
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
                layer_mode_overrides=self.layer_mode_overrides,
                use_per_head_attention=self.use_per_head_attention
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
        
        # Reshape back to 3D to extract CLS token
        reshape_3d_for_cls = SecretReshapeLayer(
            sid, "reshape_3d_for_cls", self._get_mode("reshape_3d_for_cls"),
            target_shape=[self.batch_size, self.seq_len, self.embed_dim],
            mode='view',
            manually_register_prev=True, manually_register_next=True
        )
        reshape_3d_for_cls.register_prev_layer(head_norm)
        self.layers.append(reshape_3d_for_cls)
        
        # Extract CLS token (index 0) for classification
        slice_cls = SecretSliceLayer(
            sid, "slice_cls", self._get_mode("slice_cls"),
            index=0, dim=1,
            manually_register_prev=True, manually_register_next=True
        )
        slice_cls.register_prev_layer(reshape_3d_for_cls)
        self.layers.append(slice_cls)
        
        # Classification Linear: (B, embed_dim) -> (B, num_classes)
        # Now only classifies the CLS token, not all tokens
        classifier = SGXLinearBase(
            sid, "classifier", self._get_mode("classifier"),
            batch_size=self.batch_size,  # Only batch dimension, not batch*seq
            n_output_features=self.num_classes,
            n_input_features=self.embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        classifier.register_prev_layer(slice_cls)
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
    
    def inject_params_from_pytorch(self, pytorch_vit_model):
        """
        Inject parameters from a PyTorch Vision Transformer model.
        
        This method loads weights from a pretrained PyTorch ViT model into
        the SGX-compatible layer structure.
        
        Args:
            pytorch_vit_model: PyTorch ViT model (or compatible model with
                              patch_embed, cls_token, pos_embed, blocks, etc.)
        
        Usage:
            # Load pretrained PyTorch ViT
            from torchvision.models import vit_b_16
            pytorch_model = vit_b_16(pretrained=True)
            
            # Inject into SGX model
            sgx_model = SGXViTBase(...)
            sgx_model.inject_params_from_pytorch(pytorch_model)
        """
        print("Injecting parameters from PyTorch ViT model...")
        
        # Inject CLS token
        cls_token_layer = self.get_layer_by_name("cls_token")
        if cls_token_layer and hasattr(cls_token_layer, 'inject_params_from_pytorch'):
            print("  - Injecting CLS token...")
            cls_token_layer.inject_params_from_pytorch(pytorch_vit_model)
        
        # Inject position embeddings
        pos_embed_layer = self.get_layer_by_name("pos_embed")
        if pos_embed_layer and hasattr(pos_embed_layer, 'inject_params_from_pytorch'):
            print("  - Injecting position embeddings...")
            pos_embed_layer.inject_params_from_pytorch(pytorch_vit_model)
        
        # Inject patch embedding (Conv2d)
        patch_embed_layer = self.get_layer_by_name("patch_embed")
        if patch_embed_layer and hasattr(pytorch_vit_model, 'patch_embed'):
            print("  - Injecting patch embedding...")
            if hasattr(pytorch_vit_model.patch_embed, 'proj'):
                # Standard ViT structure
                patch_embed_layer.inject_params_from_pytorch(pytorch_vit_model.patch_embed.proj)
            else:
                patch_embed_layer.inject_params_from_pytorch(pytorch_vit_model.patch_embed)
        
        # Inject transformer blocks
        if hasattr(pytorch_vit_model, 'blocks') or hasattr(pytorch_vit_model, 'encoder'):
            blocks = pytorch_vit_model.blocks if hasattr(pytorch_vit_model, 'blocks') else pytorch_vit_model.encoder.layers
            print(f"  - Injecting {len(blocks)} transformer blocks...")
            
            for i, pytorch_block in enumerate(blocks):
                # Each block contains multiple layers (norm, attn, ffn, etc.)
                # We need to match them with our SGX layers
                block_prefix = f"block{i}"
                
                # Inject attention Q/K/V projections
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                    layer_name = f"{block_prefix}_attn_{proj_name}"
                    layer = self.get_layer_by_name(layer_name)
                    if layer and hasattr(pytorch_block, 'attn'):
                        # Extract the corresponding weight from PyTorch attention
                        if proj_name == 'out_proj' and hasattr(pytorch_block.attn, 'proj'):
                            layer.inject_params_from_pytorch(pytorch_block.attn.proj)
                        elif hasattr(pytorch_block.attn, proj_name):
                            layer.inject_params_from_pytorch(getattr(pytorch_block.attn, proj_name))
                
                # Inject FFN layers
                if hasattr(pytorch_block, 'mlp'):
                    fc1_layer = self.get_layer_by_name(f"{block_prefix}_ffn_fc1")
                    if fc1_layer and hasattr(pytorch_block.mlp, 'fc1'):
                        fc1_layer.inject_params_from_pytorch(pytorch_block.mlp.fc1)
                    
                    fc2_layer = self.get_layer_by_name(f"{block_prefix}_ffn_fc2")
                    if fc2_layer and hasattr(pytorch_block.mlp, 'fc2'):
                        fc2_layer.inject_params_from_pytorch(pytorch_block.mlp.fc2)
                
                # Inject LayerNorms
                for norm_idx in [1, 2]:
                    norm_layer = self.get_layer_by_name(f"{block_prefix}_norm{norm_idx}")
                    if norm_layer and hasattr(pytorch_block, f'norm{norm_idx}'):
                        norm_layer.inject_params_from_pytorch(getattr(pytorch_block, f'norm{norm_idx}'))
        
        # Inject final norm and classifier
        head_norm = self.get_layer_by_name("head_norm")
        if head_norm and hasattr(pytorch_vit_model, 'norm'):
            print("  - Injecting final LayerNorm...")
            head_norm.inject_params_from_pytorch(pytorch_vit_model.norm)
        
        classifier_layer = self.get_layer_by_name("classifier")
        if classifier_layer and hasattr(pytorch_vit_model, 'head'):
            print("  - Injecting classifier head...")
            classifier_layer.inject_params_from_pytorch(pytorch_vit_model.head)
        
        print("Parameter injection complete!")


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



