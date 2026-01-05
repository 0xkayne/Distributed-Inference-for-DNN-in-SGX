"""
SGX DistilBERT Model - Native Implementation.

DistilBERT Architecture (from NeurIPS 2019 Workshop):
- Token Embedding + Position Embedding (NO segment embedding)
- 6 Transformer Encoder blocks (half of BERT-base)
- Each block: LayerNorm -> MHSA -> Residual -> LayerNorm -> FFN -> Residual
- Pre-classifier: Linear + ReLU
- Classifier: Linear

Key Differences from BERT:
1. Only 6 layers (vs 12 in BERT-base) - 50% reduction
2. No segment/token_type embeddings
3. Pre-classifier layer with ReLU activation
4. Trained via knowledge distillation

Reference: "DistilBERT, a distilled version of BERT: smaller, faster, 
            cheaper and lighter" (Sanh et al., NeurIPS 2019 Workshop)

This implementation provides layer-by-layer execution for profiling and
supports flexible CPU/Enclave partitioning for TEE optimization.
"""

import sys
import math
sys.path.insert(0, '.')

from typing import Dict, List, Optional

from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.layer_norm import SecretLayerNormLayer
from python.layers.gelu import SecretGELULayer
from python.layers.softmax import SecretSoftmaxLayer
from python.layers.matmul import SecretMatMulLayer
from python.layers.reshape import SecretReshapeLayer
from python.layers.add import SecretAddLayer
from python.layers.relu import SecretReLULayer
from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer
from python.utils.basic_utils import ExecutionModeOptions


class MultiHeadSelfAttention:
    """
    Multi-Head Self-Attention module for DistilBERT.
    
    Same structure as BERT attention:
    - Q, K, V projections: Linear(embed_dim, embed_dim) each
    - Attention: Q @ K^T / sqrt(d_k) -> Softmax -> @ V
    - Output projection: Linear(embed_dim, embed_dim)
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        embed_dim: int,
        num_heads: int,
        batch_size: int = 1,
        seq_len: int = 128,
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
        
        # Flatten tokens for Linear layers
        self.tokens = batch_size * seq_len
        
        # Q projection
        self.q_proj = SGXLinearBase(
            sid, f"{name_prefix}_q_proj", get_mode(f"{name_prefix}_q_proj"),
            batch_size=self.tokens,
            n_output_features=embed_dim,
            n_input_features=embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        
        # K projection
        self.k_proj = SGXLinearBase(
            sid, f"{name_prefix}_k_proj", get_mode(f"{name_prefix}_k_proj"),
            batch_size=self.tokens,
            n_output_features=embed_dim,
            n_input_features=embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        
        # V projection
        self.v_proj = SGXLinearBase(
            sid, f"{name_prefix}_v_proj", get_mode(f"{name_prefix}_v_proj"),
            batch_size=self.tokens,
            n_output_features=embed_dim,
            n_input_features=embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.extend([self.q_proj, self.k_proj, self.v_proj])
        
        # Reshape Q, K, V
        self.reshape_q = SecretReshapeLayer(
            sid, f"{name_prefix}_reshape_q", get_mode(f"{name_prefix}_reshape_q"),
            target_shape=[batch_size, seq_len, num_heads, self.head_dim],
            permute_dims=[0, 2, 1, 3],
            mode='view_permute',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.reshape_q)
        
        self.reshape_k = SecretReshapeLayer(
            sid, f"{name_prefix}_reshape_k", get_mode(f"{name_prefix}_reshape_k"),
            target_shape=[batch_size, seq_len, num_heads, self.head_dim],
            permute_dims=[0, 2, 1, 3],
            mode='view_permute',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.reshape_k)
        
        self.reshape_v = SecretReshapeLayer(
            sid, f"{name_prefix}_reshape_v", get_mode(f"{name_prefix}_reshape_v"),
            target_shape=[batch_size, seq_len, num_heads, self.head_dim],
            permute_dims=[0, 2, 1, 3],
            mode='view_permute',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.reshape_v)
        
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
        
        # Reshape back
        self.reshape_concat = SecretReshapeLayer(
            sid, f"{name_prefix}_reshape_concat", get_mode(f"{name_prefix}_reshape_concat"),
            target_shape=[batch_size, num_heads, seq_len, self.head_dim],
            permute_dims=[0, 2, 1, 3],
            mode='view_permute',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.reshape_concat)
        
        # Flatten heads
        self.flatten_heads = SecretReshapeLayer(
            sid, f"{name_prefix}_flatten_heads", get_mode(f"{name_prefix}_flatten_heads"),
            target_shape=[batch_size * seq_len, embed_dim],
            mode='view',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.flatten_heads)
        
        # Output projection
        self.out_proj = SGXLinearBase(
            sid, f"{name_prefix}_out_proj", get_mode(f"{name_prefix}_out_proj"),
            batch_size=self.tokens,
            n_output_features=embed_dim,
            n_input_features=embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.out_proj)
        
        self.input_layer = self.q_proj
        self.output_layer = self.out_proj
    
    def connect(self, prev_layer):
        """Connect attention module to previous layer."""
        self.q_proj.register_prev_layer(prev_layer)
        self.k_proj.register_prev_layer(prev_layer)
        self.v_proj.register_prev_layer(prev_layer)
        
        self.reshape_q.register_prev_layer(self.q_proj)
        self.reshape_k.register_prev_layer(self.k_proj)
        self.reshape_v.register_prev_layer(self.v_proj)
        
        self.qk_matmul.register_prev_layer(self.reshape_q)
        self.qk_matmul.register_prev_layer(self.reshape_k)
        
        self.attn_softmax.register_prev_layer(self.qk_matmul)
        
        self.attn_v_matmul.register_prev_layer(self.attn_softmax)
        self.attn_v_matmul.register_prev_layer(self.reshape_v)
        
        self.reshape_concat.register_prev_layer(self.attn_v_matmul)
        self.flatten_heads.register_prev_layer(self.reshape_concat)
        
        self.out_proj.register_prev_layer(self.flatten_heads)
        
        return self.out_proj


class FFN:
    """
    Feed-Forward Network (MLP) module for DistilBERT.
    
    Structure:
    - Linear(embed_dim, intermediate_size)
    - GELU activation
    - Linear(intermediate_size, embed_dim)
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        embed_dim: int,
        intermediate_size: int,
        batch_size: int,
        seq_len: int,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.layers = []
        self.tokens = batch_size * seq_len
        overrides = layer_mode_overrides or {}
        
        def get_mode(name):
            return overrides.get(name, enclave_mode)
        
        # FC1
        self.fc1 = SGXLinearBase(
            sid, f"{name_prefix}_fc1", get_mode(f"{name_prefix}_fc1"),
            batch_size=self.tokens,
            n_output_features=intermediate_size,
            n_input_features=embed_dim,
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
            batch_size=self.tokens,
            n_output_features=embed_dim,
            n_input_features=intermediate_size,
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


class DistilBERTEncoderBlock:
    """
    Single DistilBERT Encoder block.
    
    Structure (pre-norm style):
    - norm1 -> attn -> residual1
    - norm2 -> ffn -> residual2
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        embed_dim: int,
        num_heads: int,
        intermediate_size: int,
        batch_size: int = 1,
        seq_len: int = 128,
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
        
        # Residual 1
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
            embed_dim=embed_dim, intermediate_size=intermediate_size,
            batch_size=batch_size, seq_len=seq_len,
            layer_mode_overrides=overrides
        )
        self.layers.extend(self.ffn.layers)
        
        # Residual 2
        self.residual2 = SecretAddLayer(
            sid, f"{name_prefix}_residual2", get_mode(f"{name_prefix}_residual2"),
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.residual2)
        
        self.input_layer = self.norm1
        self.output_layer = self.residual2
    
    def connect(self, prev_layer):
        """Connect encoder block to previous layer."""
        # Norm1 -> Attention
        self.norm1.register_prev_layer(prev_layer)
        self.attn.connect(self.norm1)
        
        # Residual1: input + attention output
        self.residual1.register_prev_layer(prev_layer)
        self.residual1.register_prev_layer(self.attn.output_layer)
        
        # Norm2 -> FFN
        self.norm2.register_prev_layer(self.residual1)
        self.ffn.connect(self.norm2)
        
        # Residual2: residual1 + ffn output
        self.residual2.register_prev_layer(self.residual1)
        self.residual2.register_prev_layer(self.ffn.output_layer)
        
        return self.residual2


class SGXDistilBERTBase:
    """
    DistilBERT-base model for SGX inference.
    
    Architecture:
    - Embedding: Token + Position (NO segment embedding)
    - 6 Transformer Encoder Blocks (50% fewer than BERT-base)
    - Pre-classifier: Linear + ReLU
    - Classifier: Linear
    
    Configuration (DistilBERT-base):
    - embed_dim: 768
    - num_heads: 12
    - num_layers: 6 (key difference from BERT's 12)
    - intermediate_size: 3072
    """
    
    def __init__(
        self,
        sid: int = 0,
        num_classes: int = 2,
        enclave_mode: ExecutionModeOptions = ExecutionModeOptions.Enclave,
        batch_size: int = 1,
        seq_len: int = 128,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,  # Half of BERT-base!
        intermediate_size: int = 3072,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.sid = sid
        self.num_classes = num_classes
        self.enclave_mode = enclave_mode
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.layer_mode_overrides = layer_mode_overrides or {}
        
        self.layers: List = []
        self.model_name = f"DistilBERT-{embed_dim}"
        
        self._build_network()
    
    def _get_mode(self, name: str) -> ExecutionModeOptions:
        return self.layer_mode_overrides.get(name, self.enclave_mode)
    
    def _build_network(self):
        """Build complete DistilBERT network."""
        sid = self.sid
        tokens = self.batch_size * self.seq_len
        
        # ========== Input Layer ==========
        input_layer = SecretInputLayer(
            sid, "input",
            [tokens, self.embed_dim],
            self._get_mode("input"),
            manually_register_next=True
        )
        self.layers.append(input_layer)
        
        current_output = input_layer
        
        # ========== Encoder Blocks (6 layers) ==========
        for i in range(self.num_layers):
            block = DistilBERTEncoderBlock(
                sid, f"encoder{i}",
                self.enclave_mode,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                layer_mode_overrides=self.layer_mode_overrides
            )
            current_output = block.connect(current_output)
            self.layers.extend(block.layers)
        
        # ========== Pre-classifier (DistilBERT specific) ==========
        pre_classifier = SGXLinearBase(
            sid, "pre_classifier", self._get_mode("pre_classifier"),
            batch_size=tokens,
            n_output_features=self.embed_dim,
            n_input_features=self.embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        pre_classifier.register_prev_layer(current_output)
        self.layers.append(pre_classifier)
        
        # ReLU activation (DistilBERT uses ReLU in pre-classifier, not GELU)
        pre_classifier_relu = SecretReLULayer(
            sid, "pre_classifier_relu", self._get_mode("pre_classifier_relu"),
            manually_register_prev=True, manually_register_next=True
        )
        pre_classifier_relu.register_prev_layer(pre_classifier)
        self.layers.append(pre_classifier_relu)
        
        # ========== Classifier ==========
        classifier = SGXLinearBase(
            sid, "classifier", self._get_mode("classifier"),
            batch_size=tokens,
            n_output_features=self.num_classes,
            n_input_features=self.embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        classifier.register_prev_layer(pre_classifier_relu)
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
            "intermediate_size": self.intermediate_size,
            "seq_len": self.seq_len,
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
        print(f"  - Num layers: {info['num_layers']} (50% fewer than BERT-base)")
        print(f"  - Intermediate size: {info['intermediate_size']}")
        print(f"  - Sequence length: {info['seq_len']}")
        print(f"\nLayer counts:")
        for layer_type, count in sorted(info['layer_type_counts'].items()):
            print(f"  - {layer_type}: {count}")
        print(f"\nTotal layers: {info['total_layers']}")
        print(f"{'='*60}\n")


def create_distilbert_base(num_classes=2, **kwargs):
    """Create DistilBERT-base model."""
    return SGXDistilBERTBase(
        num_classes=num_classes,
        embed_dim=768,
        num_heads=12,
        num_layers=6,  # Key difference: 6 layers instead of 12
        intermediate_size=3072,
        seq_len=128,
        **kwargs
    )


if __name__ == "__main__":
    print("Creating DistilBERT model...")
    
    model = create_distilbert_base(num_classes=2, enclave_mode=ExecutionModeOptions.CPU)
    model.print_architecture()
