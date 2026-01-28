"""
SGX BERT-base Model - Native Implementation.

BERT-base Architecture:
- Token Embedding + Segment Embedding + Position Embedding
- 12 Transformer Encoder blocks, each containing:
  - LayerNorm -> Multi-Head Self-Attention -> Residual
  - LayerNorm -> FFN (MLP) -> Residual
- Pooler: Linear + Tanh (for [CLS] token)
- Classifier: Linear

Reference: "BERT: Pre-training of Deep Bidirectional Transformers for
            Language Understanding" (Devlin et al., NAACL 2019)

This implementation provides layer-by-layer execution for profiling and
supports flexible CPU/Enclave partitioning for TEE optimization.
"""

import sys
import math
sys.path.insert(0, '.')

from typing import Dict, List, Optional, Tuple

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
from python.utils.basic_utils import ExecutionModeOptions
from python.layers.attention import create_multi_head_attention


# Note: MultiHeadSelfAttention class is now replaced by the unified
# create_multi_head_attention factory from python.layers.attention


class FFN:
    """
    Feed-Forward Network (MLP) module for BERT.
    
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
        
        # FC1: (tokens, embed_dim) -> (tokens, intermediate_size)
        self.fc1 = SGXLinearBase(
            sid, f"{name_prefix}_fc1", get_mode(f"{name_prefix}_fc1"),
            batch_size=self.tokens,
            n_output_features=intermediate_size,
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
        
        # FC2: (tokens, intermediate_size) -> (tokens, embed_dim)
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


class BERTEncoderBlock:
    """
    Single BERT Encoder block.
    
    Structure (post-norm like original BERT):
    - attn_out = Attention(input)
    - x = LayerNorm(input + attn_out)
    - ffn_out = FFN(x)
    - output = LayerNorm(x + ffn_out)
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
        use_per_head_attention: bool = False,
    ):
        self.layers = []
        overrides = layer_mode_overrides or {}
        
        def get_mode(name):
            return overrides.get(name, enclave_mode)
        
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
        
        # LayerNorm 1 (post-norm)
        self.norm1 = SecretLayerNormLayer(
            sid, f"{name_prefix}_norm1", get_mode(f"{name_prefix}_norm1"),
            normalized_shape=[embed_dim],
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.norm1)
        
        # FFN
        self.ffn = FFN(
            sid, f"{name_prefix}_ffn", enclave_mode,
            embed_dim=embed_dim, intermediate_size=intermediate_size,
            batch_size=batch_size, seq_len=seq_len,
            layer_mode_overrides=overrides
        )
        self.layers.extend(self.ffn.layers)
        
        # Residual 2 (norm1 + ffn output)
        self.residual2 = SecretAddLayer(
            sid, f"{name_prefix}_residual2", get_mode(f"{name_prefix}_residual2"),
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.residual2)
        
        # LayerNorm 2 (post-norm)
        self.norm2 = SecretLayerNormLayer(
            sid, f"{name_prefix}_norm2", get_mode(f"{name_prefix}_norm2"),
            normalized_shape=[embed_dim],
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.norm2)
        
        self.input_layer = self.attn.input_layer
        self.output_layer = self.norm2
    
    def connect(self, prev_layer):
        """Connect encoder block to previous layer."""
        # Attention (using unified interface)
        attn_output = self.attn.connect(prev_layer)
        
        # Residual1: input + attention output
        self.residual1.register_prev_layer(prev_layer)  # Skip connection
        self.residual1.register_prev_layer(attn_output)
        
        # LayerNorm 1
        self.norm1.register_prev_layer(self.residual1)
        
        # FFN
        self.ffn.connect(self.norm1)
        
        # Residual2: norm1 + ffn output
        self.residual2.register_prev_layer(self.norm1)  # Skip connection
        self.residual2.register_prev_layer(self.ffn.output_layer)
        
        # LayerNorm 2
        self.norm2.register_prev_layer(self.residual2)
        
        return self.norm2


class SGXBERTBase:
    """
    BERT-base model for SGX inference.
    
    Architecture (BERT-base):
    - Embedding: Token + Segment + Position (simulated as Linear for profiling)
    - 12 Transformer Encoder Blocks
    - Pooler: Linear + Tanh (simplified to Linear)
    - Classifier: Linear
    
    Configurations:
    - BERT-mini: embed_dim=256, num_heads=4, num_layers=4
    - BERT-base: embed_dim=768, num_heads=12, num_layers=12
    - BERT-large: embed_dim=1024, num_heads=16, num_layers=24
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
        num_layers: int = 12,
        intermediate_size: int = 3072,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
        use_per_head_attention: bool = False,
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
        self.use_per_head_attention = use_per_head_attention
        
        self.layers: List = []
        self.model_name = f"BERT-{embed_dim}"
        
        self._build_network()
    
    def _get_mode(self, name: str) -> ExecutionModeOptions:
        return self.layer_mode_overrides.get(name, self.enclave_mode)
    
    def _build_network(self):
        """Build complete BERT network."""
        sid = self.sid
        tokens = self.batch_size * self.seq_len
        
        # ========== Input Layer ==========
        # For profiling, we assume embeddings are already computed
        # Input shape: (batch_size * seq_len, embed_dim)
        input_layer = SecretInputLayer(
            sid, "input",
            [tokens, self.embed_dim],
            self._get_mode("input"),
            manually_register_next=True
        )
        self.layers.append(input_layer)
        
        current_output = input_layer
        
        # ========== Encoder Blocks ==========
        for i in range(self.num_layers):
            block = BERTEncoderBlock(
                sid, f"encoder{i}",
                self.enclave_mode,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                layer_mode_overrides=self.layer_mode_overrides,
                use_per_head_attention=self.use_per_head_attention
            )
            current_output = block.connect(current_output)
            self.layers.extend(block.layers)
        
        # ========== Pooler ==========
        # Take [CLS] token (first token) - simplified as processing all tokens
        # In practice, we would slice the first token, but for profiling
        # we process the full sequence and the classifier handles the shape
        
        # Pooler Linear
        pooler = SGXLinearBase(
            sid, "pooler", self._get_mode("pooler"),
            batch_size=tokens,  # Process all tokens for profiling
            n_output_features=self.embed_dim,
            n_input_features=self.embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        pooler.register_prev_layer(current_output)
        self.layers.append(pooler)
        
        # ========== Classifier ==========
        classifier = SGXLinearBase(
            sid, "classifier", self._get_mode("classifier"),
            batch_size=tokens,  # Process all tokens
            n_output_features=self.num_classes,
            n_input_features=self.embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        classifier.register_prev_layer(pooler)
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
        print(f"  - Num layers: {info['num_layers']}")
        print(f"  - Intermediate size: {info['intermediate_size']}")
        print(f"  - Sequence length: {info['seq_len']}")
        print(f"\nLayer counts:")
        for layer_type, count in sorted(info['layer_type_counts'].items()):
            print(f"  - {layer_type}: {count}")
        print(f"\nTotal layers: {info['total_layers']}")
        print(f"{'='*60}\n")


def create_bert_mini(num_classes=2, **kwargs):
    """Create BERT-mini model."""
    return SGXBERTBase(
        num_classes=num_classes,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        intermediate_size=1024,
        seq_len=128,
        **kwargs
    )


def create_bert_base(num_classes=2, **kwargs):
    """Create BERT-base model."""
    return SGXBERTBase(
        num_classes=num_classes,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        intermediate_size=3072,
        seq_len=128,
        **kwargs
    )


def create_bert_large(num_classes=2, **kwargs):
    """Create BERT-large model."""
    return SGXBERTBase(
        num_classes=num_classes,
        embed_dim=1024,
        num_heads=16,
        num_layers=24,
        intermediate_size=4096,
        seq_len=128,
        **kwargs
    )


if __name__ == "__main__":
    # Test model creation
    print("Creating BERT models...")
    
    for name, factory in [
        ("BERT-mini", create_bert_mini),
        ("BERT-base", create_bert_base),
    ]:
        print(f"\n{name}:")
        model = factory(num_classes=2, enclave_mode=ExecutionModeOptions.CPU)
        model.print_architecture()
