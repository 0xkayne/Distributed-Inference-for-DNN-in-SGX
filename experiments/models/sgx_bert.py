"""
BERT-base implementation for SGX distributed inference.

This module provides a modular BERT implementation that can be partitioned
across Enclave and CPU execution environments, following the same patterns
as sgx_vit.py and sgx_inception.py.

BERT-base Architecture:
- Token Embedding: vocab_size -> embed_dim (768)
- Segment Embedding: 2 -> embed_dim
- Position Embedding: max_seq_len -> embed_dim
- 12 Transformer Encoder blocks
- Each block: LayerNorm -> MHSA -> Residual -> LayerNorm -> FFN -> Residual
- Pooler: Linear (for classification tasks)

Key Differences from ViT:
- Text input (token IDs) instead of image patches
- Embedding layer instead of patch embedding
- Segment embeddings for sentence pair tasks
- Typically longer sequences (up to 512 tokens)

Reference: "BERT: Pre-training of Deep Bidirectional Transformers for
            Language Understanding" (Devlin et al., NAACL 2019)
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
class BERTConfig:
    """Configuration for BERT model."""
    vocab_size: int = 30522          # BERT vocabulary size
    max_seq_len: int = 512           # Maximum sequence length
    embed_dim: int = 768             # Hidden size
    num_heads: int = 12              # Number of attention heads
    num_layers: int = 12             # Number of transformer layers
    intermediate_size: int = 3072    # FFN intermediate size (4 * embed_dim)
    num_segments: int = 2            # Number of segment types (A, B)
    dropout: float = 0.1             # Dropout probability
    layer_norm_eps: float = 1e-12    # LayerNorm epsilon
    num_classes: int = 2             # For classification tasks
    
    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


@dataclass
class BERTBaseConfig(BERTConfig):
    """BERT-base: 12 layers, 768 hidden, 12 heads."""
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    intermediate_size: int = 3072


@dataclass
class BERTLargeConfig(BERTConfig):
    """BERT-large: 24 layers, 1024 hidden, 16 heads."""
    embed_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    intermediate_size: int = 4096


@dataclass
class BERTMiniConfig(BERTConfig):
    """BERT-mini: 4 layers, 256 hidden, 4 heads (for testing)."""
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    intermediate_size: int = 1024
    max_seq_len: int = 128


# ==============================================================================
# Core Components
# ==============================================================================

class BERTEmbedding(nn.Module):
    """
    BERT Embedding layer.
    
    Combines:
    - Token embeddings (word embeddings)
    - Segment embeddings (sentence A vs B)
    - Position embeddings
    
    Output: (B, seq_len, embed_dim)
    """
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # Segment embeddings (for sentence A/B)
        self.segment_embed = nn.Embedding(config.num_segments, config.embed_dim)
        
        # Position embeddings (learnable)
        self.position_embed = nn.Embedding(config.max_seq_len, config.embed_dim)
        
        # LayerNorm and Dropout
        self.layer_norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        # Register position IDs buffer
        self.register_buffer(
            'position_ids',
            torch.arange(config.max_seq_len).unsqueeze(0)
        )
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (B, seq_len)
            segment_ids: Segment IDs (B, seq_len), defaults to zeros
        
        Returns:
            Embeddings (B, seq_len, embed_dim)
        """
        B, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embed(input_ids)
        
        # Segment embeddings
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        segment_embeds = self.segment_embed(segment_ids)
        
        # Position embeddings
        position_ids = self.position_ids[:, :seq_len]
        position_embeds = self.position_embed(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + segment_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BERTAttention(nn.Module):
    """
    Multi-Head Self-Attention for BERT.
    
    Same as ViT attention but with BERT-specific conventions:
    - Separate Q, K, V projections (instead of fused QKV)
    - Attention mask support for padding
    """
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.embed_dim = config.embed_dim
        self.scale = self.head_dim ** -0.5
        
        # Separate Q, K, V projections
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input (B, seq_len, embed_dim)
            attention_mask: Optional mask (B, 1, 1, seq_len) or (B, 1, seq_len, seq_len)
        
        Returns:
            Output (B, seq_len, embed_dim)
        """
        B, N, D = x.shape
        
        # Q, K, V projections
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask (for padding)
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out


class BERTFFN(nn.Module):
    """
    BERT Feed-Forward Network.
    
    Structure: Linear(D -> 4D) -> GELU -> Linear(4D -> D)
    """
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class BERTEncoderLayer(nn.Module):
    """
    Single BERT Encoder layer.
    
    Structure (post-norm, like original BERT):
        x = LayerNorm(x + Attention(x))
        x = LayerNorm(x + FFN(x))
    """
    
    def __init__(self, config: BERTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention = BERTAttention(config)
        self.ffn = BERTFFN(config)
        
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(x, attention_mask)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class BERTPooler(nn.Module):
    """
    BERT Pooler for classification tasks.
    
    Takes the [CLS] token representation and applies a dense layer + tanh.
    """
    
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.dense = nn.Linear(config.embed_dim, config.embed_dim)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Take the [CLS] token (first token)
        cls_token = hidden_states[:, 0]
        pooled = self.dense(cls_token)
        pooled = self.activation(pooled)
        return pooled


class SGXBERTModel(nn.Module):
    """
    BERT model designed for SGX distributed inference.
    
    This class mirrors the structure of SGXVisionTransformer
    to integrate with the existing distributed inference framework.
    """
    
    def __init__(self, config: Optional[BERTConfig] = None):
        super().__init__()
        
        self.config = config or BERTBaseConfig()
        
        # Embeddings
        self.embeddings = BERTEmbedding(self.config)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            BERTEncoderLayer(self.config, i)
            for i in range(self.config.num_layers)
        ])
        
        # Pooler (for classification)
        self.pooler = BERTPooler(self.config)
        
        # Classification head
        self.classifier = nn.Linear(self.config.embed_dim, self.config.num_classes)
        
        # Layer registry for profiling
        self.layers = []
        self._build_layer_registry()
    
    def _build_layer_registry(self):
        """Build layer registry for profiling and distributed execution."""
        self.layers = []
        
        # Embedding layer
        self.layers.append({
            'name': 'embeddings',
            'module': self.embeddings,
            'type': 'Embedding',
            'dependencies': ['input'],
            'input_shape': (1, self.config.max_seq_len),  # token IDs
            'output_shape': (1, self.config.max_seq_len, self.config.embed_dim)
        })
        
        # Encoder layers
        prev_layer = 'embeddings'
        for i, layer in enumerate(self.encoder_layers):
            layer_name = f'encoder_{i}'
            self.layers.append({
                'name': layer_name,
                'module': layer,
                'type': 'EncoderLayer',
                'dependencies': [prev_layer],
                'input_shape': (1, self.config.max_seq_len, self.config.embed_dim),
                'output_shape': (1, self.config.max_seq_len, self.config.embed_dim)
            })
            prev_layer = layer_name
        
        # Pooler
        self.layers.append({
            'name': 'pooler',
            'module': self.pooler,
            'type': 'Pooler',
            'dependencies': [prev_layer],
            'input_shape': (1, self.config.max_seq_len, self.config.embed_dim),
            'output_shape': (1, self.config.embed_dim)
        })
        
        # Classifier
        self.layers.append({
            'name': 'classifier',
            'module': self.classifier,
            'type': 'Linear',
            'dependencies': ['pooler'],
            'output_shape': (1, self.config.num_classes)
        })
    
    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (B, seq_len)
            segment_ids: Segment IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len), 1 for valid, 0 for padding
        
        Returns:
            Classification logits (B, num_classes)
        """
        # Convert attention mask to additive mask
        if attention_mask is not None:
            # (B, seq_len) -> (B, 1, 1, seq_len)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Embeddings
        hidden_states = self.embeddings(input_ids, segment_ids)
        
        # Encoder layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Pooler and classifier
        pooled = self.pooler(hidden_states)
        logits = self.classifier(pooled)
        
        return logits
    
    def forward_layer_by_layer(
        self,
        input_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_timing_callback: Optional[Callable[[str, float], None]] = None
    ) -> torch.Tensor:
        """Execute model layer by layer with timing callback."""
        import time
        
        # Convert attention mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Embeddings
        start = time.perf_counter()
        hidden_states = self.embeddings(input_ids, segment_ids)
        if layer_timing_callback:
            layer_timing_callback('embeddings', (time.perf_counter() - start) * 1000)
        
        # Encoder layers
        for i, layer in enumerate(self.encoder_layers):
            start = time.perf_counter()
            hidden_states = layer(hidden_states, attention_mask)
            if layer_timing_callback:
                layer_timing_callback(f'encoder_{i}', (time.perf_counter() - start) * 1000)
        
        # Pooler
        start = time.perf_counter()
        pooled = self.pooler(hidden_states)
        if layer_timing_callback:
            layer_timing_callback('pooler', (time.perf_counter() - start) * 1000)
        
        # Classifier
        start = time.perf_counter()
        logits = self.classifier(pooled)
        if layer_timing_callback:
            layer_timing_callback('classifier', (time.perf_counter() - start) * 1000)
        
        return logits
    
    def get_memory_footprint(self, batch_size: int = 1, seq_len: int = 128) -> Dict[str, Any]:
        """Estimate memory footprint for TEE execution planning."""
        D = self.config.embed_dim
        H = self.config.intermediate_size
        num_heads = self.config.num_heads
        head_dim = self.config.head_dim
        
        return {
            'config': {
                'batch_size': batch_size,
                'seq_length': seq_len,
                'embed_dim': D,
                'num_heads': num_heads,
                'num_layers': self.config.num_layers
            },
            'activations': {
                'embeddings_output': batch_size * seq_len * D * 4,
                'attention_qkv': batch_size * seq_len * D * 3 * 4,
                'attention_matrix': batch_size * num_heads * seq_len * seq_len * 4,
                'ffn_hidden': batch_size * seq_len * H * 4,
            },
            'parameters': {
                'embeddings': sum(p.numel() for p in self.embeddings.parameters()) * 4,
                'per_layer': sum(p.numel() for p in self.encoder_layers[0].parameters()) * 4,
                'total_layers': sum(p.numel() for p in self.encoder_layers.parameters()) * 4,
                'pooler': sum(p.numel() for p in self.pooler.parameters()) * 4,
                'classifier': sum(p.numel() for p in self.classifier.parameters()) * 4,
                'total': sum(p.numel() for p in self.parameters()) * 4
            }
        }


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_bert_mini(num_classes: int = 2) -> SGXBERTModel:
    """Create BERT-mini for fast experiments."""
    config = BERTMiniConfig(num_classes=num_classes)
    return SGXBERTModel(config)


def create_bert_base(num_classes: int = 2) -> SGXBERTModel:
    """Create BERT-base model."""
    config = BERTBaseConfig(num_classes=num_classes)
    return SGXBERTModel(config)


def create_bert_large(num_classes: int = 2) -> SGXBERTModel:
    """Create BERT-large model."""
    config = BERTLargeConfig(num_classes=num_classes)
    return SGXBERTModel(config)


# ==============================================================================
# Quick Test
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SGX BERT Model - Architecture Analysis")
    print("=" * 70)
    
    # Create BERT-base
    model = create_bert_base(num_classes=2)
    
    print(f"\n[Model Configuration]")
    print(f"  - Vocab Size: {model.config.vocab_size}")
    print(f"  - Max Seq Len: {model.config.max_seq_len}")
    print(f"  - Embed Dim: {model.config.embed_dim}")
    print(f"  - Num Heads: {model.config.num_heads}")
    print(f"  - Num Layers: {model.config.num_layers}")
    print(f"  - Intermediate Size: {model.config.intermediate_size}")
    
    # Memory footprint
    print(f"\n[Memory Footprint (seq_len=128)]")
    mem = model.get_memory_footprint(batch_size=1, seq_len=128)
    print(f"  - Total Parameters: {mem['parameters']['total'] / 1024 / 1024:.2f} MB")
    print(f"  - Embeddings: {mem['parameters']['embeddings'] / 1024 / 1024:.2f} MB")
    print(f"  - Per Layer: {mem['parameters']['per_layer'] / 1024 / 1024:.2f} MB")
    print(f"  - Attention Matrix: {mem['activations']['attention_matrix'] / 1024:.2f} KB")
    
    # Test forward pass
    print(f"\n[Forward Pass Test]")
    batch_size = 1
    seq_len = 128
    
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    segment_ids = torch.zeros_like(input_ids)
    attention_mask = torch.ones_like(input_ids)
    
    timings = {}
    def record_timing(name, ms):
        timings[name] = ms
    
    with torch.no_grad():
        output = model.forward_layer_by_layer(
            input_ids, segment_ids, attention_mask, record_timing
        )
    
    print(f"  - Output shape: {output.shape}")
    print(f"  - Total time: {sum(timings.values()):.2f} ms")
    print(f"\n  Layer timings:")
    for name, ms in sorted(timings.items(), key=lambda x: -x[1])[:5]:
        print(f"    {name}: {ms:.2f} ms")
    
    print("\n" + "=" * 70)
    print("✓ SGX BERT Model ready for distributed inference experiments")
    print("=" * 70)
