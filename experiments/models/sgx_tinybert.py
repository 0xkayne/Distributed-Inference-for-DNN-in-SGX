"""
TinyBERT implementation for SGX distributed inference.

TinyBERT is a distilled version of BERT that is:
- 7.5x smaller in parameters
- 9.4x faster in inference
- Achieves competitive performance on NLU tasks

Architecture Variants:
1. TinyBERT-4L-312D (Tiny):
   - 4 Transformer layers
   - 312 hidden dimension
   - 1200 intermediate size
   - 12 attention heads
   
2. TinyBERT-6L-768D (Small):
   - 6 Transformer layers  
   - 768 hidden dimension
   - 3072 intermediate size
   - 12 attention heads

Key Features:
- Novel transformer distillation at both pre-training and task-specific stages
- Embedding layer distillation
- Transformer layer distillation (attention + hidden states)
- Prediction layer distillation

Reference: "TinyBERT: Distilling BERT for Natural Language Understanding"
           (Jiao et al., EMNLP 2020 Findings)
GitHub: https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT
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
class TinyBERTConfig:
    """Configuration for TinyBERT model."""
    vocab_size: int = 30522          # Same as BERT
    max_seq_len: int = 512           # Maximum sequence length
    embed_dim: int = 312             # Hidden size
    num_heads: int = 12              # Number of attention heads
    num_layers: int = 4              # Number of Transformer layers
    intermediate_size: int = 1200    # FFN intermediate size
    dropout: float = 0.1             # Dropout probability
    attention_dropout: float = 0.1   # Attention dropout
    layer_norm_eps: float = 1e-12    # LayerNorm epsilon
    num_classes: int = 2             # For classification tasks
    
    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


@dataclass
class TinyBERT4LConfig(TinyBERTConfig):
    """TinyBERT-4L-312D: 4 layers, 312 hidden, 12 heads, 1200 intermediate."""
    embed_dim: int = 312
    num_heads: int = 12
    num_layers: int = 4
    intermediate_size: int = 1200


@dataclass
class TinyBERT6LConfig(TinyBERTConfig):
    """TinyBERT-6L-768D: 6 layers, 768 hidden, 12 heads, 3072 intermediate."""
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 6
    intermediate_size: int = 3072


# ==============================================================================
# Core Components
# ==============================================================================

class TinyBERTEmbedding(nn.Module):
    """
    TinyBERT Embedding layer.
    
    Combines:
    - Token embeddings (word embeddings)
    - Position embeddings
    - Segment embeddings (token type)
    
    Output: (B, seq_len, embed_dim)
    """
    
    def __init__(self, config: TinyBERTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # Position embeddings
        self.position_embed = nn.Embedding(config.max_seq_len, config.embed_dim)
        
        # Segment embeddings (token type)
        self.segment_embed = nn.Embedding(2, config.embed_dim)
        
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
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (B, seq_len)
            token_type_ids: Segment IDs (B, seq_len), defaults to zeros
        
        Returns:
            Embeddings (B, seq_len, embed_dim)
        """
        B, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embed(input_ids)
        
        # Position embeddings
        position_ids = self.position_ids[:, :seq_len]
        position_embeds = self.position_embed(position_ids)
        
        # Segment embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        segment_embeds = self.segment_embed(token_type_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds + segment_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class TinyBERTAttention(nn.Module):
    """
    Multi-Head Self-Attention for TinyBERT.
    
    Standard BERT-style attention with separate Q, K, V projections.
    """
    
    def __init__(self, config: TinyBERTConfig):
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
        self.dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input (B, seq_len, embed_dim)
            attention_mask: Optional mask (B, 1, 1, seq_len)
        
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
        
        # Apply attention mask
        if attention_mask is not None:
            attn = attn + attention_mask
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class TinyBERTFFN(nn.Module):
    """
    TinyBERT Feed-Forward Network.
    
    Structure: Linear(D -> intermediate) -> GELU -> Linear(intermediate -> D)
    """
    
    def __init__(self, config: TinyBERTConfig):
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


class TinyBERTEncoderLayer(nn.Module):
    """
    Single TinyBERT Encoder layer.
    
    Structure (post-norm style like BERT):
        x = LayerNorm(x + Attention(x))
        x = LayerNorm(x + FFN(x))
    """
    
    def __init__(self, config: TinyBERTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Attention
        self.attention = TinyBERTAttention(config)
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        
        # FFN
        self.ffn = TinyBERTFFN(config)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual (post-norm)
        attn_out = self.attention(x, attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual (post-norm)
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class SGXTinyBERTModel(nn.Module):
    """
    TinyBERT model designed for SGX distributed inference.
    
    Key features:
    - Significantly smaller than BERT (7.5x smaller)
    - Much faster inference (9.4x faster)
    - Supports both 4-layer and 6-layer variants
    """
    
    def __init__(self, config: Optional[TinyBERTConfig] = None):
        super().__init__()
        
        self.config = config or TinyBERT4LConfig()
        
        # Embeddings
        self.embeddings = TinyBERTEmbedding(self.config)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TinyBERTEncoderLayer(self.config, i)
            for i in range(self.config.num_layers)
        ])
        
        # Pooler (for CLS token)
        self.pooler = nn.Linear(self.config.embed_dim, self.config.embed_dim)
        self.pooler_activation = nn.Tanh()
        
        # Classifier
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
            'input_shape': (1, self.config.max_seq_len),
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
            'type': 'Linear+Tanh',
            'dependencies': [prev_layer],
            'input_shape': (1, self.config.embed_dim),
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
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len), 1 for valid, 0 for padding
            token_type_ids: Segment IDs (B, seq_len)
        
        Returns:
            Classification logits (B, num_classes)
        """
        # Convert attention mask to additive mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Embeddings
        hidden_states = self.embeddings(input_ids, token_type_ids)
        
        # Encoder layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Get CLS token and apply pooler
        cls_token = hidden_states[:, 0]  # (B, embed_dim)
        pooled = self.pooler(cls_token)
        pooled = self.pooler_activation(pooled)
        
        # Classifier
        logits = self.classifier(pooled)
        
        return logits
    
    def forward_layer_by_layer(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
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
        hidden_states = self.embeddings(input_ids, token_type_ids)
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
        cls_token = hidden_states[:, 0]
        pooled = self.pooler(cls_token)
        pooled = self.pooler_activation(pooled)
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

def create_tinybert_4l(num_classes: int = 2) -> SGXTinyBERTModel:
    """Create TinyBERT-4L-312D model (smallest, fastest)."""
    config = TinyBERT4LConfig(num_classes=num_classes)
    return SGXTinyBERTModel(config)


def create_tinybert_6l(num_classes: int = 2) -> SGXTinyBERTModel:
    """Create TinyBERT-6L-768D model (same as DistilBERT size)."""
    config = TinyBERT6LConfig(num_classes=num_classes)
    return SGXTinyBERTModel(config)


# ==============================================================================
# Quick Test
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SGX TinyBERT Model - Architecture Analysis")
    print("=" * 70)
    
    # Create TinyBERT-4L (smallest)
    model_4l = create_tinybert_4l(num_classes=2)
    
    print(f"\n[TinyBERT-4L-312D Configuration]")
    print(f"  - Vocab Size: {model_4l.config.vocab_size}")
    print(f"  - Max Seq Len: {model_4l.config.max_seq_len}")
    print(f"  - Embed Dim: {model_4l.config.embed_dim}")
    print(f"  - Num Heads: {model_4l.config.num_heads}")
    print(f"  - Num Layers: {model_4l.config.num_layers} (vs 12 in BERT-base)")
    print(f"  - Intermediate Size: {model_4l.config.intermediate_size}")
    
    # Memory footprint
    print(f"\n[Memory Footprint (seq_len=128)]")
    mem = model_4l.get_memory_footprint(batch_size=1, seq_len=128)
    print(f"  - Total Parameters: {mem['parameters']['total'] / 1024 / 1024:.2f} MB")
    print(f"  - Embeddings: {mem['parameters']['embeddings'] / 1024 / 1024:.2f} MB")
    print(f"  - Per Layer: {mem['parameters']['per_layer'] / 1024 / 1024:.2f} MB")
    print(f"  - Attention Matrix: {mem['activations']['attention_matrix'] / 1024:.2f} KB")
    
    # Test forward pass
    print(f"\n[Forward Pass Test - TinyBERT-4L]")
    batch_size = 1
    seq_len = 128
    
    input_ids = torch.randint(0, model_4l.config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    timings = {}
    def record_timing(name, ms):
        timings[name] = ms
    
    with torch.no_grad():
        output = model_4l.forward_layer_by_layer(
            input_ids, attention_mask, layer_timing_callback=record_timing
        )
    
    print(f"  - Output shape: {output.shape}")
    print(f"  - Total time: {sum(timings.values()):.2f} ms")
    print(f"\n  Layer timings:")
    for name, ms in sorted(timings.items(), key=lambda x: -x[1]):
        print(f"    {name}: {ms:.2f} ms")
    
    # Compare with TinyBERT-6L
    print(f"\n{'='*70}")
    print(f"[TinyBERT-6L-768D Configuration]")
    model_6l = create_tinybert_6l(num_classes=2)
    print(f"  - Embed Dim: {model_6l.config.embed_dim}")
    print(f"  - Num Layers: {model_6l.config.num_layers}")
    print(f"  - Intermediate Size: {model_6l.config.intermediate_size}")
    mem_6l = model_6l.get_memory_footprint(batch_size=1, seq_len=128)
    print(f"  - Total Parameters: {mem_6l['parameters']['total'] / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 70)
    print("✓ SGX TinyBERT Model ready for distributed inference experiments")
    print("  TinyBERT-4L: 7.5x smaller, 9.4x faster than BERT-base")
    print("=" * 70)
