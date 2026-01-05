"""
ALBERT implementation for SGX distributed inference.

ALBERT (A Lite BERT) is a parameter-efficient version of BERT that uses:
1. Cross-layer parameter sharing - All Transformer layers share the same parameters
2. Factorized embedding parameterization - Separate embedding dim (E) from hidden dim (H)
3. Sentence Order Prediction (SOP) instead of NSP

Architecture Variants:
1. ALBERT-base: 12 layers, H=768, E=128, 12 heads, ~12M params
2. ALBERT-large: 24 layers, H=1024, E=128, 16 heads, ~18M params
3. ALBERT-xlarge: 24 layers, H=2048, E=128, 16 heads, ~60M params
4. ALBERT-xxlarge: 12 layers, H=4096, E=128, 64 heads, ~235M params

Key Insight: Due to parameter sharing, ALBERT has significantly fewer unique
parameters than BERT, even with more layers. This is beneficial for TEE
execution where memory is limited.

Reference: "ALBERT: A Lite BERT for Self-supervised Learning of Language
            Representations" (Lan et al., ICLR 2020)
GitHub: https://github.com/google-research/albert
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
class ALBERTConfig:
    """Configuration for ALBERT model."""
    vocab_size: int = 30000          # Vocabulary size (SentencePiece)
    max_seq_len: int = 512           # Maximum sequence length
    embedding_dim: int = 128         # Embedding dimension E (factorized!)
    hidden_dim: int = 768            # Hidden dimension H
    num_heads: int = 12              # Number of attention heads
    num_layers: int = 12             # Number of Transformer layers (shared!)
    intermediate_size: int = 3072    # FFN intermediate size
    dropout: float = 0.1             # Dropout probability
    attention_dropout: float = 0.1   # Attention dropout
    layer_norm_eps: float = 1e-12    # LayerNorm epsilon
    num_classes: int = 2             # For classification tasks
    
    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


@dataclass
class ALBERTBaseConfig(ALBERTConfig):
    """ALBERT-base: 12 layers, H=768, E=128, 12 heads."""
    embedding_dim: int = 128
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    intermediate_size: int = 3072


@dataclass
class ALBERTLargeConfig(ALBERTConfig):
    """ALBERT-large: 24 layers, H=1024, E=128, 16 heads."""
    embedding_dim: int = 128
    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    intermediate_size: int = 4096


@dataclass
class ALBERTXLargeConfig(ALBERTConfig):
    """ALBERT-xlarge: 24 layers, H=2048, E=128, 16 heads."""
    embedding_dim: int = 128
    hidden_dim: int = 2048
    num_heads: int = 16
    num_layers: int = 24
    intermediate_size: int = 8192


# ==============================================================================
# Core Components
# ==============================================================================

class ALBERTEmbedding(nn.Module):
    """
    ALBERT Embedding layer with factorized parameterization.
    
    Key difference from BERT: Uses a smaller embedding dimension E and
    projects to hidden dimension H. This reduces parameters from V*H to V*E + E*H.
    
    For ALBERT-base: V*H = 30000*768 = 23M params
                     V*E + E*H = 30000*128 + 128*768 = 3.8M + 0.1M = 3.9M params
    
    Output: (B, seq_len, hidden_dim)
    """
    
    def __init__(self, config: ALBERTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings (smaller dimension E)
        self.token_embed = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # Position embeddings (also use E, then project)
        self.position_embed = nn.Embedding(config.max_seq_len, config.embedding_dim)
        
        # Segment embeddings (token type)
        self.segment_embed = nn.Embedding(2, config.embedding_dim)
        
        # Projection from E to H (the factorization!)
        self.embedding_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        
        # LayerNorm and Dropout
        self.layer_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
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
            Embeddings (B, seq_len, hidden_dim)
        """
        B, seq_len = input_ids.shape
        
        # Token embeddings (in E-dimensional space)
        token_embeds = self.token_embed(input_ids)
        
        # Position embeddings
        position_ids = self.position_ids[:, :seq_len]
        position_embeds = self.position_embed(position_ids)
        
        # Segment embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        segment_embeds = self.segment_embed(token_type_ids)
        
        # Combine in E-dimensional space
        embeddings = token_embeds + position_embeds + segment_embeds
        
        # Project from E to H (factorized embedding!)
        embeddings = self.embedding_projection(embeddings)
        
        # LayerNorm and Dropout in H-dimensional space
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class ALBERTAttention(nn.Module):
    """
    Multi-Head Self-Attention for ALBERT.
    
    Same as BERT attention, but parameters are shared across all layers.
    """
    
    def __init__(self, config: ALBERTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim
        self.scale = self.head_dim ** -0.5
        
        # Separate Q, K, V projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input (B, seq_len, hidden_dim)
            attention_mask: Optional mask (B, 1, 1, seq_len)
        
        Returns:
            Output (B, seq_len, hidden_dim)
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


class ALBERTFFN(nn.Module):
    """
    ALBERT Feed-Forward Network.
    
    Structure: Linear(H -> intermediate) -> GELU -> Linear(intermediate -> H)
    Parameters are shared across all layers.
    """
    
    def __init__(self, config: ALBERTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ALBERTSharedLayer(nn.Module):
    """
    ALBERT Shared Encoder Layer.
    
    This is the single layer that gets reused for all Transformer layers.
    Structure (post-norm style like BERT):
        x = LayerNorm(x + Attention(x))
        x = LayerNorm(x + FFN(x))
    """
    
    def __init__(self, config: ALBERTConfig):
        super().__init__()
        self.config = config
        
        # Attention
        self.attention = ALBERTAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # FFN
        self.ffn = ALBERTFFN(config)
        self.norm2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
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


class SGXALBERTModel(nn.Module):
    """
    ALBERT model designed for SGX distributed inference.
    
    Key features:
    - Parameter sharing: ONE shared layer reused N times
    - Factorized embeddings: E-dimensional embeddings projected to H
    - Much fewer unique parameters than BERT
    
    This is particularly efficient for TEE execution because:
    - Lower memory footprint due to parameter sharing
    - Embedding projection can be done outside TEE
    """
    
    def __init__(self, config: Optional[ALBERTConfig] = None):
        super().__init__()
        
        self.config = config or ALBERTBaseConfig()
        
        # Embeddings with factorization
        self.embeddings = ALBERTEmbedding(self.config)
        
        # SINGLE shared encoder layer (reused for all layers!)
        self.shared_layer = ALBERTSharedLayer(self.config)
        
        # Pooler (for CLS token)
        self.pooler = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.pooler_activation = nn.Tanh()
        
        # Classifier
        self.classifier = nn.Linear(self.config.hidden_dim, self.config.num_classes)
        
        # Layer registry for profiling
        self.layers = []
        self._build_layer_registry()
    
    def _build_layer_registry(self):
        """Build layer registry for profiling and distributed execution."""
        self.layers = []
        
        # Embedding layer (includes projection)
        self.layers.append({
            'name': 'embeddings',
            'module': self.embeddings,
            'type': 'FactorizedEmbedding',
            'dependencies': ['input'],
            'input_shape': (1, self.config.max_seq_len),
            'output_shape': (1, self.config.max_seq_len, self.config.hidden_dim),
            'note': f'E={self.config.embedding_dim} -> H={self.config.hidden_dim}'
        })
        
        # Encoder layers (all share the same parameters!)
        prev_layer = 'embeddings'
        for i in range(self.config.num_layers):
            layer_name = f'encoder_{i}'
            self.layers.append({
                'name': layer_name,
                'module': self.shared_layer,  # Same module!
                'type': 'SharedEncoderLayer',
                'dependencies': [prev_layer],
                'input_shape': (1, self.config.max_seq_len, self.config.hidden_dim),
                'output_shape': (1, self.config.max_seq_len, self.config.hidden_dim),
                'shared': True
            })
            prev_layer = layer_name
        
        # Pooler
        self.layers.append({
            'name': 'pooler',
            'module': self.pooler,
            'type': 'Linear+Tanh',
            'dependencies': [prev_layer],
            'input_shape': (1, self.config.hidden_dim),
            'output_shape': (1, self.config.hidden_dim)
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
        
        # Embeddings (with projection from E to H)
        hidden_states = self.embeddings(input_ids, token_type_ids)
        
        # Encoder layers (reuse same shared_layer!)
        for _ in range(self.config.num_layers):
            hidden_states = self.shared_layer(hidden_states, attention_mask)
        
        # Get CLS token and apply pooler
        cls_token = hidden_states[:, 0]  # (B, hidden_dim)
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
        
        # Encoder layers (reuse shared_layer)
        for i in range(self.config.num_layers):
            start = time.perf_counter()
            hidden_states = self.shared_layer(hidden_states, attention_mask)
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
        E = self.config.embedding_dim
        H = self.config.hidden_dim
        I = self.config.intermediate_size
        num_heads = self.config.num_heads
        
        # Unique parameters (key benefit of ALBERT!)
        embedding_params = sum(p.numel() for p in self.embeddings.parameters()) * 4
        shared_layer_params = sum(p.numel() for p in self.shared_layer.parameters()) * 4
        pooler_params = sum(p.numel() for p in self.pooler.parameters()) * 4
        classifier_params = sum(p.numel() for p in self.classifier.parameters()) * 4
        
        return {
            'config': {
                'batch_size': batch_size,
                'seq_length': seq_len,
                'embedding_dim': E,
                'hidden_dim': H,
                'num_heads': num_heads,
                'num_layers': self.config.num_layers,
                'parameter_sharing': True
            },
            'activations': {
                'embeddings_output': batch_size * seq_len * H * 4,
                'attention_qkv': batch_size * seq_len * H * 3 * 4,
                'attention_matrix': batch_size * num_heads * seq_len * seq_len * 4,
                'ffn_hidden': batch_size * seq_len * I * 4,
            },
            'parameters': {
                'embeddings': embedding_params,
                'shared_layer': shared_layer_params,
                'pooler': pooler_params,
                'classifier': classifier_params,
                'total_unique': embedding_params + shared_layer_params + pooler_params + classifier_params,
                # Compare to BERT-like model without sharing
                'bert_equivalent': embedding_params + shared_layer_params * self.config.num_layers + pooler_params + classifier_params
            }
        }


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_albert_base(num_classes: int = 2) -> SGXALBERTModel:
    """Create ALBERT-base model."""
    config = ALBERTBaseConfig(num_classes=num_classes)
    return SGXALBERTModel(config)


def create_albert_large(num_classes: int = 2) -> SGXALBERTModel:
    """Create ALBERT-large model."""
    config = ALBERTLargeConfig(num_classes=num_classes)
    return SGXALBERTModel(config)


# ==============================================================================
# Quick Test
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SGX ALBERT Model - Architecture Analysis")
    print("=" * 70)
    
    # Create ALBERT-base
    model = create_albert_base(num_classes=2)
    
    print(f"\n[ALBERT-base Configuration]")
    print(f"  - Vocab Size: {model.config.vocab_size}")
    print(f"  - Max Seq Len: {model.config.max_seq_len}")
    print(f"  - Embedding Dim E: {model.config.embedding_dim} (factorized!)")
    print(f"  - Hidden Dim H: {model.config.hidden_dim}")
    print(f"  - Num Heads: {model.config.num_heads}")
    print(f"  - Num Layers: {model.config.num_layers} (shared parameters!)")
    print(f"  - Intermediate Size: {model.config.intermediate_size}")
    
    # Memory footprint
    print(f"\n[Memory Footprint (seq_len=128)]")
    mem = model.get_memory_footprint(batch_size=1, seq_len=128)
    print(f"  - Total UNIQUE Parameters: {mem['parameters']['total_unique'] / 1024 / 1024:.2f} MB")
    print(f"  - BERT-equivalent (no sharing): {mem['parameters']['bert_equivalent'] / 1024 / 1024:.2f} MB")
    print(f"  - Parameter Reduction: {(1 - mem['parameters']['total_unique'] / mem['parameters']['bert_equivalent']) * 100:.1f}%")
    print(f"  - Shared Layer: {mem['parameters']['shared_layer'] / 1024 / 1024:.2f} MB (reused {model.config.num_layers}x)")
    print(f"  - Attention Matrix: {mem['activations']['attention_matrix'] / 1024:.2f} KB")
    
    # Test forward pass
    print(f"\n[Forward Pass Test]")
    batch_size = 1
    seq_len = 128
    
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    timings = {}
    def record_timing(name, ms):
        timings[name] = ms
    
    with torch.no_grad():
        output = model.forward_layer_by_layer(
            input_ids, attention_mask, layer_timing_callback=record_timing
        )
    
    print(f"  - Output shape: {output.shape}")
    print(f"  - Total time: {sum(timings.values()):.2f} ms")
    print(f"\n  Layer timings:")
    for name, ms in sorted(timings.items(), key=lambda x: -x[1]):
        print(f"    {name}: {ms:.2f} ms")
    
    print("\n" + "=" * 70)
    print("✓ SGX ALBERT Model ready for distributed inference experiments")
    print("  Key advantage: Cross-layer parameter sharing reduces memory!")
    print("=" * 70)
