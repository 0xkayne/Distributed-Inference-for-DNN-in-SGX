"""
DistilBERT implementation for SGX distributed inference.

DistilBERT is a distilled version of BERT that is:
- 40% smaller in parameters
- 60% faster in inference
- Retains 97% of BERT's language understanding

Architecture (DistilBERT-base):
- Token Embedding + Position Embedding (no segment embedding)
- 6 Transformer Encoder blocks (half of BERT-base)
- Each block: LayerNorm -> MHSA -> Residual -> LayerNorm -> FFN -> Residual
- Pre-classifier: Linear + ReLU + Dropout
- Classifier: Linear

Key Differences from BERT:
1. 6 layers instead of 12
2. No segment (token type) embeddings
3. Pre-classifier layer with ReLU activation
4. Trained using knowledge distillation from BERT

Reference: "DistilBERT, a distilled version of BERT: smaller, faster, 
            cheaper and lighter" (Sanh et al., NeurIPS 2019 Workshop)
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
class DistilBERTConfig:
    """Configuration for DistilBERT model."""
    vocab_size: int = 30522          # Same as BERT
    max_seq_len: int = 512           # Maximum sequence length
    embed_dim: int = 768             # Hidden size (same as BERT-base)
    num_heads: int = 12              # Number of attention heads
    num_layers: int = 6              # Half of BERT-base (key difference!)
    intermediate_size: int = 3072    # FFN intermediate size (4 * embed_dim)
    dropout: float = 0.1             # Dropout probability
    attention_dropout: float = 0.1   # Attention dropout
    layer_norm_eps: float = 1e-12    # LayerNorm epsilon
    num_classes: int = 2             # For classification tasks
    sinusoidal_pos_embds: bool = False  # Use learned position embeddings
    
    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


@dataclass
class DistilBERTBaseConfig(DistilBERTConfig):
    """DistilBERT-base: 6 layers, 768 hidden, 12 heads."""
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 6
    intermediate_size: int = 3072


# ==============================================================================
# Core Components
# ==============================================================================

class DistilBERTEmbedding(nn.Module):
    """
    DistilBERT Embedding layer.
    
    Unlike BERT, DistilBERT does NOT use segment embeddings.
    Only combines:
    - Token embeddings (word embeddings)
    - Position embeddings
    
    Output: (B, seq_len, embed_dim)
    """
    
    def __init__(self, config: DistilBERTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # Position embeddings (learnable, like BERT)
        self.position_embed = nn.Embedding(config.max_seq_len, config.embed_dim)
        
        # LayerNorm and Dropout
        self.layer_norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        # Register position IDs buffer
        self.register_buffer(
            'position_ids',
            torch.arange(config.max_seq_len).unsqueeze(0)
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (B, seq_len)
        
        Returns:
            Embeddings (B, seq_len, embed_dim)
        """
        B, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embed(input_ids)
        
        # Position embeddings
        position_ids = self.position_ids[:, :seq_len]
        position_embeds = self.position_embed(position_ids)
        
        # Combine embeddings (no segment embeddings in DistilBERT!)
        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class DistilBERTAttention(nn.Module):
    """
    Multi-Head Self-Attention for DistilBERT.
    
    Same structure as BERT but with separate Q, K, V projections.
    """
    
    def __init__(self, config: DistilBERTConfig):
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


class DistilBERTFFN(nn.Module):
    """
    DistilBERT Feed-Forward Network.
    
    Structure: Linear(D -> 4D) -> GELU -> Linear(4D -> D)
    """
    
    def __init__(self, config: DistilBERTConfig):
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


class DistilBERTEncoderLayer(nn.Module):
    """
    Single DistilBERT Encoder layer.
    
    Structure (pre-norm style like original DistilBERT):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    
    Note: DistilBERT uses pre-norm (LayerNorm before sublayer),
    which is slightly different from BERT's post-norm.
    """
    
    def __init__(self, config: DistilBERTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Pre-norm for attention
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.attention = DistilBERTAttention(config)
        
        # Pre-norm for FFN
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.ffn = DistilBERTFFN(config)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual (pre-norm)
        attn_out = self.attention(self.norm1(x), attention_mask)
        x = x + self.dropout(attn_out)
        
        # FFN with residual (pre-norm)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x


class SGXDistilBERTModel(nn.Module):
    """
    DistilBERT model designed for SGX distributed inference.
    
    Key differences from BERT:
    1. Only 6 transformer layers (vs 12 in BERT-base)
    2. No segment embeddings
    3. Pre-classifier layer with ReLU
    """
    
    def __init__(self, config: Optional[DistilBERTConfig] = None):
        super().__init__()
        
        self.config = config or DistilBERTBaseConfig()
        
        # Embeddings (no segment embeddings!)
        self.embeddings = DistilBERTEmbedding(self.config)
        
        # Encoder layers (only 6 layers)
        self.encoder_layers = nn.ModuleList([
            DistilBERTEncoderLayer(self.config, i)
            for i in range(self.config.num_layers)
        ])
        
        # Pre-classifier (DistilBERT specific)
        self.pre_classifier = nn.Linear(self.config.embed_dim, self.config.embed_dim)
        self.pre_classifier_act = nn.ReLU()
        self.pre_classifier_dropout = nn.Dropout(self.config.dropout)
        
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
        
        # Encoder layers (6 layers)
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
        
        # Pre-classifier
        self.layers.append({
            'name': 'pre_classifier',
            'module': self.pre_classifier,
            'type': 'Linear+ReLU',
            'dependencies': [prev_layer],
            'input_shape': (1, self.config.embed_dim),
            'output_shape': (1, self.config.embed_dim)
        })
        
        # Classifier
        self.layers.append({
            'name': 'classifier',
            'module': self.classifier,
            'type': 'Linear',
            'dependencies': ['pre_classifier'],
            'output_shape': (1, self.config.num_classes)
        })
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len), 1 for valid, 0 for padding
        
        Returns:
            Classification logits (B, num_classes)
        """
        # Convert attention mask to additive mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Encoder layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Get CLS token and apply pre-classifier
        cls_token = hidden_states[:, 0]  # (B, embed_dim)
        pooled = self.pre_classifier(cls_token)
        pooled = self.pre_classifier_act(pooled)
        pooled = self.pre_classifier_dropout(pooled)
        
        # Classifier
        logits = self.classifier(pooled)
        
        return logits
    
    def forward_layer_by_layer(
        self,
        input_ids: torch.Tensor,
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
        hidden_states = self.embeddings(input_ids)
        if layer_timing_callback:
            layer_timing_callback('embeddings', (time.perf_counter() - start) * 1000)
        
        # Encoder layers
        for i, layer in enumerate(self.encoder_layers):
            start = time.perf_counter()
            hidden_states = layer(hidden_states, attention_mask)
            if layer_timing_callback:
                layer_timing_callback(f'encoder_{i}', (time.perf_counter() - start) * 1000)
        
        # Pre-classifier
        start = time.perf_counter()
        cls_token = hidden_states[:, 0]
        pooled = self.pre_classifier(cls_token)
        pooled = self.pre_classifier_act(pooled)
        pooled = self.pre_classifier_dropout(pooled)
        if layer_timing_callback:
            layer_timing_callback('pre_classifier', (time.perf_counter() - start) * 1000)
        
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
                'num_layers': self.config.num_layers  # 6 layers!
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
                'pre_classifier': sum(p.numel() for p in self.pre_classifier.parameters()) * 4,
                'classifier': sum(p.numel() for p in self.classifier.parameters()) * 4,
                'total': sum(p.numel() for p in self.parameters()) * 4
            }
        }


# ==============================================================================
# Factory Functions
# ==============================================================================

def create_distilbert_base(num_classes: int = 2) -> SGXDistilBERTModel:
    """Create DistilBERT-base model."""
    config = DistilBERTBaseConfig(num_classes=num_classes)
    return SGXDistilBERTModel(config)


# ==============================================================================
# Quick Test
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SGX DistilBERT Model - Architecture Analysis")
    print("=" * 70)
    
    # Create DistilBERT-base
    model = create_distilbert_base(num_classes=2)
    
    print(f"\n[Model Configuration]")
    print(f"  - Vocab Size: {model.config.vocab_size}")
    print(f"  - Max Seq Len: {model.config.max_seq_len}")
    print(f"  - Embed Dim: {model.config.embed_dim}")
    print(f"  - Num Heads: {model.config.num_heads}")
    print(f"  - Num Layers: {model.config.num_layers} (vs 12 in BERT-base)")
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
    attention_mask = torch.ones_like(input_ids)
    
    timings = {}
    def record_timing(name, ms):
        timings[name] = ms
    
    with torch.no_grad():
        output = model.forward_layer_by_layer(
            input_ids, attention_mask, record_timing
        )
    
    print(f"  - Output shape: {output.shape}")
    print(f"  - Total time: {sum(timings.values()):.2f} ms")
    print(f"\n  Layer timings:")
    for name, ms in sorted(timings.items(), key=lambda x: -x[1]):
        print(f"    {name}: {ms:.2f} ms")
    
    print("\n" + "=" * 70)
    print("✓ SGX DistilBERT Model ready for distributed inference experiments")
    print("  Key advantage: Only 6 layers (50% fewer than BERT-base)")
    print("=" * 70)
