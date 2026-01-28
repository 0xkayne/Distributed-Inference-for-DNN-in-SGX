"""
Unified Multi-Head Attention modules for all Transformer-based models.

Provides:
- BatchedMultiHeadAttention: All heads computed in single operation (efficient)
- PerHeadMultiHeadAttention: Each head computed independently (fine-grained profiling)
- create_multi_head_attention: Factory function for easy instantiation

Supports: BERT, ALBERT, DistilBERT, TinyBERT, ViT, Swin Transformer, etc.
"""

from .base_attention import BaseMultiHeadAttention
from .batched_attention import BatchedMultiHeadAttention
from .per_head_attention import PerHeadMultiHeadAttention
from .attention_factory import create_multi_head_attention

__all__ = [
    'BaseMultiHeadAttention',
    'BatchedMultiHeadAttention',
    'PerHeadMultiHeadAttention',
    'create_multi_head_attention',
]
