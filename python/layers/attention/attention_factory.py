"""
Factory for creating attention modules.
Provides unified interface for all models.
"""
import sys
sys.path.insert(0, '.')

from typing import Dict, Optional
from python.utils.basic_utils import ExecutionModeOptions

from .batched_attention import BatchedMultiHeadAttention
from .per_head_attention import PerHeadMultiHeadAttention


def create_multi_head_attention(
    sid: int,
    name_prefix: str,
    enclave_mode: ExecutionModeOptions,
    embed_dim: int,
    num_heads: int,
    batch_size: int,
    seq_len: int,
    per_head_mode: bool = False,
    use_shared_qkv: bool = True,
    layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
):
    """
    Factory function to create Multi-Head Attention module.
    
    Args:
        sid: Session ID
        name_prefix: Layer name prefix
        enclave_mode: Default execution mode
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        batch_size: Batch size
        seq_len: Sequence length
        per_head_mode: If True, compute each head independently
        use_shared_qkv: If True (and per_head_mode), share Q/K/V projections
        layer_mode_overrides: Per-layer execution mode overrides
    
    Returns:
        Attention module instance
    
    Examples:
        # BERT-style attention (batched)
        attn = create_multi_head_attention(
            sid=0, name_prefix="encoder0_attn",
            enclave_mode=ExecutionModeOptions.Enclave,
            embed_dim=768, num_heads=12,
            batch_size=1, seq_len=128,
            per_head_mode=False
        )
        
        # ViT-style attention (per-head for profiling)
        attn = create_multi_head_attention(
            sid=0, name_prefix="block0_attn",
            enclave_mode=ExecutionModeOptions.Enclave,
            embed_dim=768, num_heads=12,
            batch_size=1, seq_len=197,  # 196 patches + 1 CLS
            per_head_mode=True,
            use_shared_qkv=True
        )
        
        # Swin window attention (per-head)
        num_windows = 64
        window_tokens = 49  # 7x7
        attn = create_multi_head_attention(
            sid=0, name_prefix="stage0_block0_attn",
            enclave_mode=ExecutionModeOptions.Enclave,
            embed_dim=96, num_heads=3,
            batch_size=num_windows,  # Treat each window as a batch
            seq_len=window_tokens,
            per_head_mode=True
        )
    """
    if per_head_mode:
        return PerHeadMultiHeadAttention(
            sid=sid,
            name_prefix=name_prefix,
            enclave_mode=enclave_mode,
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_size=batch_size,
            seq_len=seq_len,
            use_shared_qkv=use_shared_qkv,
            layer_mode_overrides=layer_mode_overrides,
        )
    else:
        return BatchedMultiHeadAttention(
            sid=sid,
            name_prefix=name_prefix,
            enclave_mode=enclave_mode,
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_size=batch_size,
            seq_len=seq_len,
            layer_mode_overrides=layer_mode_overrides,
        )
