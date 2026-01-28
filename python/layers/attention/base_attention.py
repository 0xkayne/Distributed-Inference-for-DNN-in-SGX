"""
Base Multi-Head Attention interface for all models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import sys
sys.path.insert(0, '.')

from python.utils.basic_utils import ExecutionModeOptions


class BaseMultiHeadAttention(ABC):
    """
    Base class for Multi-Head Attention implementations.
    
    Provides common interface for both batched and per-head modes.
    All Transformer-based models (BERT, ViT, Swin, etc.) can use this.
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        embed_dim: int,
        num_heads: int,
        batch_size: int,
        seq_len: int,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        """
        Args:
            sid: Session ID
            name_prefix: Layer name prefix (e.g., "encoder0_attn")
            enclave_mode: Default execution mode
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            batch_size: Batch size
            seq_len: Sequence length (tokens per batch)
            layer_mode_overrides: Per-layer execution mode overrides
        """
        self.sid = sid
        self.name_prefix = name_prefix
        self.enclave_mode = enclave_mode
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.head_dim = embed_dim // num_heads
        self.layer_mode_overrides = layer_mode_overrides or {}
        
        # Total flattened tokens for Linear layers
        self.tokens = batch_size * seq_len
        
        # Layers list to be populated by subclasses
        self.layers: List = []
        self.input_layer = None
        self.output_layer = None
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
    
    def _get_mode(self, name: str) -> ExecutionModeOptions:
        """Get execution mode for a specific layer."""
        full_name = f"{self.name_prefix}_{name}"
        return self.layer_mode_overrides.get(full_name, self.enclave_mode)
    
    @abstractmethod
    def connect(self, prev_layer):
        """
        Connect attention module to previous layer.
        
        Returns:
            Output layer of the attention module
        """
        pass
    
    def get_all_layers(self) -> List:
        """Return all layers in this attention module."""
        return self.layers
