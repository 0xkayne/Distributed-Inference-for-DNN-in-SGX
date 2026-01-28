"""
Multi-Head Attention with batched computation (current implementation).
All heads computed in a single matmul operation.
"""
import sys
import math
sys.path.insert(0, '.')

from typing import Dict, Optional, List

from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.softmax import SecretSoftmaxLayer
from python.layers.matmul import SecretMatMulLayer
from python.layers.reshape import SecretReshapeLayer
from python.utils.basic_utils import ExecutionModeOptions

from .base_attention import BaseMultiHeadAttention


class BatchedMultiHeadAttention(BaseMultiHeadAttention):
    """
    Multi-Head Attention with batched computation.
    
    All heads are computed simultaneously in a single operation.
    More efficient but less granular profiling.
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
        super().__init__(
            sid, name_prefix, enclave_mode, embed_dim, num_heads,
            batch_size, seq_len, layer_mode_overrides
        )
        self._build_layers()
    
    def _build_layers(self):
        """Build attention layers (batched mode)."""
        # Q, K, V projections
        self.q_proj = SGXLinearBase(
            self.sid, f"{self.name_prefix}_q_proj", 
            self._get_mode("q_proj"),
            batch_size=self.tokens,
            n_output_features=self.embed_dim,
            n_input_features=self.embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        
        self.k_proj = SGXLinearBase(
            self.sid, f"{self.name_prefix}_k_proj",
            self._get_mode("k_proj"),
            batch_size=self.tokens,
            n_output_features=self.embed_dim,
            n_input_features=self.embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        
        self.v_proj = SGXLinearBase(
            self.sid, f"{self.name_prefix}_v_proj",
            self._get_mode("v_proj"),
            batch_size=self.tokens,
            n_output_features=self.embed_dim,
            n_input_features=self.embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.extend([self.q_proj, self.k_proj, self.v_proj])
        
        # Reshape Q, K, V to multi-head format
        self.reshape_q = SecretReshapeLayer(
            self.sid, f"{self.name_prefix}_reshape_q",
            self._get_mode("reshape_q"),
            target_shape=[self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            permute_dims=[0, 2, 1, 3],  # (B, H, N, D)
            mode='view_permute',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.reshape_q)
        
        self.reshape_k = SecretReshapeLayer(
            self.sid, f"{self.name_prefix}_reshape_k",
            self._get_mode("reshape_k"),
            target_shape=[self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            permute_dims=[0, 2, 1, 3],
            mode='view_permute',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.reshape_k)
        
        self.reshape_v = SecretReshapeLayer(
            self.sid, f"{self.name_prefix}_reshape_v",
            self._get_mode("reshape_v"),
            target_shape=[self.batch_size, self.seq_len, self.num_heads, self.head_dim],
            permute_dims=[0, 2, 1, 3],
            mode='view_permute',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.reshape_v)
        
        # Q @ K^T (all heads at once)
        self.qk_matmul = SecretMatMulLayer(
            self.sid, f"{self.name_prefix}_qk_matmul",
            self._get_mode("qk_matmul"),
            transpose_b=True,
            scale=1.0 / math.sqrt(self.head_dim),
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.qk_matmul)
        
        # Softmax
        self.attn_softmax = SecretSoftmaxLayer(
            self.sid, f"{self.name_prefix}_attn_softmax",
            self._get_mode("attn_softmax"),
            dim=-1,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.attn_softmax)
        
        # Attn @ V
        self.attn_v_matmul = SecretMatMulLayer(
            self.sid, f"{self.name_prefix}_attn_v_matmul",
            self._get_mode("attn_v_matmul"),
            transpose_b=False,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.attn_v_matmul)
        
        # Reshape back and concat heads
        self.reshape_concat = SecretReshapeLayer(
            self.sid, f"{self.name_prefix}_reshape_concat",
            self._get_mode("reshape_concat"),
            target_shape=[self.batch_size, self.num_heads, self.seq_len, self.head_dim],
            permute_dims=[0, 2, 1, 3],  # (B, N, H, D)
            mode='view_permute',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.reshape_concat)
        
        self.flatten_heads = SecretReshapeLayer(
            self.sid, f"{self.name_prefix}_flatten_heads",
            self._get_mode("flatten_heads"),
            target_shape=[self.batch_size * self.seq_len, self.embed_dim],
            mode='view',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.flatten_heads)
        
        # Output projection
        self.out_proj = SGXLinearBase(
            self.sid, f"{self.name_prefix}_out_proj",
            self._get_mode("out_proj"),
            batch_size=self.tokens,
            n_output_features=self.embed_dim,
            n_input_features=self.embed_dim,
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.out_proj)
        
        self.input_layer = self.q_proj
        self.output_layer = self.out_proj
    
    def connect(self, prev_layer):
        """Connect attention module to previous layer."""
        # Q/K/V projections from input
        self.q_proj.register_prev_layer(prev_layer)
        self.k_proj.register_prev_layer(prev_layer)
        self.v_proj.register_prev_layer(prev_layer)
        
        # Reshape
        self.reshape_q.register_prev_layer(self.q_proj)
        self.reshape_k.register_prev_layer(self.k_proj)
        self.reshape_v.register_prev_layer(self.v_proj)
        
        # QK matmul
        self.qk_matmul.register_prev_layer(self.reshape_q)
        self.qk_matmul.register_prev_layer(self.reshape_k)
        
        # Softmax
        self.attn_softmax.register_prev_layer(self.qk_matmul)
        
        # Attn @ V
        self.attn_v_matmul.register_prev_layer(self.attn_softmax)
        self.attn_v_matmul.register_prev_layer(self.reshape_v)
        
        # Concat and output
        self.reshape_concat.register_prev_layer(self.attn_v_matmul)
        self.flatten_heads.register_prev_layer(self.reshape_concat)
        self.out_proj.register_prev_layer(self.flatten_heads)
        
        return self.out_proj
