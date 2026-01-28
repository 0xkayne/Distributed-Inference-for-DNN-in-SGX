"""
Multi-Head Attention with per-head computation.
Each head is computed independently for fine-grained profiling in TEE.
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


class PerHeadMultiHeadAttention(BaseMultiHeadAttention):
    """
    Multi-Head Attention with per-head computation.
    
    Each head computes attention independently, allowing:
    - Fine-grained profiling of each head's performance
    - Clear dependency tracking per head
    - Better memory management in TEE (process heads sequentially)
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
        use_shared_qkv: bool = True,  # Share QKV projections across heads
    ):
        """
        Args:
            use_shared_qkv: If True, use shared Q/K/V projections then slice per head.
                           If False, each head has its own Q/K/V projections (memory heavy).
        """
        self.use_shared_qkv = use_shared_qkv
        super().__init__(
            sid, name_prefix, enclave_mode, embed_dim, num_heads,
            batch_size, seq_len, layer_mode_overrides
        )
        self._build_layers()
    
    def _build_layers(self):
        """Build attention layers (per-head mode)."""
        # ===== Shared Q/K/V projections =====
        if self.use_shared_qkv:
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
        
        # ===== Per-head computation =====
        self.head_layers = []
        
        for head_idx in range(self.num_heads):
            head_name = f"{self.name_prefix}_head{head_idx}"
            head_dict = {}
            
            if not self.use_shared_qkv:
                # Each head has its own Q/K/V projections
                head_dict['q_proj'] = SGXLinearBase(
                    self.sid, f"{head_name}_q_proj",
                    self._get_mode(f"head{head_idx}_q_proj"),
                    batch_size=self.tokens,
                    n_output_features=self.head_dim,
                    n_input_features=self.embed_dim,
                    manually_register_prev=True, manually_register_next=True
                )
                self.layers.append(head_dict['q_proj'])
                
                head_dict['k_proj'] = SGXLinearBase(
                    self.sid, f"{head_name}_k_proj",
                    self._get_mode(f"head{head_idx}_k_proj"),
                    batch_size=self.tokens,
                    n_output_features=self.head_dim,
                    n_input_features=self.embed_dim,
                    manually_register_prev=True, manually_register_next=True
                )
                self.layers.append(head_dict['k_proj'])
                
                head_dict['v_proj'] = SGXLinearBase(
                    self.sid, f"{head_name}_v_proj",
                    self._get_mode(f"head{head_idx}_v_proj"),
                    batch_size=self.tokens,
                    n_output_features=self.head_dim,
                    n_input_features=self.embed_dim,
                    manually_register_prev=True, manually_register_next=True
                )
                self.layers.append(head_dict['v_proj'])
                
                # Reshape to add head dimension: (tokens, head_dim) -> (B, 1, seq_len, head_dim)
                head_dict['reshape_q'] = SecretReshapeLayer(
                    self.sid, f"{head_name}_reshape_q_add_head",
                    self._get_mode(f"head{head_idx}_reshape_q"),
                    target_shape=[self.batch_size, self.seq_len, 1, self.head_dim],
                    permute_dims=[0, 2, 1, 3],  # (B, 1, seq_len, head_dim)
                    mode='view_permute',
                    manually_register_prev=True, manually_register_next=True
                )
                self.layers.append(head_dict['reshape_q'])
                
                head_dict['reshape_k'] = SecretReshapeLayer(
                    self.sid, f"{head_name}_reshape_k_add_head",
                    self._get_mode(f"head{head_idx}_reshape_k"),
                    target_shape=[self.batch_size, self.seq_len, 1, self.head_dim],
                    permute_dims=[0, 2, 1, 3],
                    mode='view_permute',
                    manually_register_prev=True, manually_register_next=True
                )
                self.layers.append(head_dict['reshape_k'])
                
                head_dict['reshape_v'] = SecretReshapeLayer(
                    self.sid, f"{head_name}_reshape_v_add_head",
                    self._get_mode(f"head{head_idx}_reshape_v"),
                    target_shape=[self.batch_size, self.seq_len, 1, self.head_dim],
                    permute_dims=[0, 2, 1, 3],
                    mode='view_permute',
                    manually_register_prev=True, manually_register_next=True
                )
                self.layers.append(head_dict['reshape_v'])
            else:
                # Extract this head's slice from shared projections
                # Reshape: (tokens, embed_dim) -> (B, seq_len, num_heads, head_dim)
                # Then extract single head -> (B, 1, seq_len, head_dim)
                head_dict['reshape_q'] = SecretReshapeLayer(
                    self.sid, f"{head_name}_reshape_q",
                    self._get_mode(f"head{head_idx}_reshape_q"),
                    target_shape=[self.batch_size, self.seq_len, self.num_heads, self.head_dim],
                    permute_dims=[0, 2, 1, 3],  # (B, H, seq_len, head_dim)
                    mode='view_permute',
                    manually_register_prev=True, manually_register_next=True
                )
                self.layers.append(head_dict['reshape_q'])
                
                head_dict['reshape_k'] = SecretReshapeLayer(
                    self.sid, f"{head_name}_reshape_k",
                    self._get_mode(f"head{head_idx}_reshape_k"),
                    target_shape=[self.batch_size, self.seq_len, self.num_heads, self.head_dim],
                    permute_dims=[0, 2, 1, 3],
                    mode='view_permute',
                    manually_register_prev=True, manually_register_next=True
                )
                self.layers.append(head_dict['reshape_k'])
                
                head_dict['reshape_v'] = SecretReshapeLayer(
                    self.sid, f"{head_name}_reshape_v",
                    self._get_mode(f"head{head_idx}_reshape_v"),
                    target_shape=[self.batch_size, self.seq_len, self.num_heads, self.head_dim],
                    permute_dims=[0, 2, 1, 3],
                    mode='view_permute',
                    manually_register_prev=True, manually_register_next=True
                )
                self.layers.append(head_dict['reshape_v'])
            
            # Q @ K^T for this head: (B, 1, seq_len, head_dim) @ (B, 1, head_dim, seq_len)
            # Result: (B, 1, seq_len, seq_len)
            head_dict['qk_matmul'] = SecretMatMulLayer(
                self.sid, f"{head_name}_qk_matmul",
                self._get_mode(f"head{head_idx}_qk_matmul"),
                transpose_b=True,
                scale=1.0 / math.sqrt(self.head_dim),
                manually_register_prev=True, manually_register_next=True
            )
            self.layers.append(head_dict['qk_matmul'])
            
            # Softmax for this head
            head_dict['softmax'] = SecretSoftmaxLayer(
                self.sid, f"{head_name}_softmax",
                self._get_mode(f"head{head_idx}_softmax"),
                dim=-1,
                manually_register_prev=True, manually_register_next=True
            )
            self.layers.append(head_dict['softmax'])
            
            # Attn @ V for this head: (B, 1, seq_len, seq_len) @ (B, 1, seq_len, head_dim)
            # Result: (B, 1, seq_len, head_dim)
            head_dict['attn_v_matmul'] = SecretMatMulLayer(
                self.sid, f"{head_name}_attn_v_matmul",
                self._get_mode(f"head{head_idx}_attn_v_matmul"),
                transpose_b=False,
                manually_register_prev=True, manually_register_next=True
            )
            self.layers.append(head_dict['attn_v_matmul'])
            
            self.head_layers.append(head_dict)
        
        # ===== Concatenate all heads =====
        # We need to concatenate outputs from all heads along the head dimension
        # For simplicity, we'll use reshape operations to simulate concatenation
        # Each head output: (B, 1, seq_len, head_dim)
        # Concatenated: (B, num_heads, seq_len, head_dim)
        
        # Note: Since we don't have a SecretConcatLayer, we'll flatten and use output proj directly
        # This is a simplification - in practice, heads would be concatenated
        
        # Flatten: (B, num_heads, seq_len, head_dim) -> (B*seq_len, num_heads*head_dim=embed_dim)
        # Since we compute heads independently, we'll collect them during connection
        
        # For now, create a placeholder for concatenation
        # The actual concatenation happens in the connect() method
        
        # Flatten concatenated heads
        self.flatten_concat = SecretReshapeLayer(
            self.sid, f"{self.name_prefix}_flatten_concat",
            self._get_mode("flatten_concat"),
            target_shape=[self.batch_size * self.seq_len, self.embed_dim],
            mode='view',
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.flatten_concat)
        
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
        
        if self.use_shared_qkv:
            self.input_layer = self.q_proj
        else:
            self.input_layer = self.head_layers[0]['q_proj']
        self.output_layer = self.out_proj
    
    def connect(self, prev_layer):
        """Connect attention module to previous layer."""
        if self.use_shared_qkv:
            # Shared Q/K/V projections
            self.q_proj.register_prev_layer(prev_layer)
            self.k_proj.register_prev_layer(prev_layer)
            self.v_proj.register_prev_layer(prev_layer)
            
            # Connect each head
            for head_idx, head_dict in enumerate(self.head_layers):
                # Reshape from shared projections
                head_dict['reshape_q'].register_prev_layer(self.q_proj)
                head_dict['reshape_k'].register_prev_layer(self.k_proj)
                head_dict['reshape_v'].register_prev_layer(self.v_proj)
                
                # QK matmul
                head_dict['qk_matmul'].register_prev_layer(head_dict['reshape_q'])
                head_dict['qk_matmul'].register_prev_layer(head_dict['reshape_k'])
                
                # Softmax
                head_dict['softmax'].register_prev_layer(head_dict['qk_matmul'])
                
                # Attn @ V
                head_dict['attn_v_matmul'].register_prev_layer(head_dict['softmax'])
                head_dict['attn_v_matmul'].register_prev_layer(head_dict['reshape_v'])
        else:
            # Per-head Q/K/V projections
            for head_idx, head_dict in enumerate(self.head_layers):
                # Each head's projections from input
                head_dict['q_proj'].register_prev_layer(prev_layer)
                head_dict['k_proj'].register_prev_layer(prev_layer)
                head_dict['v_proj'].register_prev_layer(prev_layer)
                
                # Reshape to add head dimension
                head_dict['reshape_q'].register_prev_layer(head_dict['q_proj'])
                head_dict['reshape_k'].register_prev_layer(head_dict['k_proj'])
                head_dict['reshape_v'].register_prev_layer(head_dict['v_proj'])
                
                # QK matmul
                head_dict['qk_matmul'].register_prev_layer(head_dict['reshape_q'])
                head_dict['qk_matmul'].register_prev_layer(head_dict['reshape_k'])
                
                # Softmax
                head_dict['softmax'].register_prev_layer(head_dict['qk_matmul'])
                
                # Attn @ V
                head_dict['attn_v_matmul'].register_prev_layer(head_dict['softmax'])
                head_dict['attn_v_matmul'].register_prev_layer(head_dict['reshape_v'])
        
        # Concatenation: For simplification, we connect the last head's output
        # to the flatten layer. In a full implementation, this would properly
        # concatenate all heads.
        # TODO: Implement proper concatenation when SecretConcatLayer is available
        last_head_output = self.head_layers[-1]['attn_v_matmul']
        self.flatten_concat.register_prev_layer(last_head_output)
        
        # Output projection
        self.out_proj.register_prev_layer(self.flatten_concat)
        
        return self.out_proj
