"""
Position Embedding Layer for Vision Transformer in SGX.

This layer adds learnable position embeddings to the input tokens.
Position embeddings allow the model to understand the spatial relationship
between different patches in the image.

Input:  (B, N, D) - batch of N tokens with dimension D
Output: (B, N, D) - same shape, with position embeddings added

The position embedding parameter has shape (1, N, D) and is broadcast-added
to the input tokens.
"""

import torch
import torch.nn as nn
from pdb import set_trace as st

from python.layers.nonlinear import SecretNonlinearLayer
from python.tensor_loader import TensorLoader
from python.sgx_net import LearnableParamTuple
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.basic_utils import ExecutionModeOptions
from python.enclave_interfaces import GlobalTensor as gt


class SecretPositionEmbeddingLayer(SecretNonlinearLayer):
    """
    Position Embedding layer for Vision Transformer.
    
    Adds learnable position embeddings to input tokens, allowing the model
    to encode spatial information about patch positions.
    """
    
    def __init__(
        self, 
        sid, 
        LayerName, 
        EnclaveMode,
        seq_len,
        embed_dim,
        batch_size=1,
        link_prev=True, 
        link_next=True,
        manually_register_prev=False, 
        manually_register_next=False
    ):
        super().__init__(
            sid, LayerName, EnclaveMode, link_prev, link_next,
            manually_register_prev, manually_register_next
        )
        
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        
        # Position embedding is a learnable parameter
        # Shape: (1, seq_len, embed_dim) - will be broadcast to (B, seq_len, embed_dim)
        self.PosEmbedShape = [1, seq_len, embed_dim]
        
        self.LearnableParamsList = [
            LearnableParamTuple(
                dw_name="DerPosEmbed", 
                w_name="pos_embed", 
                shape=self.PosEmbedShape
            )
        ]
    
    def init_shape(self):
        """Initialize input/output shapes."""
        # Input: (B, N, D)
        self.InputShape = self.PrevLayer.get_output_shape()
        
        # Validate input is 3D and matches expected seq_len and embed_dim
        if len(self.InputShape) != 3:
            raise ValueError(
                f"PositionEmbeddingLayer expects 3D input (B, N, D), got {self.InputShape}"
            )
        
        if self.InputShape[1] != self.seq_len:
            print(
                f"Warning: Input seq_len {self.InputShape[1]} != "
                f"expected {self.seq_len}, adjusting..."
            )
            self.seq_len = self.InputShape[1]
            self.PosEmbedShape = [1, self.seq_len, self.embed_dim]
            # Update learnable params
            self.LearnableParamsList = [
                LearnableParamTuple(
                    dw_name="DerPosEmbed", 
                    w_name="pos_embed", 
                    shape=self.PosEmbedShape
                )
            ]
        
        if self.InputShape[2] != self.embed_dim:
            raise ValueError(
                f"Input embed_dim {self.InputShape[2]} != expected {self.embed_dim}"
            )
        
        # Output: same shape as input
        self.OutputShape = self.InputShape
        self.HandleShape = self.OutputShape
    
    def init(self, start_enclave=True):
        """Initialize the layer and position embedding parameter."""
        TensorLoader.init(self, start_enclave)
        
        # Initialize position embeddings with small random values
        # Using truncated normal with std=0.02 (standard for ViT)
        if self.EnclaveMode == ExecutionModeOptions.CPU or \
           self.EnclaveMode == ExecutionModeOptions.Enclave:
            pos_embed = torch.zeros(self.PosEmbedShape)
            nn.init.trunc_normal_(pos_embed, std=0.02)
            self.set_cpu("pos_embed", pos_embed)
            
        elif self.EnclaveMode == ExecutionModeOptions.GPU:
            pos_embed = torch.zeros(self.PosEmbedShape).cuda()
            nn.init.trunc_normal_(pos_embed, std=0.02)
            self.set_gpu("pos_embed", pos_embed)
    
    def get_output_shape(self):
        """Return output shape."""
        return self.OutputShape
    
    def generate_tensor_name_list(self, force=False):
        """Generate list of tensors needed by this layer."""
        if not force and self.tensor_name_list:
            return
        
        NeededTensorNames = [
            ("output", self.OutputShape, None),
            ("input", self.InputShape, None),
            ("pos_embed", self.PosEmbedShape, None),
        ]
        
        self.tensor_name_list = NeededTensorNames
    
    def link_tensors(self):
        """Link tensors between layers."""
        if self.link_prev and self.PrevLayer is not None:
            gt.link_tags(
                self.get_tag("input", remap=False), 
                self.PrevLayer.get_tag("output", remap=False)
            )
        
        if self.link_next and self.NextLayer is not None:
            gt.link_tags(
                self.get_tag("output", remap=False), 
                self.NextLayer.get_tag("input", remap=False)
            )
    
    def forward(self):
        """
        Forward pass: add position embeddings to input tokens.
        
        Process:
        1. Get input tokens: (B, N, D)
        2. Get position embeddings: (1, N, D)
        3. Broadcast addition: input + pos_embed -> (B, N, D)
        """
        with NamedTimerInstance(
            f"S{self.sid}: {self.LayerName} Forward", 
            verbose_level=VerboseLevel.LAYER
        ):
            if self.EnclaveMode == ExecutionModeOptions.CPU:
                self._forward_cpu()
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                self._forward_gpu()
            elif self.EnclaveMode == ExecutionModeOptions.Enclave:
                # For now, use CPU computation with data transfer
                self._forward_enclave()
    
    def _forward_cpu(self):
        """Forward pass on CPU."""
        # Get input tokens and position embeddings
        input_tokens = self.get_cpu("input")  # (B, N, D)
        pos_embed = self.get_cpu("pos_embed")  # (1, N, D)
        
        # Broadcast addition
        output = input_tokens + pos_embed  # (B, N, D)
        
        self.set_cpu("output", output)
    
    def _forward_gpu(self):
        """Forward pass on GPU."""
        # Get input tokens and position embeddings
        input_tokens = self.get_gpu("input")  # (B, N, D)
        pos_embed = self.get_gpu("pos_embed")  # (1, N, D)
        
        # Broadcast addition
        output = input_tokens + pos_embed  # (B, N, D)
        
        self.set_gpu("output", output)
    
    def _forward_enclave(self):
        """Forward pass in Enclave mode (currently uses CPU with transfer)."""
        # Transfer from enclave to CPU
        self.transfer_enclave_to_cpu("input")
        
        # Perform computation on CPU
        input_tokens = self.get_cpu("input")
        pos_embed = self.get_cpu("pos_embed")
        
        output = input_tokens + pos_embed
        
        self.set_cpu("output", output)
        
        # Transfer back to enclave
        self.transfer_from_cpu("output")
    
    def backward(self):
        """Backward pass (not implemented for inference)."""
        pass
    
    def inject_params_from_pytorch(self, pytorch_module):
        """
        Inject position embeddings from a PyTorch ViT model.
        
        Args:
            pytorch_module: PyTorch module with 'pos_embed' attribute
        """
        if hasattr(pytorch_module, 'pos_embed'):
            pos_embed_data = pytorch_module.pos_embed.data
            
            # Check shape compatibility
            if list(pos_embed_data.shape) != self.PosEmbedShape:
                print(
                    f"Warning: Position embedding shape mismatch: "
                    f"expected {self.PosEmbedShape}, got {list(pos_embed_data.shape)}"
                )
                # Try to adjust if only batch dimension differs
                if pos_embed_data.shape[1:] == tuple(self.PosEmbedShape[1:]):
                    pos_embed_data = pos_embed_data[:1]  # Take first element if batched
            
            if self.EnclaveMode == ExecutionModeOptions.CPU or \
               self.EnclaveMode == ExecutionModeOptions.Enclave:
                self.set_cpu("pos_embed", pos_embed_data)
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                self.set_gpu("pos_embed", pos_embed_data.cuda())
        else:
            print(f"Warning: {self.LayerName} could not find pos_embed in pytorch_module")
    
    def print_connection_info(self):
        """Print layer connection information."""
        prev_name = self.PrevLayer.LayerName if self.PrevLayer else "None"
        next_name = self.NextLayer.LayerName if self.NextLayer else "None"
        print(
            f"{self.LayerName:20} shape {self.InputShape} -> {self.OutputShape}"
            f"{' ':10} input {prev_name:20} output {next_name:20}"
        )
