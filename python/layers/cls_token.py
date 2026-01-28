"""
CLS Token Layer for Vision Transformer in SGX.

This layer prepends a learnable CLS (classification) token to the patch tokens.
The CLS token is used for final classification in ViT models.

Input:  (B, N, D) - batch of N tokens with dimension D
Output: (B, N+1, D) - batch with CLS token prepended

The CLS token parameter has shape (1, 1, D) and is expanded to (B, 1, D) 
before concatenation with the input tokens.
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


class SecretCLSTokenLayer(SecretNonlinearLayer):
    """
    CLS Token layer for Vision Transformer.
    
    Prepends a learnable classification token to the sequence of patch tokens.
    This is a critical component of standard ViT architecture.
    """
    
    def __init__(
        self, 
        sid, 
        LayerName, 
        EnclaveMode,
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
        
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        
        # CLS token is a learnable parameter
        # Shape: (1, 1, embed_dim) - will be expanded to (B, 1, embed_dim)
        self.CLSTokenShape = [1, 1, embed_dim]
        
        self.LearnableParamsList = [
            LearnableParamTuple(
                dw_name="DerCLSToken", 
                w_name="cls_token", 
                shape=self.CLSTokenShape
            )
        ]
    
    def init_shape(self):
        """Initialize input/output shapes."""
        # Input: (B, N, D) where N = num_patches
        self.InputShape = self.PrevLayer.get_output_shape()
        
        # Validate input is 3D
        if len(self.InputShape) != 3:
            raise ValueError(
                f"CLSTokenLayer expects 3D input (B, N, D), got {self.InputShape}"
            )
        
        # Output: (B, N+1, D) - one more token (the CLS token)
        self.OutputShape = list(self.InputShape)
        self.OutputShape[1] = self.InputShape[1] + 1  # N+1 tokens
        
        self.HandleShape = self.OutputShape
    
    def init(self, start_enclave=True):
        """Initialize the layer and CLS token parameter."""
        TensorLoader.init(self, start_enclave)
        
        # Initialize CLS token with small random values (like ViT does)
        # Using truncated normal with std=0.02
        if self.EnclaveMode == ExecutionModeOptions.CPU or \
           self.EnclaveMode == ExecutionModeOptions.Enclave:
            cls_token = torch.zeros(self.CLSTokenShape)
            nn.init.trunc_normal_(cls_token, std=0.02)
            self.set_cpu("cls_token", cls_token)
            
        elif self.EnclaveMode == ExecutionModeOptions.GPU:
            cls_token = torch.zeros(self.CLSTokenShape).cuda()
            nn.init.trunc_normal_(cls_token, std=0.02)
            self.set_gpu("cls_token", cls_token)
    
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
            ("cls_token", self.CLSTokenShape, None),
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
        Forward pass: prepend CLS token to input tokens.
        
        Process:
        1. Get input tokens: (B, N, D)
        2. Get CLS token: (1, 1, D)
        3. Expand CLS token: (1, 1, D) -> (B, 1, D)
        4. Concatenate: cat([cls, tokens], dim=1) -> (B, N+1, D)
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
        # Get input tokens
        input_tokens = self.get_cpu("input")  # (B, N, D)
        cls_token = self.get_cpu("cls_token")  # (1, 1, D)
        
        # Expand CLS token to batch size
        B = input_tokens.shape[0]
        cls_expanded = cls_token.expand(B, -1, -1)  # (B, 1, D)
        
        # Concatenate CLS token with input tokens
        output = torch.cat([cls_expanded, input_tokens], dim=1)  # (B, N+1, D)
        
        self.set_cpu("output", output)
    
    def _forward_gpu(self):
        """Forward pass on GPU."""
        # Get input tokens
        input_tokens = self.get_gpu("input")  # (B, N, D)
        cls_token = self.get_gpu("cls_token")  # (1, 1, D)
        
        # Expand CLS token to batch size
        B = input_tokens.shape[0]
        cls_expanded = cls_token.expand(B, -1, -1)  # (B, 1, D)
        
        # Concatenate CLS token with input tokens
        output = torch.cat([cls_expanded, input_tokens], dim=1)  # (B, N+1, D)
        
        self.set_gpu("output", output)
    
    def _forward_enclave(self):
        """Forward pass in Enclave mode (currently uses CPU with transfer)."""
        # Transfer from enclave to CPU
        self.transfer_enclave_to_cpu("input")
        
        # Perform computation on CPU
        input_tokens = self.get_cpu("input")
        cls_token = self.get_cpu("cls_token")
        
        B = input_tokens.shape[0]
        cls_expanded = cls_token.expand(B, -1, -1)
        output = torch.cat([cls_expanded, input_tokens], dim=1)
        
        self.set_cpu("output", output)
        
        # Transfer back to enclave
        self.transfer_from_cpu("output")
    
    def backward(self):
        """Backward pass (not implemented for inference)."""
        pass
    
    def inject_params_from_pytorch(self, pytorch_module):
        """
        Inject CLS token from a PyTorch ViT model.
        
        Args:
            pytorch_module: PyTorch module with 'cls_token' attribute
        """
        if hasattr(pytorch_module, 'cls_token'):
            cls_token_data = pytorch_module.cls_token.data
            
            if self.EnclaveMode == ExecutionModeOptions.CPU or \
               self.EnclaveMode == ExecutionModeOptions.Enclave:
                self.set_cpu("cls_token", cls_token_data)
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                self.set_gpu("cls_token", cls_token_data.cuda())
        else:
            print(f"Warning: {self.LayerName} could not find cls_token in pytorch_module")
    
    def print_connection_info(self):
        """Print layer connection information."""
        prev_name = self.PrevLayer.LayerName if self.PrevLayer else "None"
        next_name = self.NextLayer.LayerName if self.NextLayer else "None"
        print(
            f"{self.LayerName:20} shape {self.InputShape} -> {self.OutputShape}"
            f"{' ':10} input {prev_name:20} output {next_name:20}"
        )
