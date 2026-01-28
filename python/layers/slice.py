"""
Slice Layer for extracting specific tokens from sequences in SGX.

This layer extracts a specific token from a sequence, typically used to
extract the CLS token (at index 0) for classification in Vision Transformers.

Input:  (B, N, D) - batch of N tokens with dimension D
Output: (B, D) - extracted token at specified index

The layer slices along a specified dimension to extract one element.
"""

import torch
from pdb import set_trace as st

from python.layers.nonlinear import SecretNonlinearLayer
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.basic_utils import ExecutionModeOptions
from python.enclave_interfaces import GlobalTensor as gt


class SecretSliceLayer(SecretNonlinearLayer):
    """
    Slice layer to extract a specific token from a sequence.
    
    Commonly used to extract the CLS token (index 0) for classification
    in Vision Transformer models.
    """
    
    def __init__(
        self, 
        sid, 
        LayerName, 
        EnclaveMode,
        index=0,
        dim=1,
        link_prev=True, 
        link_next=True,
        manually_register_prev=False, 
        manually_register_next=False
    ):
        super().__init__(
            sid, LayerName, EnclaveMode, link_prev, link_next,
            manually_register_prev, manually_register_next
        )
        
        self.index = index  # Which element to extract
        self.dim = dim      # Along which dimension to slice
    
    def init_shape(self):
        """Initialize input/output shapes."""
        # Get input shape from previous layer
        self.InputShape = self.PrevLayer.get_output_shape()
        
        # Validate dimension
        if self.dim >= len(self.InputShape) or self.dim < -len(self.InputShape):
            raise ValueError(
                f"Slice dimension {self.dim} out of range for "
                f"input shape {self.InputShape}"
            )
        
        # Normalize negative dimension
        if self.dim < 0:
            self.dim = len(self.InputShape) + self.dim
        
        # Output shape: remove the sliced dimension
        self.OutputShape = list(self.InputShape)
        del self.OutputShape[self.dim]
        
        self.HandleShape = self.OutputShape
    
    def init(self, start_enclave=True):
        """Initialize the layer."""
        TensorLoader.init(self, start_enclave)
    
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
        Forward pass: extract token at specified index.
        
        For example, to extract CLS token from (B, N, D):
        - dim=1, index=0 -> output[:, 0, :] -> (B, D)
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
        input_tensor = self.get_cpu("input")
        
        # Slice along the specified dimension
        output = self._slice_tensor(input_tensor)
        
        self.set_cpu("output", output)
    
    def _forward_gpu(self):
        """Forward pass on GPU."""
        input_tensor = self.get_gpu("input")
        
        # Slice along the specified dimension
        output = self._slice_tensor(input_tensor)
        
        self.set_gpu("output", output)
    
    def _forward_enclave(self):
        """Forward pass in Enclave mode (currently uses CPU with transfer)."""
        # Transfer from enclave to CPU
        self.transfer_enclave_to_cpu("input")
        
        # Perform slicing on CPU
        input_tensor = self.get_cpu("input")
        output = self._slice_tensor(input_tensor)
        
        self.set_cpu("output", output)
        
        # Transfer back to enclave
        self.transfer_from_cpu("output")
    
    def _slice_tensor(self, tensor):
        """
        Slice tensor along specified dimension at specified index.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Sliced tensor with one dimension removed
        """
        # Create slice object for all dimensions
        slices = [slice(None)] * len(tensor.shape)
        
        # Set the slice for the target dimension
        slices[self.dim] = self.index
        
        # Apply slicing
        return tensor[tuple(slices)]
    
    def backward(self):
        """Backward pass (not implemented for inference)."""
        pass
    
    def print_connection_info(self):
        """Print layer connection information."""
        prev_name = self.PrevLayer.LayerName if self.PrevLayer else "None"
        next_name = self.NextLayer.LayerName if self.NextLayer else "None"
        print(
            f"{self.LayerName:20} shape {self.InputShape} -> {self.OutputShape} "
            f"(slice dim={self.dim}, idx={self.index})"
            f"{' ':5} input {prev_name:20} output {next_name:20}"
        )
