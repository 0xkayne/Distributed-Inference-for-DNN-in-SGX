"""
3D Cyclic Roll Layer for Video Swin Transformer.

Performs cyclic shift of a 3D feature map along specified dimensions.
This is used in Shifted Window (SW-MSA) attention to enable cross-window connections.

Reference: Video Swin Transformer
https://github.com/SwinTransformer/Video-Swin-Transformer
Line 335-340: torch.roll for cyclic shift
"""

import torch
from pdb import set_trace as st

from python.layers.nonlinear import SecretNonlinearLayer
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.basic_utils import ExecutionModeOptions
from python.enclave_interfaces import GlobalTensor as gt


class SecretCyclicRoll3DLayer(SecretNonlinearLayer):
    """
    Cyclic roll (shift) operation for 3D tensors.
    
    Uses torch.roll to cyclically shift the tensor along specified dimensions.
    This is a key operation for Shifted Window Multi-Head Self-Attention (SW-MSA).
    
    In SW-MSA, the feature map is shifted by (window_size//2) before partitioning
    to create cross-window connections.
    
    Example:
        Input: (B, D, H, W, C) = (1, 8, 56, 56, 96)
        shifts: (-1, -3, -3) - shift by half window (2//2, 7//2, 7//2)
        dims: (1, 2, 3) - shift along D, H, W dimensions
        Output: (1, 8, 56, 56, 96) - cyclically shifted
    """
    
    def __init__(
        self,
        sid,
        LayerName,
        EnclaveMode,
        shifts=(0, -3, -3),  # Shift amounts for (D, H, W)
        dims=(1, 2, 3),      # Dimensions to shift (1=D, 2=H, 3=W for BDHWC)
        link_prev=True,
        link_next=True,
        manually_register_prev=False,
        manually_register_next=False
    ):
        super().__init__(
            sid, LayerName, EnclaveMode, link_prev, link_next,
            manually_register_prev, manually_register_next
        )
        
        self.shifts = tuple(shifts)
        self.dims = tuple(dims)
        
        assert len(self.shifts) == len(self.dims), \
            "shifts and dims must have same length"
    
    def init_shape(self):
        """Initialize input/output shapes."""
        # Get input shape from previous layer
        self.InputShape = self.PrevLayer.get_output_shape()
        
        # Output shape is same as input (roll doesn't change shape)
        self.OutputShape = self.InputShape
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
        Forward pass: cyclic roll.
        
        Uses torch.roll(x, shifts, dims) to perform cyclic shift.
        For example:
        - roll(x, shifts=(-3, -3), dims=(2, 3)) shifts H and W by -3
        - Negative shifts move elements towards beginning
        - Positive shifts move elements towards end
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
                self._forward_enclave()
    
    def _forward_cpu(self):
        """Forward pass on CPU."""
        input_tensor = self.get_cpu("input")
        output = self._cyclic_roll(input_tensor)
        self.set_cpu("output", output)
    
    def _forward_gpu(self):
        """Forward pass on GPU."""
        input_tensor = self.get_gpu("input")
        output = self._cyclic_roll(input_tensor)
        self.set_gpu("output", output)
    
    def _forward_enclave(self):
        """
        Forward pass in Enclave mode.
        
        Note: torch.roll is not available in enclave, so we use CPU computation.
        For production, this could be implemented in C++ for the enclave.
        """
        # Transfer from enclave to CPU
        self.transfer_enclave_to_cpu("input")
        
        # Perform roll on CPU
        input_tensor = self.get_cpu("input")
        output = self._cyclic_roll(input_tensor)
        
        self.set_cpu("output", output)
        
        # Transfer back to enclave
        self.transfer_from_cpu("output")
    
    def _cyclic_roll(self, x):
        """
        Perform cyclic roll using torch.roll.
        
        Args:
            x: Input tensor
        
        Returns:
            Cyclically shifted tensor
        """
        # Check if any shift is non-zero
        if all(s == 0 for s in self.shifts):
            # No shift needed
            return x
        
        # Use torch.roll for cyclic shift
        return torch.roll(x, shifts=self.shifts, dims=self.dims)
    
    def backward(self):
        """Backward pass (not implemented for inference)."""
        pass
    
    def print_connection_info(self):
        """Print layer connection information."""
        prev_name = self.PrevLayer.LayerName if self.PrevLayer else "None"
        next_name = self.NextLayer.LayerName if self.NextLayer else "None"
        print(
            f"{self.LayerName:20} shape {self.InputShape} -> {self.OutputShape} "
            f"(shifts={self.shifts}, dims={self.dims})"
            f"{' ':5} input {prev_name:20} output {next_name:20}"
        )
