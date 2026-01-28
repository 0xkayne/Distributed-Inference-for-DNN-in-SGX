"""
3D Window Reverse Layer for Video Swin Transformer.

Reverses the window partition operation to restore the original 3D feature map.
This is the inverse of SecretWindowPartition3DLayer.

Reference: Video Swin Transformer
https://github.com/SwinTransformer/Video-Swin-Transformer
"""

import torch
from pdb import set_trace as st

from python.layers.nonlinear import SecretNonlinearLayer
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.basic_utils import ExecutionModeOptions
from python.enclave_interfaces import GlobalTensor as gt


class SecretWindowReverse3DLayer(SecretNonlinearLayer):
    """
    Reverse window partition to restore 3D feature map.
    
    Transforms (num_windows*B, Wd*Wh*Ww, C) back to (B, D, H, W, C)
    
    This is the inverse operation of window partition, used after
    window attention to merge the windows back.
    
    Example:
        Input: (64, 98, 96) where 64 = 1 * 4 * 8 * 8, 98 = 2 * 7 * 7
        Output: (1, 8, 56, 56, 96) with window_size=(2, 7, 7)
    """
    
    def __init__(
        self,
        sid,
        LayerName,
        EnclaveMode,
        window_size=(2, 7, 7),  # (Wd, Wh, Ww)
        output_shape=None,  # (B, D, H, W, C) - must be provided
        link_prev=True,
        link_next=True,
        manually_register_prev=False,
        manually_register_next=False
    ):
        super().__init__(
            sid, LayerName, EnclaveMode, link_prev, link_next,
            manually_register_prev, manually_register_next
        )
        
        self.window_size = tuple(window_size)
        assert len(self.window_size) == 3, "Window size must be 3D (Wd, Wh, Ww)"
        
        self.target_output_shape = output_shape  # Will be set in init_shape if None
    
    def init_shape(self):
        """Initialize input/output shapes."""
        # Get input shape from previous layer
        self.InputShape = self.PrevLayer.get_output_shape()
        
        # Validate input is 3D: (num_windows*B, Wd*Wh*Ww, C)
        assert len(self.InputShape) == 3, \
            f"Input must be 3D (num_windows*B, tokens, C), got {self.InputShape}"
        
        num_windows_times_B, window_tokens, C = self.InputShape
        Wd, Wh, Ww = self.window_size
        
        # Validate window tokens match window size
        expected_tokens = Wd * Wh * Ww
        assert window_tokens == expected_tokens, \
            f"Window tokens {window_tokens} != {Wd}*{Wh}*{Ww}={expected_tokens}"
        
        # If output shape not provided, infer from input
        if self.target_output_shape is None:
            # We need additional context to infer B, D, H, W
            # For now, try to get from layer metadata or use heuristic
            # This is typically set explicitly when creating the layer
            raise ValueError(
                "output_shape must be provided for WindowReverse3D. "
                "Cannot infer (B, D, H, W) from input alone."
            )
        
        self.OutputShape = list(self.target_output_shape)
        B, D, H, W, C_out = self.OutputShape
        
        # Validate dimensions are compatible
        num_windows = (D // Wd) * (H // Wh) * (W // Ww)
        expected_input_batch = num_windows * B
        assert num_windows_times_B == expected_input_batch, \
            f"Input batch {num_windows_times_B} != expected {expected_input_batch}"
        assert C == C_out, f"Channel mismatch: input {C} != output {C_out}"
        
        self.HandleShape = self.OutputShape
        
        # Store for forward pass
        self.B = B
        self.D = D
        self.H = H
        self.W = W
    
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
        Forward pass: reverse window partition.
        
        Algorithm (inverse of window_partition):
        1. Reshape: (B * num_windows, Wd*Wh*Ww, C) 
                 -> (B, D//Wd, H//Wh, W//Ww, Wd, Wh, Ww, C)
        2. Permute: (B, D//Wd, Wd, H//Wh, Wh, W//Ww, Ww, C)
        3. Reshape: (B, D, H, W, C)
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
        output = self._reverse_windows(input_tensor)
        self.set_cpu("output", output)
    
    def _forward_gpu(self):
        """Forward pass on GPU."""
        input_tensor = self.get_gpu("input")
        output = self._reverse_windows(input_tensor)
        self.set_gpu("output", output)
    
    def _forward_enclave(self):
        """Forward pass in Enclave mode (currently uses CPU with transfer)."""
        # Transfer from enclave to CPU
        self.transfer_enclave_to_cpu("input")
        
        # Perform reverse on CPU
        input_tensor = self.get_cpu("input")
        output = self._reverse_windows(input_tensor)
        
        self.set_cpu("output", output)
        
        # Transfer back to enclave
        self.transfer_from_cpu("output")
    
    def _reverse_windows(self, windows):
        """
        Reverse window partition to restore original feature map.
        
        Args:
            windows: (num_windows*B, Wd*Wh*Ww, C)
        
        Returns:
            x: (B, D, H, W, C)
        """
        Wd, Wh, Ww = self.window_size
        B, D, H, W, C = self.B, self.D, self.H, self.W, self.OutputShape[-1]
        
        # Reshape from flat windows
        # (num_windows*B, Wd*Wh*Ww, C) -> (B, D//Wd, H//Wh, W//Ww, Wd, Wh, Ww, C)
        x = windows.view(B, D // Wd, H // Wh, W // Ww, Wd, Wh, Ww, -1)
        
        # Permute to restore original dimension order
        # (B, D//Wd, H//Wh, W//Ww, Wd, Wh, Ww, C)
        # -> (B, D//Wd, Wd, H//Wh, Wh, W//Ww, Ww, C)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        
        # Merge window dimensions back
        # (B, D//Wd, Wd, H//Wh, Wh, W//Ww, Ww, C) -> (B, D, H, W, C)
        x = x.view(B, D, H, W, -1)
        
        return x
    
    def backward(self):
        """Backward pass (not implemented for inference)."""
        pass
    
    def print_connection_info(self):
        """Print layer connection information."""
        prev_name = self.PrevLayer.LayerName if self.PrevLayer else "None"
        next_name = self.NextLayer.LayerName if self.NextLayer else "None"
        print(
            f"{self.LayerName:20} shape {self.InputShape} -> {self.OutputShape} "
            f"(window_size={self.window_size})"
            f"{' ':5} input {prev_name:20} output {next_name:20}"
        )
