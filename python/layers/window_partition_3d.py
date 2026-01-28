"""
3D Window Partition Layer for Video Swin Transformer.

Partitions a 3D feature map into non-overlapping 3D windows.
This is a key operation in Video Swin Transformer for efficient local attention.

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
from functools import reduce
from operator import mul


class SecretWindowPartition3DLayer(SecretNonlinearLayer):
    """
    Partition 3D feature map into non-overlapping windows.
    
    Transforms (B, D, H, W, C) into (num_windows*B, Wd*Wh*Ww, C)
    where num_windows = (D//Wd) * (H//Wh) * (W//Ww)
    
    This enables efficient local attention within windows instead of global attention.
    
    Example:
        Input: (1, 8, 56, 56, 96) with window_size=(2, 7, 7)
        Output: (64, 98, 96) where 64 = 1 * (8//2) * (56//7) * (56//7)
                                   98 = 2 * 7 * 7
    """
    
    def __init__(
        self,
        sid,
        LayerName,
        EnclaveMode,
        window_size=(2, 7, 7),  # (Wd, Wh, Ww)
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
    
    def init_shape(self):
        """Initialize input/output shapes."""
        # Get input shape from previous layer
        self.InputShape = self.PrevLayer.get_output_shape()
        
        # Validate input is 5D: (B, D, H, W, C)
        assert len(self.InputShape) == 5, \
            f"Input must be 5D (B, D, H, W, C), got {self.InputShape}"
        
        B, D, H, W, C = self.InputShape
        Wd, Wh, Ww = self.window_size
        
        # Validate dimensions are divisible by window size
        assert D % Wd == 0, f"D={D} must be divisible by window_size[0]={Wd}"
        assert H % Wh == 0, f"H={H} must be divisible by window_size[1]={Wh}"
        assert W % Ww == 0, f"W={W} must be divisible by window_size[2]={Ww}"
        
        # Calculate number of windows
        num_windows = (D // Wd) * (H // Wh) * (W // Ww)
        window_tokens = Wd * Wh * Ww
        
        # Output: (num_windows*B, Wd*Wh*Ww, C)
        self.OutputShape = [num_windows * B, window_tokens, C]
        self.HandleShape = self.OutputShape
        
        # Store for forward pass
        self.num_windows = num_windows
        self.window_tokens = window_tokens
    
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
        Forward pass: partition into windows.
        
        Algorithm (from Video Swin Transformer):
        1. Reshape: (B, D, H, W, C) -> (B, D//Wd, Wd, H//Wh, Wh, W//Ww, Ww, C)
        2. Permute: (B, D//Wd, H//Wh, W//Ww, Wd, Wh, Ww, C)
        3. Reshape: (B * num_windows, Wd*Wh*Ww, C)
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
        output = self._partition_windows(input_tensor)
        self.set_cpu("output", output)
    
    def _forward_gpu(self):
        """Forward pass on GPU."""
        input_tensor = self.get_gpu("input")
        output = self._partition_windows(input_tensor)
        self.set_gpu("output", output)
    
    def _forward_enclave(self):
        """Forward pass in Enclave mode (currently uses CPU with transfer)."""
        # Transfer from enclave to CPU
        self.transfer_enclave_to_cpu("input")
        
        # Perform partitioning on CPU
        input_tensor = self.get_cpu("input")
        output = self._partition_windows(input_tensor)
        
        self.set_cpu("output", output)
        
        # Transfer back to enclave
        self.transfer_from_cpu("output")
    
    def _partition_windows(self, x):
        """
        Partition feature map into windows.
        
        Args:
            x: (B, D, H, W, C)
        
        Returns:
            windows: (num_windows*B, Wd*Wh*Ww, C)
        """
        B, D, H, W, C = x.shape
        Wd, Wh, Ww = self.window_size
        
        # Reshape to separate windows
        # (B, D, H, W, C) -> (B, D//Wd, Wd, H//Wh, Wh, W//Ww, Ww, C)
        x = x.view(B, D // Wd, Wd, H // Wh, Wh, W // Ww, Ww, C)
        
        # Permute to group window dimensions together
        # (B, D//Wd, Wd, H//Wh, Wh, W//Ww, Ww, C)
        # -> (B, D//Wd, H//Wh, W//Ww, Wd, Wh, Ww, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        
        # Flatten to get windows
        # (B, D//Wd, H//Wh, W//Ww, Wd, Wh, Ww, C)
        # -> (B * num_windows, Wd*Wh*Ww, C)
        windows = windows.view(-1, reduce(mul, self.window_size), C)
        
        return windows
    
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
