"""
MatMul (Batch Matrix Multiplication) Layer for SGX Transformer inference.

This layer performs batch matrix multiplication for attention computation:
- Q @ K^T: (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
- Attention @ V: (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)

Unlike Linear layer, MatMul has no learnable parameters.
It requires two input tensors from two previous layers.
"""

import torch
from pdb import set_trace as st

from python.layers.nonlinear import SecretNonlinearLayer
from python.tensor_loader import TensorLoader
from python.enclave_interfaces import GlobalTensor as gt
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.torch_utils import compare_expected_actual
from python.utils.basic_utils import ExecutionModeOptions
from python.global_config import SecretConfig


class SecretMatMulLayer(SecretNonlinearLayer):
    """
    Batch Matrix Multiplication layer for SGX.
    
    Performs: output = input1 @ input2
    where input1 and input2 are tensors from two previous layers.
    
    For attention: 
    - Q @ K^T: set transpose_b=True
    - Attention @ V: set transpose_b=False
    """
    
    def __init__(
        self, sid, LayerName, EnclaveMode,
        transpose_a=False,  # Transpose first input
        transpose_b=False,  # Transpose second input
        scale=None,  # Optional scaling factor (e.g., 1/sqrt(d_k) for attention)
        link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False
    ):
        super().__init__(
            sid, LayerName, EnclaveMode, link_prev, link_next,
            manually_register_prev, manually_register_next
        )
        
        self.ForwardFuncName = "MatMul"
        self.BackwardFuncName = "DerMatMul"
        
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.scale = scale
        
        # Two input layers for matrix multiplication
        self.PrevLayer = []
        assert link_prev
    
    def register_prev_layer(self, layer):
        """Register a previous layer (needs exactly 2)."""
        if layer not in self.PrevLayer:
            self.PrevLayer.append(layer)
    
    def init_shape(self):
        """Initialize shapes based on previous layers."""
        assert len(self.PrevLayer) == 2, f"MatMul requires exactly 2 input layers, got {len(self.PrevLayer)}"
        
        shape1 = list(self.PrevLayer[0].get_output_shape())
        shape2 = list(self.PrevLayer[1].get_output_shape())
        
        # Apply transpose if needed
        if self.transpose_a:
            shape1 = shape1[:-2] + [shape1[-1], shape1[-2]]
        if self.transpose_b:
            shape2 = shape2[:-2] + [shape2[-1], shape2[-2]]
        
        # Compute output shape for batch matmul: (..., M, K) @ (..., K, N) -> (..., M, N)
        # Batch dimensions should match (or be broadcastable)
        batch_dims = shape1[:-2]
        M = shape1[-2]
        K1 = shape1[-1]
        K2 = shape2[-2]
        N = shape2[-1]
        
        assert K1 == K2, f"Inner dimensions must match: {K1} vs {K2}"
        
        self.InputShape1 = list(self.PrevLayer[0].get_output_shape())
        self.InputShape2 = list(self.PrevLayer[1].get_output_shape())
        self.InputShape = self.InputShape1  # For compatibility
        self.OutputShape = batch_dims + [M, N]
        self.HandleShape = self.OutputShape
    
    def init(self, start_enclave=True):
        """Initialize the layer."""
        TensorLoader.init(self, start_enclave)
    
    def get_output_shape(self):
        return self.OutputShape
    
    def generate_tensor_name_list(self, force=False):
        """Generate list of tensors needed for this layer."""
        if not force and self.tensor_name_list:
            return
        
        NeededTensorNames = [
            ("input1", self.InputShape1, None),
            ("input2", self.InputShape2, None),
            ("input", self.InputShape1, None),  # Alias for compatibility
            ("output", self.OutputShape, None),
        ]
        
        self.tensor_name_list = NeededTensorNames
    
    def link_tensors(self):
        """Link input/output tensors with previous/next layers."""
        if self.link_prev and self.PrevLayer is not None:
            gt.link_tags(self.get_tag("input1", remap=False), self.PrevLayer[0].get_tag("output", remap=False))
            gt.link_tags(self.get_tag("input2", remap=False), self.PrevLayer[1].get_tag("output", remap=False))
        if self.link_next and self.NextLayer is not None:
            gt.link_tags(self.get_tag("output", remap=False), self.NextLayer.get_tag("input", remap=False))
    
    def _matmul_cpu(self, a, b):
        """Perform batch matrix multiplication on CPU."""
        if self.transpose_a:
            a = a.transpose(-2, -1)
        if self.transpose_b:
            b = b.transpose(-2, -1)
        
        result = torch.matmul(a, b)
        
        if self.scale is not None:
            result = result * self.scale
        
        return result
    
    def forward(self):
        """Forward pass."""
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            assert self.PrevLayer[0] is not None and self.PrevLayer[1] is not None
            
            prev0_mode = self.PrevLayer[0].EnclaveMode
            prev1_mode = self.PrevLayer[1].EnclaveMode
            
            if prev0_mode == ExecutionModeOptions.Enclave and prev1_mode == ExecutionModeOptions.Enclave:
                # Both inputs from Enclave
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Enclave->CPU", verbose_level=VerboseLevel.LAYER):
                    self.transfer_enclave_to_cpu("input1")
                    self.transfer_enclave_to_cpu("input2")
                
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} CPU Compute", verbose_level=VerboseLevel.LAYER):
                    input1 = self.get_cpu("input1")
                    input2 = self.get_cpu("input2")
                    output = self._matmul_cpu(input1, input2)
                    self.set_cpu("output", output)
                
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} CPU->Enclave", verbose_level=VerboseLevel.LAYER):
                    self.transfer_cpu_to_enclave("output")
                    
            elif prev0_mode == ExecutionModeOptions.CPU and prev1_mode == ExecutionModeOptions.CPU:
                # Both inputs from CPU
                input1 = self.get_cpu("input1")
                input2 = self.get_cpu("input2")
                output = self._matmul_cpu(input1, input2)
                self.set_cpu("output", output)
                
            elif prev0_mode == ExecutionModeOptions.GPU and prev1_mode == ExecutionModeOptions.GPU:
                # Both inputs from GPU
                input1 = self.get_gpu("input1")
                input2 = self.get_gpu("input2")
                
                if self.transpose_a:
                    input1 = input1.transpose(-2, -1)
                if self.transpose_b:
                    input2 = input2.transpose(-2, -1)
                
                output = torch.matmul(input1, input2)
                if self.scale is not None:
                    output = output * self.scale
                
                self.set_gpu("output", output)
                
            else:
                # Mixed modes - transfer all to CPU
                if prev0_mode == ExecutionModeOptions.Enclave:
                    self.transfer_enclave_to_cpu("input1")
                elif prev0_mode == ExecutionModeOptions.GPU:
                    self.transfer_gpu_to_cpu("input1")
                    
                if prev1_mode == ExecutionModeOptions.Enclave:
                    self.transfer_enclave_to_cpu("input2")
                elif prev1_mode == ExecutionModeOptions.GPU:
                    self.transfer_gpu_to_cpu("input2")
                
                input1 = self.get_cpu("input1")
                input2 = self.get_cpu("input2")
                output = self._matmul_cpu(input1, input2)
                self.set_cpu("output", output)
                
                # Transfer back if needed
                if self.EnclaveMode == ExecutionModeOptions.Enclave:
                    self.transfer_cpu_to_enclave("output")
    
    def backward(self):
        """Backward pass (not implemented for inference-only)."""
        raise NotImplementedError("MatMul backward not implemented")
    
    def forward_tensor_transfer(self):
        """Handle tensor transfers between execution modes."""
        if self.PrevLayer[0] is not None and self.PrevLayer[0].EnclaveMode == ExecutionModeOptions.Enclave:
            if self.EnclaveMode != ExecutionModeOptions.Enclave:
                self.transfer_enclave_to_cpu("input1")
        if self.PrevLayer[1] is not None and self.PrevLayer[1].EnclaveMode == ExecutionModeOptions.Enclave:
            if self.EnclaveMode != ExecutionModeOptions.Enclave:
                self.transfer_enclave_to_cpu("input2")
    
    def print_connection_info(self):
        """Print layer connection information."""
        prev_names = [p.LayerName for p in self.PrevLayer] if self.PrevLayer else []
        next_name = self.NextLayer.LayerName if self.NextLayer else "None"
        print(f"{self.LayerName:20} shape{self.OutputShape} mode{self.EnclaveMode} input {prev_names} output {next_name}")


