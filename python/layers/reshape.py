"""
Reshape Layer for SGX Transformer inference.

Used in attention to reshape tensors between different views:
- (B, N, embed_dim) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
- And the reverse operation for concatenation

This is a zero-cost operation that just changes the view of the tensor.
"""

import torch
from pdb import set_trace as st

from python.layers.activation import SecretActivationLayer
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.torch_utils import compare_expected_actual
from python.utils.basic_utils import ExecutionModeOptions
from python.global_config import SecretConfig


class SecretReshapeLayer(SecretActivationLayer):
    """
    Reshape layer implementation for SGX.
    
    Performs: output = input.view(target_shape) or input.permute(dims)
    
    Modes:
    - 'view': Simple reshape (default)
    - 'permute': Transpose dimensions
    - 'view_permute': Reshape then transpose
    """
    
    def __init__(
        self, sid, LayerName, EnclaveMode,
        target_shape=None,  # Target shape for view (use -1 for inferred dim)
        permute_dims=None,  # Dimensions for permute
        mode='view',  # 'view', 'permute', or 'view_permute'
        link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False,
        merge_own_tensors=False
    ):
        super().__init__(
            sid, LayerName, EnclaveMode, link_prev, link_next,
            manually_register_prev, manually_register_next, merge_own_tensors
        )
        
        self.ForwardFuncName = "Reshape"
        self.BackwardFuncName = "DerReshape"
        
        self.target_shape = target_shape
        self.permute_dims = permute_dims
        self.mode = mode
        
        self.ForwardFunc = self._cpu_forward
    
    def init_shape(self):
        """Initialize shapes based on previous layer."""
        self.InputShape = list(self.PrevLayer.get_output_shape())
        
        if self.mode == 'view':
            # Compute output shape from target_shape
            output_shape = list(self.target_shape)
            # Replace -1 with computed value
            if -1 in output_shape:
                total_elements = 1
                for s in self.InputShape:
                    total_elements *= s
                known_elements = 1
                unknown_idx = -1
                for i, s in enumerate(output_shape):
                    if s == -1:
                        unknown_idx = i
                    else:
                        known_elements *= s
                output_shape[unknown_idx] = total_elements // known_elements
            self.OutputShape = output_shape
            
        elif self.mode == 'permute':
            # Permute dimensions
            self.OutputShape = [self.InputShape[d] for d in self.permute_dims]
            
        elif self.mode == 'view_permute':
            # First apply view, then permute
            temp_shape = list(self.target_shape)
            if -1 in temp_shape:
                total_elements = 1
                for s in self.InputShape:
                    total_elements *= s
                known_elements = 1
                unknown_idx = -1
                for i, s in enumerate(temp_shape):
                    if s == -1:
                        unknown_idx = i
                    else:
                        known_elements *= s
                temp_shape[unknown_idx] = total_elements // known_elements
            self.OutputShape = [temp_shape[d] for d in self.permute_dims]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
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
        
        if self.sid == 2:
            self.tensor_name_list = {}
            return
        
        NeededTensorNames = [
            ("input", self.InputShape, None),
            ("output", self.OutputShape, None),
        ]
        
        self.tensor_name_list = NeededTensorNames
    
    def _cpu_forward(self, input_tensor):
        """CPU-based forward computation."""
        if self.mode == 'view':
            return input_tensor.view(*self.OutputShape)
        elif self.mode == 'permute':
            return input_tensor.permute(*self.permute_dims).contiguous()
        elif self.mode == 'view_permute':
            # Apply view first
            temp_shape = list(self.target_shape)
            if -1 in temp_shape:
                # Recompute for runtime
                total_elements = input_tensor.numel()
                known_elements = 1
                unknown_idx = -1
                for i, s in enumerate(temp_shape):
                    if s == -1:
                        unknown_idx = i
                    else:
                        known_elements *= s
                temp_shape[unknown_idx] = total_elements // known_elements
            temp = input_tensor.view(*temp_shape)
            return temp.permute(*self.permute_dims).contiguous()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def forward(self):
        """Forward pass."""
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            if self.EnclaveMode == ExecutionModeOptions.Enclave:
                # For Enclave mode: transfer to CPU, compute, transfer back
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Enclave->CPU", verbose_level=VerboseLevel.LAYER):
                    self.forward_tensor_transfer()
                    self.transfer_enclave_to_cpu("input")
                
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} CPU Compute", verbose_level=VerboseLevel.LAYER):
                    input_data = self.get_cpu("input")
                    output = self._cpu_forward(input_data)
                    self.set_cpu("output", output)
                
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} CPU->Enclave", verbose_level=VerboseLevel.LAYER):
                    self.transfer_cpu_to_enclave("output")
                    
            elif self.EnclaveMode == ExecutionModeOptions.CPU:
                self.forward_tensor_transfer()
                input_data = self.get_cpu("input")
                self.set_cpu("output", self._cpu_forward(input_data))
                
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                self.forward_tensor_transfer()
                input_data = self.get_gpu("input")
                if self.mode == 'view':
                    output = input_data.view(*self.OutputShape)
                elif self.mode == 'permute':
                    output = input_data.permute(*self.permute_dims).contiguous()
                elif self.mode == 'view_permute':
                    temp_shape = list(self.target_shape)
                    if -1 in temp_shape:
                        total_elements = input_data.numel()
                        known_elements = 1
                        unknown_idx = -1
                        for i, s in enumerate(temp_shape):
                            if s == -1:
                                unknown_idx = i
                            else:
                                known_elements *= s
                        temp_shape[unknown_idx] = total_elements // known_elements
                    temp = input_data.view(*temp_shape)
                    output = temp.permute(*self.permute_dims).contiguous()
                self.set_gpu("output", output)
    
    def plain_forward(self, NeedBackward=False):
        """Plain forward for verification."""
        self.make_sure_cpu_is_latest("input")
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainForward"):
            self.PlainForwardResult = self._cpu_forward(self.get_cpu("input"))
    
    def show_plain_error_forward(self):
        """Show error between plain and SGX forward."""
        self.make_sure_cpu_is_latest("output")
        err = compare_expected_actual(
            self.PlainForwardResult, self.get_cpu("output"), 
            get_relative=False, show_values=False
        )
        print(f"S{self.sid}: {self.LayerName} Forward Error: {err}")


