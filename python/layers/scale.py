"""
Scale Layer for SGX Transformer inference.

Simple element-wise multiplication by a scalar value.
Used in attention: Q @ K^T * (1/sqrt(d_k))
"""

import torch
from pdb import set_trace as st

from python.layers.activation import SecretActivationLayer
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.torch_utils import compare_expected_actual
from python.utils.basic_utils import ExecutionModeOptions
from python.global_config import SecretConfig


class SecretScaleLayer(SecretActivationLayer):
    """
    Scale layer implementation for SGX.
    
    Performs: output = input * scale_factor
    """
    
    def __init__(
        self, sid, LayerName, EnclaveMode,
        scale_factor=1.0,
        link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False,
        merge_own_tensors=False
    ):
        super().__init__(
            sid, LayerName, EnclaveMode, link_prev, link_next,
            manually_register_prev, manually_register_next, merge_own_tensors
        )
        
        self.ForwardFuncName = "Scale"
        self.BackwardFuncName = "DerScale"
        self.scale_factor = scale_factor
        self.PlainFunc = lambda x: x * scale_factor
        
        self.ForwardFunc = self._cpu_forward
    
    def init_shape(self):
        """Initialize shapes based on previous layer."""
        self.InputShape = self.PrevLayer.get_output_shape()
        self.OutputShape = self.InputShape
        self.HandleShape = self.InputShape
    
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
        return input_tensor * self.scale_factor
    
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
                    output = input_data * self.scale_factor
                    self.set_cpu("output", output)
                
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} CPU->Enclave", verbose_level=VerboseLevel.LAYER):
                    self.transfer_cpu_to_enclave("output")
                    
            elif self.EnclaveMode == ExecutionModeOptions.CPU:
                self.forward_tensor_transfer()
                input_data = self.get_cpu("input")
                self.set_cpu("output", input_data * self.scale_factor)
                
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                self.forward_tensor_transfer()
                input_data = self.get_gpu("input")
                self.set_gpu("output", input_data * self.scale_factor)
    
    def plain_forward(self, NeedBackward=False):
        """Plain forward for verification."""
        self.make_sure_cpu_is_latest("input")
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainForward"):
            self.PlainForwardResult = self.PlainFunc(self.get_cpu("input"))
    
    def show_plain_error_forward(self):
        """Show error between plain and SGX forward."""
        self.make_sure_cpu_is_latest("output")
        err = compare_expected_actual(
            self.PlainForwardResult, self.get_cpu("output"), 
            get_relative=False, show_values=False
        )
        print(f"S{self.sid}: {self.LayerName} Forward Error: {err}")



