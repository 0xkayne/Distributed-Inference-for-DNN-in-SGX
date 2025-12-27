"""
GELU (Gaussian Error Linear Unit) Layer for SGX Transformer inference.

GELU is the default activation function in Transformer FFN blocks.

Exact formula: GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
Approximate formula (faster):
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

Reference: "Gaussian Error Linear Units (GELUs)" by Hendrycks and Gimpel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pdb import set_trace as st

from python.layers.activation import SecretActivationLayer
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.torch_utils import compare_expected_actual
from python.utils.basic_utils import ExecutionModeOptions
from python.global_config import SecretConfig


class SecretGELULayer(SecretActivationLayer):
    """
    GELU activation layer implementation for SGX.
    
    Supports both exact and approximate GELU computation.
    For Enclave mode, uses CPU computation with data transfer until native 
    Enclave implementation is available.
    """
    
    # Constants for approximate GELU
    SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
    GELU_COEFF = 0.044715
    
    def __init__(
        self, sid, LayerName, EnclaveMode,
        approximate=True,  # Use fast approximation (default in most Transformers)
        link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False,
        merge_own_tensors=False
    ):
        super().__init__(
            sid, LayerName, EnclaveMode, link_prev, link_next,
            manually_register_prev, manually_register_next, merge_own_tensors
        )
        
        self.ForwardFuncName = "GELU"
        self.BackwardFuncName = "DerGELU"
        self.approximate = approximate
        
        if approximate:
            self.PlainFunc = lambda x: F.gelu(x, approximate='tanh')
        else:
            self.PlainFunc = F.gelu
        
        # Use CPU-based computation for now
        self.ForwardFunc = self._cpu_forward
    
    def init_shape(self):
        """Initialize shapes based on previous layer."""
        self.InputShape = self.PrevLayer.get_output_shape()
        self.OutputShape = self.InputShape
        self.HandleShape = self.InputShape
    
    def init(self, start_enclave=True):
        """Initialize the layer."""
        TensorLoader.init(self, start_enclave)
        
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            # TODO: Initialize native enclave GELU when implemented
            pass
        elif self.EnclaveMode is ExecutionModeOptions.GPU:
            # Nothing special needed for GPU
            pass
    
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
    
    def _gelu_approximate(self, x):
        """
        Fast approximate GELU using tanh.
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        """
        return 0.5 * x * (1.0 + torch.tanh(
            self.SQRT_2_OVER_PI * (x + self.GELU_COEFF * x.pow(3))
        ))
    
    def _gelu_exact(self, x):
        """
        Exact GELU using error function.
        GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
    def _cpu_forward(self, input_tensor):
        """CPU-based forward computation."""
        if self.approximate:
            return self._gelu_approximate(input_tensor)
        else:
            return self._gelu_exact(input_tensor)
    
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
                if self.approximate:
                    output = F.gelu(input_data, approximate='tanh')
                else:
                    output = F.gelu(input_data)
                self.set_cpu("output", output)
                
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                self.forward_tensor_transfer()
                input_data = self.get_gpu("input")
                if self.approximate:
                    output = F.gelu(input_data, approximate='tanh')
                else:
                    output = F.gelu(input_data)
                self.set_gpu("output", output)
    
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


