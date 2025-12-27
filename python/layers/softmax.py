"""
Softmax Layer for SGX Transformer inference.

Softmax computes: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
The max-subtraction is used for numerical stability.

Used in Transformer attention: Attention = softmax(Q @ K^T / sqrt(d_k))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as st

from python.layers.activation import SecretActivationLayer
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.torch_utils import compare_expected_actual
from python.utils.basic_utils import ExecutionModeOptions
from python.global_config import SecretConfig


class SecretSoftmaxLayer(SecretActivationLayer):
    """
    Softmax layer implementation for SGX.
    
    Supports softmax along a specified dimension (default: -1 for last dimension).
    For Enclave mode, uses CPU computation with data transfer until native 
    Enclave implementation is available.
    """
    
    def __init__(
        self, sid, LayerName, EnclaveMode,
        dim=-1,  # Dimension along which to apply softmax
        link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False,
        merge_own_tensors=False
    ):
        super().__init__(
            sid, LayerName, EnclaveMode, link_prev, link_next,
            manually_register_prev, manually_register_next, merge_own_tensors
        )
        
        self.ForwardFuncName = "Softmax"
        self.BackwardFuncName = "DerSoftmax"
        self.PlainFunc = lambda x: F.softmax(x, dim=self.dim)
        
        self.dim = dim
        
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
            # TODO: Initialize native enclave Softmax when implemented
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
    
    def _cpu_forward(self, input_tensor):
        """CPU-based forward computation with numerical stability."""
        return F.softmax(input_tensor, dim=self.dim)
    
    def _stable_softmax(self, x, dim=-1):
        """
        Numerically stable softmax computation.
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        """
        max_val = x.max(dim=dim, keepdim=True)[0]
        exp_x = torch.exp(x - max_val)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)
    
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
                    output = self._stable_softmax(input_data, dim=self.dim)
                    self.set_cpu("output", output)
                
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} CPU->Enclave", verbose_level=VerboseLevel.LAYER):
                    self.transfer_cpu_to_enclave("output")
                    
            elif self.EnclaveMode == ExecutionModeOptions.CPU:
                self.forward_tensor_transfer()
                input_data = self.get_cpu("input")
                self.set_cpu("output", F.softmax(input_data, dim=self.dim))
                
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                self.forward_tensor_transfer()
                input_data = self.get_gpu("input")
                self.set_gpu("output", F.softmax(input_data, dim=self.dim))
    
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


