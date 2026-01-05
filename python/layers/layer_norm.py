"""
LayerNorm Layer for SGX Transformer inference.

LayerNorm normalizes across the feature dimension (last dimension),
unlike BatchNorm which normalizes across the batch dimension.

Formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
where mean and var are computed across the last dimension.
"""

import torch
import torch.nn as nn
from pdb import set_trace as st

from python.layers.activation import SecretActivationLayer
from python.tensor_loader import TensorLoader
from python.sgx_net import LearnableParamTuple
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.utils.torch_utils import compare_expected_actual
from python.utils.basic_utils import ExecutionModeOptions
from python.global_config import SecretConfig


class SecretLayerNormLayer(SecretActivationLayer):
    """
    LayerNorm layer implementation for SGX.
    
    Supports both CPU and Enclave execution modes.
    For Enclave mode, falls back to CPU computation with data transfer
    until native Enclave implementation is available.
    """
    
    def __init__(
        self, sid, LayerName, EnclaveMode, 
        normalized_shape=None,  # int or list of ints for the last dimensions
        eps=1e-5,
        elementwise_affine=True,
        link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False,
        merge_own_tensors=False
    ):
        super().__init__(
            sid, LayerName, EnclaveMode, link_prev, link_next,
            manually_register_prev, manually_register_next, merge_own_tensors
        )
        
        self.ForwardFuncName = "LayerNorm"
        self.BackwardFuncName = "DerLayerNorm"
        self.PlainFunc = nn.LayerNorm
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        # Weight and bias shapes
        self.WeightShape = None
        self.BiasShape = None
        
        # For now, use CPU-based computation for all modes
        # Native Enclave implementation will be added later
        self.ForwardFunc = self._cpu_forward
    
    def init_shape(self):
        """Initialize shapes based on previous layer."""
        self.InputShape = self.PrevLayer.get_output_shape()
        self.OutputShape = self.InputShape
        self.HandleShape = self.InputShape
        
        # Determine normalized_shape from input if not specified
        if self.normalized_shape is None:
            # Default: normalize over the last dimension
            self.normalized_shape = [self.InputShape[-1]]
        elif isinstance(self.normalized_shape, int):
            self.normalized_shape = [self.normalized_shape]
        
        # Weight and bias have shape of normalized_shape
        self.WeightShape = list(self.normalized_shape)
        self.BiasShape = list(self.normalized_shape)
        
        if self.elementwise_affine:
            self.LearnableParamsList = [
                LearnableParamTuple(dw_name="DerWeight", w_name="weight", shape=self.WeightShape),
                LearnableParamTuple(dw_name="DerBias", w_name="bias", shape=self.BiasShape),
            ]
    
    def init(self, start_enclave=True):
        """Initialize the layer."""
        TensorLoader.init(self, start_enclave)
        
        # Create PyTorch LayerNorm for reference/CPU execution
        self.PlainFunc = nn.LayerNorm(
            self.normalized_shape, 
            eps=self.eps, 
            elementwise_affine=self.elementwise_affine
        )
        
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            # Initialize weights in enclave
            if self.elementwise_affine:
                self.get_cpu("weight").data.copy_(self.PlainFunc.weight.data)
                self.get_cpu("bias").data.copy_(self.PlainFunc.bias.data)
                self.transfer_cpu_to_enclave("weight")
                self.transfer_cpu_to_enclave("bias")
            
            # Calculate batch and seq_len from InputShape
            # InputShape is typically [batch, seq_len, embed_dim] or [batch * seq_len, embed_dim]
            if len(self.InputShape) == 3:
                batch_size = self.InputShape[0]
                seq_len = self.InputShape[1]
                embed_dim = self.InputShape[2]
            elif len(self.InputShape) == 2:
                # Flattened [batch * seq_len, embed_dim]
                batch_size = 1
                seq_len = self.InputShape[0]
                embed_dim = self.InputShape[1]
            else:
                raise ValueError(f"Unsupported InputShape for LayerNorm: {self.InputShape}")
            
            # Initialize native enclave LayerNorm
            self.layernorm_init(
                self.LayerName,
                "input", "output", "weight", "bias",
                batch_size, seq_len, embed_dim,
                self.eps
            )
        else:
            # CPU or GPU mode
            self.ForwardFuncModule = nn.LayerNorm(
                self.normalized_shape,
                eps=self.eps,
                elementwise_affine=self.elementwise_affine
            )
            self.ForwardFuncModule.weight.data.copy_(self.PlainFunc.weight.data)
            self.ForwardFuncModule.bias.data.copy_(self.PlainFunc.bias.data)
            
            if self.EnclaveMode is ExecutionModeOptions.GPU:
                self.ForwardFuncModule = self.ForwardFuncModule.cuda()
    
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
        
        if self.elementwise_affine:
            NeededTensorNames.extend([
                ("weight", self.WeightShape, None),
                ("bias", self.BiasShape, None),
            ])
        
        self.tensor_name_list = NeededTensorNames
    
    def _cpu_forward(self, input_tensor):
        """CPU-based forward computation."""
        return self.ForwardFuncModule(input_tensor)
    
    def forward(self):
        """Forward pass."""
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            if self.EnclaveMode == ExecutionModeOptions.Enclave:
                # Native Enclave execution
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Input Preprocess", verbose_level=VerboseLevel.LAYER):
                    self.forward_tensor_transfer()
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} layernorm_forward", verbose_level=VerboseLevel.LAYER):
                    self.layernorm_forward(self.LayerName)
                    
            elif self.EnclaveMode == ExecutionModeOptions.CPU:
                self.forward_tensor_transfer()
                input_data = self.get_cpu("input")
                self.set_cpu("output", self.ForwardFuncModule(input_data))
                
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                self.forward_tensor_transfer()
                input_data = self.get_gpu("input")
                self.set_gpu("output", self.ForwardFuncModule(input_data))
    
    def inject_params(self, params):
        """Inject parameters from a PyTorch module."""
        if not self.elementwise_affine:
            return
            
        if self.EnclaveMode in [ExecutionModeOptions.CPU, ExecutionModeOptions.Enclave]:
            self.get_cpu("weight").copy_(params.weight.data)
            self.get_cpu("bias").copy_(params.bias.data)
            if self.EnclaveMode is ExecutionModeOptions.Enclave:
                self.transfer_cpu_to_enclave("weight")
                self.transfer_cpu_to_enclave("bias")
        elif self.EnclaveMode is ExecutionModeOptions.GPU:
            self.get_gpu("weight").copy_(params.weight.data)
            self.get_gpu("bias").copy_(params.bias.data)
    
    def plain_forward(self, NeedBackward=False):
        """Plain forward for verification."""
        self.make_sure_cpu_is_latest("input")
        if self.elementwise_affine:
            self.make_sure_cpu_is_latest("weight")
            self.make_sure_cpu_is_latest("bias")
            self.PlainFunc.weight.data.copy_(self.get_cpu("weight"))
            self.PlainFunc.bias.data.copy_(self.get_cpu("bias"))
        
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



