"""
3D Convolution layer for Video Swin Transformer in SGX.

Extends SGXConvBase to support 3D convolution operations for video inputs.
Key differences from 2D:
- Input: (B, C, D, H, W) - includes temporal dimension D
- Kernel: (Kd, Kh, Kw) - 3D filter
- SGX format: (B, D, H, W, C) instead of (B, H, W, C)
"""

from python.layers.base import SecretLayerBase
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.sgx_net import LearnableParamTuple
from python.utils.torch_utils import compare_expected_actual
from python.utils.basic_utils import ExecutionModeOptions
from python.global_config import SecretConfig

import torch
from pdb import set_trace as st


def calc_conv3d_output_shape_stride(input_shape, weight_shape, padding, stride):
    """
    Calculate output shape for 3D convolution.
    
    Args:
        input_shape: [B, C_in, D, H, W]
        weight_shape: [C_out, C_in, Kd, Kh, Kw]
        padding: int or tuple (pd, ph, pw)
        stride: int or tuple (sd, sh, sw)
    
    Returns:
        output_shape: [B, C_out, D', H', W']
    """
    B, C_in, D, H, W = input_shape
    C_out, _, Kd, Kh, Kw = weight_shape
    
    # Handle padding
    if isinstance(padding, int):
        pd = ph = pw = padding
    else:
        pd, ph, pw = padding
    
    # Handle stride
    if isinstance(stride, int):
        sd = sh = sw = stride
    else:
        sd, sh, sw = stride
    
    D_out = (D + 2 * pd - Kd) // sd + 1
    H_out = (H + 2 * ph - Kh) // sh + 1
    W_out = (W + 2 * pw - Kw) // sw + 1
    
    return [B, C_out, D_out, H_out, W_out]


class SGXConv3DBase(SecretLayerBase):
    """
    3D Convolution layer for Video Swin Transformer.
    
    Supports video inputs with temporal dimension.
    
    PyTorch format: (B, C, D, H, W)
    SGX format: (B, D, H, W, C)
    Weight PyTorch: (C_out, C_in, Kd, Kh, Kw)
    Weight SGX: (C_out, Kd, Kh, Kw, C_in)
    """
    
    batch_size = None
    pytorch_x_shape, sgx_x_shape = None, None
    pytorch_w_shape, sgx_w_shape = None, None
    bias_shape = None
    pytorch_y_shape, sgx_y_shape = None, None
    
    def __init__(
        self, sid, LayerName, EnclaveMode,
        n_output_channel, filter_dhw, stride, padding,
        batch_size=None, n_input_channel=None,
        video_dhw=None, bias=True,
        is_enclave_mode=False, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False
    ):
        """
        Initialize 3D Convolution layer.
        
        Args:
            filter_dhw: tuple (Kd, Kh, Kw) or int (cubic kernel)
            video_dhw: tuple (D, H, W) - input video dimensions
            stride: tuple (sd, sh, sw) or int
            padding: tuple (pd, ph, pw) or int
        """
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next,
                        manually_register_prev, manually_register_next)
        
        self.ForwardFuncName = "SGXConv3D"
        self.BackwardFuncName = "DerSGXConv3D"
        self.PlainFunc = torch.nn.Conv3d
        self.is_enclave_mode = is_enclave_mode
        
        self.batch_size = batch_size
        self.n_input_channel = n_input_channel
        self.n_output_channel = n_output_channel
        
        # Handle filter size
        if isinstance(filter_dhw, int):
            self.filter_dhw = (filter_dhw, filter_dhw, filter_dhw)
        else:
            self.filter_dhw = tuple(filter_dhw)
        
        # Handle video dimensions
        if isinstance(video_dhw, int):
            self.video_dhw = (video_dhw, video_dhw, video_dhw)
        elif video_dhw is not None:
            self.video_dhw = tuple(video_dhw)
        else:
            self.video_dhw = None
        
        # Handle stride
        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = tuple(stride)
        
        # Handle padding
        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = tuple(padding)
        
        self.bias = bias
        
        if EnclaveMode in [ExecutionModeOptions.CPU, ExecutionModeOptions.GPU]:
            self.ForwardFunc = torch.nn.Conv3d
    
    def init_shape(self):
        """Initialize shapes for 3D convolution."""
        if self.batch_size is None and self.PrevLayer is not None:
            self.pytorch_x_shape = self.PrevLayer.get_output_shape()
            self.batch_size = self.pytorch_x_shape[0]
            self.n_input_channel = self.pytorch_x_shape[1]
            self.video_dhw = tuple(self.pytorch_x_shape[2:5])
        else:
            D, H, W = self.video_dhw
            self.pytorch_x_shape = [self.batch_size, self.n_input_channel, D, H, W]
        
        # SGX format: BDHWC instead of BCDHW
        B, C, D, H, W = self.pytorch_x_shape
        self.sgx_x_shape = [B, D, H, W, C]
        
        # Weight shape
        # PyTorch: (out, in, d, h, w)
        Kd, Kh, Kw = self.filter_dhw
        self.pytorch_w_shape = [self.n_output_channel, self.n_input_channel, Kd, Kh, Kw]
        
        # SGX format: (out, d, h, w, in)
        self.sgx_w_shape = [self.n_output_channel, Kd, Kh, Kw, self.n_input_channel]
        
        # Output shape
        self.pytorch_y_shape = calc_conv3d_output_shape_stride(
            self.pytorch_x_shape, self.pytorch_w_shape,
            self.padding, self.stride
        )
        
        # SGX output: BDHWC
        B, C_out, D_out, H_out, W_out = self.pytorch_y_shape
        self.sgx_y_shape = [B, D_out, H_out, W_out, C_out]
        
        self.bias_shape = [self.n_output_channel]
        
        self.LearnableParamsList = [
            LearnableParamTuple(dw_name="DerWeight", w_name="weight", shape=self.sgx_w_shape),
            LearnableParamTuple(dw_name="DerBias", w_name="bias", shape=self.bias_shape),
        ]
    
    def init(self, start_enclave=True):
        """Initialize layer and load weights."""
        TensorLoader.init(self, start_enclave)
        
        Kd, Kh, Kw = self.filter_dhw
        
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            # For Enclave mode, use CPU computation (3D conv not implemented in enclave yet)
            self.PlainFunc = self.PlainFunc(
                self.n_input_channel, self.n_output_channel, self.filter_dhw,
                stride=self.stride, padding=self.padding, bias=self.bias
            )
            
            weight_pytorch_form = self.PlainFunc.weight.data
            weight_tf_form = self.weight_pytorch2tf(weight_pytorch_form)
            self.get_cpu("weight").data.copy_(weight_tf_form)
            self.transfer_cpu_to_enclave("weight")
            
            # Bias
            if self.bias:
                bias_data = self.PlainFunc.bias.data
            else:
                bias_data = torch.zeros(self.bias_shape)
            self.get_cpu("bias").data.copy_(bias_data)
            self.transfer_cpu_to_enclave("bias")
            
        elif self.EnclaveMode in [ExecutionModeOptions.CPU, ExecutionModeOptions.GPU]:
            self.ForwardFunc = self.ForwardFunc(
                self.n_input_channel, self.n_output_channel, self.filter_dhw,
                stride=self.stride, padding=self.padding, bias=self.bias
            )
            self.PlainFunc = torch.nn.Conv3d(
                self.n_input_channel, self.n_output_channel, self.filter_dhw,
                stride=self.stride, padding=self.padding, bias=self.bias
            )
            
            self.ForwardFunc.weight.data.copy_(self.PlainFunc.weight.data)
            weight_pytorch_form = list(self.ForwardFunc.parameters())[0].data
            weight_tf_form = self.weight_pytorch2tf(weight_pytorch_form)
            
            if self.EnclaveMode is ExecutionModeOptions.CPU:
                self.set_cpu("weight", weight_tf_form)
                if self.bias:
                    self.ForwardFunc.bias.data.copy_(self.PlainFunc.bias.data)
                    bias_data = self.PlainFunc.bias.data
                    self.set_cpu("bias", bias_data)
            elif self.EnclaveMode is ExecutionModeOptions.GPU:
                self.set_gpu("weight", weight_tf_form)
                if self.bias:
                    self.ForwardFunc.bias.data.copy_(self.PlainFunc.bias.data)
                    bias_data = self.PlainFunc.bias.data
                    self.set_gpu("bias", bias_data)
                self.ForwardFunc.cuda()
    
    def link_tensors(self):
        """Link tensors between layers."""
        super().link_tensors()
    
    def get_output_shape(self):
        """Return output shape in PyTorch format."""
        return self.pytorch_y_shape
    
    def weight_pytorch2tf(self, weight_pytorch_form):
        """
        Convert weight from PyTorch to SGX format.
        PyTorch: (out, in, d, h, w)
        SGX: (out, d, h, w, in)
        """
        weight_tf_form = weight_pytorch_form.permute(0, 2, 3, 4, 1).contiguous()
        return weight_tf_form
    
    def weight_tf2pytorch(self, weight_tf_form):
        """
        Convert weight from SGX to PyTorch format.
        SGX: (out, d, h, w, in)
        PyTorch: (out, in, d, h, w)
        """
        weight_pytorch_form = weight_tf_form.permute(0, 4, 1, 2, 3).contiguous()
        return weight_pytorch_form
    
    def feature_pytorch2tf(self, tensor_pytorch_form):
        """
        Convert feature from PyTorch to SGX format.
        PyTorch: (b, c, d, h, w)
        SGX: (b, d, h, w, c)
        """
        tensor_tf_form = tensor_pytorch_form.permute(0, 2, 3, 4, 1).contiguous()
        return tensor_tf_form
    
    def feature_tf2pytorch(self, tensor_tf_form):
        """
        Convert feature from SGX to PyTorch format.
        SGX: (b, d, h, w, c)
        PyTorch: (b, c, d, h, w)
        """
        tensor_pytorch_form = tensor_tf_form.permute(0, 4, 1, 2, 3).contiguous()
        return tensor_pytorch_form
    
    def inject_params(self, params):
        """Inject parameters from a PyTorch Conv3d module."""
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            cpu_w = self.get_cpu("weight")
            weight_pytorch_form = params.weight.data
            weight_tf_form = self.weight_pytorch2tf(weight_pytorch_form)
            cpu_w.copy_(weight_tf_form)
            self.transfer_cpu_to_enclave("weight")
            
            # Bias
            assert (
                (self.bias and params.bias is not None) or
                (not self.bias and params.bias is None)
            )
            if self.bias:
                bias_data = params.bias.data
            else:
                bias_data = torch.zeros(self.n_output_channel)
            cpu_b = self.get_cpu("bias")
            cpu_b.copy_(bias_data)
            self.transfer_cpu_to_enclave("bias")
            
        elif self.EnclaveMode is ExecutionModeOptions.CPU:
            weight_pytorch_form = params.weight.data
            weight_tf_form = self.weight_pytorch2tf(weight_pytorch_form)
            self.get_cpu("weight").copy_(weight_tf_form)
            
            if self.bias:
                self.get_cpu("bias").copy_(params.bias.data)
            
            # Update ForwardFunc
            weight_tf_form = self.get_cpu("weight")
            weight_pytorch_form = self.weight_tf2pytorch(weight_tf_form)
            self.ForwardFunc.weight.data.copy_(weight_pytorch_form)
            
        elif self.EnclaveMode is ExecutionModeOptions.GPU:
            weight_pytorch_form = params.weight.data
            weight_tf_form = self.weight_pytorch2tf(weight_pytorch_form)
            self.get_gpu("weight").copy_(weight_tf_form)
            
            if self.bias:
                self.get_gpu("bias").copy_(params.bias.data)
            
            # Update ForwardFunc
            weight_tf_form = self.get_gpu("weight")
            weight_pytorch_form = self.weight_tf2pytorch(weight_tf_form)
            self.ForwardFunc.weight.data.copy_(weight_pytorch_form)
    
    def generate_tensor_name_list(self, force=False):
        """Generate list of tensors needed by this layer."""
        if not force and self.tensor_name_list:
            return
        
        NeededTensorNames = [
            ("output", self.pytorch_y_shape, None),
            ("sgx_output", self.sgx_y_shape, None),
            ("DerInput", self.pytorch_x_shape, None),
            ("sgx_DerInput", self.sgx_x_shape, None),
            ("input", self.pytorch_x_shape, None),
            ("sgx_input", self.sgx_x_shape, None),
            ("DerOutput", self.pytorch_y_shape, None),
            ("sgx_DerOutput", self.sgx_y_shape, None),
            ("weight", self.sgx_w_shape, None),
            ("bias", self.bias_shape, None),
        ]
        self.tensor_name_list = NeededTensorNames
    
    def forward(self):
        """Forward pass for 3D convolution."""
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            self.forward_tensor_transfer("input")
            
            if self.EnclaveMode == ExecutionModeOptions.Enclave:
                # Enclave mode: use CPU computation (3D conv not in enclave yet)
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Input Preprocess", verbose_level=VerboseLevel.LAYER):
                    if self.PrevLayer.EnclaveMode is ExecutionModeOptions.Enclave:
                        self.transfer_enclave_to_cpu("input")
                    input_pytorch_form = self.get_cpu("input")
                    
                    if torch.sum(input_pytorch_form.abs()) == 0:
                        # Input is zero - fill with random data for profiling
                        input_pytorch_form = torch.randn_like(input_pytorch_form)
                        self.set_cpu("input", input_pytorch_form)
                    
                    input_tf_form = self.feature_pytorch2tf(input_pytorch_form)
                    self.set_cpu("sgx_input", input_tf_form)
                    self.transfer_cpu_to_enclave("sgx_input")
                
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} CPU Conv3D", verbose_level=VerboseLevel.LAYER):
                    # Use CPU PyTorch Conv3d
                    weight_tf_form = self.get_cpu("weight")
                    weight_pytorch_form = self.weight_tf2pytorch(weight_tf_form)
                    self.PlainFunc.weight.data.copy_(weight_pytorch_form)
                    if self.bias:
                        bias_data = self.get_cpu("bias")
                        self.PlainFunc.bias.data.copy_(bias_data)
                    
                    output_pytorch_form = self.PlainFunc(input_pytorch_form)
                    self.set_cpu("output", output_pytorch_form)
                
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Output Postprocess", verbose_level=VerboseLevel.LAYER):
                    output_tf_form = self.feature_pytorch2tf(output_pytorch_form)
                    self.set_cpu("sgx_output", output_tf_form)
                    self.transfer_cpu_to_enclave("output")
                    self.transfer_cpu_to_enclave("sgx_output")
                    
            elif self.EnclaveMode == ExecutionModeOptions.CPU:
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Weight Transfer", verbose_level=VerboseLevel.LAYER):
                    self.forward_tensor_transfer()
                    weight_tf_form = self.get_cpu("weight")
                    weight_pytorch_form = self.weight_tf2pytorch(weight_tf_form)
                    self.ForwardFunc.weight.data.copy_(weight_pytorch_form)
                
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} CPU Conv3D Forward", verbose_level=VerboseLevel.LAYER):
                    self.set_cpu("output", self.ForwardFunc(self.get_cpu("input")))
                    
            elif self.EnclaveMode == ExecutionModeOptions.GPU:
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} Weight Transfer", verbose_level=VerboseLevel.LAYER):
                    self.forward_tensor_transfer()
                    weight_tf_form = self.get_gpu("weight")
                    weight_pytorch_form = self.weight_tf2pytorch(weight_tf_form)
                    self.ForwardFunc.weight.data.copy_(weight_pytorch_form)
                
                with NamedTimerInstance(f"  S{self.sid}: {self.LayerName} GPU Conv3D Forward", verbose_level=VerboseLevel.LAYER):
                    self.set_gpu("output", self.ForwardFunc(
                        self.get_gpu("input").type(SecretConfig.dtypeForCpuOp)
                    ))
    
    def plain_forward(self, NeedBackward=False):
        """Plain forward pass for verification."""
        if self.EnclaveMode == ExecutionModeOptions.Enclave:
            self.make_sure_cpu_is_latest("input")
            self.make_sure_cpu_is_latest("weight")
            if self.bias:
                self.make_sure_cpu_is_latest("bias")
            
            weight_tf_form = self.get_cpu("weight")
            weight_pytorch_form = self.weight_tf2pytorch(weight_tf_form)
            self.PlainFunc.weight.data.copy_(weight_pytorch_form)
            if self.bias:
                bias_data = self.get_cpu("bias")
                self.PlainFunc.bias.data.copy_(bias_data)
                
        elif self.EnclaveMode in [ExecutionModeOptions.CPU, ExecutionModeOptions.GPU]:
            self.make_sure_cpu_is_latest("input")
        
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} PlainForward"):
            self.PlainForwardResult = self.PlainFunc(self.get_cpu("input"))
    
    def show_plain_error_forward(self):
        """Show forward error compared to plain implementation."""
        err = compare_expected_actual(
            self.PlainForwardResult, self.get_cpu("output"),
            get_relative=True
        )
        print(f"S{self.sid}: {self.LayerName} Forward Error: {err}")
    
    def print_connection_info(self):
        """Print layer connection information."""
        prev_name = self.PrevLayer.LayerName if self.PrevLayer else "None"
        next_name = self.NextLayer.LayerName if self.NextLayer else "None"
        print(
            f"{self.LayerName:20} shape{self.pytorch_x_shape}{' ':20} "
            f"mode{self.EnclaveMode}{' ':20} "
            f"input {prev_name:20} output {next_name:20}"
        )
