"""
SGX Layer implementations for DNN inference.

This module provides layer implementations that can execute in both
CPU mode and SGX Enclave mode.
"""

# Base layers
from python.layers.base import SecretLayerBase
from python.layers.nonlinear import SecretNonlinearLayer
from python.layers.activation import SecretActivationLayer

# Input/Output layers
from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer

# Linear layers
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.sgx_conv_base import SGXConvBase

# Normalization layers
from python.layers.batch_norm_1d import SecretBatchNorm1dLayer
from python.layers.batch_norm_2d import SecretBatchNorm2dLayer
from python.layers.layer_norm import SecretLayerNormLayer

# Activation layers
from python.layers.relu import SecretReLULayer
from python.layers.gelu import SecretGELULayer
from python.layers.quant_relu import SecretEnclaveQuantReLULayer

# Pooling layers
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.layers.avgpool2d import SecretAvgpool2dLayer

# Tensor operations
from python.layers.add import SecretAddLayer
from python.layers.weighted_add import SecretWeightedAddLayer
from python.layers.concatenate import SecretConcatenateLayer
from python.layers.flatten import SecretFlattenLayer
from python.layers.identity import SecretIdentityLayer

# Transformer-specific layers
from python.layers.softmax import SecretSoftmaxLayer
from python.layers.matmul import SecretMatMulLayer
from python.layers.scale import SecretScaleLayer
from python.layers.reshape import SecretReshapeLayer

# Vision Transformer-specific layers
from python.layers.cls_token import SecretCLSTokenLayer
from python.layers.position_embedding import SecretPositionEmbeddingLayer
from python.layers.slice import SecretSliceLayer

# Video Swin Transformer 3D layers
from python.layers.sgx_conv3d_base import SGXConv3DBase
from python.layers.window_partition_3d import SecretWindowPartition3DLayer
from python.layers.window_reverse_3d import SecretWindowReverse3DLayer
from python.layers.cyclic_roll_3d import SecretCyclicRoll3DLayer
from python.layers.swin_window_attention_3d import (
    SwinWindowAttention3D,
    create_swin_window_attention_3d
)

__all__ = [
    # Base
    'SecretLayerBase',
    'SecretNonlinearLayer',
    'SecretActivationLayer',
    # Input/Output
    'SecretInputLayer',
    'SecretOutputLayer',
    # Linear
    'SGXLinearBase',
    'SGXConvBase',
    'SGXConv3DBase',
    # Normalization
    'SecretBatchNorm1dLayer',
    'SecretBatchNorm2dLayer',
    'SecretLayerNormLayer',
    # Activation
    'SecretReLULayer',
    'SecretGELULayer',
    'SecretEnclaveQuantReLULayer',
    # Pooling
    'SecretMaxpool2dLayer',
    'SecretAvgpool2dLayer',
    # Tensor operations
    'SecretAddLayer',
    'SecretWeightedAddLayer',
    'SecretConcatenateLayer',
    'SecretFlattenLayer',
    'SecretIdentityLayer',
    # Transformer
    'SecretSoftmaxLayer',
    'SecretMatMulLayer',
    'SecretScaleLayer',
    'SecretReshapeLayer',
    # Vision Transformer
    'SecretCLSTokenLayer',
    'SecretPositionEmbeddingLayer',
    'SecretSliceLayer',
    # Video Swin Transformer 3D
    'SecretWindowPartition3DLayer',
    'SecretWindowReverse3DLayer',
    'SecretCyclicRoll3DLayer',
    'SwinWindowAttention3D',
    'create_swin_window_attention_3d',
]

