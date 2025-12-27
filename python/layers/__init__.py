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
]

