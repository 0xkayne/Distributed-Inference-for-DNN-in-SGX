"""
Mini Inception Model for Distributed Inference Verification.

Structure:
        Input
          |
    +-----+-----+
    |           |
  Branch1     Branch2
  (Conv)      (Conv)
    |           |
    +-----+-----+
          |
       Concat
          |
         FC
          |
        Output
"""

import sys
sys.path.insert(0, '.')

from typing import Dict, Optional
from python.layers.sgx_conv_base import SGXConvBase
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer
from python.layers.concatenate import SecretConcatenateLayer
from python.utils.basic_utils import ExecutionModeOptions


class SGXMiniInception:
    def __init__(self, sid=0, num_classes=10, enclave_mode=ExecutionModeOptions.Enclave,
                 batch_size=1, input_size=32, layer_mode_overrides=None):
        self.sid = sid
        self.num_classes = num_classes
        self.enclave_mode = enclave_mode
        self.batch_size = batch_size
        self.input_size = input_size
        self.layer_mode_overrides = layer_mode_overrides or {}
        self.layers = []
        self.model_name = "MiniInception"
        
        self._build_network()

    def _get_mode(self, name):
        return self.layer_mode_overrides.get(name, self.enclave_mode)

    def _build_network(self):
        sid = self.sid
        
        # Input
        input_layer = SecretInputLayer(sid, "input", 
                                       [self.batch_size, 3, self.input_size, self.input_size],
                                       self._get_mode("input"), manually_register_next=True)
        self.layers.append(input_layer)

        # Branch 1: Conv 3x3
        b1_conv = SGXConvBase(sid, "branch1_conv", self._get_mode("branch1_conv"), 
                              n_output_channel=16, filter_hw=3, stride=1, padding=1, 
                              manually_register_prev=True, manually_register_next=True)
        b1_conv.register_prev_layer(input_layer)
        self.layers.append(b1_conv)

        # Branch 2: Conv 1x1 -> Conv 3x3 (Simulate heavier load)
        b2_conv1 = SGXConvBase(sid, "branch2_conv1", self._get_mode("branch2_conv1"), 
                               n_output_channel=16, filter_hw=1, stride=1, padding=0, 
                               manually_register_prev=True, manually_register_next=True)
        b2_conv1.register_prev_layer(input_layer)
        self.layers.append(b2_conv1)
        
        b2_conv2 = SGXConvBase(sid, "branch2_conv2", self._get_mode("branch2_conv2"), 
                               n_output_channel=16, filter_hw=3, stride=1, padding=1, 
                               manually_register_prev=True, manually_register_next=True)
        b2_conv2.register_prev_layer(b2_conv1)
        self.layers.append(b2_conv2)

        # Concatenate
        concat = SecretConcatenateLayer(
            sid, "concat", self._get_mode("concat"),
            manually_register_prev=True, manually_register_next=True, dim=1
        )
        concat.register_prev_layer(b1_conv)
        concat.register_prev_layer(b2_conv2)
        self.layers.append(concat)
        
        # FC Layer (16+16=32 channels * input_size * input_size)
        # Flatten is implicit in SGXLinearBase if input is 4D? No, usually need explicit flatten.
        # Let's add Flatten.
        from python.layers.flatten import SecretFlattenLayer
        flatten = SecretFlattenLayer(sid, "flatten", self._get_mode("flatten"),
                                     manually_register_prev=True, manually_register_next=True)
        flatten.register_prev_layer(concat)
        self.layers.append(flatten)
        
        fc_input_dim = 32 * self.input_size * self.input_size
        fc = SGXLinearBase(sid, "fc", self._get_mode("fc"), batch_size=self.batch_size,
                           n_output_features=self.num_classes, n_input_features=fc_input_dim,
                           manually_register_prev=True, manually_register_next=True)
        fc.register_prev_layer(flatten)
        self.layers.append(fc)
        
        # Output
        output = SecretOutputLayer(sid, "output", self._get_mode("output"), inference=True,
                                   manually_register_prev=True)
        output.register_prev_layer(fc)
        self.layers.append(output)

    def print_architecture(self):
        print(f"SGXMiniInception with {len(self.layers)} layers")
        for l in self.layers:
            print(f"{l.LayerName:20} {type(l).__name__:20} {l.EnclaveMode.name}")

