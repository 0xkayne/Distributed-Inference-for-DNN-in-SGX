"""
SGX InceptionV3 Model with Parallel Branches.

Structure:
- Stem: Conv -> MaxPool
- Inception Block A (x2): 4 parallel branches
  - Branch1: 1x1
  - Branch2: 1x1 -> 3x3
  - Branch3: 1x1 -> 3x3 -> 3x3
  - Branch4: Pool -> 1x1
- MaxPool (reduction)
- Flatten -> FC
"""

import sys
sys.path.insert(0, '.')

from typing import Dict, List, Optional
from python.layers.sgx_conv_base import SGXConvBase
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.relu import SecretReLULayer
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.layers.avgpool2d import SecretAvgpool2dLayer
from python.layers.flatten import SecretFlattenLayer
from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer
from python.layers.concatenate import SecretConcatenateLayer
from python.utils.basic_utils import ExecutionModeOptions


class InceptionBlock:
    """
    Inception Block with 4 parallel branches.
    All branches execute in parallel and merge at Concatenate layer.
    """
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        in_channels: int,
        # Output channels for each branch
        ch1x1: int,           # Branch 1
        ch3x3_red: int, ch3x3: int,  # Branch 2
        ch5x5_red: int, ch5x5: int,  # Branch 3 (simulated 5x5 via 3x3)
        pool_proj: int,       # Branch 4
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.layers = []
        overrides = layer_mode_overrides or {}
        
        def get_mode(name):
            return overrides.get(name, enclave_mode)

        # Branch 1: 1x1 conv
        b1_conv = SGXConvBase(sid, f"{name_prefix}_b1_1x1", get_mode(f"{name_prefix}_b1_1x1"),
                              n_output_channel=ch1x1, filter_hw=1, stride=1, padding=0,
                              manually_register_prev=True, manually_register_next=True)
        b1_relu = SecretReLULayer(sid, f"{name_prefix}_b1_relu", get_mode(f"{name_prefix}_b1_relu"),
                                  manually_register_prev=True, manually_register_next=True)
        self.layers.extend([b1_conv, b1_relu])

        # Branch 2: 1x1 -> 3x3
        b2_conv1 = SGXConvBase(sid, f"{name_prefix}_b2_1x1", get_mode(f"{name_prefix}_b2_1x1"),
                               n_output_channel=ch3x3_red, filter_hw=1, stride=1, padding=0,
                               manually_register_prev=True, manually_register_next=True)
        b2_relu1 = SecretReLULayer(sid, f"{name_prefix}_b2_relu1", get_mode(f"{name_prefix}_b2_relu1"),
                                   manually_register_prev=True, manually_register_next=True)
        b2_conv2 = SGXConvBase(sid, f"{name_prefix}_b2_3x3", get_mode(f"{name_prefix}_b2_3x3"),
                               n_output_channel=ch3x3, filter_hw=3, stride=1, padding=1,
                               manually_register_prev=True, manually_register_next=True)
        b2_relu2 = SecretReLULayer(sid, f"{name_prefix}_b2_relu2", get_mode(f"{name_prefix}_b2_relu2"),
                                   manually_register_prev=True, manually_register_next=True)
        self.layers.extend([b2_conv1, b2_relu1, b2_conv2, b2_relu2])

        # Branch 3: 1x1 -> 3x3 -> 3x3 (simulating 5x5)
        b3_conv1 = SGXConvBase(sid, f"{name_prefix}_b3_1x1", get_mode(f"{name_prefix}_b3_1x1"),
                               n_output_channel=ch5x5_red, filter_hw=1, stride=1, padding=0,
                               manually_register_prev=True, manually_register_next=True)
        b3_relu1 = SecretReLULayer(sid, f"{name_prefix}_b3_relu1", get_mode(f"{name_prefix}_b3_relu1"),
                                   manually_register_prev=True, manually_register_next=True)
        b3_conv2 = SGXConvBase(sid, f"{name_prefix}_b3_3x3_1", get_mode(f"{name_prefix}_b3_3x3_1"),
                               n_output_channel=ch5x5, filter_hw=3, stride=1, padding=1,
                               manually_register_prev=True, manually_register_next=True)
        b3_relu2 = SecretReLULayer(sid, f"{name_prefix}_b3_relu2", get_mode(f"{name_prefix}_b3_relu2"),
                                   manually_register_prev=True, manually_register_next=True)
        b3_conv3 = SGXConvBase(sid, f"{name_prefix}_b3_3x3_2", get_mode(f"{name_prefix}_b3_3x3_2"),
                               n_output_channel=ch5x5, filter_hw=3, stride=1, padding=1,
                               manually_register_prev=True, manually_register_next=True)
        b3_relu3 = SecretReLULayer(sid, f"{name_prefix}_b3_relu3", get_mode(f"{name_prefix}_b3_relu3"),
                                   manually_register_prev=True, manually_register_next=True)
        self.layers.extend([b3_conv1, b3_relu1, b3_conv2, b3_relu2, b3_conv3, b3_relu3])

        # Branch 4: Pool -> 1x1
        b4_pool = SecretMaxpool2dLayer(sid, f"{name_prefix}_b4_pool", get_mode(f"{name_prefix}_b4_pool"),
                                       filter_hw=3, stride=1, padding=1,
                                       manually_register_prev=True, manually_register_next=True)
        b4_conv = SGXConvBase(sid, f"{name_prefix}_b4_1x1", get_mode(f"{name_prefix}_b4_1x1"),
                              n_output_channel=pool_proj, filter_hw=1, stride=1, padding=0,
                              manually_register_prev=True, manually_register_next=True)
        b4_relu = SecretReLULayer(sid, f"{name_prefix}_b4_relu", get_mode(f"{name_prefix}_b4_relu"),
                                  manually_register_prev=True, manually_register_next=True)
        self.layers.extend([b4_pool, b4_conv, b4_relu])

        # Concatenate
        self.concat = SecretConcatenateLayer(
            sid, f"{name_prefix}_concat", get_mode(f"{name_prefix}_concat"),
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.concat)

        # Store branch heads and tails for connection
        self.branch_heads = [b1_conv, b2_conv1, b3_conv1, b4_pool]
        self.branch_tails = [b1_relu, b2_relu2, b3_relu3, b4_relu]

    def connect(self, prev_layer):
        # Connect all branch heads to prev_layer
        for head in self.branch_heads:
            head.register_prev_layer(prev_layer)

        # Connect internal layers
        # Branch 1
        self.layers[1].register_prev_layer(self.layers[0]) # relu -> conv
        
        # Branch 2
        self.layers[3].register_prev_layer(self.layers[2]) # relu1 -> conv1
        self.layers[4].register_prev_layer(self.layers[3]) # conv2 -> relu1
        self.layers[5].register_prev_layer(self.layers[4]) # relu2 -> conv2
        
        # Branch 3
        self.layers[7].register_prev_layer(self.layers[6]) # relu1 -> conv1
        self.layers[8].register_prev_layer(self.layers[7]) # conv2 -> relu1
        self.layers[9].register_prev_layer(self.layers[8]) # relu2 -> conv2
        self.layers[10].register_prev_layer(self.layers[9]) # conv3 -> relu2
        self.layers[11].register_prev_layer(self.layers[10]) # relu3 -> conv3
        
        # Branch 4
        self.layers[13].register_prev_layer(self.layers[12]) # conv -> pool
        self.layers[14].register_prev_layer(self.layers[13]) # relu -> conv

        # Connect all branch tails to concat
        for tail in self.branch_tails:
            self.concat.register_prev_layer(tail)

        return self.concat


class SGXInceptionV3:
    def __init__(self, sid=0, num_classes=10, enclave_mode=ExecutionModeOptions.Enclave,
                 batch_size=1, input_size=96, layer_mode_overrides=None):
        self.sid = sid
        self.num_classes = num_classes
        self.enclave_mode = enclave_mode
        self.batch_size = batch_size
        self.input_size = input_size
        self.layer_mode_overrides = layer_mode_overrides or {}
        self.layers = []
        self.model_name = "InceptionV3"
        
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

        # Stem: Conv -> ReLU -> MaxPool
        conv1 = SGXConvBase(sid, "conv1", self._get_mode("conv1"), n_output_channel=32,
                            filter_hw=3, stride=2, padding=0, manually_register_prev=True, manually_register_next=True)
        conv1.register_prev_layer(input_layer)
        
        relu1 = SecretReLULayer(sid, "relu1", self._get_mode("relu1"), manually_register_prev=True, manually_register_next=True)
        relu1.register_prev_layer(conv1)
        
        pool1 = SecretMaxpool2dLayer(sid, "pool1", self._get_mode("pool1"), filter_hw=3, stride=2, padding=0,
                                     manually_register_prev=True, manually_register_next=True)
        pool1.register_prev_layer(relu1)
        
        self.layers.extend([conv1, relu1, pool1])
        
        # Inception Block 1
        # 32 -> 64+128+32+32 = 256 channels
        block1 = InceptionBlock(sid, "inc1", self.enclave_mode, in_channels=32,
                                ch1x1=64, ch3x3_red=96, ch3x3=128, ch5x5_red=16, ch5x5=32, pool_proj=32,
                                layer_mode_overrides=self.layer_mode_overrides)
        output1 = block1.connect(pool1)
        self.layers.extend(block1.layers)
        
        # Inception Block 2
        # 256 -> 128+192+96+64 = 480 channels
        block2 = InceptionBlock(sid, "inc2", self.enclave_mode, in_channels=256,
                                ch1x1=128, ch3x3_red=128, ch3x3=192, ch5x5_red=32, ch5x5=96, pool_proj=64,
                                layer_mode_overrides=self.layer_mode_overrides)
        output2 = block2.connect(output1)
        self.layers.extend(block2.layers)
        
        # Classifier
        # Adaptive AvgPool
        final_size = self.input_size // 4 # approximate
        avgpool = SecretAvgpool2dLayer(sid, "avgpool", self._get_mode("avgpool"),
                                       filter_hw=max(1, final_size), stride=1, padding=0,
                                       manually_register_prev=True, manually_register_next=True)
        avgpool.register_prev_layer(output2)
        
        flatten = SecretFlattenLayer(sid, "flatten", self._get_mode("flatten"),
                                     manually_register_prev=True, manually_register_next=True)
        flatten.register_prev_layer(avgpool)
        
        fc = SGXLinearBase(sid, "fc", self._get_mode("fc"), batch_size=self.batch_size,
                           n_output_features=self.num_classes, n_input_features=480,
                           manually_register_prev=True, manually_register_next=True)
        fc.register_prev_layer(flatten)
        
        output = SecretOutputLayer(sid, "output", self._get_mode("output"), inference=True,
                                   manually_register_prev=True)
        output.register_prev_layer(fc)
        
        self.layers.extend([avgpool, flatten, fc, output])

    def print_architecture(self):
        print(f"SGXInceptionV3 with {len(self.layers)} layers")
        for l in self.layers:
            print(f"{l.LayerName:20} {type(l).__name__:20} {l.EnclaveMode.name}")

def test_inception():
    model = SGXInceptionV3(layer_mode_overrides={"input": ExecutionModeOptions.CPU})
    model.print_architecture()

if __name__ == "__main__":
    test_inception()
