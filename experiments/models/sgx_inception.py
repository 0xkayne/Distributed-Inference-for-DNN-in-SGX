"""
SGX InceptionV3 Model - Standard Implementation.

Standard Inception V3 Architecture:
- Stem: Conv 3x3 stride2 -> Conv 3x3 -> Conv 3x3 padded -> MaxPool -> Conv 1x1 -> Conv 3x3 -> MaxPool
- Inception-A (x3): 4 parallel branches with concatenation
- Reduction-A (x1): Special stride=2 reduction module
- Inception-B (x4): 4 parallel branches with concatenation
- Reduction-B (x1): Special stride=2 reduction module
- Inception-C (x2): 4 parallel branches with concatenation
- Global Average Pooling
- Fully Connected Layer (1000 classes)

Reference: Rethinking the Inception Architecture for Computer Vision (Szegedy et al., 2016)
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


class ReductionBlock:
    """
    Reduction Block for Inception V3.
    Special Inception module with stride=2 in some branches for downsampling.
    
    Structure (based on Reduction-A):
    - Branch 1: 3x3 conv stride=2
    - Branch 2: 1x1 -> 3x3 -> 3x3 stride=2
    - Branch 3: 3x3 maxpool stride=2
    """
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        in_channels: int,
        # Output channels for each branch
        ch3x3: int,           # Branch 1: 3x3 stride=2
        ch3x3_red: int, ch3x3_2: int,  # Branch 2: 1x1 -> 3x3 -> 3x3 stride=2
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        self.layers = []
        overrides = layer_mode_overrides or {}
        
        def get_mode(name):
            return overrides.get(name, enclave_mode)

        # Branch 1: 3x3 conv stride=2
        b1_conv = SGXConvBase(sid, f"{name_prefix}_b1_3x3", get_mode(f"{name_prefix}_b1_3x3"),
                              n_output_channel=ch3x3, filter_hw=3, stride=2, padding=0,
                              manually_register_prev=True, manually_register_next=True)
        b1_relu = SecretReLULayer(sid, f"{name_prefix}_b1_relu", get_mode(f"{name_prefix}_b1_relu"),
                                  manually_register_prev=True, manually_register_next=True)
        self.layers.extend([b1_conv, b1_relu])

        # Branch 2: 1x1 -> 3x3 -> 3x3 stride=2
        b2_conv1 = SGXConvBase(sid, f"{name_prefix}_b2_1x1", get_mode(f"{name_prefix}_b2_1x1"),
                               n_output_channel=ch3x3_red, filter_hw=1, stride=1, padding=0,
                               manually_register_prev=True, manually_register_next=True)
        b2_relu1 = SecretReLULayer(sid, f"{name_prefix}_b2_relu1", get_mode(f"{name_prefix}_b2_relu1"),
                                   manually_register_prev=True, manually_register_next=True)
        b2_conv2 = SGXConvBase(sid, f"{name_prefix}_b2_3x3", get_mode(f"{name_prefix}_b2_3x3"),
                               n_output_channel=ch3x3_red, filter_hw=3, stride=1, padding=1,
                               manually_register_prev=True, manually_register_next=True)
        b2_relu2 = SecretReLULayer(sid, f"{name_prefix}_b2_relu2", get_mode(f"{name_prefix}_b2_relu2"),
                                   manually_register_prev=True, manually_register_next=True)
        b2_conv3 = SGXConvBase(sid, f"{name_prefix}_b2_3x3_stride2", get_mode(f"{name_prefix}_b2_3x3_stride2"),
                               n_output_channel=ch3x3_2, filter_hw=3, stride=2, padding=0,
                               manually_register_prev=True, manually_register_next=True)
        b2_relu3 = SecretReLULayer(sid, f"{name_prefix}_b2_relu3", get_mode(f"{name_prefix}_b2_relu3"),
                                   manually_register_prev=True, manually_register_next=True)
        self.layers.extend([b2_conv1, b2_relu1, b2_conv2, b2_relu2, b2_conv3, b2_relu3])

        # Branch 3: 3x3 maxpool stride=2
        b3_pool = SecretMaxpool2dLayer(sid, f"{name_prefix}_b3_pool", get_mode(f"{name_prefix}_b3_pool"),
                                       filter_hw=3, stride=2, padding=0,
                                       manually_register_prev=True, manually_register_next=True)
        self.layers.append(b3_pool)

        # Concatenate
        self.concat = SecretConcatenateLayer(
            sid, f"{name_prefix}_concat", get_mode(f"{name_prefix}_concat"),
            manually_register_prev=True, manually_register_next=True
        )
        self.layers.append(self.concat)

        # Store branch heads and tails for connection
        self.branch_heads = [b1_conv, b2_conv1, b3_pool]
        self.branch_tails = [b1_relu, b2_relu3, b3_pool]

    def connect(self, prev_layer):
        # Connect all branch heads to prev_layer
        for head in self.branch_heads:
            head.register_prev_layer(prev_layer)

        # Connect internal layers
        # Branch 1
        self.layers[1].register_prev_layer(self.layers[0])  # relu -> conv
        
        # Branch 2
        self.layers[3].register_prev_layer(self.layers[2])  # relu1 -> conv1
        self.layers[4].register_prev_layer(self.layers[3])  # conv2 -> relu1
        self.layers[5].register_prev_layer(self.layers[4])  # relu2 -> conv2
        self.layers[6].register_prev_layer(self.layers[5])  # conv3 -> relu2
        self.layers[7].register_prev_layer(self.layers[6])  # relu3 -> conv3

        # Connect all branch tails to concat
        for tail in self.branch_tails:
            self.concat.register_prev_layer(tail)

        return self.concat


class SGXInceptionV3:
    """
    Standard Inception V3 Model Implementation for SGX.
    
    Complete architecture following the standard Inception V3 design:
    - Stem: 7 layers (conv + pool operations)
    - 3x Inception-A modules
    - 1x Reduction-A module
    - 4x Inception-B modules
    - 1x Reduction-B module
    - 2x Inception-C modules
    - Global Average Pooling
    - Fully Connected Layer (1000 classes)
    """
    def __init__(self, sid=0, num_classes=1000, enclave_mode=ExecutionModeOptions.Enclave,
                 batch_size=1, input_size=299, layer_mode_overrides=None):
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
        """Build complete standard Inception V3 network."""
        sid = self.sid
        
        # ========== Input Layer ==========
        input_layer = SecretInputLayer(sid, "input", 
                                       [self.batch_size, 3, self.input_size, self.input_size],
                                       self._get_mode("input"), manually_register_next=True)
        self.layers.append(input_layer)

        # ========== Stem: Standard 7-layer stem ==========
        # Conv 3x3 stride=2
        stem_conv1 = SGXConvBase(sid, "stem_conv1", self._get_mode("stem_conv1"), 
                                 n_output_channel=32, n_input_channel=3,
                                 filter_hw=3, stride=2, padding=0,
                                 batch_size=self.batch_size, img_hw=self.input_size,
                                 manually_register_prev=True, manually_register_next=True)
        stem_conv1.register_prev_layer(input_layer)
        stem_relu1 = SecretReLULayer(sid, "stem_relu1", self._get_mode("stem_relu1"),
                                     manually_register_prev=True, manually_register_next=True)
        stem_relu1.register_prev_layer(stem_conv1)
        
        # Conv 3x3
        stem_conv2 = SGXConvBase(sid, "stem_conv2", self._get_mode("stem_conv2"),
                                 n_output_channel=32, filter_hw=3, stride=1, padding=0,
                                 manually_register_prev=True, manually_register_next=True)
        stem_conv2.register_prev_layer(stem_relu1)
        stem_relu2 = SecretReLULayer(sid, "stem_relu2", self._get_mode("stem_relu2"),
                                     manually_register_prev=True, manually_register_next=True)
        stem_relu2.register_prev_layer(stem_conv2)
        
        # Conv 3x3 padded
        stem_conv3 = SGXConvBase(sid, "stem_conv3", self._get_mode("stem_conv3"),
                                 n_output_channel=64, filter_hw=3, stride=1, padding=1,
                                 manually_register_prev=True, manually_register_next=True)
        stem_conv3.register_prev_layer(stem_relu2)
        stem_relu3 = SecretReLULayer(sid, "stem_relu3", self._get_mode("stem_relu3"),
                                     manually_register_prev=True, manually_register_next=True)
        stem_relu3.register_prev_layer(stem_conv3)
        
        # MaxPool
        stem_pool1 = SecretMaxpool2dLayer(sid, "stem_pool1", self._get_mode("stem_pool1"),
                                          filter_hw=3, stride=2, padding=0,
                                          manually_register_prev=True, manually_register_next=True)
        stem_pool1.register_prev_layer(stem_relu3)
        
        # Conv 1x1
        stem_conv4 = SGXConvBase(sid, "stem_conv4", self._get_mode("stem_conv4"),
                                 n_output_channel=80, filter_hw=1, stride=1, padding=0,
                                 manually_register_prev=True, manually_register_next=True)
        stem_conv4.register_prev_layer(stem_pool1)
        stem_relu4 = SecretReLULayer(sid, "stem_relu4", self._get_mode("stem_relu4"),
                                     manually_register_prev=True, manually_register_next=True)
        stem_relu4.register_prev_layer(stem_conv4)
        
        # Conv 3x3
        stem_conv5 = SGXConvBase(sid, "stem_conv5", self._get_mode("stem_conv5"),
                                 n_output_channel=192, filter_hw=3, stride=1, padding=0,
                                 manually_register_prev=True, manually_register_next=True)
        stem_conv5.register_prev_layer(stem_relu4)
        stem_relu5 = SecretReLULayer(sid, "stem_relu5", self._get_mode("stem_relu5"),
                                     manually_register_prev=True, manually_register_next=True)
        stem_relu5.register_prev_layer(stem_conv5)
        
        # MaxPool
        stem_pool2 = SecretMaxpool2dLayer(sid, "stem_pool2", self._get_mode("stem_pool2"),
                                          filter_hw=3, stride=2, padding=0,
                                          manually_register_prev=True, manually_register_next=True)
        stem_pool2.register_prev_layer(stem_relu5)
        
        self.layers.extend([stem_conv1, stem_relu1, stem_conv2, stem_relu2, 
                           stem_conv3, stem_relu3, stem_pool1, stem_conv4, 
                           stem_relu4, stem_conv5, stem_relu5, stem_pool2])
        
        current_output = stem_pool2
        
        # ========== Inception-A (x3) ==========
        # Inception-A: 192 -> 256 channels (64+64+96+32)
        for i in range(3):
            inc_a = InceptionBlock(sid, f"inception_a{i+1}", self.enclave_mode, in_channels=192,
                                   ch1x1=64, ch3x3_red=64, ch3x3=96, ch5x5_red=48, ch5x5=64, pool_proj=32,
                                   layer_mode_overrides=self.layer_mode_overrides)
            current_output = inc_a.connect(current_output)
            self.layers.extend(inc_a.layers)
        
        # ========== Reduction-A ==========
        # Reduction-A: 256 -> 768 channels with stride=2
        red_a = ReductionBlock(sid, "reduction_a", self.enclave_mode, in_channels=256,
                              ch3x3=384, ch3x3_red=192, ch3x3_2=384,
                              layer_mode_overrides=self.layer_mode_overrides)
        current_output = red_a.connect(current_output)
        self.layers.extend(red_a.layers)
        
        # ========== Inception-B (x4) ==========
        # Inception-B: 768 -> 768 channels (192+192+192+192)
        for i in range(4):
            inc_b = InceptionBlock(sid, f"inception_b{i+1}", self.enclave_mode, in_channels=768,
                                   ch1x1=192, ch3x3_red=128, ch3x3=192, ch5x5_red=128, ch5x5=192, pool_proj=192,
                                   layer_mode_overrides=self.layer_mode_overrides)
            current_output = inc_b.connect(current_output)
            self.layers.extend(inc_b.layers)
        
        # ========== Reduction-B ==========
        # Reduction-B: 768 -> 1280 channels with stride=2
        red_b = ReductionBlock(sid, "reduction_b", self.enclave_mode, in_channels=768,
                              ch3x3=192, ch3x3_red=192, ch3x3_2=320,
                              layer_mode_overrides=self.layer_mode_overrides)
        current_output = red_b.connect(current_output)
        self.layers.extend(red_b.layers)
        
        # ========== Inception-C (x2) ==========
        # Inception-C: 1280 -> 2048 channels (320+384+384+448)
        for i in range(2):
            inc_c = InceptionBlock(sid, f"inception_c{i+1}", self.enclave_mode, in_channels=1280,
                                   ch1x1=320, ch3x3_red=384, ch3x3=384, ch5x5_red=448, ch5x5=384, pool_proj=192,
                                   layer_mode_overrides=self.layer_mode_overrides)
            current_output = inc_c.connect(current_output)
            self.layers.extend(inc_c.layers)
        
        # ========== Classifier ==========
        # Global Average Pooling
        # For 299x299 input, after all reductions, feature map should be ~8x8
        # Use adaptive pooling to handle different input sizes
        final_size = max(1, self.input_size // 32)  # Approximate final feature map size
        avgpool = SecretAvgpool2dLayer(sid, "avgpool", self._get_mode("avgpool"),
                                       filter_hw=final_size, stride=1, padding=0,
                                       manually_register_prev=True, manually_register_next=True)
        avgpool.register_prev_layer(current_output)
        
        flatten = SecretFlattenLayer(sid, "flatten", self._get_mode("flatten"),
                                     manually_register_prev=True, manually_register_next=True)
        flatten.register_prev_layer(avgpool)
        
        # FC layer: 2048 -> num_classes
        # Note: Inception-C output is 2048 channels
        fc = SGXLinearBase(sid, "fc", self._get_mode("fc"), batch_size=self.batch_size,
                           n_output_features=self.num_classes, n_input_features=2048,
                           manually_register_prev=True, manually_register_next=True)
        fc.register_prev_layer(flatten)
        
        output = SecretOutputLayer(sid, "output", self._get_mode("output"), inference=True,
                                   manually_register_prev=True)
        output.register_prev_layer(fc)
        
        self.layers.extend([avgpool, flatten, fc, output])

    def print_architecture(self):
        """Print the complete architecture of the model."""
        print(f"\n{'='*80}")
        print(f"SGXInceptionV3 - Standard Implementation")
        print(f"{'='*80}")
        print(f"Input size: {self.input_size}x{self.input_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Total layers: {len(self.layers)}")
        print(f"Enclave mode: {self.enclave_mode.name}")
        print(f"\n{'Layer Name':<30} {'Type':<25} {'Mode':<15}")
        print(f"{'-'*70}")
        for l in self.layers:
            print(f"{l.LayerName:<30} {type(l).__name__:<25} {l.EnclaveMode.name:<15}")
        print(f"{'='*80}\n")

def test_inception():
    """Test function for standard Inception V3."""
    print("Testing Standard Inception V3 Model...")
    model = SGXInceptionV3(
        sid=0,
        num_classes=1000,
        enclave_mode=ExecutionModeOptions.Enclave,
        batch_size=1,
        input_size=299,
        layer_mode_overrides={"input": ExecutionModeOptions.Enclave}
    )
    model.print_architecture()
    print(f"Model created successfully with {len(model.layers)} layers!")

if __name__ == "__main__":
    test_inception()
