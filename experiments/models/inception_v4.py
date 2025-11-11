"""
Inception V4 Model
Paper: Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (Szegedy et al., 2017)
"""

import sys
sys.path.insert(0, '.')

from python.layers.sgx_conv_base import SGXConvBase
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.relu import SecretReLULayer
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.layers.avgpool2d import SecretAvgpool2dLayer
from python.layers.flatten import SecretFlattenLayer
from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer
from python.utils.basic_utils import ExecutionModeOptions


class InceptionV4ModuleA:
    """Inception-A module for Inception V4"""
    
    def __init__(self, sid, name_prefix, enclave_mode):
        self.layers = []
        
        # Branch 1: 1x1 conv
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_1x1", enclave_mode,
            n_output_channel=96,
            filter_hw=1, stride=1, padding=0
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_1x1_relu", enclave_mode))
        
        # Branch 2: 1x1 -> 3x3
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_3x3_reduce", enclave_mode,
            n_output_channel=64,
            filter_hw=1, stride=1, padding=0
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_3x3_reduce_relu", enclave_mode))
        
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_3x3", enclave_mode,
            n_output_channel=96,
            filter_hw=3, stride=1, padding=1
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_3x3_relu", enclave_mode))
        
        # Branch 3: 1x1 -> 3x3 -> 3x3
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_double_3x3_reduce", enclave_mode,
            n_output_channel=64,
            filter_hw=1, stride=1, padding=0
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_double_3x3_reduce_relu", enclave_mode))
        
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_double_3x3_1", enclave_mode,
            n_output_channel=96,
            filter_hw=3, stride=1, padding=1
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_double_3x3_1_relu", enclave_mode))
        
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_double_3x3_2", enclave_mode,
            n_output_channel=96,
            filter_hw=3, stride=1, padding=1
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_double_3x3_2_relu", enclave_mode))
        
        # Branch 4: avgpool -> 1x1
        self.layers.append(SecretAvgpool2dLayer(
            sid, f"{name_prefix}_pool", enclave_mode,
            kernel_size=3, stride=1, padding=1
        ))
        
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_pool_proj", enclave_mode,
            n_output_channel=96,
            filter_hw=1, stride=1, padding=0
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_pool_proj_relu", enclave_mode))


class InceptionV4ModuleB:
    """Inception-B module for Inception V4"""
    
    def __init__(self, sid, name_prefix, enclave_mode):
        self.layers = []
        
        # Branch 1: 1x1 conv
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_1x1", enclave_mode,
            n_output_channel=384,
            filter_hw=1, stride=1, padding=0
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_1x1_relu", enclave_mode))
        
        # Branch 2: 1x1 -> 1x7 -> 7x1
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_7x7_reduce", enclave_mode,
            n_output_channel=192,
            filter_hw=1, stride=1, padding=0
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_7x7_reduce_relu", enclave_mode))
        
        # Approximate 1x7 and 7x1 with 3x3 for simplicity
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_7x7_1", enclave_mode,
            n_output_channel=224,
            filter_hw=3, stride=1, padding=1
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_7x7_1_relu", enclave_mode))
        
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_7x7_2", enclave_mode,
            n_output_channel=256,
            filter_hw=3, stride=1, padding=1
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_7x7_2_relu", enclave_mode))
        
        # Branch 3: avgpool -> 1x1
        self.layers.append(SecretAvgpool2dLayer(
            sid, f"{name_prefix}_pool", enclave_mode,
            filter_hw=3, stride=1, padding=1
        ))
        
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_pool_proj", enclave_mode,
            n_output_channel=128,
            filter_hw=1, stride=1, padding=0
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_pool_proj_relu", enclave_mode))


class SGXInceptionV4:
    """
    Inception V4 Model (Simplified version for SGX)
    
    Note: This is a simplified implementation. Full Inception V4 has:
    - Stem with multiple branches
    - Inception-A, Inception-B, Inception-C modules
    - Reduction modules
    - More complex structure
    """
    
    def __init__(self, sid=0, num_classes=1000,
                 enclave_mode=ExecutionModeOptions.Enclave,
                 batch_size=1, input_size=299):
        """
        Args:
            sid: Session ID
            num_classes: Number of output classes
            enclave_mode: Execution mode
            batch_size: Batch size
            input_size: Input image size (299x299 for Inception V4)
        """
        self.sid = sid
        self.num_classes = num_classes
        self.enclave_mode = enclave_mode
        self.batch_size = batch_size
        self.input_size = input_size
        
        self.layers = self._build_network()
        self.model_name = 'InceptionV4'
    
    def _build_stem(self):
        """Build Inception V4 stem (simplified)"""
        layers = []
        sid = self.sid
        mode = self.enclave_mode
        
        # Initial convolutions
        layers.append(SGXConvBase(
            sid, "stem_conv1", mode,
            n_output_channel=32,
            n_input_channel=3,
            filter_hw=3, stride=2, padding=0,
            batch_size=self.batch_size,
            img_hw=self.input_size
        ))
        layers.append(SecretReLULayer(sid, "stem_conv1_relu", mode))
        
        layers.append(SGXConvBase(
            sid, "stem_conv2", mode,
            n_output_channel=32,
            filter_hw=3, stride=1, padding=0
        ))
        layers.append(SecretReLULayer(sid, "stem_conv2_relu", mode))
        
        layers.append(SGXConvBase(
            sid, "stem_conv3", mode,
            n_output_channel=64,
            filter_hw=3, stride=1, padding=1
        ))
        layers.append(SecretReLULayer(sid, "stem_conv3_relu", mode))
        
        # Additional stem layers
        layers.append(SGXConvBase(
            sid, "stem_conv4", mode,
            n_output_channel=96,
            filter_hw=3, stride=2, padding=0
        ))
        layers.append(SecretReLULayer(sid, "stem_conv4_relu", mode))
        
        layers.append(SGXConvBase(
            sid, "stem_conv5", mode,
            n_output_channel=192,
            filter_hw=3, stride=1, padding=1
        ))
        layers.append(SecretReLULayer(sid, "stem_conv5_relu", mode))
        
        return layers
    
    def _build_network(self):
        """Build Inception V4 network (simplified)"""
        layers = []
        sid = self.sid
        mode = self.enclave_mode
        
        # Input layer
        layers.append(SecretInputLayer(
            sid, "input", mode, self.input_size, self.input_size, 3
        ))
        
        # Stem
        layers.extend(self._build_stem())
        
        # Inception-A modules (4x)
        for i in range(2):  # Simplified: use 2 instead of 4
            inception_a = InceptionV4ModuleA(sid, f"inception_a{i+1}", mode)
            layers.extend(inception_a.layers)
        
        # Reduction-A (simplified with strided conv)
        layers.append(SGXConvBase(
            sid, "reduction_a", mode,
            n_output_channel=384,
            filter_hw=3, stride=2, padding=0
        ))
        layers.append(SecretReLULayer(sid, "reduction_a_relu", mode))
        
        # Inception-B modules (7x, simplified to 2x)
        for i in range(2):
            inception_b = InceptionV4ModuleB(sid, f"inception_b{i+1}", mode)
            layers.extend(inception_b.layers)
        
        # Reduction-B (simplified)
        layers.append(SGXConvBase(
            sid, "reduction_b", mode,
            n_output_channel=512,
            filter_hw=3, stride=2, padding=0
        ))
        layers.append(SecretReLULayer(sid, "reduction_b_relu", mode))
        
        # Final convolutions
        layers.append(SGXConvBase(
            sid, "final_conv", mode,
            n_output_channel=1536,
            filter_hw=3, stride=1, padding=1
        ))
        layers.append(SecretReLULayer(sid, "final_conv_relu", mode))
        
        # Global Average Pooling
        layers.append(SecretAvgpool2dLayer(
            sid, "global_avgpool", mode,
            filter_hw=8, stride=1, padding=0
        ))
        
        # Flatten
        layers.append(SecretFlattenLayer(sid, "flatten", mode))
        
        # FC layer
        layers.append(SGXLinearBase(
            sid, "fc", mode,
            batch_size=self.batch_size,
            input_size=1536,
            output_size=self.num_classes
        ))
        
        # Output layer
        layers.append(SecretOutputLayer(sid, "output", mode, self.num_classes))
        
        return layers
    
    def __str__(self):
        info = f"SGXInceptionV4 Model\n"
        info += f"  Input size: {self.input_size}x{self.input_size}\n"
        info += f"  Num classes: {self.num_classes}\n"
        info += f"  Num layers: {len(self.layers)}\n"
        info += f"  Enclave mode: {self.enclave_mode}\n"
        info += f"  Note: Simplified version for SGX (reduced modules)\n"
        return info


def test_inception_v4():
    """Test Inception V4 model creation"""
    print("Testing Inception V4 model...")
    
    model = SGXInceptionV4(sid=0, enclave_mode=ExecutionModeOptions.CPU)
    print(model)
    print(f"Successfully created {len(model.layers)} layers")


if __name__ == '__main__':
    test_inception_v4()

