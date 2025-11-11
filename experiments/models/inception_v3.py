"""
Inception V3 Model
Paper: Rethinking the Inception Architecture for Computer Vision (Szegedy et al., 2016)
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
from python.layers.add import SecretAddLayer
from python.utils.basic_utils import ExecutionModeOptions


class InceptionModule:
    """
    Basic Inception Module with 4 branches:
    - 1x1 conv
    - 1x1 conv -> 3x3 conv
    - 1x1 conv -> 5x5 conv (or two 3x3)
    - 3x3 maxpool -> 1x1 conv
    """
    
    def __init__(self, sid, name_prefix, enclave_mode,
                 in_channels, out_1x1, out_3x3_reduce, out_3x3,
                 out_5x5_reduce, out_5x5, out_pool_proj):
        """
        Args:
            sid: Session ID
            name_prefix: Prefix for layer names
            enclave_mode: Execution mode
            in_channels: Input channels
            out_1x1: Output channels for 1x1 branch
            out_3x3_reduce: Reduction channels for 3x3 branch
            out_3x3: Output channels for 3x3 branch
            out_5x5_reduce: Reduction channels for 5x5 branch
            out_5x5: Output channels for 5x5 branch
            out_pool_proj: Output channels for pooling branch
        """
        self.layers = []
        
        # Branch 1: 1x1 convolution
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_1x1", enclave_mode,
            n_output_channel=out_1x1,
            filter_hw=1, stride=1, padding=0
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_1x1_relu", enclave_mode))
        
        # Branch 2: 1x1 -> 3x3 convolution
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_3x3_reduce", enclave_mode,
            n_output_channel=out_3x3_reduce,
            filter_hw=1, stride=1, padding=0
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_3x3_reduce_relu", enclave_mode))
        
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_3x3", enclave_mode,
            n_output_channel=out_3x3,
            filter_hw=3, stride=1, padding=1
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_3x3_relu", enclave_mode))
        
        # Branch 3: 1x1 -> 5x5 (or two 3x3) convolution
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_5x5_reduce", enclave_mode,
            n_output_channel=out_5x5_reduce,
            filter_hw=1, stride=1, padding=0
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_5x5_reduce_relu", enclave_mode))
        
        # Use two 3x3 instead of 5x5 for efficiency (Inception V3 style)
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_5x5_1", enclave_mode,
            n_output_channel=out_5x5,
            filter_hw=3, stride=1, padding=1
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_5x5_1_relu", enclave_mode))
        
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_5x5_2", enclave_mode,
            n_output_channel=out_5x5,
            filter_hw=3, stride=1, padding=1
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_5x5_2_relu", enclave_mode))
        
        # Branch 4: maxpool -> 1x1 convolution
        self.layers.append(SecretMaxpool2dLayer(
            sid, f"{name_prefix}_pool", enclave_mode,
            filter_hw=3, stride=1, padding=1
        ))
        
        self.layers.append(SGXConvBase(
            sid, f"{name_prefix}_pool_proj", enclave_mode,
            n_output_channel=out_pool_proj,
            filter_hw=1, stride=1, padding=0
        ))
        self.layers.append(SecretReLULayer(sid, f"{name_prefix}_pool_proj_relu", enclave_mode))
        
        # Note: Concatenation of branches is implicit in the sequential model
        # In a real implementation, we'd need explicit concatenation layers


class SGXInceptionV3:
    """
    Inception V3 Model (Simplified version for SGX)
    
    Note: This is a simplified implementation focusing on the main structure.
    Full Inception V3 includes auxiliary classifiers and more complex modules.
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
            input_size: Input image size (299x299 for Inception V3)
        """
        self.sid = sid
        self.num_classes = num_classes
        self.enclave_mode = enclave_mode
        self.batch_size = batch_size
        self.input_size = input_size
        
        self.layers = self._build_network()
        self.model_name = 'InceptionV3'
    
    def _build_network(self):
        """Build Inception V3 network (simplified)"""
        layers = []
        sid = self.sid
        mode = self.enclave_mode
        
        # Input layer
        layers.append(SecretInputLayer(
            sid, "input", mode, self.input_size, self.input_size, 3
        ))
        
        # === Initial Convolutions ===
        # Conv1: 3x3, stride 2
        layers.append(SGXConvBase(
            sid, "conv1", mode,
            n_output_channel=32,
            n_input_channel=3,
            filter_hw=3, stride=2, padding=0,
            batch_size=self.batch_size,
            img_hw=self.input_size
        ))
        layers.append(SecretReLULayer(sid, "conv1_relu", mode))
        
        # Conv2: 3x3
        layers.append(SGXConvBase(
            sid, "conv2", mode,
            n_output_channel=32,
            filter_hw=3, stride=1, padding=0
        ))
        layers.append(SecretReLULayer(sid, "conv2_relu", mode))
        
        # Conv3: 3x3, padded
        layers.append(SGXConvBase(
            sid, "conv3", mode,
            n_output_channel=64,
            filter_hw=3, stride=1, padding=1
        ))
        layers.append(SecretReLULayer(sid, "conv3_relu", mode))
        
        # MaxPool
        layers.append(SecretMaxpool2dLayer(
            sid, "pool1", mode,
            filter_hw=3, stride=2, padding=0
        ))
        
        # Conv4: 1x1
        layers.append(SGXConvBase(
            sid, "conv4", mode,
            n_output_channel=80,
            filter_hw=1, stride=1, padding=0
        ))
        layers.append(SecretReLULayer(sid, "conv4_relu", mode))
        
        # Conv5: 3x3
        layers.append(SGXConvBase(
            sid, "conv5", mode,
            n_output_channel=192,
            filter_hw=3, stride=1, padding=0
        ))
        layers.append(SecretReLULayer(sid, "conv5_relu", mode))
        
        # MaxPool
        layers.append(SecretMaxpool2dLayer(
            sid, "pool2", mode,
            filter_hw=3, stride=2, padding=0
        ))
        
        # === Inception Modules (Simplified) ===
        # For simplicity, we use basic sequential inception-style blocks
        # instead of full parallel branches with concatenation
        
        # Inception 3a
        inception_3a = InceptionModule(
            sid, "inception_3a", mode,
            in_channels=192,
            out_1x1=64,
            out_3x3_reduce=96, out_3x3=128,
            out_5x5_reduce=16, out_5x5=32,
            out_pool_proj=32
        )
        layers.extend(inception_3a.layers)
        
        # Inception 3b
        inception_3b = InceptionModule(
            sid, "inception_3b", mode,
            in_channels=256,  # 64+128+32+32
            out_1x1=128,
            out_3x3_reduce=128, out_3x3=192,
            out_5x5_reduce=32, out_5x5=96,
            out_pool_proj=64
        )
        layers.extend(inception_3b.layers)
        
        # MaxPool
        layers.append(SecretMaxpool2dLayer(
            sid, "pool3", mode,
            filter_hw=3, stride=2, padding=0
        ))
        
        # Inception 4a (simplified)
        layers.append(SGXConvBase(
            sid, "inception_4a_conv", mode,
            n_output_channel=256,
            filter_hw=3, stride=1, padding=1
        ))
        layers.append(SecretReLULayer(sid, "inception_4a_relu", mode))
        
        # Additional conv layers
        layers.append(SGXConvBase(
            sid, "conv_final", mode,
            n_output_channel=512,
            filter_hw=3, stride=1, padding=1
        ))
        layers.append(SecretReLULayer(sid, "conv_final_relu", mode))
        
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
            input_size=512,  # Depends on final feature map size
            output_size=self.num_classes
        ))
        
        # Output layer
        layers.append(SecretOutputLayer(sid, "output", mode, self.num_classes))
        
        return layers
    
    def __str__(self):
        info = f"SGXInceptionV3 Model\n"
        info += f"  Input size: {self.input_size}x{self.input_size}\n"
        info += f"  Num classes: {self.num_classes}\n"
        info += f"  Num layers: {len(self.layers)}\n"
        info += f"  Enclave mode: {self.enclave_mode}\n"
        info += f"  Note: Simplified version for SGX\n"
        return info


def test_inception_v3():
    """Test Inception V3 model creation"""
    print("Testing Inception V3 model...")
    
    model = SGXInceptionV3(sid=0, enclave_mode=ExecutionModeOptions.CPU)
    print(model)
    print(f"Successfully created {len(model.layers)} layers")


if __name__ == '__main__':
    test_inception_v3()

