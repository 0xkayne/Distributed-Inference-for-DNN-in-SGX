"""
VGG16 Model for ImageNet
Paper: Very Deep Convolutional Networks for Large-Scale Image Recognition (Simonyan & Zisserman, 2014)
"""

import sys
sys.path.insert(0, '.')

from python.layers.sgx_conv_base import SGXConvBase
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.relu import SecretReLULayer
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.layers.flatten import SecretFlattenLayer
from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer
from python.utils.basic_utils import ExecutionModeOptions


class SGXVGG16:
    """
    VGG16 Model
    
    Architecture:
    - Conv Block 1: Conv3x3(64)-Conv3x3(64)-MaxPool
    - Conv Block 2: Conv3x3(128)-Conv3x3(128)-MaxPool
    - Conv Block 3: Conv3x3(256)-Conv3x3(256)-Conv3x3(256)-MaxPool
    - Conv Block 4: Conv3x3(512)-Conv3x3(512)-Conv3x3(512)-MaxPool
    - Conv Block 5: Conv3x3(512)-Conv3x3(512)-Conv3x3(512)-MaxPool
    - FC: FC(4096)-FC(4096)-FC(num_classes)
    
    Total: 13 conv layers + 3 FC layers
    """
    
    def __init__(self, sid=0, num_classes=1000,
                 enclave_mode=ExecutionModeOptions.Enclave,
                 batch_size=1, input_size=224):
        """
        Args:
            sid: Session ID
            num_classes: Number of output classes
            enclave_mode: Execution mode (CPU/GPU/Enclave)
            batch_size: Batch size
            input_size: Input image size (224 for ImageNet)
        """
        self.sid = sid
        self.num_classes = num_classes
        self.enclave_mode = enclave_mode
        self.batch_size = batch_size
        self.input_size = input_size
        
        self.layers = self._build_network()
        self.model_name = 'VGG16'
    
    def _make_conv_block(self, start_idx, in_channels, out_channels, 
                        num_convs, has_pool=True):
        """
        Helper function to create a conv block
        
        Args:
            start_idx: Starting index for layer naming
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_convs: Number of conv layers in this block
            has_pool: Whether to add pooling layer
            
        Returns:
            List of layers
        """
        layers = []
        sid = self.sid
        mode = self.enclave_mode
        
        for i in range(num_convs):
            layer_name = f"conv{start_idx}_{i+1}"
            layers.append(SGXConvBase(
                sid, layer_name, mode,
                n_output_channel=out_channels,
                n_input_channel=in_channels if i == 0 else out_channels,
                filter_hw=3, stride=1, padding=1
            ))
            layers.append(SecretReLULayer(sid, f"relu{start_idx}_{i+1}", mode))
        
        if has_pool:
            layers.append(SecretMaxpool2dLayer(
                sid, f"pool{start_idx}", mode,
                filter_hw=2, stride=2, padding=0
            ))
        
        return layers
    
    def _build_network(self):
        """Build VGG16 network layers"""
        layers = []
        sid = self.sid
        mode = self.enclave_mode
        
        # Input layer
        layers.append(SecretInputLayer(
            sid, "input", mode, self.input_size, self.input_size, 3
        ))
        
        # Block 1: 64 channels, 2 conv layers
        layers.extend(self._make_conv_block(1, 3, 64, 2, has_pool=True))
        
        # Block 2: 128 channels, 2 conv layers
        layers.extend(self._make_conv_block(2, 64, 128, 2, has_pool=True))
        
        # Block 3: 256 channels, 3 conv layers
        layers.extend(self._make_conv_block(3, 128, 256, 3, has_pool=True))
        
        # Block 4: 512 channels, 3 conv layers
        layers.extend(self._make_conv_block(4, 256, 512, 3, has_pool=True))
        
        # Block 5: 512 channels, 3 conv layers
        layers.extend(self._make_conv_block(5, 512, 512, 3, has_pool=True))
        
        # Flatten
        layers.append(SecretFlattenLayer(sid, "flatten", mode))
        
        # FC layers
        # After 5 pooling layers: 224/(2^5) = 7, so feature map is 7x7x512 = 25088
        layers.append(SGXLinearBase(
            sid, "fc1", mode,
            batch_size=self.batch_size,
            input_size=7*7*512,  # 25088
            output_size=4096
        ))
        layers.append(SecretReLULayer(sid, "relu_fc1", mode))
        
        layers.append(SGXLinearBase(
            sid, "fc2", mode,
            batch_size=self.batch_size,
            input_size=4096,
            output_size=4096
        ))
        layers.append(SecretReLULayer(sid, "relu_fc2", mode))
        
        layers.append(SGXLinearBase(
            sid, "fc3", mode,
            batch_size=self.batch_size,
            input_size=4096,
            output_size=self.num_classes
        ))
        
        # Output layer
        layers.append(SecretOutputLayer(sid, "output", mode, self.num_classes))
        
        return layers
    
    def __str__(self):
        info = f"SGXVGG16 Model\n"
        info += f"  Input size: {self.input_size}x{self.input_size}\n"
        info += f"  Num classes: {self.num_classes}\n"
        info += f"  Num layers: {len(self.layers)}\n"
        info += f"  Enclave mode: {self.enclave_mode}\n"
        return info


def test_vgg16():
    """Test VGG16 model creation"""
    print("Testing VGG16 model...")
    
    model = SGXVGG16(sid=0, enclave_mode=ExecutionModeOptions.CPU)
    print(model)
    print(f"Successfully created {len(model.layers)} layers")


if __name__ == '__main__':
    test_vgg16()

