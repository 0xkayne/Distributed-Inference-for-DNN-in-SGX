"""
AlexNet Model for ImageNet
Paper: ImageNet Classification with Deep Convolutional Neural Networks (Krizhevsky et al., 2012)
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


class SGXAlexNet:
    """
    AlexNet Model
    
    Architecture:
    - Conv1: 11x11, 96 filters, stride 4, padding 2
    - MaxPool: 3x3, stride 2
    - Conv2: 5x5, 256 filters, padding 2
    - MaxPool: 3x3, stride 2
    - Conv3: 3x3, 384 filters, padding 1
    - Conv4: 3x3, 384 filters, padding 1
    - Conv5: 3x3, 256 filters, padding 1
    - MaxPool: 3x3, stride 2
    - FC1: 4096
    - FC2: 4096
    - FC3: num_classes
    
    Total: 5 conv layers + 3 FC layers
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
        self.model_name = 'AlexNet'
    
    def _build_network(self):
        """Build AlexNet network layers"""
        layers = []
        sid = self.sid
        mode = self.enclave_mode
        
        # Input layer
        layers.append(SecretInputLayer(
            sid, "input", mode, self.input_size, self.input_size, 3
        ))
        
        # Conv1: 11x11, 96 filters, stride 4
        layers.append(SGXConvBase(
            sid, "conv1", mode,
            n_output_channel=96,
            n_input_channel=3,
            filter_hw=11,
            stride=4,
            padding=2,
            batch_size=self.batch_size,
            img_hw=self.input_size
        ))
        layers.append(SecretReLULayer(sid, "relu1", mode))
        
        # MaxPool1: 3x3, stride 2
        layers.append(SecretMaxpool2dLayer(
            sid, "pool1", mode,
            filter_hw=3, stride=2, padding=0
        ))
        
        # Conv2: 5x5, 256 filters
        layers.append(SGXConvBase(
            sid, "conv2", mode,
            n_output_channel=256,
            filter_hw=5,
            stride=1,
            padding=2
        ))
        layers.append(SecretReLULayer(sid, "relu2", mode))
        
        # MaxPool2: 3x3, stride 2
        layers.append(SecretMaxpool2dLayer(
            sid, "pool2", mode,
            filter_hw=3, stride=2, padding=0
        ))
        
        # Conv3: 3x3, 384 filters
        layers.append(SGXConvBase(
            sid, "conv3", mode,
            n_output_channel=384,
            filter_hw=3,
            stride=1,
            padding=1
        ))
        layers.append(SecretReLULayer(sid, "relu3", mode))
        
        # Conv4: 3x3, 384 filters
        layers.append(SGXConvBase(
            sid, "conv4", mode,
            n_output_channel=384,
            filter_hw=3,
            stride=1,
            padding=1
        ))
        layers.append(SecretReLULayer(sid, "relu4", mode))
        
        # Conv5: 3x3, 256 filters
        layers.append(SGXConvBase(
            sid, "conv5", mode,
            n_output_channel=256,
            filter_hw=3,
            stride=1,
            padding=1
        ))
        layers.append(SecretReLULayer(sid, "relu5", mode))
        
        # MaxPool3: 3x3, stride 2
        layers.append(SecretMaxpool2dLayer(
            sid, "pool3", mode,
            filter_hw=3, stride=2, padding=0
        ))
        
        # Flatten
        layers.append(SecretFlattenLayer(sid, "flatten", mode))
        
        # FC layers
        # After conv and pooling: 224 -> 55 -> 27 -> 13 -> 6
        # Feature size: 6x6x256 = 9216
        layers.append(SGXLinearBase(
            sid, "fc1", mode,
            batch_size=self.batch_size,
            input_size=6*6*256,  # 9216
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
        info = f"SGXAlexNet Model\n"
        info += f"  Input size: {self.input_size}x{self.input_size}\n"
        info += f"  Num classes: {self.num_classes}\n"
        info += f"  Num layers: {len(self.layers)}\n"
        info += f"  Enclave mode: {self.enclave_mode}\n"
        return info


def test_alexnet():
    """Test AlexNet model creation"""
    print("Testing AlexNet model...")
    
    model = SGXAlexNet(sid=0, enclave_mode=ExecutionModeOptions.CPU)
    print(model)
    print(f"Successfully created {len(model.layers)} layers")


if __name__ == '__main__':
    test_alexnet()

