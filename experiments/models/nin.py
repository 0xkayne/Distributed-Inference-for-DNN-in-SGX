"""
Network in Network (NiN) for CIFAR-10
Paper: Network In Network (Lin et al., 2013)
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
from python.sgx_net import SecretNeuralNetwork


class SGXNiN:
    """
    Network in Network for CIFAR-10 (32x32 input)
    
    Architecture:
    - Conv Block 1: Conv3x3-Conv1x1-Conv1x1-MaxPool
    - Conv Block 2: Conv3x3-Conv1x1-Conv1x1-MaxPool  
    - Conv Block 3: Conv3x3-Conv1x1-Conv1x1-AvgPool
    - Output: 10 classes
    """
    
    def __init__(self, sid=0, num_classes=10, 
                 enclave_mode=ExecutionModeOptions.Enclave,
                 batch_size=1, input_size=32):
        """
        Args:
            sid: Session ID
            num_classes: Number of output classes
            enclave_mode: Execution mode (CPU/GPU/Enclave)
            batch_size: Batch size
            input_size: Input image size (default 32 for CIFAR-10)
        """
        self.sid = sid
        self.num_classes = num_classes
        self.enclave_mode = enclave_mode
        self.batch_size = batch_size
        self.input_size = input_size
        
        self.layers = self._build_network()
        self.model_name = 'NiN'
    
    def _build_network(self):
        """Build NiN network layers"""
        layers = []
        sid = self.sid
        mode = self.enclave_mode
        
        # Input layer
        layers.append(SecretInputLayer(
            sid, "input", mode, self.input_size, self.input_size, 3
        ))
        
        # ===== Conv Block 1 =====
        # Conv 5x5, 192 filters
        layers.append(SGXConvBase(
            sid, "conv1_1", mode,
            n_output_channel=192, filter_hw=5, stride=1, padding=2,
            batch_size=self.batch_size, n_input_channel=3,
            img_hw=self.input_size
        ))
        layers.append(SecretReLULayer(sid, "relu1_1", mode))
        
        # Conv 1x1, 160 filters (mlpconv)
        layers.append(SGXConvBase(
            sid, "conv1_2", mode,
            n_output_channel=160, filter_hw=1, stride=1, padding=0
        ))
        layers.append(SecretReLULayer(sid, "relu1_2", mode))
        
        # Conv 1x1, 96 filters (mlpconv)
        layers.append(SGXConvBase(
            sid, "conv1_3", mode,
            n_output_channel=96, filter_hw=1, stride=1, padding=0
        ))
        layers.append(SecretReLULayer(sid, "relu1_3", mode))
        
        # MaxPool 3x3, stride 2
        layers.append(SecretMaxpool2dLayer(
            sid, "pool1", mode,
            filter_hw=3, stride=2, padding=1
        ))
        
        # ===== Conv Block 2 =====
        # Conv 5x5, 192 filters
        layers.append(SGXConvBase(
            sid, "conv2_1", mode,
            n_output_channel=192, filter_hw=5, stride=1, padding=2
        ))
        layers.append(SecretReLULayer(sid, "relu2_1", mode))
        
        # Conv 1x1, 192 filters (mlpconv)
        layers.append(SGXConvBase(
            sid, "conv2_2", mode,
            n_output_channel=192, filter_hw=1, stride=1, padding=0
        ))
        layers.append(SecretReLULayer(sid, "relu2_2", mode))
        
        # Conv 1x1, 192 filters (mlpconv)
        layers.append(SGXConvBase(
            sid, "conv2_3", mode,
            n_output_channel=192, filter_hw=1, stride=1, padding=0
        ))
        layers.append(SecretReLULayer(sid, "relu2_3", mode))
        
        # MaxPool 3x3, stride 2
        layers.append(SecretMaxpool2dLayer(
            sid, "pool2", mode,
            filter_hw=3, stride=2, padding=1
        ))
        
        # ===== Conv Block 3 =====
        # Conv 3x3, 192 filters
        layers.append(SGXConvBase(
            sid, "conv3_1", mode,
            n_output_channel=192, filter_hw=3, stride=1, padding=1
        ))
        layers.append(SecretReLULayer(sid, "relu3_1", mode))
        
        # Conv 1x1, 192 filters (mlpconv)
        layers.append(SGXConvBase(
            sid, "conv3_2", mode,
            n_output_channel=192, filter_hw=1, stride=1, padding=0
        ))
        layers.append(SecretReLULayer(sid, "relu3_2", mode))
        
        # Conv 1x1, num_classes filters (mlpconv for classification)
        layers.append(SGXConvBase(
            sid, "conv3_3", mode,
            n_output_channel=self.num_classes, filter_hw=1, stride=1, padding=0
        ))
        layers.append(SecretReLULayer(sid, "relu3_3", mode))
        
        # Global Average Pooling (8x8 -> 1x1)
        layers.append(SecretAvgpool2dLayer(
            sid, "global_avgpool", mode,
            filter_hw=8, stride=1, padding=0
        ))
        
        # Flatten
        layers.append(SecretFlattenLayer(sid, "flatten", mode))
        
        # Output layer
        layers.append(SecretOutputLayer(sid, "output", mode, self.num_classes))
        
        return layers
    
    def __str__(self):
        info = f"SGXNiN Model\n"
        info += f"  Input size: {self.input_size}x{self.input_size}\n"
        info += f"  Num classes: {self.num_classes}\n"
        info += f"  Num layers: {len(self.layers)}\n"
        info += f"  Enclave mode: {self.enclave_mode}\n"
        return info


def test_nin():
    """Test NiN model creation"""
    print("Testing NiN model...")
    
    # Test with different execution modes
    for mode in [ExecutionModeOptions.CPU, ExecutionModeOptions.Enclave]:
        print(f"\nTesting mode: {mode}")
        model = SGXNiN(sid=0, enclave_mode=mode)
        print(model)
        print(f"Successfully created {len(model.layers)} layers")


if __name__ == '__main__':
    test_nin()

