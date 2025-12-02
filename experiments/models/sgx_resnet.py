"""
ResNet-18 Model for SGX with flexible layer partitioning support.

ResNet-18 Architecture:
- Conv1 + BN + ReLU + MaxPool
- Layer1: 2 x BasicBlock (64 channels)  
- Layer2: 2 x BasicBlock (128 channels, stride=2)
- Layer3: 2 x BasicBlock (256 channels, stride=2)
- Layer4: 2 x BasicBlock (512 channels, stride=2)
- AvgPool + FC

BasicBlock structure:
    Input
      |
      +------------------+
      |                  |
    Conv3x3            Identity/Conv1x1 (skip)
      |                  |
      BN                 |
      |                  |
     ReLU                |
      |                  |
    Conv3x3              |
      |                  |
      BN                 |
      |                  |
      +--------Add-------+
               |
             ReLU
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
from python.layers.add import SecretAddLayer
from python.layers.identity import SecretIdentityLayer
from python.utils.basic_utils import ExecutionModeOptions


class BasicBlock:
    """
    ResNet Basic Block
    
    Structure:
        x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+skip) -> ReLU -> out
        |                                                   ^
        +----> Identity or Conv1x1 (downsample) -----------+
    """
    
    def __init__(
        self,
        sid: int,
        name_prefix: str,
        enclave_mode: ExecutionModeOptions,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        """
        Args:
            sid: Session ID
            name_prefix: Prefix for layer names (e.g., "layer1_block0")
            enclave_mode: Default execution mode
            in_channels: Input channels
            out_channels: Output channels
            stride: Stride for first conv (used for downsampling)
            layer_mode_overrides: Override execution mode for specific layers
        """
        self.layers = []
        self.name_prefix = name_prefix
        overrides = layer_mode_overrides or {}
        
        def get_mode(layer_name: str) -> ExecutionModeOptions:
            return overrides.get(layer_name, enclave_mode)
        
        # Main path: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
        # Note: We omit BN for simplicity in SGX (can be added later)
        
        # Conv1: 3x3 conv with stride
        conv1_name = f"{name_prefix}_conv1"
        self.layers.append(SGXConvBase(
            sid, conv1_name, get_mode(conv1_name),
            n_output_channel=out_channels,
            filter_hw=3,
            stride=stride,
            padding=1,
            manually_register_prev=True,
            manually_register_next=True,
        ))
        
        # ReLU1
        relu1_name = f"{name_prefix}_relu1"
        self.layers.append(SecretReLULayer(
            sid, relu1_name, get_mode(relu1_name),
            manually_register_prev=True,
            manually_register_next=True,
        ))
        
        # Conv2: 3x3 conv with stride=1
        conv2_name = f"{name_prefix}_conv2"
        self.layers.append(SGXConvBase(
            sid, conv2_name, get_mode(conv2_name),
            n_output_channel=out_channels,
            filter_hw=3,
            stride=1,
            padding=1,
            manually_register_prev=True,
            manually_register_next=True,
        ))
        
        # Skip connection path
        # If dimensions change (stride != 1 or channels change), use Conv1x1
        # Otherwise, use Identity
        if stride != 1 or in_channels != out_channels:
            downsample_name = f"{name_prefix}_downsample"
            self.layers.append(SGXConvBase(
                sid, downsample_name, get_mode(downsample_name),
                n_output_channel=out_channels,
                filter_hw=1,
                stride=stride,
                padding=0,
                manually_register_prev=True,
                manually_register_next=True,
            ))
        else:
            skip_name = f"{name_prefix}_skip"
            self.layers.append(SecretIdentityLayer(
                sid, skip_name, get_mode(skip_name),
                manually_register_prev=True,
                manually_register_next=True,
            ))
        
        # Add: main + skip
        add_name = f"{name_prefix}_add"
        self.layers.append(SecretAddLayer(
            sid, add_name, get_mode(add_name),
            manually_register_prev=True,
            manually_register_next=True,
        ))
        
        # ReLU after add
        relu2_name = f"{name_prefix}_relu2"
        self.layers.append(SecretReLULayer(
            sid, relu2_name, get_mode(relu2_name),
            manually_register_prev=True,
            manually_register_next=True,
        ))
    
    def build_connections(self, prev_layer):
        """
        Build the residual connections for this block.
        
        Args:
            prev_layer: The layer before this block
        
        Returns:
            The output layer of this block
        """
        # Indices in self.layers:
        # 0: conv1, 1: relu1, 2: conv2, 3: skip/downsample, 4: add, 5: relu2
        
        conv1 = self.layers[0]
        relu1 = self.layers[1]
        conv2 = self.layers[2]
        skip = self.layers[3]
        add = self.layers[4]
        relu2 = self.layers[5]
        
        # Main path: prev -> conv1 -> relu1 -> conv2
        conv1.register_prev_layer(prev_layer)
        relu1.register_prev_layer(conv1)
        conv2.register_prev_layer(relu1)
        
        # Skip path: prev -> skip
        skip.register_prev_layer(prev_layer)
        
        # Merge: conv2 + skip -> add -> relu2
        add.register_prev_layer(conv2)
        add.register_prev_layer(skip)
        relu2.register_prev_layer(add)
        
        return relu2


class SGXResNet18:
    """
    ResNet-18 Model with flexible layer-wise execution mode control.
    
    Supports:
    - CPU, Enclave, or GPU execution for any layer
    - Distributed inference via layer_mode_overrides
    - Standard ResNet-18 architecture with BasicBlocks
    """
    
    def __init__(
        self,
        sid: int = 0,
        num_classes: int = 1000,
        enclave_mode: ExecutionModeOptions = ExecutionModeOptions.Enclave,
        batch_size: int = 1,
        input_size: int = 224,
        layer_mode_overrides: Optional[Dict[str, ExecutionModeOptions]] = None,
    ):
        """
        Args:
            sid: Session ID
            num_classes: Number of output classes
            enclave_mode: Default execution mode for all layers
            batch_size: Batch size
            input_size: Input image size (224x224 for ImageNet)
            layer_mode_overrides: Dict mapping layer names to ExecutionModeOptions
                                 to override the default mode
        """
        self.sid = sid
        self.num_classes = num_classes
        self.enclave_mode = enclave_mode
        self.batch_size = batch_size
        self.input_size = input_size
        self.layer_mode_overrides = layer_mode_overrides or {}
        
        self.layers = []
        self.model_name = 'ResNet18'
        
        self._build_network()
    
    def _get_mode(self, layer_name: str) -> ExecutionModeOptions:
        """Get execution mode for a layer, considering overrides."""
        return self.layer_mode_overrides.get(layer_name, self.enclave_mode)
    
    def _build_network(self):
        """Build ResNet-18 network."""
        sid = self.sid
        
        # Input layer
        input_name = "input"
        input_layer = SecretInputLayer(
            sid,
            input_name,
            [self.batch_size, 3, self.input_size, self.input_size],
            self._get_mode(input_name),
            manually_register_next=True,
        )
        self.layers.append(input_layer)
        
        # === Stem: Conv1 + ReLU + MaxPool ===
        conv1_name = "conv1"
        conv1 = SGXConvBase(
            sid, conv1_name, self._get_mode(conv1_name),
            n_output_channel=64,
            filter_hw=7,
            stride=2,
            padding=3,
            manually_register_prev=True,
            manually_register_next=True,
        )
        self.layers.append(conv1)
        
        relu_name = "relu"
        relu = SecretReLULayer(
            sid, relu_name, self._get_mode(relu_name),
            manually_register_prev=True,
            manually_register_next=True,
        )
        self.layers.append(relu)
        
        maxpool_name = "maxpool"
        maxpool = SecretMaxpool2dLayer(
            sid, maxpool_name, self._get_mode(maxpool_name),
            filter_hw=3,
            stride=2,
            padding=1,
            manually_register_prev=True,
            manually_register_next=True,
        )
        self.layers.append(maxpool)
        
        # Connect stem
        conv1.register_prev_layer(input_layer)
        relu.register_prev_layer(conv1)
        maxpool.register_prev_layer(relu)
        
        # === Layer1: 2 x BasicBlock (64 channels) ===
        prev_layer = maxpool
        for i in range(2):
            block = BasicBlock(
                sid, f"layer1_block{i}", self.enclave_mode,
                in_channels=64,
                out_channels=64,
                stride=1,
                layer_mode_overrides=self.layer_mode_overrides,
            )
            self.layers.extend(block.layers)
            prev_layer = block.build_connections(prev_layer)
        
        # === Layer2: 2 x BasicBlock (128 channels, first with stride=2) ===
        for i in range(2):
            stride = 2 if i == 0 else 1
            in_ch = 64 if i == 0 else 128
            block = BasicBlock(
                sid, f"layer2_block{i}", self.enclave_mode,
                in_channels=in_ch,
                out_channels=128,
                stride=stride,
                layer_mode_overrides=self.layer_mode_overrides,
            )
            self.layers.extend(block.layers)
            prev_layer = block.build_connections(prev_layer)
        
        # === Layer3: 2 x BasicBlock (256 channels, first with stride=2) ===
        for i in range(2):
            stride = 2 if i == 0 else 1
            in_ch = 128 if i == 0 else 256
            block = BasicBlock(
                sid, f"layer3_block{i}", self.enclave_mode,
                in_channels=in_ch,
                out_channels=256,
                stride=stride,
                layer_mode_overrides=self.layer_mode_overrides,
            )
            self.layers.extend(block.layers)
            prev_layer = block.build_connections(prev_layer)
        
        # === Layer4: 2 x BasicBlock (512 channels, first with stride=2) ===
        for i in range(2):
            stride = 2 if i == 0 else 1
            in_ch = 256 if i == 0 else 512
            block = BasicBlock(
                sid, f"layer4_block{i}", self.enclave_mode,
                in_channels=in_ch,
                out_channels=512,
                stride=stride,
                layer_mode_overrides=self.layer_mode_overrides,
            )
            self.layers.extend(block.layers)
            prev_layer = block.build_connections(prev_layer)
        
        # === Classifier: AvgPool + Flatten + FC ===
        # Calculate feature map size after all downsampling
        # Input: input_size
        # After conv1 (stride=2): input_size / 2
        # After maxpool (stride=2): input_size / 4
        # After layer2 (stride=2): input_size / 8
        # After layer3 (stride=2): input_size / 16
        # After layer4 (stride=2): input_size / 32
        final_feature_size = self.input_size // 32
        
        # Use adaptive average pooling (pool to 1x1)
        avgpool_name = "avgpool"
        avgpool = SecretAvgpool2dLayer(
            sid, avgpool_name, self._get_mode(avgpool_name),
            filter_hw=max(1, final_feature_size),  # Ensure at least 1x1
            stride=1,
            padding=0,
            manually_register_prev=True,
            manually_register_next=True,
        )
        self.layers.append(avgpool)
        avgpool.register_prev_layer(prev_layer)
        
        flatten_name = "flatten"
        flatten = SecretFlattenLayer(
            sid, flatten_name, self._get_mode(flatten_name),
            manually_register_prev=True,
            manually_register_next=True,
        )
        self.layers.append(flatten)
        flatten.register_prev_layer(avgpool)
        
        fc_name = "fc"
        fc = SGXLinearBase(
            sid, fc_name, self._get_mode(fc_name),
            batch_size=self.batch_size,
            n_output_features=self.num_classes,
            n_input_features=512,
            manually_register_prev=True,
            manually_register_next=True,
        )
        self.layers.append(fc)
        fc.register_prev_layer(flatten)
        
        output_name = "output"
        output_layer = SecretOutputLayer(
            sid, output_name, self._get_mode(output_name),
            inference=True,
            manually_register_prev=True,
        )
        self.layers.append(output_layer)
        output_layer.register_prev_layer(fc)
    
    def print_architecture(self):
        """Print model architecture summary."""
        print(f"\n{'='*60}")
        print(f"SGXResNet18 Architecture")
        print(f"{'='*60}")
        print(f"Total layers: {len(self.layers)}")
        print(f"Input size: {self.input_size}x{self.input_size}")
        print(f"Num classes: {self.num_classes}")
        print(f"Default mode: {self.enclave_mode}")
        print(f"\nLayer Details:")
        print(f"{'#':<4} {'Name':<30} {'Type':<20} {'Mode':<10}")
        print(f"{'-'*70}")
        
        for idx, layer in enumerate(self.layers):
            layer_type = type(layer).__name__
            print(f"{idx:<4} {layer.LayerName:<30} {layer_type:<20} {layer.EnclaveMode.name:<10}")
        
        # Count modes
        mode_counts = {}
        for layer in self.layers:
            mode = layer.EnclaveMode.name
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        print(f"\nMode Distribution:")
        for mode, count in mode_counts.items():
            print(f"  {mode}: {count} layers")
        print(f"{'='*60}\n")


def test_resnet18():
    """Test ResNet-18 model creation."""
    print("Testing ResNet-18 model creation...")
    
    # Test 1: All CPU
    print("\n[Test 1] Creating ResNet-18 in CPU mode...")
    model_cpu = SGXResNet18(
        sid=0,
        enclave_mode=ExecutionModeOptions.CPU,
        batch_size=1,
        input_size=224,
        num_classes=1000,
    )
    model_cpu.print_architecture()
    
    # Test 2: Mixed mode (first half Enclave, second half CPU)
    print("\n[Test 2] Creating ResNet-18 with mixed mode...")
    overrides = {
        "input": ExecutionModeOptions.CPU,  # Input must be CPU
        # Layer1-2 in Enclave, Layer3-4 in CPU
        **{f"layer3_block{i}_{suffix}": ExecutionModeOptions.CPU
           for i in range(2)
           for suffix in ["conv1", "relu1", "conv2", "downsample", "skip", "add", "relu2"]},
        **{f"layer4_block{i}_{suffix}": ExecutionModeOptions.CPU
           for i in range(2)
           for suffix in ["conv1", "relu1", "conv2", "downsample", "skip", "add", "relu2"]},
        "avgpool": ExecutionModeOptions.CPU,
        "flatten": ExecutionModeOptions.CPU,
        "fc": ExecutionModeOptions.CPU,
        "output": ExecutionModeOptions.CPU,
    }
    
    model_mixed = SGXResNet18(
        sid=0,
        enclave_mode=ExecutionModeOptions.Enclave,
        batch_size=1,
        input_size=224,
        num_classes=1000,
        layer_mode_overrides=overrides,
    )
    model_mixed.print_architecture()
    
    print("✓ ResNet-18 model creation successful!")


if __name__ == '__main__':
    test_resnet18()

