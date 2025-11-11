"""
ResNet18 Model - wrapper around existing TAOISM ResNet implementation
"""

import sys
sys.path.insert(0, '.')

from teeslice.sgx_resnet_cifar import secret_resnet18
from python.utils.basic_utils import ExecutionModeOptions


class SGXResNet18:
    """
    ResNet18 wrapper for experiments
    Uses the existing TAOISM ResNet18 implementation
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
            input_size: Input image size (32 for CIFAR-10)
        """
        self.sid = sid
        self.num_classes = num_classes
        self.enclave_mode = enclave_mode
        self.batch_size = batch_size
        self.input_size = input_size
        
        # Use existing ResNet18 function implementation
        self.base_model = secret_resnet18(
            pretrained=False,
            EnclaveMode=enclave_mode,
            sid=sid,
            batch_size=batch_size,
            num_classes=num_classes
        )
        
        self.layers = self.base_model.layers
        self.model_name = 'ResNet18'
    
    def __str__(self):
        info = f"SGXResNet18 Model\n"
        info += f"  Input size: {self.input_size}x{self.input_size}\n"
        info += f"  Num classes: {self.num_classes}\n"
        info += f"  Num layers: {len(self.layers)}\n"
        info += f"  Enclave mode: {self.enclave_mode}\n"
        return info


def test_resnet18():
    """Test ResNet18 model creation"""
    print("Testing ResNet18 model...")
    
    model = SGXResNet18(sid=0, enclave_mode=ExecutionModeOptions.CPU)
    print(model)
    print(f"Successfully created {len(model.layers)} layers")


if __name__ == '__main__':
    test_resnet18()

