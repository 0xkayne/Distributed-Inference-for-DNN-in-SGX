"""
Network in Network (NiN) for CIFAR-10
Paper: Network In Network (Lin et al., 2013)
"""

import importlib
import sys
import time
from statistics import mean
sys.path.insert(0, '.')

from python.enclave_interfaces import GlobalTensor

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
        self.input_channels = 3
        self.input_shape = [self.batch_size, self.input_channels, self.input_size, self.input_size]
        
        self.layers = self._build_network()
        self.model_name = 'NiN'
    
    def _build_network(self):
        """Build NiN network layers"""
        layers = []
        sid = self.sid
        mode = self.enclave_mode
        
        # Input layer
        layers.append(SecretInputLayer(
            sid, "input", self.input_shape, mode
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
        
        # Conv 1x1, 160 filters (mlpconv)
        layers.append(SGXConvBase(
            sid, "conv2_2", mode,
            n_output_channel=160, filter_hw=1, stride=1, padding=0
        ))
        layers.append(SecretReLULayer(sid, "relu2_2", mode))
        
        # Conv 1x1, 96 filters (mlpconv)
        layers.append(SGXConvBase(
            sid, "conv2_3", mode,
            n_output_channel=96, filter_hw=1, stride=1, padding=0
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


def _ensure_torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("运行 NiN 测试需要安装 PyTorch") from exc


def _run_inference_benchmark(model, warmup_runs=1, measured_runs=3):
    if model.enclave_mode is ExecutionModeOptions.Enclave:
        return _run_enclave_benchmark(model, warmup_runs, measured_runs)
    if model.enclave_mode is ExecutionModeOptions.CPU:
        return _run_plain_cpu_benchmark(model, warmup_runs, measured_runs)
    raise ValueError(f"Unsupported execution mode: {model.enclave_mode}")


def _run_enclave_benchmark(model, warmup_runs, measured_runs):
    torch = _ensure_torch()
    inference_times = []
    try:
        if not GlobalTensor.is_init_global_tensor:
            GlobalTensor.init()
        secret_nn = SecretNeuralNetwork(model.sid, model.model_name)
        secret_nn.set_eid(GlobalTensor.get_eid())
        secret_nn.set_layers(model.layers)

        total_runs = warmup_runs + measured_runs
        input_shape = tuple(model.layers[0].shape)
        for run_idx in range(total_runs):
            test_input = torch.randn(*input_shape)
            model.layers[0].set_input(test_input)
            elapsed_ms = secret_nn.forward_with_time()
            if run_idx >= warmup_runs:
                inference_times.append(elapsed_ms)
                print(f"Inference run {run_idx - warmup_runs + 1}: {elapsed_ms:.3f} ms")
    finally:
        if GlobalTensor.is_init_global_tensor:
            GlobalTensor.destroy()
    return inference_times


def _build_plain_nin(num_classes, input_size, input_channels=3):
    torch = _ensure_torch()
    nn = torch.nn
    gap_kernel = max(1, input_size // 4)

    layers = [
        nn.Conv2d(input_channels, 192, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(192, 160, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(160, 96, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(192, 160, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(160, 96, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(192, 192, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(192, num_classes, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=gap_kernel, stride=1),
        nn.Flatten(1),
    ]
    return nn.Sequential(*layers)


def _run_plain_cpu_benchmark(model, warmup_runs, measured_runs):
    torch = _ensure_torch()
    device = torch.device("cpu")
    plain_model = _build_plain_nin(model.num_classes, model.input_size, model.input_channels)
    plain_model.eval().to(device)

    inference_times = []
    total_runs = warmup_runs + measured_runs
    input_shape = (model.batch_size, model.input_channels, model.input_size, model.input_size)

    with torch.no_grad():
        for run_idx in range(total_runs):
            test_input = torch.randn(*input_shape, device=device)
            start = time.perf_counter()
            _ = plain_model(test_input)
            elapsed_ms = (time.perf_counter() - start) * 1e3
            if run_idx >= warmup_runs:
                inference_times.append(elapsed_ms)
                print(f"Inference run {run_idx - warmup_runs + 1}: {elapsed_ms:.3f} ms")
    return inference_times


def test_nin():
    """Test NiN model creation"""
    print("Testing NiN model...")

    batch_size = 1
    input_size = 32
    warmup_runs = 1
    measured_runs = 3

    for mode in [ExecutionModeOptions.CPU, ExecutionModeOptions.Enclave]:
        print(f"\nTesting mode: {mode}")
        model = SGXNiN(sid=0, enclave_mode=mode, batch_size=batch_size, input_size=input_size)
        print(model)
        print(f"Successfully created {len(model.layers)} layers")

        inference_times = _run_inference_benchmark(
            model,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )

        if inference_times:
            avg_time = mean(inference_times)
            min_time = min(inference_times)
            max_time = max(inference_times)
            print(
                f"{mode.name} 平均推理时间: {avg_time:.3f} ms "
                f"(最小 {min_time:.3f} ms, 最大 {max_time:.3f} ms)"
            )


if __name__ == '__main__':
    test_nin()

