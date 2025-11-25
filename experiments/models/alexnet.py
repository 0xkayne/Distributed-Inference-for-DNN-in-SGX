"""
AlexNet Model for ImageNet
Paper: ImageNet Classification with Deep Convolutional Neural Networks (Krizhevsky et al., 2012)
"""

import importlib
import inspect
import time
from statistics import mean
import sys
sys.path.insert(0, '.')

from python.enclave_interfaces import GlobalTensor
from python.layers.sgx_conv_base import SGXConvBase
from python.layers.sgx_linear_base import SGXLinearBase
from python.layers.relu import SecretReLULayer
from python.layers.maxpool2d import SecretMaxpool2dLayer
from python.layers.flatten import SecretFlattenLayer
from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer
from python.sgx_net import SecretNeuralNetwork
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
        self.input_shape = [self.batch_size, 3, self.input_size, self.input_size]

        self.layers = self._build_network()
        self.model_name = 'AlexNet'

    def _build_network(self):
        """Build AlexNet network layers"""
        layers = []
        sid = self.sid
        mode = self.enclave_mode
        pool_mode = ExecutionModeOptions.CPU if mode is ExecutionModeOptions.Enclave else mode

        # Input layer
        layers.append(SecretInputLayer(
            sid, "input", self.input_shape, mode
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

        # MaxPool1: 3x3, stride 2 (CPU fallback in enclave mode)
        layers.append(SecretMaxpool2dLayer(
            sid, "pool1", pool_mode,
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
            sid, "pool2", pool_mode,
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
            sid, "pool3", pool_mode,
            filter_hw=3, stride=2, padding=0
        ))

        # Flatten
        layers.append(SecretFlattenLayer(sid, "flatten", ExecutionModeOptions.CPU if mode is ExecutionModeOptions.Enclave else mode))

        # FC layers (feature size after pooling: 6x6x256 = 9216)
        layers.append(SGXLinearBase(
            sid, "fc1", mode,
            batch_size=self.batch_size,
            n_output_features=4096,
            n_input_features=6 * 6 * 256
        ))
        layers.append(SecretReLULayer(sid, "relu_fc1", mode))

        layers.append(SGXLinearBase(
            sid, "fc2", mode,
            batch_size=self.batch_size,
            n_output_features=4096,
            n_input_features=4096
        ))
        layers.append(SecretReLULayer(sid, "relu_fc2", mode))

        layers.append(SGXLinearBase(
            sid, "fc3", mode,
            batch_size=self.batch_size,
            n_output_features=self.num_classes,
            n_input_features=4096
        ))

        # Output layer (loss computed on CPU)
        layers.append(SecretOutputLayer(sid, "output", ExecutionModeOptions.CPU, self.num_classes))

        return layers

    def __str__(self):
        info = f"SGXAlexNet Model\n"
        info += f"  Input size: {self.input_size}x{self.input_size}\n"
        info += f"  Num classes: {self.num_classes}\n"
        info += f"  Num layers: {len(self.layers)}\n"
        info += f"  Enclave mode: {self.enclave_mode}\n"
        return info


def _ensure_torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("运行 AlexNet 测试需要安装 PyTorch") from exc


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


def _build_plain_alexnet(num_classes):
    torch = _ensure_torch()
    try:
        torchvision_models = importlib.import_module("torchvision.models")
    except ImportError as exc:
        raise RuntimeError("CPU 模式需要安装 torchvision") from exc

    alex_sig = inspect.signature(torchvision_models.alexnet)
    if "weights" in alex_sig.parameters:
        plain_model = torchvision_models.alexnet(weights=None)
    else:
        plain_model = torchvision_models.alexnet(pretrained=False)
    in_features = plain_model.classifier[-1].in_features
    plain_model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return plain_model


def _run_plain_cpu_benchmark(model, warmup_runs, measured_runs):
    torch = _ensure_torch()
    device = torch.device("cpu")
    plain_model = _build_plain_alexnet(model.num_classes)
    plain_model.eval().to(device)

    inference_times = []
    total_runs = warmup_runs + measured_runs
    input_shape = (model.batch_size, 3, model.input_size, model.input_size)

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


def test_alexnet():
    """Benchmark AlexNet model creation and forward inference in CPU/Enclave modes."""
    print("Testing AlexNet model...")

    batch_size = 1
    input_size = 224
    warmup_runs = 1
    measured_runs = 3

    for mode in [ExecutionModeOptions.CPU, ExecutionModeOptions.Enclave]:
        print(f"\nTesting mode: {mode}")
        model = SGXAlexNet(
            sid=0,
            enclave_mode=mode,
            batch_size=batch_size,
            input_size=input_size,
            num_classes=1000,
        )
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
    test_alexnet()


