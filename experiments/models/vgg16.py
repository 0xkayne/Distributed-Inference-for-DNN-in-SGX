"""
VGG16 Model for ImageNet
Paper: Very Deep Convolutional Networks for Large-Scale Image Recognition (Simonyan & Zisserman, 2014)
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


class SGXVGG16:
    """
    VGG16 Model

    Architecture:
    - Conv Block 1: Conv3x3(64)-Conv3x3(64)-MaxPool
    - Conv Block 2: Conv3x3(128)-Conv3x3(128)-MaxPool
    - Conv Block 3: Conv3x3(256)-Conv3x3(256)-Conv3x3(256)-MaxPool
    - Conv Block 4: Conv3x3(512)-Conv3x3(512)-Conv3x3(512)-MaxPool
    - Conv Block 5: Conv3x3(512)-Conv3x3(512)-Conv3x3(512)-MaxPool
    - FC: FC(2048)-FC(2048)-FC(num_classes)

    Total: 13 conv layers + 3 FC layers
    """

    def __init__(self, sid=0, num_classes=1000,
                 enclave_mode=ExecutionModeOptions.Enclave,
                 batch_size=1, input_size=112):
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
        pool_mode = ExecutionModeOptions.CPU if mode is ExecutionModeOptions.Enclave else mode

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
                sid, f"pool{start_idx}", pool_mode,
                filter_hw=2, stride=2, padding=0
            ))

        return layers

    def _build_network(self):
        """Build VGG16 network layers"""
        layers = []
        sid = self.sid
        mode = self.enclave_mode
        flatten_mode = ExecutionModeOptions.CPU if mode is ExecutionModeOptions.Enclave else mode

        # Input layer
        layers.append(SecretInputLayer(
            sid, "input", self.input_shape, mode
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
        layers.append(SecretFlattenLayer(sid, "flatten", flatten_mode))

        # FC layers
        # After 5 pooling layers, spatial size reduces by 2^5 = 32
        feature_hw = max(self.input_size // (2 ** 5), 1)
        flattened_features = feature_hw * feature_hw * 512
        fc_hidden = 2048
        layers.append(SGXLinearBase(
            sid, "fc1", mode,
            batch_size=self.batch_size,
            n_output_features=fc_hidden,
            n_input_features=flattened_features
        ))
        layers.append(SecretReLULayer(sid, "relu_fc1", mode))

        layers.append(SGXLinearBase(
            sid, "fc2", mode,
            batch_size=self.batch_size,
            n_output_features=fc_hidden,
            n_input_features=fc_hidden
        ))
        layers.append(SecretReLULayer(sid, "relu_fc2", mode))

        layers.append(SGXLinearBase(
            sid, "fc3", mode,
            batch_size=self.batch_size,
            n_output_features=self.num_classes,
            n_input_features=4096
        ))

        # Output layer
        layers.append(SecretOutputLayer(sid, "output", ExecutionModeOptions.CPU, self.num_classes))

        return layers

    def __str__(self):
        info = f"SGXVGG16 Model\n"
        info += f"  Input size: {self.input_size}x{self.input_size}\n"
        info += f"  Num classes: {self.num_classes}\n"
        info += f"  Num layers: {len(self.layers)}\n"
        info += f"  Enclave mode: {self.enclave_mode}\n"
        return info


def _ensure_torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("运行 VGG16 测试需要安装 PyTorch") from exc


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


def _build_plain_vgg16(num_classes):
    torch = _ensure_torch()
    try:
        torchvision_models = importlib.import_module("torchvision.models")
    except ImportError as exc:
        raise RuntimeError("CPU 模式需要安装 torchvision") from exc

    vgg_sig = inspect.signature(torchvision_models.vgg16)
    if "weights" in vgg_sig.parameters:
        plain_model = torchvision_models.vgg16(weights=None)
    else:
        plain_model = torchvision_models.vgg16(pretrained=False)
    in_features = plain_model.classifier[-1].in_features
    plain_model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return plain_model


def _run_plain_cpu_benchmark(model, warmup_runs, measured_runs):
    torch = _ensure_torch()
    device = torch.device("cpu")
    plain_model = _build_plain_vgg16(model.num_classes)
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


def test_vgg16():
    """Benchmark VGG16 model creation and forward inference in CPU/Enclave modes."""
    print("Testing VGG16 model...")

    batch_size = 1
    input_size = 224
    warmup_runs = 1
    measured_runs = 3

    for mode in [ExecutionModeOptions.CPU, ExecutionModeOptions.Enclave]:
        print(f"\nTesting mode: {mode}")
        model = SGXVGG16(
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
    test_vgg16()
