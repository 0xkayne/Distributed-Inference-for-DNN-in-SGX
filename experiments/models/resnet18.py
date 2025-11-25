"""
ResNet18 Model - wrapper around existing TAOISM ResNet implementation
"""

import importlib
import time
from statistics import mean
import sys
sys.path.insert(0, '.')

from python.enclave_interfaces import GlobalTensor
from python.sgx_net import SecretNeuralNetwork
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
        
        # The underlying constructor exposes SGX-specific layer list via sgx_layers
        if hasattr(self.base_model, "sgx_layers"):
            self.layers = self.base_model.sgx_layers
        else:
            self.layers = getattr(self.base_model, "layers")
        self.model_name = 'ResNet18'
    
    def __str__(self):
        info = f"SGXResNet18 Model\n"
        info += f"  Input size: {self.input_size}x{self.input_size}\n"
        info += f"  Num classes: {self.num_classes}\n"
        info += f"  Num layers: {len(self.layers)}\n"
        info += f"  Enclave mode: {self.enclave_mode}\n"
        return info


def _ensure_torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("运行 ResNet18 测试需要安装 PyTorch") from exc


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


def _build_plain_resnet18(num_classes, input_size):
    torch = _ensure_torch()
    try:
        resnet_cifar = importlib.import_module("teeslice.resnet_cifar")
    except ImportError as exc:
        raise RuntimeError("CPU 模式需要 teeslice.resnet_cifar") from exc

    plain_model = resnet_cifar.resnet18(pretrained=False, num_classes=num_classes, img_size=input_size)
    return plain_model


def _run_plain_cpu_benchmark(model, warmup_runs, measured_runs):
    torch = _ensure_torch()
    device = torch.device("cpu")
    plain_model = _build_plain_resnet18(model.num_classes, model.input_size)
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


def test_resnet18():
    """Benchmark ResNet18 model creation and forward inference on CPU/Enclave."""
    print("Testing ResNet18 model...")

    batch_size = 1
    input_size = 32
    warmup_runs = 1
    measured_runs = 3

    for mode in [ExecutionModeOptions.CPU, ExecutionModeOptions.Enclave]:
    # for mode in [ExecutionModeOptions.CPU]:
        print(f"\nTesting mode: {mode}")
        model = SGXResNet18(
            sid=0,
            enclave_mode=mode,
            batch_size=batch_size,
            input_size=input_size,
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
    test_resnet18()

