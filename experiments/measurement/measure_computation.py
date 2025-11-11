"""
Measure Computation Cost
测量各层在不同设备(CPU/GPU/Enclave)上的计算开销
"""

import sys
import os
sys.path.insert(0, '.')

import torch
import argparse
from datetime import datetime

# Import models
from experiments.models.nin import SGXNiN
from experiments.models.vgg16 import SGXVGG16
from experiments.models.resnet18 import SGXResNet18
from experiments.models.alexnet import SGXAlexNet

# Import utilities
from experiments.utils.layer_profiler import LayerProfiler
from experiments.utils.data_collector import DataCollector
from python.utils.basic_utils import ExecutionModeOptions
from python.enclave_interfaces import GlobalTensor


# Model registry
MODEL_REGISTRY = {
    'NiN': SGXNiN,
    'VGG16': SGXVGG16,
    'ResNet18': SGXResNet18,
    'AlexNet': SGXAlexNet,
}

# Note: InceptionV3/V4 can be added when needed:
# from experiments.models.inception_v3 import SGXInceptionV3
# from experiments.models.inception_v4 import SGXInceptionV4
# MODEL_REGISTRY['InceptionV3'] = SGXInceptionV3
# MODEL_REGISTRY['InceptionV4'] = SGXInceptionV4

# Device mode mapping
DEVICE_MODES = {
    'CPU': ExecutionModeOptions.CPU,
    'GPU': ExecutionModeOptions.GPU,
    'Enclave': ExecutionModeOptions.Enclave,
}


def load_model(model_name, device='CPU', batch_size=1):
    """
    Load model with specified device
    
    Args:
        model_name: Name of the model (NiN, VGG16, etc.)
        device: Device type (CPU, GPU, Enclave)
        batch_size: Batch size
        
    Returns:
        Model instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    enclave_mode = DEVICE_MODES[device]
    
    # Adjust parameters based on model
    if model_name in ['VGG16', 'AlexNet']:
        input_size = 224
        num_classes = 1000
    else:
        input_size = 32
        num_classes = 10
    
    print(f"Loading {model_name} for {device} mode...")
    model = model_class(
        sid=0,
        num_classes=num_classes,
        enclave_mode=enclave_mode,
        batch_size=batch_size,
        input_size=input_size
    )
    
    print(f"  Model created with {len(model.layers)} layers")
    return model


def measure_model_computation(model_name, devices=['CPU'], 
                              batch_sizes=[1], num_iterations=100):
    """
    Measure computation cost for a model across devices and batch sizes
    
    Args:
        model_name: Name of the model
        devices: List of devices to test
        batch_sizes: List of batch sizes to test
        num_iterations: Number of iterations for each measurement
        
    Returns:
        Dictionary with all results
    """
    print(f"\n{'='*60}")
    print(f"Measuring Computation Cost: {model_name}")
    print(f"{'='*60}\n")
    
    all_results = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'num_iterations': num_iterations,
        'devices': {}
    }
    
    data_collector = DataCollector()
    
    for device in devices:
        print(f"\n--- Device: {device} ---")
        all_results['devices'][device] = {}
        
        # Initialize Enclave if needed
        if device == 'Enclave':
            try:
                print("Initializing SGX Enclave...")
                GlobalTensor.init()
                print("Enclave initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize Enclave: {e}")
                print("Skipping Enclave measurements")
                continue
        
        for batch_size in batch_sizes:
            print(f"\n  Batch size: {batch_size}")
            
            try:
                # Load model
                model = load_model(model_name, device, batch_size)
                
                # Profile layers
                profiler = LayerProfiler(model, device)
                layer_results = profiler.profile_all_layers(
                    batch_size=batch_size,
                    num_iterations=num_iterations
                )
                
                # Get model summary
                summary = profiler.get_model_summary(layer_results)
                
                # Store results
                all_results['devices'][device][f'batch_{batch_size}'] = {
                    'layers': layer_results,
                    'summary': summary
                }
                
                # Save intermediate results for this device-batch combination
                data_collector.save_layer_results(
                    model_name=model_name,
                    device=device,
                    results=layer_results,
                    cost_type=f'computation_batch{batch_size}'
                )
                
                print(f"  Total inference time: {summary['total_time_ms']:.2f}ms")
                print(f"  Total parameters: {summary['total_params']:,}")
                
            except Exception as e:
                print(f"  Error measuring {model_name} on {device} with batch_size {batch_size}: {e}")
                import traceback
                traceback.print_exc()
        
        # Cleanup Enclave if needed
        if device == 'Enclave':
            try:
                GlobalTensor.destroy()
                print("\nEnclave destroyed")
            except:
                pass
    
    # Save aggregated results
    filename = f'computation_cost_{model_name}_aggregated.json'
    data_collector.save_json(all_results, filename)
    
    print(f"\n{'='*60}")
    print(f"Computation measurement completed for {model_name}")
    print(f"Results saved to experiments/data/")
    print(f"{'='*60}\n")
    
    return all_results


def measure_all_models(models=['NiN', 'ResNet18'], 
                       devices=['CPU'], 
                       batch_sizes=[1],
                       num_iterations=100):
    """
    Measure computation cost for multiple models
    
    Args:
        models: List of model names
        devices: List of devices
        batch_sizes: List of batch sizes
        num_iterations: Number of iterations
    """
    print(f"\n{'#'*60}")
    print(f"# Measuring Computation Cost for All Models")
    print(f"# Models: {models}")
    print(f"# Devices: {devices}")
    print(f"# Batch sizes: {batch_sizes}")
    print(f"# Iterations: {num_iterations}")
    print(f"{'#'*60}\n")
    
    results_summary = {}
    
    for model_name in models:
        try:
            result = measure_model_computation(
                model_name=model_name,
                devices=devices,
                batch_sizes=batch_sizes,
                num_iterations=num_iterations
            )
            results_summary[model_name] = 'SUCCESS'
        except Exception as e:
            print(f"\nFailed to measure {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results_summary[model_name] = f'FAILED: {str(e)}'
    
    # Print summary
    print(f"\n{'#'*60}")
    print("# Measurement Summary")
    print(f"{'#'*60}")
    for model, status in results_summary.items():
        status_symbol = '✓' if status == 'SUCCESS' else '✗'
        print(f"  {status_symbol} {model}: {status}")
    print(f"{'#'*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Measure computation cost of DNN models')
    parser.add_argument('--models', nargs='+', 
                       default=['NiN', 'ResNet18'],
                       choices=list(MODEL_REGISTRY.keys()),
                       help='Models to measure')
    parser.add_argument('--devices', nargs='+',
                       default=['CPU'],
                       choices=['CPU', 'GPU', 'Enclave'],
                       help='Devices to test')
    parser.add_argument('--batch-sizes', nargs='+', type=int,
                       default=[1],
                       help='Batch sizes to test')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations for each measurement')
    parser.add_argument('--single-model', type=str, default=None,
                       help='Measure only a single model')
    
    args = parser.parse_args()
    
    if args.single_model:
        models = [args.single_model]
    else:
        models = args.models
    
    measure_all_models(
        models=models,
        devices=args.devices,
        batch_sizes=args.batch_sizes,
        num_iterations=args.iterations
    )


if __name__ == '__main__':
    main()

