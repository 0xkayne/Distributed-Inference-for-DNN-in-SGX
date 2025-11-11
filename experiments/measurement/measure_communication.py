"""
Measure Communication Cost
测量层间通信开销（数据传输时间）
"""

import sys
sys.path.insert(0, '.')

import torch
import time
import numpy as np
import argparse
import pickle
from datetime import datetime

from experiments.models.nin import SGXNiN
from experiments.models.vgg16 import SGXVGG16
from experiments.models.resnet18 import SGXResNet18
from experiments.models.alexnet import SGXAlexNet
from experiments.utils.data_collector import DataCollector
from python.utils.basic_utils import ExecutionModeOptions


MODEL_REGISTRY = {
    'NiN': SGXNiN,
    'VGG16': SGXVGG16,
    'ResNet18': SGXResNet18,
    'AlexNet': SGXAlexNet,
}


def measure_tensor_serialization(tensor, num_iterations=100):
    """
    Measure serialization and deserialization time
    
    Args:
        tensor: PyTorch tensor
        num_iterations: Number of iterations
        
    Returns:
        Dict with timing results
    """
    serialize_times = []
    deserialize_times = []
    
    for _ in range(num_iterations):
        # Serialize
        start = time.perf_counter()
        serialized = pickle.dumps(tensor)
        serialize_times.append(time.perf_counter() - start)
        
        # Deserialize
        start = time.perf_counter()
        _ = pickle.loads(serialized)
        deserialize_times.append(time.perf_counter() - start)
    
    return {
        'serialize_mean_ms': np.mean(serialize_times) * 1000,
        'deserialize_mean_ms': np.mean(deserialize_times) * 1000,
        'serialize_std_ms': np.std(serialize_times) * 1000,
        'deserialize_std_ms': np.std(deserialize_times) * 1000,
        'serialized_size_bytes': len(serialized),
        'serialized_size_mb': len(serialized) / (1024 * 1024),
    }


def estimate_transfer_time(data_size_bytes, bandwidth_mbps):
    """
    Estimate transfer time given data size and bandwidth
    
    Args:
        data_size_bytes: Size of data in bytes
        bandwidth_mbps: Network bandwidth in Mbps
        
    Returns:
        Transfer time in milliseconds
    """
    # Convert to bits
    data_size_bits = data_size_bytes * 8
    bandwidth_bps = bandwidth_mbps * 1_000_000
    
    # Time in seconds
    transfer_time_s = data_size_bits / bandwidth_bps
    
    # Convert to milliseconds
    return transfer_time_s * 1000


def measure_layer_communication(layer, layer_idx, bandwidths, num_iterations=100):
    """
    Measure communication cost for a single layer
    
    Args:
        layer: The layer to measure
        layer_idx: Layer index
        bandwidths: List of bandwidths to test (in Mbps)
        num_iterations: Number of iterations
        
    Returns:
        Dict with communication measurements
    """
    layer_name = getattr(layer, 'LayerName', f'layer_{layer_idx}')
    layer_type = layer.__class__.__name__
    
    # Skip Input/Output layers
    if layer_type in ['SecretInputLayer', 'SecretOutputLayer']:
        return None
    
    # Get output shape
    output_shape = None
    if hasattr(layer, 'pytorch_y_shape'):
        output_shape = layer.pytorch_y_shape
    
    # If shape not set, try to get from layer or use default
    if output_shape is None:
        # Use default shape based on common layer types
        if 'Conv' in layer_type:
            output_shape = [1, 64, 32, 32]  # Default conv output
        elif 'Linear' in layer_type:
            output_shape = [1, 512]  # Default linear output
        else:
            return None
    
    # Create dummy output tensor
    try:
        output_tensor = torch.randn(*output_shape)
    except Exception as e:
        print(f"Cannot create tensor for {layer_name}: {e}")
        return None
    
    # Measure serialization
    serial_result = measure_tensor_serialization(output_tensor, num_iterations)
    
    # Calculate transfer time for different bandwidths
    transfer_times = {}
    for bw in bandwidths:
        transfer_time = estimate_transfer_time(
            serial_result['serialized_size_bytes'], 
            bw
        )
        transfer_times[f'{bw}Mbps'] = transfer_time
    
    result = {
        'layer_index': layer_idx,
        'layer_name': layer_name,
        'layer_type': layer.__class__.__name__,
        'output_shape': list(output_shape),
        'output_size_elements': int(np.prod(output_shape)),
        'output_size_mb': int(np.prod(output_shape)) * 4 / (1024 * 1024),  # float32
        **serial_result,
        'transfer_times': transfer_times,
        'total_comm_cost': {}  # serialize + transfer + deserialize
    }
    
    # Calculate total communication cost
    for bw_key, transfer_time in transfer_times.items():
        total_time = (serial_result['serialize_mean_ms'] + 
                     transfer_time + 
                     serial_result['deserialize_mean_ms'])
        result['total_comm_cost'][bw_key] = total_time
    
    return result


def measure_model_communication(model_name, bandwidths=[10, 100, 1000], 
                                num_iterations=100):
    """
    Measure communication cost for entire model
    
    Args:
        model_name: Name of the model
        bandwidths: List of bandwidths in Mbps
        num_iterations: Number of iterations
        
    Returns:
        Dict with results
    """
    print(f"\n{'='*60}")
    print(f"Measuring Communication Cost: {model_name}")
    print(f"Bandwidths: {bandwidths} Mbps")
    print(f"{'='*60}\n")
    
    # Load model (CPU mode is sufficient, we just need shapes)
    model_class = MODEL_REGISTRY[model_name]
    
    if model_name in ['VGG16', 'AlexNet']:
        input_size = 224
        num_classes = 1000
    else:
        input_size = 32
        num_classes = 10
    
    model = model_class(
        sid=0,
        num_classes=num_classes,
        enclave_mode=ExecutionModeOptions.CPU,
        batch_size=1,
        input_size=input_size
    )
    
    # Measure each layer
    layer_results = []
    total_data_mb = 0
    
    for idx, layer in enumerate(model.layers):
        print(f"  Measuring layer {idx}: {getattr(layer, 'LayerName', 'unknown')}...", end=' ')
        
        result = measure_layer_communication(layer, idx, bandwidths, num_iterations)
        
        if result is not None:
            layer_results.append(result)
            total_data_mb += result['output_size_mb']
            print(f"{result['output_size_mb']:.2f}MB")
        else:
            print("SKIP")
    
    # Calculate model-level statistics
    model_summary = {
        'total_layers': len(layer_results),
        'total_data_mb': total_data_mb,
        'total_comm_cost': {}
    }
    
    for bw in bandwidths:
        bw_key = f'{bw}Mbps'
        total_cost = sum(r['total_comm_cost'][bw_key] for r in layer_results)
        model_summary['total_comm_cost'][bw_key] = total_cost
    
    results = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'bandwidths_mbps': bandwidths,
        'num_iterations': num_iterations,
        'layers': layer_results,
        'summary': model_summary
    }
    
    # Save results
    data_collector = DataCollector()
    filename = f'communication_cost_{model_name}.json'
    data_collector.save_json(results, filename)
    
    print(f"\n  Model Summary:")
    print(f"    Total data transferred: {total_data_mb:.2f}MB")
    for bw_key, cost in model_summary['total_comm_cost'].items():
        print(f"    Total comm cost ({bw_key}): {cost:.2f}ms")
    
    print(f"\n{'='*60}")
    print(f"Communication measurement completed for {model_name}")
    print(f"{'='*60}\n")
    
    return results


def measure_all_models(models=['NiN', 'ResNet18'], 
                       bandwidths=[10, 100, 1000],
                       num_iterations=100):
    """
    Measure communication cost for multiple models
    """
    print(f"\n{'#'*60}")
    print(f"# Measuring Communication Cost for All Models")
    print(f"# Models: {models}")
    print(f"# Bandwidths: {bandwidths} Mbps")
    print(f"# Iterations: {num_iterations}")
    print(f"{'#'*60}\n")
    
    results_summary = {}
    
    for model_name in models:
        try:
            result = measure_model_communication(
                model_name=model_name,
                bandwidths=bandwidths,
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
    parser = argparse.ArgumentParser(description='Measure communication cost of DNN models')
    parser.add_argument('--models', nargs='+',
                       default=['NiN', 'ResNet18'],
                       choices=list(MODEL_REGISTRY.keys()),
                       help='Models to measure')
    parser.add_argument('--bandwidths', nargs='+', type=int,
                       default=[10, 100, 1000],
                       help='Bandwidths to test in Mbps')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations')
    parser.add_argument('--single-model', type=str, default=None,
                       help='Measure only a single model')
    
    args = parser.parse_args()
    
    if args.single_model:
        models = [args.single_model]
    else:
        models = args.models
    
    measure_all_models(
        models=models,
        bandwidths=args.bandwidths,
        num_iterations=args.iterations
    )


if __name__ == '__main__':
    main()

