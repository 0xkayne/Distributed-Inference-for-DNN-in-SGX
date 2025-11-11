"""
Measure Security Overhead
测量TEE安全开销（Enclave vs CPU执行时间对比）
"""

import sys
sys.path.insert(0, '.')

import torch
import time
import numpy as np
import argparse
from datetime import datetime

from experiments.models.nin import SGXNiN
from experiments.models.vgg16 import SGXVGG16
from experiments.models.resnet18 import SGXResNet18
from experiments.models.alexnet import SGXAlexNet
from experiments.utils.layer_profiler import LayerProfiler
from experiments.utils.data_collector import DataCollector
from python.utils.basic_utils import ExecutionModeOptions
from python.enclave_interfaces import GlobalTensor


MODEL_REGISTRY = {
    'NiN': SGXNiN,
    'VGG16': SGXVGG16,
    'ResNet18': SGXResNet18,
    'AlexNet': SGXAlexNet,
}


def load_model(model_name, device, batch_size=1):
    """Load model with specified device"""
    model_class = MODEL_REGISTRY[model_name]
    enclave_mode = ExecutionModeOptions.CPU if device == 'CPU' else ExecutionModeOptions.Enclave
    
    if model_name in ['VGG16', 'AlexNet']:
        input_size = 224
        num_classes = 1000
    else:
        input_size = 32
        num_classes = 10
    
    return model_class(
        sid=0,
        num_classes=num_classes,
        enclave_mode=enclave_mode,
        batch_size=batch_size,
        input_size=input_size
    )


def measure_security_overhead(model_name, batch_size=1, num_iterations=100):
    """
    Measure security overhead by comparing CPU and Enclave execution
    
    Args:
        model_name: Name of the model
        batch_size: Batch size
        num_iterations: Number of iterations
        
    Returns:
        Dict with security overhead measurements
    """
    print(f"\n{'='*60}")
    print(f"Measuring Security Overhead: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    results = {
        'model': model_name,
        'batch_size': batch_size,
        'num_iterations': num_iterations,
        'timestamp': datetime.now().isoformat(),
        'layers': []
    }
    
    # === Measure CPU ===
    print("Phase 1: Measuring CPU execution...")
    cpu_model = load_model(model_name, 'CPU', batch_size)
    cpu_profiler = LayerProfiler(cpu_model, 'CPU')
    cpu_results = cpu_profiler.profile_all_layers(batch_size, num_iterations)
    cpu_summary = cpu_profiler.get_model_summary(cpu_results)
    
    print(f"  CPU total time: {cpu_summary['total_time_ms']:.2f}ms\n")
    
    # === Measure Enclave ===
    print("Phase 2: Measuring Enclave execution...")
    
    try:
        # Initialize Enclave
        print("  Initializing SGX Enclave...")
        GlobalTensor.init()
        print("  Enclave initialized\n")
        
        enclave_model = load_model(model_name, 'Enclave', batch_size)
        enclave_profiler = LayerProfiler(enclave_model, 'Enclave')
        enclave_results = enclave_profiler.profile_all_layers(batch_size, num_iterations)
        enclave_summary = enclave_profiler.get_model_summary(enclave_results)
        
        print(f"  Enclave total time: {enclave_summary['total_time_ms']:.2f}ms\n")
        
        # === Calculate Overhead ===
        print("Phase 3: Calculating security overhead...\n")
        
        # Match layers by index
        for idx, (cpu_layer, enclave_layer) in enumerate(zip(cpu_results, enclave_results)):
            cpu_time = cpu_layer['mean_ms']
            enclave_time = enclave_layer['mean_ms']
            overhead = enclave_time - cpu_time
            overhead_ratio = overhead / cpu_time if cpu_time > 0 else 0
            overhead_percent = overhead_ratio * 100
            
            layer_result = {
                'layer_index': idx,
                'layer_name': cpu_layer['name'],
                'layer_type': cpu_layer['type'],
                'cpu_time_ms': cpu_time,
                'enclave_time_ms': enclave_time,
                'overhead_ms': overhead,
                'overhead_ratio': overhead_ratio,
                'overhead_percent': overhead_percent,
                'param_count': cpu_layer.get('param_count', 0),
                'memory_mb': cpu_layer.get('memory_mb', 0),
            }
            
            results['layers'].append(layer_result)
            
            print(f"  Layer {idx} ({cpu_layer['name']}): "
                  f"CPU={cpu_time:.2f}ms, Enclave={enclave_time:.2f}ms, "
                  f"Overhead={overhead_percent:.1f}%")
        
        # Model-level summary
        total_overhead = enclave_summary['total_time_ms'] - cpu_summary['total_time_ms']
        total_overhead_ratio = total_overhead / cpu_summary['total_time_ms']
        
        results['summary'] = {
            'cpu_total_ms': cpu_summary['total_time_ms'],
            'enclave_total_ms': enclave_summary['total_time_ms'],
            'total_overhead_ms': total_overhead,
            'total_overhead_ratio': total_overhead_ratio,
            'total_overhead_percent': total_overhead_ratio * 100,
            'avg_overhead_percent': np.mean([r['overhead_percent'] for r in results['layers']]),
            'median_overhead_percent': np.median([r['overhead_percent'] for r in results['layers']]),
        }
        
        print(f"\n  Model Summary:")
        print(f"    Total overhead: {total_overhead:.2f}ms ({total_overhead_ratio*100:.1f}%)")
        print(f"    Average layer overhead: {results['summary']['avg_overhead_percent']:.1f}%")
        
        # Cleanup
        GlobalTensor.destroy()
        print("\n  Enclave destroyed")
        
    except Exception as e:
        print(f"\nError measuring Enclave: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
    
    # Save results
    data_collector = DataCollector()
    filename = f'security_cost_{model_name}.json'
    data_collector.save_json(results, filename)
    
    print(f"\n{'='*60}")
    print(f"Security overhead measurement completed for {model_name}")
    print(f"{'='*60}\n")
    
    return results


def measure_all_models(models=['NiN', 'ResNet18'], 
                       batch_size=1, 
                       num_iterations=100):
    """
    Measure security overhead for multiple models
    """
    print(f"\n{'#'*60}")
    print(f"# Measuring Security Overhead for All Models")
    print(f"# Models: {models}")
    print(f"# Batch size: {batch_size}")
    print(f"# Iterations: {num_iterations}")
    print(f"{'#'*60}\n")
    
    results_summary = {}
    
    for model_name in models:
        try:
            result = measure_security_overhead(
                model_name=model_name,
                batch_size=batch_size,
                num_iterations=num_iterations
            )
            
            if 'error' in result:
                results_summary[model_name] = f'FAILED: {result["error"]}'
            else:
                overhead_pct = result['summary']['total_overhead_percent']
                results_summary[model_name] = f'SUCCESS (Overhead: {overhead_pct:.1f}%)'
                
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
        status_symbol = '✓' if 'SUCCESS' in status else '✗'
        print(f"  {status_symbol} {model}: {status}")
    print(f"{'#'*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Measure security overhead of DNN models')
    parser.add_argument('--models', nargs='+',
                       default=['NiN', 'ResNet18'],
                       choices=list(MODEL_REGISTRY.keys()),
                       help='Models to measure')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
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
        batch_size=args.batch_size,
        num_iterations=args.iterations
    )


if __name__ == '__main__':
    main()

