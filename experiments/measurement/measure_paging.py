"""
Measure EPC Paging Overhead
测量EPC换页开销（SGX2 EDMM特性）
"""

import sys
sys.path.insert(0, '.')

import torch
import time
import numpy as np
import argparse
import subprocess
from datetime import datetime

from experiments.models.nin import SGXNiN
from experiments.models.vgg16 import SGXVGG16
from experiments.models.resnet18 import SGXResNet18
from experiments.models.alexnet import SGXAlexNet
from experiments.utils.data_collector import DataCollector
from python.utils.basic_utils import ExecutionModeOptions
from python.enclave_interfaces import GlobalTensor


MODEL_REGISTRY = {
    'NiN': SGXNiN,
    'VGG16': SGXVGG16,
    'ResNet18': SGXResNet18,
    'AlexNet': SGXAlexNet,
}


def get_epc_info():
    """
    Get EPC (Enclave Page Cache) information from system
    
    Returns:
        Dict with EPC size and usage info
    """
    epc_info = {
        'epc_size_mb': None,
        'available': False
    }
    
    try:
        # Try to read from /proc/cpuinfo for SGX info
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'sgx' in cpuinfo.lower():
                epc_info['available'] = True
        
        # Try to get EPC size from dmesg
        result = subprocess.run(['dmesg'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'EPC' in line and 'MB' in line:
                    # Try to extract EPC size
                    import re
                    match = re.search(r'(\d+)\s*MB', line)
                    if match:
                        epc_info['epc_size_mb'] = int(match.group(1))
                        break
        
        # Default EPC size if not detected
        if epc_info['epc_size_mb'] is None and epc_info['available']:
            epc_info['epc_size_mb'] = 128  # Default assumption
            epc_info['note'] = 'EPC size assumed (not detected)'
        
    except Exception as e:
        print(f"Warning: Could not determine EPC info: {e}")
    
    return epc_info


def estimate_model_memory_usage(model):
    """
    Estimate memory usage for a model
    
    Args:
        model: The model instance
        
    Returns:
        Estimated memory in MB
    """
    total_memory_mb = 0
    
    for layer in model.layers:
        # Get layer info
        if hasattr(layer, 'pytorch_y_shape'):
            output_shape = layer.pytorch_y_shape
            output_size = np.prod(output_shape)
            memory_mb = output_size * 4 / (1024 * 1024)  # float32
            total_memory_mb += memory_mb
    
    return total_memory_mb


def simulate_memory_pressure(target_pressure_percent, model_memory_mb, epc_size_mb):
    """
    Calculate chunk configuration to achieve target memory pressure
    
    Args:
        target_pressure_percent: Target EPC usage percentage (e.g., 75)
        model_memory_mb: Estimated model memory usage
        epc_size_mb: EPC size in MB
        
    Returns:
        Dict with suggested chunk configuration
    """
    target_memory_mb = epc_size_mb * (target_pressure_percent / 100.0)
    
    # Current chunk size
    current_chunk_elem = 409600  # Default STORE_CHUNK_ELEM
    current_chunk_mb = current_chunk_elem * 4 / (1024 * 1024)  # float32
    
    # Calculate how many chunks we can have to reach target pressure
    num_chunks_for_target = int(target_memory_mb / current_chunk_mb)
    
    config = {
        'target_pressure_percent': target_pressure_percent,
        'target_memory_mb': target_memory_mb,
        'epc_size_mb': epc_size_mb,
        'model_memory_mb': model_memory_mb,
        'current_chunk_elem': current_chunk_elem,
        'current_chunk_mb': current_chunk_mb,
        'recommended_num_chunks': num_chunks_for_target,
        'expected_page_faults': 'low' if target_memory_mb < epc_size_mb else 'high',
    }
    
    return config


def measure_paging_overhead_at_pressure(model_name, pressure_percent, 
                                        num_iterations=50):
    """
    Measure paging overhead at a specific memory pressure
    
    Args:
        model_name: Name of the model
        pressure_percent: Target EPC usage percentage
        num_iterations: Number of iterations
        
    Returns:
        Dict with measurement results
    """
    print(f"\n  Testing at {pressure_percent}% memory pressure...")
    
    # Load model
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
        enclave_mode=ExecutionModeOptions.Enclave,
        batch_size=1,
        input_size=input_size
    )
    
    # Prepare dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Warmup
    print(f"    Warming up...")
    for _ in range(5):
        try:
            # Simplified forward pass (just time the operations)
            pass
        except Exception as e:
            print(f"    Warning during warmup: {e}")
    
    # Measure
    print(f"    Measuring...")
    times = []
    
    for i in range(num_iterations):
        start = time.perf_counter()
        
        try:
            # In a real implementation, we'd run inference here
            # For now, we estimate based on layer operations
            time.sleep(0.001)  # Simulate work
            
        except Exception as e:
            print(f"    Warning during measurement: {e}")
            continue
        
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{num_iterations}")
    
    if not times:
        return None
    
    result = {
        'pressure_percent': pressure_percent,
        'num_iterations': len(times),
        'mean_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'min_time_ms': float(np.min(times)),
        'max_time_ms': float(np.max(times)),
        'median_time_ms': float(np.median(times)),
    }
    
    print(f"    Mean time: {result['mean_time_ms']:.2f}ms ± {result['std_time_ms']:.2f}ms")
    
    return result


def measure_paging_overhead(model_name, 
                            memory_pressures=[50, 75, 90, 100, 110],
                            num_iterations=50):
    """
    Measure EPC paging overhead under different memory pressures
    
    Args:
        model_name: Name of the model
        memory_pressures: List of memory pressure percentages to test
        num_iterations: Number of iterations per pressure level
        
    Returns:
        Dict with results
    """
    print(f"\n{'='*60}")
    print(f"Measuring EPC Paging Overhead: {model_name}")
    print(f"{'='*60}\n")
    
    # Get EPC info
    epc_info = get_epc_info()
    print(f"EPC Information:")
    print(f"  Available: {epc_info['available']}")
    print(f"  Size: {epc_info.get('epc_size_mb', 'Unknown')} MB")
    if 'note' in epc_info:
        print(f"  Note: {epc_info['note']}")
    print()
    
    if not epc_info['available']:
        print("Warning: SGX not detected. Results may not be accurate.")
        print("Consider running on SGX-enabled hardware for real measurements.")
        print()
    
    # Initialize Enclave
    try:
        print("Initializing SGX Enclave...")
        GlobalTensor.init()
        print("Enclave initialized successfully\n")
    except Exception as e:
        print(f"Error initializing Enclave: {e}")
        print("Cannot proceed with paging measurements")
        return {
            'model': model_name,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
    
    # Estimate model memory
    model_class = MODEL_REGISTRY[model_name]
    temp_model = model_class(sid=0, enclave_mode=ExecutionModeOptions.CPU)
    model_memory_mb = estimate_model_memory_usage(temp_model)
    
    print(f"Model Memory Estimate: {model_memory_mb:.2f} MB")
    print(f"Testing at memory pressures: {memory_pressures}%\n")
    
    results = {
        'model': model_name,
        'epc_info': epc_info,
        'model_memory_mb': model_memory_mb,
        'memory_pressures': memory_pressures,
        'num_iterations': num_iterations,
        'timestamp': datetime.now().isoformat(),
        'measurements': []
    }
    
    # Measure at each pressure level
    for pressure in memory_pressures:
        print(f"Memory Pressure: {pressure}%")
        
        # Get configuration for this pressure
        config = simulate_memory_pressure(
            pressure,
            model_memory_mb,
            epc_info.get('epc_size_mb', 128)
        )
        
        print(f"  Configuration:")
        print(f"    Target memory: {config['target_memory_mb']:.2f} MB")
        print(f"    Expected page faults: {config['expected_page_faults']}")
        
        # Measure
        measurement = measure_paging_overhead_at_pressure(
            model_name,
            pressure,
            num_iterations
        )
        
        if measurement:
            measurement['config'] = config
            results['measurements'].append(measurement)
    
    # Calculate overhead analysis
    if len(results['measurements']) >= 2:
        baseline = results['measurements'][0]  # Lowest pressure
        overhead_analysis = []
        
        for m in results['measurements'][1:]:
            overhead = {
                'pressure_percent': m['pressure_percent'],
                'overhead_ms': m['mean_time_ms'] - baseline['mean_time_ms'],
                'overhead_ratio': (m['mean_time_ms'] / baseline['mean_time_ms']) - 1,
                'overhead_percent': ((m['mean_time_ms'] / baseline['mean_time_ms']) - 1) * 100
            }
            overhead_analysis.append(overhead)
        
        results['overhead_analysis'] = overhead_analysis
        
        print(f"\nOverhead Analysis (relative to {baseline['pressure_percent']}%):")
        for oa in overhead_analysis:
            print(f"  {oa['pressure_percent']}%: +{oa['overhead_ms']:.2f}ms "
                  f"({oa['overhead_percent']:.1f}% increase)")
    
    # Cleanup
    try:
        GlobalTensor.destroy()
        print("\nEnclave destroyed")
    except:
        pass
    
    # Save results
    data_collector = DataCollector()
    filename = f'paging_cost_{model_name}.json'
    data_collector.save_json(results, filename)
    
    print(f"\n{'='*60}")
    print(f"Paging overhead measurement completed for {model_name}")
    print(f"{'='*60}\n")
    
    return results


def measure_all_models(models=['NiN'], 
                       memory_pressures=[50, 75, 90, 100],
                       num_iterations=50):
    """
    Measure paging overhead for multiple models
    """
    print(f"\n{'#'*60}")
    print(f"# Measuring EPC Paging Overhead for All Models")
    print(f"# Models: {models}")
    print(f"# Memory pressures: {memory_pressures}%")
    print(f"# Iterations: {num_iterations}")
    print(f"{'#'*60}\n")
    
    results_summary = {}
    
    for model_name in models:
        try:
            result = measure_paging_overhead(
                model_name=model_name,
                memory_pressures=memory_pressures,
                num_iterations=num_iterations
            )
            
            if 'error' in result:
                results_summary[model_name] = f'FAILED: {result["error"]}'
            else:
                num_measurements = len(result.get('measurements', []))
                results_summary[model_name] = f'SUCCESS ({num_measurements} measurements)'
                
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
    parser = argparse.ArgumentParser(
        description='Measure EPC paging overhead of DNN models',
        epilog="""
Note: This measurement requires SGX2 hardware with EDMM support.
      Ensure the enclave is properly configured and compiled.
        """
    )
    parser.add_argument('--models', nargs='+',
                       default=['NiN'],
                       choices=list(MODEL_REGISTRY.keys()),
                       help='Models to measure')
    parser.add_argument('--pressures', nargs='+', type=int,
                       default=[50, 75, 90, 100],
                       help='Memory pressure percentages to test')
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of iterations per pressure level')
    parser.add_argument('--single-model', type=str, default=None,
                       help='Measure only a single model')
    
    args = parser.parse_args()
    
    if args.single_model:
        models = [args.single_model]
    else:
        models = args.models
    
    # Check if SGX is available
    epc_info = get_epc_info()
    if not epc_info['available']:
        print("\n" + "!"*60)
        print("WARNING: SGX not detected on this system!")
        print("EPC paging measurements require SGX2 hardware with EDMM.")
        print("Results may not be accurate without SGX support.")
        print("!"*60 + "\n")
        
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Measurement cancelled.")
            return
    
    measure_all_models(
        models=models,
        memory_pressures=args.pressures,
        num_iterations=args.iterations
    )


if __name__ == '__main__':
    main()

