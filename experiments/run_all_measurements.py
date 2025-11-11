#!/usr/bin/env python3
"""
Batch runner for all measurements
批量运行所有测量实验
"""

import sys
sys.path.insert(0, '.')

import argparse
import time
from datetime import datetime

# Import measurement modules
from experiments.measurement.measure_computation import measure_all_models as measure_computation_all
from experiments.measurement.measure_communication import measure_all_models as measure_communication_all
from experiments.measurement.measure_security import measure_all_models as measure_security_all
from experiments.measurement.measure_paging import measure_all_models as measure_paging_all


# Default model list (start with simpler models)
DEFAULT_MODELS = ['NiN', 'ResNet18']  # Can add 'AlexNet', 'VGG16', 'InceptionV3', 'InceptionV4' later
ALL_MODELS = ['NiN', 'ResNet18', 'AlexNet', 'VGG16', 'InceptionV3', 'InceptionV4']


def print_section_header(title):
    """Print a formatted section header"""
    print(f"\n\n")
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()


def run_computation_measurements(models, devices, batch_sizes, iterations):
    """Run computation cost measurements"""
    print_section_header("PHASE 1: COMPUTATION COST MEASUREMENT")
    
    print(f"Configuration:")
    print(f"  Models: {models}")
    print(f"  Devices: {devices}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Iterations: {iterations}")
    print()
    
    start_time = time.time()
    
    try:
        measure_computation_all(
            models=models,
            devices=devices,
            batch_sizes=batch_sizes,
            num_iterations=iterations
        )
        status = "✓ COMPLETED"
    except Exception as e:
        print(f"\nError in computation measurements: {e}")
        import traceback
        traceback.print_exc()
        status = "✗ FAILED"
    
    elapsed = time.time() - start_time
    print(f"\nPhase 1 Status: {status}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    
    return status == "✓ COMPLETED"


def run_communication_measurements(models, bandwidths, iterations):
    """Run communication cost measurements"""
    print_section_header("PHASE 2: COMMUNICATION COST MEASUREMENT")
    
    print(f"Configuration:")
    print(f"  Models: {models}")
    print(f"  Bandwidths: {bandwidths} Mbps")
    print(f"  Iterations: {iterations}")
    print()
    
    start_time = time.time()
    
    try:
        measure_communication_all(
            models=models,
            bandwidths=bandwidths,
            num_iterations=iterations
        )
        status = "✓ COMPLETED"
    except Exception as e:
        print(f"\nError in communication measurements: {e}")
        import traceback
        traceback.print_exc()
        status = "✗ FAILED"
    
    elapsed = time.time() - start_time
    print(f"\nPhase 2 Status: {status}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    
    return status == "✓ COMPLETED"


def run_security_measurements(models, batch_size, iterations):
    """Run security overhead measurements"""
    print_section_header("PHASE 3: SECURITY OVERHEAD MEASUREMENT")
    
    print(f"Configuration:")
    print(f"  Models: {models}")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {iterations}")
    print()
    
    start_time = time.time()
    
    try:
        measure_security_all(
            models=models,
            batch_size=batch_size,
            num_iterations=iterations
        )
        status = "✓ COMPLETED"
    except Exception as e:
        print(f"\nError in security measurements: {e}")
        import traceback
        traceback.print_exc()
        status = "✗ FAILED"
    
    elapsed = time.time() - start_time
    print(f"\nPhase 3 Status: {status}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    
    return status == "✓ COMPLETED"


def run_all_measurements(models=None, quick_test=False, include_paging=False):
    """
    Run all measurements
    
    Args:
        models: List of models to test (None = use defaults)
        quick_test: If True, use reduced iterations for quick testing
        include_paging: If True, include EPC paging measurements
    """
    if models is None:
        models = DEFAULT_MODELS
    
    # Configuration
    if quick_test:
        print("\n*** QUICK TEST MODE - Reduced iterations ***\n")
        devices = ['CPU']  # Only CPU for quick test
        batch_sizes = [1]
        bandwidths = [100]  # Only one bandwidth
        iterations = 10  # Reduced iterations
        memory_pressures = [50, 75]  # Fewer pressure points
    else:
        devices = ['CPU']  # Start with CPU, add 'Enclave' later
        batch_sizes = [1]
        bandwidths = [10, 100, 1000]  # Low/Medium/High bandwidth
        iterations = 100
        memory_pressures = [50, 75, 90, 100]
    
    # Start timestamp
    start_time = time.time()
    start_datetime = datetime.now()
    
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#  TAOISM THESIS EXPERIMENTS - PHASE 1: MEASUREMENT SUITE         #")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print(f"\nExperiment started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models to test: {models}")
    print(f"Quick test mode: {quick_test}")
    print(f"Include paging: {include_paging}")
    print()
    
    # Track results
    results = {
        'computation': False,
        'communication': False,
        'security': False,
        'paging': False,
    }
    
    # Phase 1: Computation Cost
    results['computation'] = run_computation_measurements(
        models=models,
        devices=devices,
        batch_sizes=batch_sizes,
        iterations=iterations
    )
    
    # Phase 2: Communication Cost
    results['communication'] = run_communication_measurements(
        models=models,
        bandwidths=bandwidths,
        iterations=iterations
    )
    
    # Phase 3: Security Overhead
    results['security'] = run_security_measurements(
        models=models,
        batch_size=1,
        iterations=iterations
    )
    
    # Phase 4: EPC Paging (Optional)
    if include_paging:
        print_section_header("PHASE 4: EPC PAGING OVERHEAD MEASUREMENT")
        print(f"Configuration:")
        print(f"  Models: {models}")
        print(f"  Memory pressures: {memory_pressures}%")
        print(f"  Iterations: {iterations}")
        print()
        
        start_phase = time.time()
        try:
            measure_paging_all(
                models=models,
                memory_pressures=memory_pressures,
                num_iterations=iterations
            )
            status = "✓ COMPLETED"
        except Exception as e:
            print(f"\nError in paging measurements: {e}")
            import traceback
            traceback.print_exc()
            status = "✗ FAILED"
        
        elapsed = time.time() - start_phase
        print(f"\nPhase 4 Status: {status}")
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        results['paging'] = (status == "✓ COMPLETED")
    
    # Final summary
    total_time = time.time() - start_time
    end_datetime = datetime.now()
    
    print("\n\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#  EXPERIMENT COMPLETED - FINAL SUMMARY                           #")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    print(f"\nStarted:  {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print()
    print("Phase Results:")
    print(f"  Phase 1 (Computation):   {'✓ PASS' if results['computation'] else '✗ FAIL'}")
    print(f"  Phase 2 (Communication): {'✓ PASS' if results['communication'] else '✗ FAIL'}")
    print(f"  Phase 3 (Security):      {'✓ PASS' if results['security'] else '✗ FAIL'}")
    if include_paging:
        print(f"  Phase 4 (Paging):        {'✓ PASS' if results['paging'] else '✗ FAIL'}")
    print()
    
    all_passed = all(results.values())
    if all_passed:
        print("🎉 ALL MEASUREMENTS COMPLETED SUCCESSFULLY!")
    else:
        print("⚠️  Some measurements failed. Please check the logs above.")
    
    print()
    print("Results saved to: experiments/data/")
    print("#" * 70)
    print()
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Run all measurement experiments for TAOISM thesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 2 models and reduced iterations
  python experiments/run_all_measurements.py --quick-test

  # Full test with default models (NiN, ResNet18)
  python experiments/run_all_measurements.py

  # Test specific models
  python experiments/run_all_measurements.py --models NiN AlexNet

  # Test only specific phases
  python experiments/run_all_measurements.py --phases computation communication
        """
    )
    
    parser.add_argument('--models', nargs='+',
                       choices=['NiN', 'VGG16', 'ResNet18', 'AlexNet', 'InceptionV3', 'InceptionV4', 'all'],
                       default=None,
                       help='Models to test (default: NiN, ResNet18)')
    
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode with reduced iterations')
    
    parser.add_argument('--include-paging', action='store_true',
                       help='Include EPC paging overhead measurements (requires SGX2)')
    
    parser.add_argument('--phases', nargs='+',
                       choices=['computation', 'communication', 'security', 'paging', 'all'],
                       default=['all'],
                       help='Which measurement phases to run')
    
    args = parser.parse_args()
    
    # Handle 'all' models selection
    if args.models and 'all' in args.models:
        models = ALL_MODELS
    else:
        models = args.models
    
    # Run measurements
    success = run_all_measurements(
        models=models,
        quick_test=args.quick_test,
        include_paging=args.include_paging
    )
    
    # Exit code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

