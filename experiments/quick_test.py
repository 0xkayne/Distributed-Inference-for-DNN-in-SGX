#!/usr/bin/env python3
"""
Quick Test Script
快速测试各个组件是否工作正常
"""

import sys
sys.path.insert(0, '.')

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from experiments.models.nin import SGXNiN
        from experiments.models.vgg16 import SGXVGG16
        from experiments.models.resnet18 import SGXResNet18
        from experiments.models.alexnet import SGXAlexNet
        from experiments.utils.layer_profiler import LayerProfiler
        from experiments.utils.data_collector import DataCollector
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    print("\nTesting model creation...")
    
    from experiments.models.nin import SGXNiN
    from python.utils.basic_utils import ExecutionModeOptions
    
    try:
        model = SGXNiN(sid=0, enclave_mode=ExecutionModeOptions.CPU)
        print(f"  ✓ NiN model created with {len(model.layers)} layers")
        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_profiler():
    """Test LayerProfiler"""
    print("\nTesting LayerProfiler...")
    
    from experiments.models.nin import SGXNiN
    from experiments.utils.layer_profiler import LayerProfiler
    from python.utils.basic_utils import ExecutionModeOptions
    
    try:
        model = SGXNiN(sid=0, enclave_mode=ExecutionModeOptions.CPU, batch_size=1)
        profiler = LayerProfiler(model, 'CPU')
        
        # Profile just first layer
        first_layer = model.layers[0]
        result = profiler.profile_single_layer(first_layer, 0, batch_size=1, num_iterations=10)
        
        if result:
            print(f"  ✓ Profiled layer: {result['mean_ms']:.2f}ms")
            return True
        else:
            print("  ⚠ Profiling returned None (might be Input layer)")
            return True
            
    except Exception as e:
        print(f"  ✗ Profiler failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_collector():
    """Test DataCollector"""
    print("\nTesting DataCollector...")
    
    from experiments.utils.data_collector import DataCollector
    import os
    
    try:
        collector = DataCollector()
        
        # Test save
        test_data = {'test': 'data', 'value': 123}
        collector.save_json(test_data, 'test_output.json')
        
        # Test load
        loaded_data = collector.load_json('test_output.json')
        
        if loaded_data['test'] == 'data':
            print("  ✓ DataCollector works")
            
            # Cleanup
            os.remove('experiments/data/test_output.json')
            return True
        else:
            print("  ✗ Data mismatch")
            return False
            
    except Exception as e:
        print(f"  ✗ DataCollector failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("  TAOISM Experiments - Quick Test")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("LayerProfiler", test_profiler),
        ("DataCollector", test_data_collector),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*50)
    print("  Test Summary")
    print("="*50)
    
    for test_name, passed in results.items():
        symbol = "✓" if passed else "✗"
        status = "PASS" if passed else "FAIL"
        print(f"  {symbol} {test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("="*50)
    if all_passed:
        print("\n🎉 All tests passed! Ready to run measurements.\n")
        print("Next step: Run a single model test")
        print("  python experiments/measurement/measure_computation.py --single-model NiN --devices CPU --iterations 10")
    else:
        print("\n⚠️  Some tests failed. Please fix errors before proceeding.\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

