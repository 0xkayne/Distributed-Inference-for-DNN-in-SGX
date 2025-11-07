#!/bin/bash

# Basic EDMM Functionality Test
# This script tests that the enclave can be initialized with EDMM support

set -e

echo "=========================================="
echo "  TAOISM SGX2 EDMM Basic Functionality Test"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "enclave.signed.so" ]; then
    echo "Error: enclave.signed.so not found. Please run from project root."
    exit 1
fi

# Source SGX environment
if [ -z "$SGX_SDK" ]; then
    export SGX_SDK=/opt/intel/sgxsdk
fi

if [ -f "$SGX_SDK/environment" ]; then
    source "$SGX_SDK/environment"
    echo "✓ SGX SDK environment loaded"
else
    echo "⚠ Warning: Could not find SGX SDK environment file"
fi

# Set library path
export LD_LIBRARY_PATH=$SGX_SDK/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$(pwd):$PYTHONPATH

echo ""
echo "Testing enclave initialization with EDMM support..."
echo ""

# Create a minimal Python test script
cat > /tmp/test_edmm_init.py << 'EOF'
import sys
sys.path.insert(0, '.')

try:
    from python.enclave_interfaces import EnclaveInterface
    
    print("Attempting to initialize enclave with SGX2 EDMM support...")
    enclave = EnclaveInterface()
    
    print("\n✓ SUCCESS: Enclave initialized successfully!")
    print(f"  Enclave ID: {enclave.eid}")
    
    # Try to create a simple tensor to test memory allocation
    print("\nTesting tensor initialization (will use EDMM if available)...")
    test_id = 9999
    enclave.InitTensor(test_id, 1, 1, 32, 32)  # Small 32x32 tensor
    print("✓ Tensor initialization successful")
    
    print("\n" + "="*50)
    print("  All basic EDMM tests PASSED!")
    print("="*50)
    
except Exception as e:
    print(f"\n✗ FAILED: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

# Run the test
python3 /tmp/test_edmm_init.py

echo ""
echo "Test completed successfully!"
echo ""
echo "Next steps:"
echo "  - Run full model tests: bash teeslice/scripts/run_resnet_baseline.sh"
echo "  - Check EDMM statistics in the output logs"
echo "  - Monitor EPC usage: sudo dmesg | grep -i sgx"

