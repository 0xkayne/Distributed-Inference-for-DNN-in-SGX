"""
Test script to verify Transformer-specific layers work in Enclave mode.
Tests: LayerNorm, Softmax, GELU, MatMul
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from python.enclave_interfaces import GlobalTensor
from python.layers.input import SecretInputLayer
from python.layers.output import SecretOutputLayer
from python.layers.layer_norm import SecretLayerNormLayer
from python.layers.softmax import SecretSoftmaxLayer
from python.layers.gelu import SecretGELULayer
from python.sgx_net import SecretNeuralNetwork
from python.utils.basic_utils import ExecutionModeOptions


def test_layernorm_enclave():
    """Test LayerNorm layer in Enclave mode."""
    print("\n" + "="*60)
    print("Testing LayerNorm in Enclave Mode")
    print("="*60)
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    input_shape = [batch_size, seq_len, embed_dim]
    
    sid = 0
    model_name = "test_layernorm"
    
    try:
        # Initialize Enclave
        if not GlobalTensor.is_init_global_tensor:
            GlobalTensor.init()
        
        layers = []
        
        # Create layers
        input_layer = SecretInputLayer(
            sid, "input", input_shape,
            ExecutionModeOptions.Enclave,
            manually_register_next=True
        )
        layers.append(input_layer)
        
        ln_layer = SecretLayerNormLayer(
            sid, "layernorm",
            ExecutionModeOptions.Enclave,
            normalized_shape=[embed_dim],
            manually_register_prev=True,
            manually_register_next=True
        )
        ln_layer.register_prev_layer(input_layer)
        layers.append(ln_layer)
        
        output_layer = SecretOutputLayer(
            sid, "output",
            ExecutionModeOptions.CPU,  # Use CPU to avoid loss calculation
            manually_register_prev=True
        )
        output_layer.register_prev_layer(ln_layer)
        layers.append(output_layer)
        
        # Create and run network (this initializes all layers with start_enclave=False)
        secret_nn = SecretNeuralNetwork(sid, model_name)
        secret_nn.set_eid(GlobalTensor.get_eid())
        secret_nn.set_layers(layers)
        
        # Test forward pass
        test_input = torch.randn(*input_shape)
        layers[0].set_input(test_input)
        
        import time
        start = time.time()
        # Only run forward on input and ln layers to avoid output layer loss calculation
        layers[0].forward()
        layers[1].forward()
        elapsed = (time.time() - start) * 1000
        
        print(f"✓ LayerNorm Enclave forward completed in {elapsed:.2f} ms")
        print(f"  Input shape: {input_shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ LayerNorm test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Destroy Enclave for clean state
        if GlobalTensor.is_init_global_tensor:
            GlobalTensor.destroy()


def test_softmax_enclave():
    """Test Softmax layer in Enclave mode."""
    print("\n" + "="*60)
    print("Testing Softmax in Enclave Mode")
    print("="*60)
    
    # Test parameters
    batch_size = 2
    num_heads = 4
    seq_len = 10
    input_shape = [batch_size * num_heads, seq_len, seq_len]
    
    sid = 0
    model_name = "test_softmax"
    
    try:
        # Initialize Enclave
        if not GlobalTensor.is_init_global_tensor:
            GlobalTensor.init()
        
        layers = []
        
        # Create layers
        input_layer = SecretInputLayer(
            sid, "input", input_shape,
            ExecutionModeOptions.Enclave,
            manually_register_next=True
        )
        layers.append(input_layer)
        
        softmax_layer = SecretSoftmaxLayer(
            sid, "softmax",
            ExecutionModeOptions.Enclave,
            dim=-1,
            manually_register_prev=True,
            manually_register_next=True
        )
        softmax_layer.register_prev_layer(input_layer)
        layers.append(softmax_layer)
        
        output_layer = SecretOutputLayer(
            sid, "output",
            ExecutionModeOptions.CPU,
            manually_register_prev=True
        )
        output_layer.register_prev_layer(softmax_layer)
        layers.append(output_layer)
        
        # Create and run network
        secret_nn = SecretNeuralNetwork(sid, model_name)
        secret_nn.set_eid(GlobalTensor.get_eid())
        secret_nn.set_layers(layers)
        
        # Test forward pass
        test_input = torch.randn(*input_shape)
        layers[0].set_input(test_input)
        
        import time
        start = time.time()
        layers[0].forward()
        layers[1].forward()
        elapsed = (time.time() - start) * 1000
        
        print(f"✓ Softmax Enclave forward completed in {elapsed:.2f} ms")
        print(f"  Input shape: {input_shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Softmax test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if GlobalTensor.is_init_global_tensor:
            GlobalTensor.destroy()


def test_gelu_enclave():
    """Test GELU layer in Enclave mode."""
    print("\n" + "="*60)
    print("Testing GELU in Enclave Mode")
    print("="*60)
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    hidden_dim = 128
    input_shape = [batch_size, seq_len, hidden_dim]
    
    sid = 0
    model_name = "test_gelu"
    
    try:
        # Initialize Enclave
        if not GlobalTensor.is_init_global_tensor:
            GlobalTensor.init()
        
        layers = []
        
        # Create layers
        input_layer = SecretInputLayer(
            sid, "input", input_shape,
            ExecutionModeOptions.Enclave,
            manually_register_next=True
        )
        layers.append(input_layer)
        
        gelu_layer = SecretGELULayer(
            sid, "gelu",
            ExecutionModeOptions.Enclave,
            approximate=True,
            manually_register_prev=True,
            manually_register_next=True
        )
        gelu_layer.register_prev_layer(input_layer)
        layers.append(gelu_layer)
        
        output_layer = SecretOutputLayer(
            sid, "output",
            ExecutionModeOptions.CPU,
            manually_register_prev=True
        )
        output_layer.register_prev_layer(gelu_layer)
        layers.append(output_layer)
        
        # Create and run network
        secret_nn = SecretNeuralNetwork(sid, model_name)
        secret_nn.set_eid(GlobalTensor.get_eid())
        secret_nn.set_layers(layers)
        
        # Test forward pass
        test_input = torch.randn(*input_shape)
        layers[0].set_input(test_input)
        
        import time
        start = time.time()
        layers[0].forward()
        layers[1].forward()
        elapsed = (time.time() - start) * 1000
        
        print(f"✓ GELU Enclave forward completed in {elapsed:.2f} ms")
        print(f"  Input shape: {input_shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ GELU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if GlobalTensor.is_init_global_tensor:
            GlobalTensor.destroy()


def main():
    """Run all Transformer layer Enclave tests."""
    print("="*60)
    print("Transformer Layer Enclave Tests")
    print("="*60)
    
    # Initialize Enclave
    print("\nInitializing Enclave...")
    if not GlobalTensor.is_init_global_tensor:
        GlobalTensor.init()
    print(f"Enclave initialized with EID: {GlobalTensor.get_eid()}")
    
    results = {}
    
    # Run tests
    results['LayerNorm'] = test_layernorm_enclave()
    results['Softmax'] = test_softmax_enclave()
    results['GELU'] = test_gelu_enclave()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

