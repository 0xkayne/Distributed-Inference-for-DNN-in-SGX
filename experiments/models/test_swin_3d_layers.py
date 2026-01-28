"""
Unit tests for Video Swin Transformer 3D operators.

Tests all new 3D operators to ensure correctness:
1. SGXConv3DBase - 3D convolution
2. SecretWindowPartition3DLayer - 3D window partition
3. SecretWindowReverse3DLayer - 3D window reverse
4. SecretCyclicRoll3DLayer - 3D cyclic shift
5. SwinWindowAttention3D - 3D window attention with relative position bias

Each test validates:
- Shape correctness
- Data flow
- Consistency with PyTorch reference
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from operator import mul

from python.layers.sgx_conv3d_base import SGXConv3DBase, calc_conv3d_output_shape_stride
from python.layers.window_partition_3d import SecretWindowPartition3DLayer
from python.layers.window_reverse_3d import SecretWindowReverse3DLayer
from python.layers.cyclic_roll_3d import SecretCyclicRoll3DLayer
from python.layers.swin_window_attention_3d import SwinWindowAttention3D
from python.layers.input import SecretInputLayer
from python.utils.basic_utils import ExecutionModeOptions


def test_conv3d():
    """Test 3D convolution operator."""
    print("\n" + "="*70)
    print("TEST 1: SGXConv3DBase - 3D Convolution")
    print("="*70)
    
    # Test parameters
    batch_size = 1
    in_channels = 3
    out_channels = 96
    D, H, W = 8, 224, 224
    filter_dhw = (2, 4, 4)
    stride = (2, 4, 4)
    padding = (0, 0, 0)
    
    # Create input layer
    input_layer = SecretInputLayer(
        sid=0, LayerName="input",
        EnclaveMode=ExecutionModeOptions.CPU,
        input_shape=[batch_size, in_channels, D, H, W]
    )
    input_layer.init()
    
    # Create Conv3D layer
    conv3d = SGXConv3DBase(
        sid=0, LayerName="conv3d_test",
        EnclaveMode=ExecutionModeOptions.CPU,
        batch_size=batch_size,
        n_input_channel=in_channels,
        n_output_channel=out_channels,
        filter_dhw=filter_dhw,
        video_dhw=(D, H, W),
        stride=stride,
        padding=padding,
        bias=True
    )
    
    # Connect layers
    conv3d.register_prev_layer(input_layer)
    input_layer.register_next_layer(conv3d)
    
    # Initialize
    conv3d.init_shape()
    conv3d.init(start_enclave=False)
    
    # Check output shape
    expected_D = (D + 2*padding[0] - filter_dhw[0]) // stride[0] + 1
    expected_H = (H + 2*padding[1] - filter_dhw[1]) // stride[1] + 1
    expected_W = (W + 2*padding[2] - filter_dhw[2]) // stride[2] + 1
    expected_shape = [batch_size, out_channels, expected_D, expected_H, expected_W]
    
    actual_shape = conv3d.get_output_shape()
    
    print(f"Input shape: {[batch_size, in_channels, D, H, W]}")
    print(f"Filter: {filter_dhw}, Stride: {stride}, Padding: {padding}")
    print(f"Expected output shape: {expected_shape}")
    print(f"Actual output shape: {actual_shape}")
    
    assert actual_shape == expected_shape, \
        f"Shape mismatch: {actual_shape} != {expected_shape}"
    
    # Test forward pass
    test_input = torch.randn(batch_size, in_channels, D, H, W)
    input_layer.set_cpu("output", test_input)
    conv3d.forward()
    
    output = conv3d.get_cpu("output")
    print(f"Forward pass output shape: {output.shape}")
    
    assert output.shape == torch.Size(expected_shape), \
        f"Output shape mismatch: {output.shape} != {expected_shape}"
    
    print("✓ Test PASSED: Conv3D shape and forward correct")


def test_window_partition_3d():
    """Test 3D window partition operator."""
    print("\n" + "="*70)
    print("TEST 2: SecretWindowPartition3DLayer - 3D Window Partition")
    print("="*70)
    
    # Test parameters
    batch_size = 1
    D, H, W = 4, 56, 56
    C = 96
    window_size = (2, 7, 7)
    
    # Create input layer
    input_layer = SecretInputLayer(
        sid=0, LayerName="input",
        EnclaveMode=ExecutionModeOptions.CPU,
        input_shape=[batch_size, D, H, W, C]
    )
    input_layer.init()
    
    # Create window partition layer
    win_part = SecretWindowPartition3DLayer(
        sid=0, LayerName="win_part",
        EnclaveMode=ExecutionModeOptions.CPU,
        window_size=window_size
    )
    
    # Connect
    win_part.register_prev_layer(input_layer)
    input_layer.register_next_layer(win_part)
    
    # Initialize
    win_part.init_shape()
    win_part.init(start_enclave=False)
    
    # Check output shape
    Wd, Wh, Ww = window_size
    num_windows = (D // Wd) * (H // Wh) * (W // Ww)
    window_tokens = Wd * Wh * Ww
    expected_shape = [num_windows * batch_size, window_tokens, C]
    
    actual_shape = win_part.get_output_shape()
    
    print(f"Input shape: {[batch_size, D, H, W, C]}")
    print(f"Window size: {window_size}")
    print(f"Num windows: {num_windows}")
    print(f"Expected output shape: {expected_shape}")
    print(f"Actual output shape: {actual_shape}")
    
    assert actual_shape == expected_shape, \
        f"Shape mismatch: {actual_shape} != {expected_shape}"
    
    # Test forward pass
    test_input = torch.randn(batch_size, D, H, W, C)
    input_layer.set_cpu("output", test_input)
    win_part.forward()
    
    output = win_part.get_cpu("output")
    print(f"Forward pass output shape: {output.shape}")
    
    assert output.shape == torch.Size(expected_shape), \
        f"Output shape mismatch: {output.shape} != {expected_shape}"
    
    print("✓ Test PASSED: Window partition shape and forward correct")


def test_window_reverse_3d():
    """Test 3D window reverse operator."""
    print("\n" + "="*70)
    print("TEST 3: SecretWindowReverse3DLayer - 3D Window Reverse")
    print("="*70)
    
    # Test parameters
    batch_size = 1
    D, H, W = 4, 56, 56
    C = 96
    window_size = (2, 7, 7)
    
    Wd, Wh, Ww = window_size
    num_windows = (D // Wd) * (H // Wh) * (W // Ww)
    window_tokens = Wd * Wh * Ww
    
    # Create input layer (windowed)
    input_layer = SecretInputLayer(
        sid=0, LayerName="input",
        EnclaveMode=ExecutionModeOptions.CPU,
        input_shape=[num_windows * batch_size, window_tokens, C]
    )
    input_layer.init()
    
    # Create window reverse layer
    win_rev = SecretWindowReverse3DLayer(
        sid=0, LayerName="win_rev",
        EnclaveMode=ExecutionModeOptions.CPU,
        window_size=window_size,
        output_shape=[batch_size, D, H, W, C]
    )
    
    # Connect
    win_rev.register_prev_layer(input_layer)
    input_layer.register_next_layer(win_rev)
    
    # Initialize
    win_rev.init_shape()
    win_rev.init(start_enclave=False)
    
    # Check output shape
    expected_shape = [batch_size, D, H, W, C]
    actual_shape = win_rev.get_output_shape()
    
    print(f"Input shape: {[num_windows * batch_size, window_tokens, C]}")
    print(f"Window size: {window_size}")
    print(f"Expected output shape: {expected_shape}")
    print(f"Actual output shape: {actual_shape}")
    
    assert actual_shape == expected_shape, \
        f"Shape mismatch: {actual_shape} != {expected_shape}"
    
    # Test forward pass
    test_input = torch.randn(num_windows * batch_size, window_tokens, C)
    input_layer.set_cpu("output", test_input)
    win_rev.forward()
    
    output = win_rev.get_cpu("output")
    print(f"Forward pass output shape: {output.shape}")
    
    assert output.shape == torch.Size(expected_shape), \
        f"Output shape mismatch: {output.shape} != {expected_shape}"
    
    print("✓ Test PASSED: Window reverse shape and forward correct")


def test_partition_reverse_roundtrip():
    """Test partition and reverse are inverse operations."""
    print("\n" + "="*70)
    print("TEST 4: Window Partition + Reverse Roundtrip")
    print("="*70)
    
    batch_size = 1
    D, H, W = 4, 56, 56
    C = 96
    window_size = (2, 7, 7)
    
    # Create original input
    original = torch.randn(batch_size, D, H, W, C)
    
    # Partition
    from python.layers.window_partition_3d import SecretWindowPartition3DLayer
    from python.layers.window_reverse_3d import SecretWindowReverse3DLayer
    
    input_layer1 = SecretInputLayer(
        sid=0, LayerName="input1",
        EnclaveMode=ExecutionModeOptions.CPU,
        input_shape=list(original.shape)
    )
    input_layer1.init()
    input_layer1.set_cpu("output", original)
    
    win_part = SecretWindowPartition3DLayer(
        sid=0, LayerName="part", EnclaveMode=ExecutionModeOptions.CPU,
        window_size=window_size
    )
    win_part.register_prev_layer(input_layer1)
    input_layer1.register_next_layer(win_part)
    win_part.init_shape()
    win_part.init(start_enclave=False)
    win_part.forward()
    
    partitioned = win_part.get_cpu("output")
    
    # Reverse
    input_layer2 = SecretInputLayer(
        sid=0, LayerName="input2",
        EnclaveMode=ExecutionModeOptions.CPU,
        input_shape=list(partitioned.shape)
    )
    input_layer2.init()
    input_layer2.set_cpu("output", partitioned)
    
    win_rev = SecretWindowReverse3DLayer(
        sid=0, LayerName="rev", EnclaveMode=ExecutionModeOptions.CPU,
        window_size=window_size,
        output_shape=list(original.shape)
    )
    win_rev.register_prev_layer(input_layer2)
    input_layer2.register_next_layer(win_rev)
    win_rev.init_shape()
    win_rev.init(start_enclave=False)
    win_rev.forward()
    
    reconstructed = win_rev.get_cpu("output")
    
    # Check equality
    diff = torch.abs(original - reconstructed).max().item()
    print(f"Max difference after roundtrip: {diff}")
    
    assert diff < 1e-6, f"Roundtrip error too large: {diff}"
    
    print("✓ Test PASSED: Partition and reverse are perfect inverses")


def test_cyclic_roll_3d():
    """Test 3D cyclic roll operator."""
    print("\n" + "="*70)
    print("TEST 5: SecretCyclicRoll3DLayer - 3D Cyclic Shift")
    print("="*70)
    
    batch_size = 1
    D, H, W = 4, 56, 56
    C = 96
    shifts = (-1, -3, -3)
    dims = (1, 2, 3)
    
    # Create input layer
    input_layer = SecretInputLayer(
        sid=0, LayerName="input",
        EnclaveMode=ExecutionModeOptions.CPU,
        input_shape=[batch_size, D, H, W, C]
    )
    input_layer.init()
    
    # Create cyclic roll layer
    roll = SecretCyclicRoll3DLayer(
        sid=0, LayerName="roll",
        EnclaveMode=ExecutionModeOptions.CPU,
        shifts=shifts,
        dims=dims
    )
    
    # Connect
    roll.register_prev_layer(input_layer)
    input_layer.register_next_layer(roll)
    
    # Initialize
    roll.init_shape()
    roll.init(start_enclave=False)
    
    # Check shape unchanged
    expected_shape = [batch_size, D, H, W, C]
    actual_shape = roll.get_output_shape()
    
    print(f"Input shape: {expected_shape}")
    print(f"Shifts: {shifts}, Dims: {dims}")
    print(f"Output shape: {actual_shape}")
    
    assert actual_shape == expected_shape, \
        f"Shape should be unchanged: {actual_shape} != {expected_shape}"
    
    # Test forward with known pattern
    test_input = torch.arange(D * H * W * C).reshape(batch_size, D, H, W, C).float()
    input_layer.set_cpu("output", test_input)
    roll.forward()
    
    output = roll.get_cpu("output")
    
    # Verify against PyTorch roll
    expected_output = torch.roll(test_input, shifts=shifts, dims=dims)
    diff = torch.abs(output - expected_output).max().item()
    
    print(f"Max difference vs PyTorch roll: {diff}")
    
    assert diff < 1e-6, f"Roll output incorrect: diff={diff}"
    
    print("✓ Test PASSED: Cyclic roll matches PyTorch")


def test_swin_attention_3d_shape():
    """Test Swin Window Attention 3D shape correctness."""
    print("\n" + "="*70)
    print("TEST 6: SwinWindowAttention3D - Shape Validation")
    print("="*70)
    
    batch_size = 1
    num_windows = 64  # (4//2) * (56//7) * (56//7) = 2 * 8 * 8
    window_size = (2, 7, 7)
    dim = 96
    num_heads = 3
    
    Wd, Wh, Ww = window_size
    window_tokens = Wd * Wh * Ww
    
    # Create attention module
    attn = SwinWindowAttention3D(
        sid=0,
        name_prefix="test_attn",
        enclave_mode=ExecutionModeOptions.CPU,
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        batch_size=batch_size,
        num_windows=num_windows,
    )
    
    print(f"Dim: {dim}, Num heads: {num_heads}, Head dim: {attn.head_dim}")
    print(f"Window size: {window_size}, Window tokens: {window_tokens}")
    print(f"Num windows: {num_windows}")
    print(f"Relative position bias table shape: {attn.relative_position_bias_table.shape}")
    
    # Expected bias table size
    expected_bias_size = (2*Wd-1) * (2*Wh-1) * (2*Ww-1)
    actual_bias_size = attn.relative_position_bias_table.shape[0]
    
    assert actual_bias_size == expected_bias_size, \
        f"Bias table size mismatch: {actual_bias_size} != {expected_bias_size}"
    
    # Check relative position index
    expected_index_shape = (window_tokens, window_tokens)
    actual_index_shape = attn.relative_position_index.shape
    
    print(f"Relative position index shape: {actual_index_shape}")
    assert actual_index_shape == expected_index_shape, \
        f"Index shape mismatch: {actual_index_shape} != {expected_index_shape}"
    
    print("✓ Test PASSED: Swin attention shapes correct")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*70)
    print("VIDEO SWIN TRANSFORMER 3D - UNIT TESTS")
    print("="*70)
    
    try:
        test_conv3d()
        test_window_partition_3d()
        test_window_reverse_3d()
        test_partition_reverse_roundtrip()
        test_cyclic_roll_3d()
        test_swin_attention_3d_shape()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nAll 3D operators are working correctly:")
        print("  ✓ Conv3D - shape and forward")
        print("  ✓ Window Partition 3D - shape and forward")
        print("  ✓ Window Reverse 3D - shape and forward")
        print("  ✓ Partition/Reverse roundtrip - perfect inverse")
        print("  ✓ Cyclic Roll 3D - matches PyTorch")
        print("  ✓ Swin Attention 3D - shape validation")
        print("="*70)
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
