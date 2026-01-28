#!/usr/bin/env python
"""
Test script for new ViT layers (CLS Token, Position Embedding, Slice).

This script validates that the new layers work correctly with simple test cases.
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np

from python.layers.cls_token import SecretCLSTokenLayer
from python.layers.position_embedding import SecretPositionEmbeddingLayer
from python.layers.slice import SecretSliceLayer
from python.layers.input import SecretInputLayer
from python.utils.basic_utils import ExecutionModeOptions


def test_cls_token_layer():
    """Test CLS Token Layer functionality."""
    print("\n" + "="*60)
    print("Testing SecretCLSTokenLayer")
    print("="*60)
    
    # Configuration
    batch_size = 2
    num_patches = 196  # 14x14 patches
    embed_dim = 768
    
    # Create input layer
    input_layer = SecretInputLayer(
        sid=0,
        LayerName="input",
        input_shape=[batch_size, num_patches, embed_dim],
        EnclaveMode=ExecutionModeOptions.CPU
    )
    
    # Create CLS token layer
    cls_layer = SecretCLSTokenLayer(
        sid=0,
        LayerName="cls_token",
        EnclaveMode=ExecutionModeOptions.CPU,
        embed_dim=embed_dim,
        batch_size=batch_size
    )
    
    # Connect layers
    cls_layer.register_prev_layer(input_layer)
    
    # Initialize
    input_layer.init_shape()
    input_layer.init()
    cls_layer.init_shape()
    cls_layer.init()
    cls_layer.generate_tensor_name_list()
    
    # Create test input
    test_input = torch.randn(batch_size, num_patches, embed_dim)
    input_layer.set_cpu("output", test_input)
    
    # Forward pass
    cls_layer.forward()
    
    # Get output
    output = cls_layer.get_cpu("output")
    
    # Validate
    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     ({batch_size}, {num_patches + 1}, {embed_dim})")
    
    assert output.shape == (batch_size, num_patches + 1, embed_dim), \
        f"Shape mismatch: got {output.shape}, expected ({batch_size}, {num_patches + 1}, {embed_dim})"
    
    # Check that CLS token is prepended (first token should be different from input)
    cls_token = output[:, 0, :]
    patch_tokens = output[:, 1:, :]
    
    # Patch tokens should match input
    assert torch.allclose(patch_tokens, test_input, atol=1e-6), \
        "Patch tokens don't match input!"
    
    print(f"✓ CLS token shape: {cls_token.shape}")
    print(f"✓ Patch tokens preserved correctly")
    print("✓ SecretCLSTokenLayer test passed!\n")
    
    return True


def test_position_embedding_layer():
    """Test Position Embedding Layer functionality."""
    print("\n" + "="*60)
    print("Testing SecretPositionEmbeddingLayer")
    print("="*60)
    
    # Configuration
    batch_size = 2
    seq_len = 197  # 196 patches + 1 CLS token
    embed_dim = 768
    
    # Create input layer
    input_layer = SecretInputLayer(
        sid=0,
        LayerName="input",
        input_shape=[batch_size, seq_len, embed_dim],
        EnclaveMode=ExecutionModeOptions.CPU
    )
    
    # Create position embedding layer
    pos_embed_layer = SecretPositionEmbeddingLayer(
        sid=0,
        LayerName="pos_embed",
        EnclaveMode=ExecutionModeOptions.CPU,
        seq_len=seq_len,
        embed_dim=embed_dim,
        batch_size=batch_size
    )
    
    # Connect layers
    pos_embed_layer.register_prev_layer(input_layer)
    
    # Initialize
    input_layer.init_shape()
    input_layer.init()
    pos_embed_layer.init_shape()
    pos_embed_layer.init()
    pos_embed_layer.generate_tensor_name_list()
    
    # Create test input
    test_input = torch.randn(batch_size, seq_len, embed_dim)
    input_layer.set_cpu("output", test_input)
    
    # Forward pass
    pos_embed_layer.forward()
    
    # Get output
    output = pos_embed_layer.get_cpu("output")
    pos_embed = pos_embed_layer.get_cpu("pos_embed")
    
    # Validate
    print(f"Input shape:    {test_input.shape}")
    print(f"Pos embed shape: {pos_embed.shape}")
    print(f"Output shape:   {output.shape}")
    
    assert output.shape == test_input.shape, \
        f"Shape mismatch: got {output.shape}, expected {test_input.shape}"
    
    # Check that position embedding was added
    expected_output = test_input + pos_embed
    assert torch.allclose(output, expected_output, atol=1e-6), \
        "Output doesn't match input + pos_embed!"
    
    print(f"✓ Output = Input + Position Embedding")
    print("✓ SecretPositionEmbeddingLayer test passed!\n")
    
    return True


def test_slice_layer():
    """Test Slice Layer functionality."""
    print("\n" + "="*60)
    print("Testing SecretSliceLayer")
    print("="*60)
    
    # Configuration
    batch_size = 2
    seq_len = 197
    embed_dim = 768
    
    # Create input layer
    input_layer = SecretInputLayer(
        sid=0,
        LayerName="input",
        input_shape=[batch_size, seq_len, embed_dim],
        EnclaveMode=ExecutionModeOptions.CPU
    )
    
    # Create slice layer (extract CLS token at index 0)
    slice_layer = SecretSliceLayer(
        sid=0,
        LayerName="slice_cls",
        EnclaveMode=ExecutionModeOptions.CPU,
        index=0,
        dim=1  # Slice along sequence dimension
    )
    
    # Connect layers
    slice_layer.register_prev_layer(input_layer)
    
    # Initialize
    input_layer.init_shape()
    input_layer.init()
    slice_layer.init_shape()
    slice_layer.init()
    slice_layer.generate_tensor_name_list()
    
    # Create test input
    test_input = torch.randn(batch_size, seq_len, embed_dim)
    input_layer.set_cpu("output", test_input)
    
    # Forward pass
    slice_layer.forward()
    
    # Get output
    output = slice_layer.get_cpu("output")
    
    # Validate
    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     ({batch_size}, {embed_dim})")
    
    assert output.shape == (batch_size, embed_dim), \
        f"Shape mismatch: got {output.shape}, expected ({batch_size}, {embed_dim})"
    
    # Check that we extracted the first token (CLS token)
    expected_output = test_input[:, 0, :]
    assert torch.allclose(output, expected_output, atol=1e-6), \
        "Output doesn't match input[:, 0, :]!"
    
    print(f"✓ Correctly extracted token at index 0")
    print("✓ SecretSliceLayer test passed!\n")
    
    return True


def test_combined_flow():
    """Test all layers together in a typical ViT flow."""
    print("\n" + "="*60)
    print("Testing Combined ViT Flow")
    print("="*60)
    
    # Configuration
    batch_size = 2
    num_patches = 196
    embed_dim = 768
    
    # 1. Start with patch tokens
    patch_tokens = torch.randn(batch_size, num_patches, embed_dim)
    print(f"1. Patch tokens:          {patch_tokens.shape}")
    
    # 2. Create input layer
    input_layer = SecretInputLayer(
        sid=0,
        LayerName="input",
        input_shape=[batch_size, num_patches, embed_dim],
        EnclaveMode=ExecutionModeOptions.CPU
    )
    input_layer.init_shape()
    input_layer.init()
    input_layer.set_cpu("output", patch_tokens)
    
    # 3. Add CLS token
    cls_layer = SecretCLSTokenLayer(
        sid=0,
        LayerName="cls_token",
        EnclaveMode=ExecutionModeOptions.CPU,
        embed_dim=embed_dim,
        batch_size=batch_size
    )
    cls_layer.register_prev_layer(input_layer)
    cls_layer.init_shape()
    cls_layer.init()
    cls_layer.generate_tensor_name_list()
    cls_layer.forward()
    
    tokens_with_cls = cls_layer.get_cpu("output")
    print(f"2. After CLS token:       {tokens_with_cls.shape}")
    assert tokens_with_cls.shape == (batch_size, num_patches + 1, embed_dim)
    
    # 4. Add position embedding
    # Create intermediate input for pos embed
    input_for_pos = SecretInputLayer(
        sid=0,
        LayerName="input_pos",
        input_shape=[batch_size, num_patches + 1, embed_dim],
        EnclaveMode=ExecutionModeOptions.CPU
    )
    input_for_pos.init_shape()
    input_for_pos.init()
    input_for_pos.set_cpu("output", tokens_with_cls)
    
    pos_embed_layer = SecretPositionEmbeddingLayer(
        sid=0,
        LayerName="pos_embed",
        EnclaveMode=ExecutionModeOptions.CPU,
        seq_len=num_patches + 1,
        embed_dim=embed_dim,
        batch_size=batch_size
    )
    pos_embed_layer.register_prev_layer(input_for_pos)
    pos_embed_layer.init_shape()
    pos_embed_layer.init()
    pos_embed_layer.generate_tensor_name_list()
    pos_embed_layer.forward()
    
    embedded_tokens = pos_embed_layer.get_cpu("output")
    print(f"3. After pos embedding:   {embedded_tokens.shape}")
    assert embedded_tokens.shape == (batch_size, num_patches + 1, embed_dim)
    
    # 5. Extract CLS token for classification
    # Create intermediate input for slice
    input_for_slice = SecretInputLayer(
        sid=0,
        LayerName="input_slice",
        input_shape=[batch_size, num_patches + 1, embed_dim],
        EnclaveMode=ExecutionModeOptions.CPU
    )
    input_for_slice.init_shape()
    input_for_slice.init()
    input_for_slice.set_cpu("output", embedded_tokens)
    
    slice_layer = SecretSliceLayer(
        sid=0,
        LayerName="slice_cls",
        EnclaveMode=ExecutionModeOptions.CPU,
        index=0,
        dim=1
    )
    slice_layer.register_prev_layer(input_for_slice)
    slice_layer.init_shape()
    slice_layer.init()
    slice_layer.generate_tensor_name_list()
    slice_layer.forward()
    
    cls_token = slice_layer.get_cpu("output")
    print(f"4. Extracted CLS token:   {cls_token.shape}")
    assert cls_token.shape == (batch_size, embed_dim)
    
    print("\n✓ Combined flow test passed!")
    print(f"✓ Successfully transformed {patch_tokens.shape} -> {cls_token.shape}")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "ViT New Layers Unit Tests")
    print("="*70)
    
    try:
        # Run individual layer tests
        test_cls_token_layer()
        test_position_embedding_layer()
        test_slice_layer()
        
        # Run combined flow test
        test_combined_flow()
        
        print("\n" + "="*70)
        print("✓ All tests passed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
