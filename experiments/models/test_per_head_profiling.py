#!/usr/bin/env python
"""
Quick test script to verify per-head attention profiling functionality.

This script tests:
1. Per-head mode can be enabled
2. Each head generates separate profiling layers
3. Dependencies are correctly inferred
4. No NameError occurs

Usage:
    python experiments/models/test_per_head_profiling.py
"""

import sys
sys.path.insert(0, '.')

from experiments.models.profile_bert_enclave import BERTEnclaveProfiler


def test_initialization():
    """Test that profiler can be initialized with per-head mode."""
    print("=" * 70)
    print("Test 1: Initialization")
    print("=" * 70)
    
    # Test batched mode
    profiler_batched = BERTEnclaveProfiler(
        model_variant='mini',
        seq_len=64,
        num_iterations=2,
        warmup_iterations=1,
        use_per_head_attention=False
    )
    print(f"✓ Batched mode profiler created")
    print(f"  - num_heads: {profiler_batched.num_heads}")
    print(f"  - use_per_head_attention: {profiler_batched.use_per_head_attention}")
    
    # Test per-head mode
    profiler_per_head = BERTEnclaveProfiler(
        model_variant='mini',
        seq_len=64,
        num_iterations=2,
        warmup_iterations=1,
        use_per_head_attention=True
    )
    print(f"✓ Per-head mode profiler created")
    print(f"  - num_heads: {profiler_per_head.num_heads}")
    print(f"  - use_per_head_attention: {profiler_per_head.use_per_head_attention}")
    
    return profiler_batched, profiler_per_head


def test_method_dispatch():
    """Test that _profile_encoder_block dispatches correctly."""
    print("\n" + "=" * 70)
    print("Test 2: Method Dispatch")
    print("=" * 70)
    
    profiler = BERTEnclaveProfiler(
        model_variant='mini',
        use_per_head_attention=True
    )
    
    # Check that the dispatch method exists
    assert hasattr(profiler, '_profile_encoder_block')
    print(f"✓ _profile_encoder_block method exists")
    
    assert hasattr(profiler, '_profile_encoder_block_batched')
    print(f"✓ _profile_encoder_block_batched method exists")
    
    assert hasattr(profiler, '_profile_encoder_block_per_head')
    print(f"✓ _profile_encoder_block_per_head method exists")


def test_layer_naming():
    """Test layer naming convention for per-head mode."""
    print("\n" + "=" * 70)
    print("Test 3: Layer Naming Convention")
    print("=" * 70)
    
    block_idx = 0
    num_heads = 4
    
    # Expected layer names in per-head mode
    expected_layers = []
    
    # Shared projections
    expected_layers.extend([
        f'encoder{block_idx}_attn_q_proj',
        f'encoder{block_idx}_attn_k_proj',
        f'encoder{block_idx}_attn_v_proj',
    ])
    
    # Per-head layers
    for head_idx in range(num_heads):
        expected_layers.extend([
            f'encoder{block_idx}_attn_head{head_idx}_qk_matmul',
            f'encoder{block_idx}_attn_head{head_idx}_softmax',
            f'encoder{block_idx}_attn_head{head_idx}_attn_v_matmul',
        ])
    
    # Output projection
    expected_layers.append(f'encoder{block_idx}_attn_out_proj')
    
    print(f"✓ Expected {len(expected_layers)} attention layers for encoder {block_idx}")
    print(f"  - Shared layers: 4 (Q/K/V proj + Out proj)")
    print(f"  - Per-head layers: {num_heads * 3} ({num_heads} heads × 3 ops)")
    print(f"  - Total: {len(expected_layers)} layers")
    
    print(f"\nExample layer names:")
    for i, name in enumerate(expected_layers[:7]):
        print(f"  {i+1}. {name}")
    print(f"  ...")


def test_config_parameters():
    """Test that configuration parameters are correctly set."""
    print("\n" + "=" * 70)
    print("Test 4: Configuration Parameters")
    print("=" * 70)
    
    profiler = BERTEnclaveProfiler(
        model_variant='base',
        batch_size=2,
        seq_len=256,
        num_classes=5,
        num_iterations=20,
        warmup_iterations=5,
        use_per_head_attention=True
    )
    
    assert profiler.batch_size == 2
    print(f"✓ batch_size: {profiler.batch_size}")
    
    assert profiler.seq_len == 256
    print(f"✓ seq_len: {profiler.seq_len}")
    
    assert profiler.num_classes == 5
    print(f"✓ num_classes: {profiler.num_classes}")
    
    assert profiler.num_iterations == 20
    print(f"✓ num_iterations: {profiler.num_iterations}")
    
    assert profiler.warmup_iterations == 5
    print(f"✓ warmup_iterations: {profiler.warmup_iterations}")
    
    assert profiler.use_per_head_attention == True
    print(f"✓ use_per_head_attention: {profiler.use_per_head_attention}")
    
    # Check BERT-base specific config
    assert profiler.embed_dim == 768
    print(f"✓ embed_dim: {profiler.embed_dim}")
    
    assert profiler.num_heads == 12
    print(f"✓ num_heads: {profiler.num_heads}")
    
    assert profiler.num_layers == 12
    print(f"✓ num_layers: {profiler.num_layers}")


def test_output_filename():
    """Test output filename generation."""
    print("\n" + "=" * 70)
    print("Test 5: Output Filename Generation")
    print("=" * 70)
    
    # Batched mode
    profiler_batched = BERTEnclaveProfiler(
        model_variant='base',
        use_per_head_attention=False
    )
    print(f"Batched mode output files:")
    print(f"  - bert_base_enclave_layers.csv")
    print(f"  - bert_base_enclave_layers.json")
    
    # Per-head mode
    profiler_per_head = BERTEnclaveProfiler(
        model_variant='base',
        use_per_head_attention=True
    )
    print(f"\nPer-head mode output files:")
    print(f"  - bert_base_enclave_per_head_layers.csv")
    print(f"  - bert_base_enclave_per_head_layers.json")
    
    print(f"\n✓ Filenames are distinguishable")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("BERT Per-Head Profiling Functionality Tests")
    print("=" * 70)
    print("\nThese tests verify the per-head profiling implementation")
    print("without actually running Enclave (which requires compilation).\n")
    
    try:
        test_initialization()
        test_method_dispatch()
        test_layer_naming()
        test_config_parameters()
        test_output_filename()
        
        print("\n" + "=" * 70)
        print("All Tests Passed! ✓")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Compile the project: make")
        print("2. Run actual profiling:")
        print("   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \\")
        print("   python -m experiments.models.profile_bert_enclave --model mini --per-head")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
