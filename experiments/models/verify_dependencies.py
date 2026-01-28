#!/usr/bin/env python
"""
Verify Per-Head Attention Dependencies in Profiling Results.

This script checks that the dependencies in the profiling CSV are correct
according to Transformer architecture, especially for per-head attention.

Usage:
    python experiments/models/verify_dependencies.py \
        --input experiments/data/bert_base_enclave_per_head_layers.csv
"""

import sys
import argparse
import pandas as pd
from typing import List, Set
import ast


def verify_per_head_dependencies(csv_path: str, num_heads: int = 12, num_layers: int = 12):
    """Verify dependencies in per-head profiling results."""
    
    print("="*70)
    print("Per-Head Attention Dependencies Verification")
    print("="*70)
    print(f"Input: {csv_path}")
    print(f"Expected: {num_heads} heads per encoder, {num_layers} encoders")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    
    errors = []
    warnings = []
    successes = []
    
    # 1. Check embedding has no dependencies
    embedding = df[df['name'] == 'embedding']
    if len(embedding) > 0:
        deps = ast.literal_eval(embedding.iloc[0]['dependencies'])
        if deps == []:
            successes.append("✓ Embedding has no dependencies")
        else:
            errors.append(f"❌ Embedding should have no dependencies, got: {deps}")
    
    # 2. Check each encoder block
    for enc_idx in range(num_layers):
        prefix = f'encoder{enc_idx}'
        
        # Expected previous block output
        if enc_idx == 0:
            expected_prev = 'embedding'
        else:
            expected_prev = f'encoder{enc_idx-1}_norm2'
        
        # 2.1 Check Q/K/V projections all depend on previous block
        for proj in ['q_proj', 'k_proj', 'v_proj']:
            layer_name = f'{prefix}_attn_{proj}'
            layer_data = df[df['name'] == layer_name]
            
            if len(layer_data) == 0:
                warnings.append(f"⚠ Layer not found: {layer_name}")
                continue
            
            deps = ast.literal_eval(layer_data.iloc[0]['dependencies'])
            if deps == [expected_prev]:
                successes.append(f"✓ {layer_name} depends on {expected_prev}")
            else:
                errors.append(f"❌ {layer_name} should depend on [{expected_prev}], got: {deps}")
        
        # 2.2 Check all heads have correct dependencies
        all_head_outputs = []
        
        for head_idx in range(num_heads):
            head_prefix = f'{prefix}_attn_head{head_idx}'
            
            # QK matmul should depend on Q and K projections
            qk_name = f'{head_prefix}_qk_matmul'
            qk_data = df[df['name'] == qk_name]
            
            if len(qk_data) == 0:
                warnings.append(f"⚠ Layer not found: {qk_name}")
                continue
            
            qk_deps = set(ast.literal_eval(qk_data.iloc[0]['dependencies']))
            expected_qk_deps = {f'{prefix}_attn_q_proj', f'{prefix}_attn_k_proj'}
            
            if qk_deps == expected_qk_deps:
                if head_idx == 0 or head_idx == num_heads - 1:  # Only print first and last
                    successes.append(f"✓ {qk_name} has correct dependencies")
            else:
                errors.append(f"❌ {qk_name} should depend on {expected_qk_deps}, got: {qk_deps}")
            
            # Softmax should depend on qk_matmul
            softmax_name = f'{head_prefix}_softmax'
            softmax_data = df[df['name'] == softmax_name]
            
            if len(softmax_data) > 0:
                softmax_deps = ast.literal_eval(softmax_data.iloc[0]['dependencies'])
                if softmax_deps == [qk_name]:
                    if head_idx == 0:  # Only print once
                        successes.append(f"✓ Head softmax layers depend on their own qk_matmul")
                else:
                    errors.append(f"❌ {softmax_name} should depend on [{qk_name}], got: {softmax_deps}")
            
            # Attn @ V should depend on softmax and V projection
            attn_v_name = f'{head_prefix}_attn_v_matmul'
            attn_v_data = df[df['name'] == attn_v_name]
            
            if len(attn_v_data) > 0:
                attn_v_deps = set(ast.literal_eval(attn_v_data.iloc[0]['dependencies']))
                expected_attn_v_deps = {softmax_name, f'{prefix}_attn_v_proj'}
                
                if attn_v_deps == expected_attn_v_deps:
                    if head_idx == 0:  # Only print once
                        successes.append(f"✓ Head attn_v layers depend on softmax and v_proj")
                else:
                    errors.append(f"❌ {attn_v_name} should depend on {expected_attn_v_deps}, got: {attn_v_deps}")
                
                all_head_outputs.append(attn_v_name)
        
        # 2.3 Check out_proj depends on all heads
        out_proj_name = f'{prefix}_attn_out_proj'
        out_proj_data = df[df['name'] == out_proj_name]
        
        if len(out_proj_data) > 0:
            out_proj_deps = set(ast.literal_eval(out_proj_data.iloc[0]['dependencies']))
            expected_out_deps = set(all_head_outputs)
            
            if out_proj_deps == expected_out_deps:
                successes.append(f"✓ {out_proj_name} depends on all {num_heads} heads")
            else:
                missing = expected_out_deps - out_proj_deps
                extra = out_proj_deps - expected_out_deps
                if missing:
                    errors.append(f"❌ {out_proj_name} missing dependencies: {missing}")
                if extra:
                    errors.append(f"❌ {out_proj_name} extra dependencies: {extra}")
        
        # 2.4 Check FFN dependencies (sequential)
        ffn_chain = [
            (f'{prefix}_norm1', [f'{prefix}_attn_out_proj']),
            (f'{prefix}_ffn_fc1', [f'{prefix}_norm1']),
            (f'{prefix}_ffn_gelu', [f'{prefix}_ffn_fc1']),
            (f'{prefix}_ffn_fc2', [f'{prefix}_ffn_gelu']),
            (f'{prefix}_norm2', [f'{prefix}_ffn_fc2']),
        ]
        
        for layer_name, expected_deps in ffn_chain:
            layer_data = df[df['name'] == layer_name]
            if len(layer_data) > 0:
                actual_deps = ast.literal_eval(layer_data.iloc[0]['dependencies'])
                if actual_deps == expected_deps:
                    if enc_idx == 0:  # Only print for first encoder
                        successes.append(f"✓ FFN chain correct for {layer_name}")
                else:
                    errors.append(f"❌ {layer_name} should depend on {expected_deps}, got: {actual_deps}")
    
    # 3. Check classifier head
    pooler_data = df[df['name'] == 'pooler']
    if len(pooler_data) > 0:
        pooler_deps = ast.literal_eval(pooler_data.iloc[0]['dependencies'])
        expected_pooler_deps = [f'encoder{num_layers-1}_norm2']
        if pooler_deps == expected_pooler_deps:
            successes.append(f"✓ Pooler depends on last encoder output")
        else:
            errors.append(f"❌ Pooler should depend on {expected_pooler_deps}, got: {pooler_deps}")
    
    classifier_data = df[df['name'] == 'classifier']
    if len(classifier_data) > 0:
        classifier_deps = ast.literal_eval(classifier_data.iloc[0]['dependencies'])
        if classifier_deps == ['pooler']:
            successes.append(f"✓ Classifier depends on pooler")
        else:
            errors.append(f"❌ Classifier should depend on ['pooler'], got: {classifier_deps}")
    
    # Print results
    print(f"\n{'='*70}")
    print("Verification Results")
    print(f"{'='*70}")
    
    if successes:
        print(f"\n✓ Successes ({len(successes)}):")
        for msg in successes[:10]:  # Show first 10
            print(f"  {msg}")
        if len(successes) > 10:
            print(f"  ... and {len(successes) - 10} more")
    
    if warnings:
        print(f"\n⚠ Warnings ({len(warnings)}):")
        for msg in warnings[:10]:
            print(f"  {msg}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
    
    if errors:
        print(f"\n❌ Errors ({len(errors)}):")
        for msg in errors:
            print(f"  {msg}")
    else:
        print(f"\n🎉 No errors found!")
    
    print(f"\n{'='*70}")
    print(f"Summary: {len(successes)} successes, {len(warnings)} warnings, {len(errors)} errors")
    print(f"{'='*70}")
    
    return len(errors) == 0


def verify_parallelism(csv_path: str, num_heads: int = 12):
    """Verify that heads are truly parallel (no inter-head dependencies)."""
    
    print(f"\n{'='*70}")
    print("Parallelism Verification")
    print(f"{'='*70}")
    
    df = pd.read_csv(csv_path)
    
    # Extract all head layers
    head_layers = df[df['name'].str.contains(r'head\d+_(qk_matmul|softmax|attn_v_matmul)')]
    
    inter_head_deps = []
    
    for idx, row in head_layers.iterrows():
        layer_name = row['name']
        deps = ast.literal_eval(row['dependencies'])
        
        # Extract head number from layer name
        import re
        match = re.search(r'head(\d+)_', layer_name)
        if not match:
            continue
        
        current_head = int(match.group(1))
        
        # Check if any dependency is from a different head
        for dep in deps:
            dep_match = re.search(r'head(\d+)_', dep)
            if dep_match:
                dep_head = int(dep_match.group(1))
                if dep_head != current_head:
                    inter_head_deps.append((layer_name, dep))
    
    if inter_head_deps:
        print(f"\n❌ Found {len(inter_head_deps)} inter-head dependencies (should be 0):")
        for layer, dep in inter_head_deps[:10]:
            print(f"  {layer} depends on {dep}")
    else:
        print(f"\n✓ All heads are independent (no inter-head dependencies)")
        print(f"  This confirms that all {num_heads} heads can execute in parallel!")
    
    print(f"{'='*70}")
    
    return len(inter_head_deps) == 0


def main():
    parser = argparse.ArgumentParser(description='Verify per-head attention dependencies')
    parser.add_argument('--input', type=str, 
                       default='experiments/data/bert_base_enclave_per_head_layers.csv',
                       help='Path to profiling CSV file')
    parser.add_argument('--heads', type=int, default=12,
                       help='Number of attention heads')
    parser.add_argument('--layers', type=int, default=12,
                       help='Number of encoder layers')
    
    args = parser.parse_args()
    
    # Verify dependencies
    deps_ok = verify_per_head_dependencies(args.input, args.heads, args.layers)
    
    # Verify parallelism
    parallel_ok = verify_parallelism(args.input, args.heads)
    
    # Overall result
    print(f"\n{'='*70}")
    if deps_ok and parallel_ok:
        print("✅ All checks passed! Dependencies are correct.")
        print("="*70)
        sys.exit(0)
    else:
        print("❌ Some checks failed. Please review the errors above.")
        print("="*70)
        sys.exit(1)


if __name__ == '__main__':
    main()
