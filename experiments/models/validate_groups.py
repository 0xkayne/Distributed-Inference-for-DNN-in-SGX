#!/usr/bin/env python3
"""
Validate Inception V3 Group Configurations

This script validates the group configurations for STORE_CHUNK_ELEM
without requiring PyTorch or SGX dependencies.

It checks:
1. Memory safety (STORE_CHUNK_ELEM size)
2. Divisibility constraints for different layer types
3. Group coverage
"""

import sys
import math
from typing import Dict, List, Tuple, Optional

# Maximum safe STORE_CHUNK_ELEM (16M elements = 64MB for float32)
MAX_SAFE_STORE_CHUNK_ELEM = 16 * 1024 * 1024

# Inception V3 layer structure with computed shapes
# Format: (layer_name, layer_type, input_shape, output_shape, constraints)
# Constraints: list of numbers that STORE_CHUNK_ELEM must be divisible by

INCEPTION_V3_LAYERS = [
    # Input
    ("input", "SecretInputLayer", [1, 3, 299, 299], [1, 3, 299, 299], []),
    
    # Stem Part 1: input -> conv1 -> conv2
    ("stem_conv1", "SGXConvBase", [1, 3, 299, 299], [1, 32, 149, 149], [299*3*2, 32]),  # stride=2
    ("stem_relu1", "SecretReLULayer", [1, 32, 149, 149], [1, 32, 149, 149], []),
    ("stem_conv2", "SGXConvBase", [1, 32, 149, 149], [1, 32, 147, 147], [149*32, 32]),  # stride=1
    ("stem_relu2", "SecretReLULayer", [1, 32, 147, 147], [1, 32, 147, 147], []),
    
    # Stem Part 2: conv3 -> pool1
    ("stem_conv3", "SGXConvBase", [1, 32, 147, 147], [1, 64, 147, 147], [147*32, 64]),  # stride=1, pad=1
    ("stem_relu3", "SecretReLULayer", [1, 64, 147, 147], [1, 64, 147, 147], []),
    ("stem_pool1", "SecretMaxpool2dLayer", [1, 64, 147, 147], [1, 64, 73, 73], [147*147]),  # stride=2
    
    # Stem Part 3: conv4 -> conv5
    ("stem_conv4", "SGXConvBase", [1, 64, 73, 73], [1, 80, 73, 73], [73*64, 80]),  # 1x1, stride=1
    ("stem_relu4", "SecretReLULayer", [1, 80, 73, 73], [1, 80, 73, 73], []),
    ("stem_conv5", "SGXConvBase", [1, 80, 73, 73], [1, 192, 71, 71], [73*80, 192]),  # 3x3, stride=1
    ("stem_relu5", "SecretReLULayer", [1, 192, 71, 71], [1, 192, 71, 71], []),
    
    # Stem Part 4: pool2
    ("stem_pool2", "SecretMaxpool2dLayer", [1, 192, 71, 71], [1, 192, 35, 35], [71*71]),  # stride=2
]

# After stem: 35x35x192
# Inception-A blocks: 35x35, conv constraints involve 35*channels
INCEPTION_A_CONSTRAINTS = [35*35, 35*192, 64, 96, 32]  # MaxPool input, conv row sizes, output channels

# After Inception-A: 35x35x256
# Reduction-A: 35x35 -> 17x17
REDUCTION_A_CONSTRAINTS = [35*35, 35*256, 384]  # MaxPool, conv, output

# Inception-B blocks: 17x17x768
INCEPTION_B_CONSTRAINTS = [17*17, 17*768, 192]

# Reduction-B: 17x17 -> 8x8
REDUCTION_B_CONSTRAINTS = [17*17, 17*768, 192, 320]

# Inception-C blocks: 8x8x1280
INCEPTION_C_CONSTRAINTS = [8*8, 8*1280, 320, 384, 192]

# Classifier: avgpool 8x8, fc
CLASSIFIER_CONSTRAINTS = [8*8, 2048, 1000]


def gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Least common multiple."""
    return abs(a * b) // gcd(a, b)


def lcm_list(numbers: List[int]) -> int:
    """LCM of a list of numbers."""
    if not numbers:
        return 1
    result = numbers[0]
    for n in numbers[1:]:
        result = lcm(result, n)
    return result


def check_divisibility(value: int, constraints: List[int]) -> Tuple[bool, List[str]]:
    """Check if value is divisible by all constraints."""
    errors = []
    for c in constraints:
        if c > 0 and value % c != 0:
            errors.append(f"Not divisible by {c} (remainder: {value % c})")
    return len(errors) == 0, errors


def suggest_store_chunk_elem(constraints: List[int], max_value: int = MAX_SAFE_STORE_CHUNK_ELEM) -> Optional[int]:
    """Suggest a STORE_CHUNK_ELEM value that satisfies all constraints."""
    if not constraints:
        return 65536  # Default small value
    
    # Filter out zeros
    constraints = [c for c in constraints if c > 0]
    if not constraints:
        return 65536
    
    # Calculate LCM
    base_lcm = lcm_list(constraints)
    
    # Find smallest multiple of LCM that's reasonable
    # Prefer values aligned to 64 for memory efficiency
    suggested = base_lcm
    if suggested < 64:
        suggested = 64 * (64 // suggested + 1)
    
    # If too large, just return the LCM
    if suggested > max_value:
        return base_lcm
    
    return suggested


# Group configurations matching profile_inception.py (UPDATED with correct LCM values)
GROUP_CONFIGS = {
    'Stem-Part1': {
        'store_chunk_elem': 4276896,  # LCM(1794, 4768, 32)
        'constraints': [299*3*2, 149*32, 32],  # conv1 row*stride, conv2 row, output_c
    },
    'Stem-Part2': {
        'store_chunk_elem': 1382976,  # LCM(4704, 64, 21609)
        'constraints': [147*32, 64, 147*147],  # conv3, output_c, pool1 input_hw
    },
    'Stem-Part3': {
        'store_chunk_elem': 70080,  # LCM(4672, 5840, 80, 192)
        'constraints': [73*64, 73*80, 80, 192],  # conv4, conv5, output_c
    },
    'Stem-Part4': {
        'store_chunk_elem': 322624,  # 5041*64
        'constraints': [71*71],  # pool2 input_hw
    },
    'Inception-A1': {'store_chunk_elem': 235200, 'constraints': [35*35, 64, 96, 32]},  # LCM
    'Inception-A2': {'store_chunk_elem': 235200, 'constraints': [35*35, 64, 96, 32]},
    'Inception-A3': {'store_chunk_elem': 235200, 'constraints': [35*35, 64, 96, 32]},
    'Reduction-A': {'store_chunk_elem': 470400, 'constraints': [35*35, 384]},  # LCM(1225, 384)
    'Inception-B1': {'store_chunk_elem': 55488, 'constraints': [17*17, 192]},  # LCM(289, 192) ✓
    'Inception-B2': {'store_chunk_elem': 55488, 'constraints': [17*17, 192]},
    'Inception-B3': {'store_chunk_elem': 55488, 'constraints': [17*17, 192]},
    'Inception-B4': {'store_chunk_elem': 55488, 'constraints': [17*17, 192]},
    'Reduction-B': {'store_chunk_elem': 277440, 'constraints': [17*17, 192, 320]},  # LCM(289, 192, 320)
    'Inception-C1': {'store_chunk_elem': 1920, 'constraints': [8*8, 320, 384, 192]},  # LCM(64, 320, 384, 192)
    'Inception-C2': {'store_chunk_elem': 1920, 'constraints': [8*8, 320, 384, 192]},
    'Classifier': {'store_chunk_elem': 64000, 'constraints': [8*8, 1000]},  # ✓
}

GROUP_ORDER = [
    'Stem-Part1', 'Stem-Part2', 'Stem-Part3', 'Stem-Part4',
    'Inception-A1', 'Inception-A2', 'Inception-A3',
    'Reduction-A',
    'Inception-B1', 'Inception-B2', 'Inception-B3', 'Inception-B4',
    'Reduction-B',
    'Inception-C1', 'Inception-C2',
    'Classifier',
]


def validate_all_groups():
    """Validate all group configurations."""
    print("="*80)
    print("Validating Inception V3 Group Configurations")
    print("="*80)
    
    all_valid = True
    total_mem = 0
    
    print("\n" + "-"*80)
    print(f"{'Group':<20} {'STORE_CHUNK_ELEM':>15} {'Memory':>10} {'Status':>15}")
    print("-"*80)
    
    for group_name in GROUP_ORDER:
        config = GROUP_CONFIGS[group_name]
        store_elem = config['store_chunk_elem']
        constraints = config['constraints']
        
        mem_mb = store_elem * 4 / 1024 / 1024
        total_mem += mem_mb
        
        # Check memory safety
        if store_elem > MAX_SAFE_STORE_CHUNK_ELEM:
            status = "⚠ TOO LARGE"
            all_valid = False
        else:
            # Check divisibility
            is_valid, errors = check_divisibility(store_elem, constraints)
            if is_valid:
                status = "✓ VALID"
            else:
                status = "✗ INVALID"
                all_valid = False
        
        print(f"{group_name:<20} {store_elem:>15} {mem_mb:>9.2f}MB {status:>15}")
    
    print("-"*80)
    print(f"{'Total memory (per group):':<36} {total_mem:>9.2f}MB")
    print("-"*80)
    
    # Detailed validation
    print("\n" + "="*80)
    print("Detailed Constraint Validation")
    print("="*80)
    
    for group_name in GROUP_ORDER:
        config = GROUP_CONFIGS[group_name]
        store_elem = config['store_chunk_elem']
        constraints = config['constraints']
        
        print(f"\n{group_name}:")
        print(f"  STORE_CHUNK_ELEM = {store_elem}")
        print(f"  Constraints: {constraints}")
        
        is_valid, errors = check_divisibility(store_elem, constraints)
        if is_valid:
            print(f"  ✓ All constraints satisfied")
        else:
            print(f"  ✗ Constraint violations:")
            for err in errors:
                print(f"      - {err}")
            
            # Suggest a better value
            suggested = suggest_store_chunk_elem(constraints)
            if suggested:
                print(f"  💡 Suggested value: {suggested} ({suggested * 4 / 1024 / 1024:.2f} MB)")
    
    print("\n" + "="*80)
    if all_valid:
        print("✓ All group configurations are VALID and MEMORY-SAFE")
    else:
        print("⚠ Some group configurations need attention")
    print("="*80)
    
    return all_valid


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Inception V3 group configurations')
    parser.add_argument('--suggest', action='store_true', help='Suggest optimal STORE_CHUNK_ELEM values')
    args = parser.parse_args()
    
    if args.suggest:
        print("="*80)
        print("Suggested STORE_CHUNK_ELEM Values")
        print("="*80)
        
        for group_name in GROUP_ORDER:
            config = GROUP_CONFIGS[group_name]
            constraints = config['constraints']
            suggested = suggest_store_chunk_elem(constraints)
            current = config['store_chunk_elem']
            
            status = "✓" if suggested == current else "→"
            print(f"{group_name:<20}: current={current:>10}, suggested={suggested:>10} {status}")
    else:
        validate_all_groups()


if __name__ == "__main__":
    main()

