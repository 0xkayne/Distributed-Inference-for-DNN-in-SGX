#!/usr/bin/env python3
"""
Check Layer-to-Group Mapping for Inception V3

This script verifies that all layers in Inception V3 are correctly
assigned to groups, without requiring SGX libraries.
"""

# Complete list of Inception V3 layers based on sgx_inception.py
INCEPTION_V3_LAYERS = [
    # Input
    "input",
    
    # Stem Part 1
    "stem_conv1", "stem_relu1",
    "stem_conv2", "stem_relu2",
    
    # Stem Part 2
    "stem_conv3", "stem_relu3", "stem_pool1",
    
    # Stem Part 3
    "stem_conv4", "stem_relu4",
    "stem_conv5", "stem_relu5",
    
    # Stem Part 4
    "stem_pool2",
    
    # Inception-A1 (18 layers per block: 4 branches)
    "inception_a1_b1_1x1", "inception_a1_b1_relu",
    "inception_a1_b2_1x1", "inception_a1_b2_relu1", "inception_a1_b2_3x3", "inception_a1_b2_relu2",
    "inception_a1_b3_1x1", "inception_a1_b3_relu1", "inception_a1_b3_3x3_1", "inception_a1_b3_relu2",
    "inception_a1_b3_3x3_2", "inception_a1_b3_relu3",
    "inception_a1_b4_pool", "inception_a1_b4_1x1", "inception_a1_b4_relu",
    "inception_a1_concat",
    
    # Inception-A2
    "inception_a2_b1_1x1", "inception_a2_b1_relu",
    "inception_a2_b2_1x1", "inception_a2_b2_relu1", "inception_a2_b2_3x3", "inception_a2_b2_relu2",
    "inception_a2_b3_1x1", "inception_a2_b3_relu1", "inception_a2_b3_3x3_1", "inception_a2_b3_relu2",
    "inception_a2_b3_3x3_2", "inception_a2_b3_relu3",
    "inception_a2_b4_pool", "inception_a2_b4_1x1", "inception_a2_b4_relu",
    "inception_a2_concat",
    
    # Inception-A3
    "inception_a3_b1_1x1", "inception_a3_b1_relu",
    "inception_a3_b2_1x1", "inception_a3_b2_relu1", "inception_a3_b2_3x3", "inception_a3_b2_relu2",
    "inception_a3_b3_1x1", "inception_a3_b3_relu1", "inception_a3_b3_3x3_1", "inception_a3_b3_relu2",
    "inception_a3_b3_3x3_2", "inception_a3_b3_relu3",
    "inception_a3_b4_pool", "inception_a3_b4_1x1", "inception_a3_b4_relu",
    "inception_a3_concat",
    
    # Reduction-A (9 layers)
    "reduction_a_b1_3x3", "reduction_a_b1_relu",
    "reduction_a_b2_1x1", "reduction_a_b2_relu1", "reduction_a_b2_3x3", "reduction_a_b2_relu2",
    "reduction_a_b2_3x3_stride2", "reduction_a_b2_relu3",
    "reduction_a_b3_pool",
    "reduction_a_concat",
    
    # Inception-B1
    "inception_b1_b1_1x1", "inception_b1_b1_relu",
    "inception_b1_b2_1x1", "inception_b1_b2_relu1", "inception_b1_b2_3x3", "inception_b1_b2_relu2",
    "inception_b1_b3_1x1", "inception_b1_b3_relu1", "inception_b1_b3_3x3_1", "inception_b1_b3_relu2",
    "inception_b1_b3_3x3_2", "inception_b1_b3_relu3",
    "inception_b1_b4_pool", "inception_b1_b4_1x1", "inception_b1_b4_relu",
    "inception_b1_concat",
    
    # Inception-B2
    "inception_b2_b1_1x1", "inception_b2_b1_relu",
    "inception_b2_b2_1x1", "inception_b2_b2_relu1", "inception_b2_b2_3x3", "inception_b2_b2_relu2",
    "inception_b2_b3_1x1", "inception_b2_b3_relu1", "inception_b2_b3_3x3_1", "inception_b2_b3_relu2",
    "inception_b2_b3_3x3_2", "inception_b2_b3_relu3",
    "inception_b2_b4_pool", "inception_b2_b4_1x1", "inception_b2_b4_relu",
    "inception_b2_concat",
    
    # Inception-B3
    "inception_b3_b1_1x1", "inception_b3_b1_relu",
    "inception_b3_b2_1x1", "inception_b3_b2_relu1", "inception_b3_b2_3x3", "inception_b3_b2_relu2",
    "inception_b3_b3_1x1", "inception_b3_b3_relu1", "inception_b3_b3_3x3_1", "inception_b3_b3_relu2",
    "inception_b3_b3_3x3_2", "inception_b3_b3_relu3",
    "inception_b3_b4_pool", "inception_b3_b4_1x1", "inception_b3_b4_relu",
    "inception_b3_concat",
    
    # Inception-B4
    "inception_b4_b1_1x1", "inception_b4_b1_relu",
    "inception_b4_b2_1x1", "inception_b4_b2_relu1", "inception_b4_b2_3x3", "inception_b4_b2_relu2",
    "inception_b4_b3_1x1", "inception_b4_b3_relu1", "inception_b4_b3_3x3_1", "inception_b4_b3_relu2",
    "inception_b4_b3_3x3_2", "inception_b4_b3_relu3",
    "inception_b4_b4_pool", "inception_b4_b4_1x1", "inception_b4_b4_relu",
    "inception_b4_concat",
    
    # Reduction-B
    "reduction_b_b1_3x3", "reduction_b_b1_relu",
    "reduction_b_b2_1x1", "reduction_b_b2_relu1", "reduction_b_b2_3x3", "reduction_b_b2_relu2",
    "reduction_b_b2_3x3_stride2", "reduction_b_b2_relu3",
    "reduction_b_b3_pool",
    "reduction_b_concat",
    
    # Inception-C1
    "inception_c1_b1_1x1", "inception_c1_b1_relu",
    "inception_c1_b2_1x1", "inception_c1_b2_relu1", "inception_c1_b2_3x3", "inception_c1_b2_relu2",
    "inception_c1_b3_1x1", "inception_c1_b3_relu1", "inception_c1_b3_3x3_1", "inception_c1_b3_relu2",
    "inception_c1_b3_3x3_2", "inception_c1_b3_relu3",
    "inception_c1_b4_pool", "inception_c1_b4_1x1", "inception_c1_b4_relu",
    "inception_c1_concat",
    
    # Inception-C2
    "inception_c2_b1_1x1", "inception_c2_b1_relu",
    "inception_c2_b2_1x1", "inception_c2_b2_relu1", "inception_c2_b2_3x3", "inception_c2_b2_relu2",
    "inception_c2_b3_1x1", "inception_c2_b3_relu1", "inception_c2_b3_3x3_1", "inception_c2_b3_relu2",
    "inception_c2_b3_3x3_2", "inception_c2_b3_relu3",
    "inception_c2_b4_pool", "inception_c2_b4_1x1", "inception_c2_b4_relu",
    "inception_c2_concat",
    
    # Classifier
    "avgpool", "flatten", "fc", "output",
]

# Group configurations from profile_inception.py
GROUP_CONFIGS = {
    'Stem-Part1': {
        'store_chunk_elem': 4276896,
        'layer_names': ['input', 'stem_conv1', 'stem_relu1', 'stem_conv2', 'stem_relu2'],
    },
    'Stem-Part2': {
        'store_chunk_elem': 1382976,
        'layer_names': ['stem_conv3', 'stem_relu3', 'stem_pool1'],
    },
    'Stem-Part3': {
        'store_chunk_elem': 70080,
        'layer_names': ['stem_conv4', 'stem_relu4', 'stem_conv5', 'stem_relu5'],
    },
    'Stem-Part4': {
        'store_chunk_elem': 322624,
        'layer_names': ['stem_pool2'],
    },
    'Inception-A1': {'store_chunk_elem': 235200, 'layer_prefixes': ['inception_a1_']},
    'Inception-A2': {'store_chunk_elem': 235200, 'layer_prefixes': ['inception_a2_']},
    'Inception-A3': {'store_chunk_elem': 235200, 'layer_prefixes': ['inception_a3_']},
    'Reduction-A': {'store_chunk_elem': 470400, 'layer_prefixes': ['reduction_a']},
    'Inception-B1': {'store_chunk_elem': 55488, 'layer_prefixes': ['inception_b1_']},
    'Inception-B2': {'store_chunk_elem': 55488, 'layer_prefixes': ['inception_b2_']},
    'Inception-B3': {'store_chunk_elem': 55488, 'layer_prefixes': ['inception_b3_']},
    'Inception-B4': {'store_chunk_elem': 55488, 'layer_prefixes': ['inception_b4_']},
    'Reduction-B': {'store_chunk_elem': 277440, 'layer_prefixes': ['reduction_b']},
    'Inception-C1': {'store_chunk_elem': 1920, 'layer_prefixes': ['inception_c1_']},
    'Inception-C2': {'store_chunk_elem': 1920, 'layer_prefixes': ['inception_c2_']},
    'Classifier': {'store_chunk_elem': 64000, 'layer_names': ['avgpool', 'flatten', 'fc', 'output']},
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


def get_layer_group(layer_name):
    """Determine which group a layer belongs to."""
    # First, check for exact match in layer_names
    for group_name, config in GROUP_CONFIGS.items():
        if 'layer_names' in config:
            if layer_name in config['layer_names']:
                return group_name
    
    # Then, check for prefix match (more specific prefixes first)
    sorted_groups = sorted(
        GROUP_CONFIGS.items(),
        key=lambda x: max(len(p) for p in x[1].get('layer_prefixes', [''])),
        reverse=True
    )
    
    for group_name, config in sorted_groups:
        for prefix in config.get('layer_prefixes', []):
            if layer_name.startswith(prefix):
                return group_name
    
    return None


def main():
    print("="*80)
    print("Checking Layer-to-Group Mapping for Inception V3")
    print("="*80)
    
    print(f"\nTotal layers expected: {len(INCEPTION_V3_LAYERS)}")
    print(f"Total groups: {len(GROUP_ORDER)}")
    
    # Check each layer
    group_counts = {g: 0 for g in GROUP_ORDER}
    unassigned = []
    assignments = []
    
    print("\n" + "-"*80)
    print("Layer Assignments:")
    print("-"*80)
    
    for layer_name in INCEPTION_V3_LAYERS:
        group = get_layer_group(layer_name)
        if group:
            group_counts[group] += 1
            assignments.append((layer_name, group))
        else:
            unassigned.append(layer_name)
    
    # Print by group
    for group_name in GROUP_ORDER:
        config = GROUP_CONFIGS[group_name]
        mem_mb = config['store_chunk_elem'] * 4 / 1024 / 1024
        layers_in_group = [a[0] for a in assignments if a[1] == group_name]
        print(f"\n{group_name} ({mem_mb:.2f} MB, {group_counts[group_name]} layers):")
        for layer in layers_in_group:
            print(f"  - {layer}")
    
    # Summary
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    
    total_assigned = sum(group_counts.values())
    print(f"Total layers assigned: {total_assigned}/{len(INCEPTION_V3_LAYERS)}")
    
    print("\nLayers per group:")
    for group_name in GROUP_ORDER:
        print(f"  {group_name:20}: {group_counts[group_name]:3} layers")
    
    if unassigned:
        print(f"\n⚠ WARNING: {len(unassigned)} layers not assigned:")
        for name in unassigned:
            print(f"  - {name}")
    else:
        print(f"\n✓ All {len(INCEPTION_V3_LAYERS)} layers correctly assigned!")
    
    # Memory summary
    print("\n" + "-"*80)
    print("Memory Summary:")
    print("-"*80)
    max_mem = 0
    for group_name in GROUP_ORDER:
        config = GROUP_CONFIGS[group_name]
        mem_mb = config['store_chunk_elem'] * 4 / 1024 / 1024
        max_mem = max(max_mem, mem_mb)
        print(f"  {group_name:20}: {mem_mb:6.2f} MB")
    
    print(f"\nMaximum group memory: {max_mem:.2f} MB")
    print("="*80)


if __name__ == "__main__":
    main()



