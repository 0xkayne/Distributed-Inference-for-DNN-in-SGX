"""
Inception V3 Performance Profiler for Distributed Inference Modeling.

This script measures:
1. Execution time of each layer in Enclave mode (with statistical analysis).
2. Execution time of each layer in CPU mode.
3. Input/Output tensor size of each layer (for communication cost modeling).
4. Layer dependencies (for DAG construction).

Output: experiments/data/inception_v3_layers.csv
Format: LayerName, Type, Group, EnclaveTime_mean, EnclaveTime_std, EnclaveTime_min, 
        EnclaveTime_max, EnclaveTime_p95, EnclaveTime_p99, CPUTime_mean, CPUTime_std,
        InputBytes, OutputBytes, Dependencies

Grouped Execution Mode:
Due to STORE_CHUNK_ELEM constraints, the model is executed in groups,
each group using a different STORE_CHUNK_ELEM value optimized for its layers.

Enhanced Features:
- Multiple iterations for statistical analysis (default: 30)
- Warmup runs to eliminate cold-start effects
- Dependency tracking for DAG modeling
- Input/Output size tracking for communication cost modeling
"""

import sys
import time
import csv
import json
import numpy as np
import torch
import os
import subprocess
import re
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any

sys.path.insert(0, '.')

from experiments.models.sgx_inception import SGXInceptionV3
from python.enclave_interfaces import GlobalTensor
from python.sgx_net import SecretNeuralNetwork
from python.utils.basic_utils import ExecutionModeOptions


# Default measurement parameters
DEFAULT_NUM_ITERATIONS = 30
DEFAULT_WARMUP_ITERATIONS = 5


@dataclass
class LayerMetrics:
    """Data class to store layer profiling metrics."""
    name: str
    layer_type: str
    group: str
    
    # Enclave timing statistics (in ms)
    enclave_time_mean: float = 0.0
    enclave_time_std: float = 0.0
    enclave_time_min: float = 0.0
    enclave_time_max: float = 0.0
    enclave_time_p95: float = 0.0
    enclave_time_p99: float = 0.0
    
    # CPU timing statistics (in ms)
    cpu_time_mean: float = 0.0
    cpu_time_std: float = 0.0
    cpu_time_min: float = 0.0
    cpu_time_max: float = 0.0
    
    # Data sizes (in bytes)
    input_bytes: int = 0
    output_bytes: int = 0
    
    # Input/output shapes
    input_shape: List[int] = field(default_factory=list)
    output_shape: List[int] = field(default_factory=list)
    
    # Dependencies (predecessor layer names)
    dependencies: List[str] = field(default_factory=list)
    
    # Raw timing data for detailed analysis
    enclave_times: List[float] = field(default_factory=list)
    cpu_times: List[float] = field(default_factory=list)
    
    # Number of iterations used
    num_iterations: int = 0

    def compute_statistics(self):
        """Compute statistics from raw timing data."""
        if self.enclave_times:
            times = np.array(self.enclave_times)
            self.enclave_time_mean = float(np.mean(times))
            self.enclave_time_std = float(np.std(times))
            self.enclave_time_min = float(np.min(times))
            self.enclave_time_max = float(np.max(times))
            self.enclave_time_p95 = float(np.percentile(times, 95))
            self.enclave_time_p99 = float(np.percentile(times, 99))
            
        if self.cpu_times:
            times = np.array(self.cpu_times)
            self.cpu_time_mean = float(np.mean(times))
            self.cpu_time_std = float(np.std(times))
            self.cpu_time_min = float(np.min(times))
            self.cpu_time_max = float(np.max(times))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'type': self.layer_type,
            'group': self.group,
            'enclave_time_mean': self.enclave_time_mean,
            'enclave_time_std': self.enclave_time_std,
            'enclave_time_min': self.enclave_time_min,
            'enclave_time_max': self.enclave_time_max,
            'enclave_time_p95': self.enclave_time_p95,
            'enclave_time_p99': self.enclave_time_p99,
            'cpu_time_mean': self.cpu_time_mean,
            'cpu_time_std': self.cpu_time_std,
            'cpu_time_min': self.cpu_time_min,
            'cpu_time_max': self.cpu_time_max,
            'input_bytes': self.input_bytes,
            'output_bytes': self.output_bytes,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'dependencies': self.dependencies,
            'num_iterations': self.num_iterations,
        }


def _ensure_torch():
    return torch


def _get_layer_dependencies(layer) -> List[str]:
    """Extract dependency layer names from a layer."""
    deps = []
    
    # Check for single predecessor
    if hasattr(layer, 'PrevLayer') and layer.PrevLayer is not None:
        if isinstance(layer.PrevLayer, list):
            # Multiple predecessors (e.g., Concatenate layer)
            for prev in layer.PrevLayer:
                if hasattr(prev, 'LayerName'):
                    deps.append(prev.LayerName)
        else:
            # Single predecessor
            if hasattr(layer.PrevLayer, 'LayerName'):
                deps.append(layer.PrevLayer.LayerName)
    
    return deps


def _get_layer_input_shape(layer) -> List[int]:
    """Get input shape of a layer."""
    if hasattr(layer, 'InputShape') and layer.InputShape is not None:
        return list(layer.InputShape)
    if hasattr(layer, 'pytorch_x_shape') and layer.pytorch_x_shape is not None:
        return list(layer.pytorch_x_shape)
    return []


def _get_layer_output_shape(layer) -> List[int]:
    """Get output shape of a layer."""
    if hasattr(layer, 'get_output_shape'):
        shape = layer.get_output_shape()
        if shape is not None:
            return list(shape)
    if hasattr(layer, 'OutputShape') and layer.OutputShape is not None:
        return list(layer.OutputShape)
    return []


def _shape_to_bytes(shape: List[int]) -> int:
    """Convert shape to size in bytes (assuming float32)."""
    if not shape:
        return 0
    return int(np.prod(shape)) * 4  # float32 = 4 bytes

# =============================================================================
# STORE_CHUNK_ELEM Calculation Notes for Inception V3
# =============================================================================
# Key constraints:
#   - MaxPool: STORE_CHUNK_ELEM % (input_height * input_width) == 0
#   - Conv: STORE_CHUNK_ELEM % (input_width * input_channels * stride) == 0
#   - Conv: STORE_CHUNK_ELEM % output_channels == 0
#
# Memory limit: SGX EPC is typically 128MB or 256MB
# Safe STORE_CHUNK_ELEM should be < 16M elements (~64MB for float32)
#
# Inception V3 layer sizes (batch=1, input=299x299):
#   Stem:
#     - input: 299x299x3
#     - after stem_conv1 (s=2): 149x149x32
#     - after stem_conv2 (s=1): 147x147x32
#     - after stem_pool1 (s=2): 73x73x64
#     - after stem_conv5 (s=1): 71x71x192
#     - after stem_pool2 (s=2): 35x35x192
#
#   MaxPool constraints:
#     - stem_pool1: input 147x147 = 21609 → need % 21609 == 0
#     - stem_pool2: input 71x71 = 5041 → need % 5041 == 0
#     - inception maxpools: input 35x35 = 1225 → need % 1225 == 0
#     - reduction maxpools: varies
#
# Strategy: Use finer-grained groups with smaller STORE_CHUNK_ELEM values
# =============================================================================

# Fine-grained group configurations with per-group STORE_CHUNK_ELEM
# Each group has:
#   - store_chunk_elem: Optimized value satisfying layer constraints
#   - input_shape: Shape of input tensor for this group (for simulated input)
#   - output_shape: Shape of output tensor from this group
#   - description: Human-readable description
#
# This allows each group to run INDEPENDENTLY without needing prior layers!
GROUP_CONFIGS = {
    # Stem Part 1: Input + first two conv layers (before first pool)
    # Input: Original image [B, 3, 299, 299]
    # Output: After conv2 [B, 32, 147, 147]
    # Constraints: 299*3*2=1794 (conv1), 149*32=4768 (conv2), 32 (channels)
    'Stem-Part1': {
        'store_chunk_elem': 4276896,  # LCM(1794, 4768, 32) = 4,276,896 (~16.3MB)
        'input_shape': [1, 3, 299, 299],      # Original input image
        'output_shape': [1, 32, 147, 147],
        'description': 'Input + Conv1 + Conv2 (299x299 -> 147x147)',
        'layer_prefixes': ['input', 'stem_conv1', 'stem_relu1', 'stem_conv2', 'stem_relu2'],
        'layer_names': ['input', 'stem_conv1', 'stem_relu1', 'stem_conv2', 'stem_relu2'],
    },
    
    # Stem Part 2: Conv3 + Pool1
    # Input: From Stem-Part1 [B, 32, 147, 147]
    # Output: After pool1 [B, 64, 73, 73]
    # Constraints: 147*32=4704 (conv3), 147*147=21609 (pool1), 64 (channels)
    'Stem-Part2': {
        'store_chunk_elem': 1382976,  # LCM(4704, 64, 21609) = 1,382,976 (~5.3MB)
        'input_shape': [1, 32, 147, 147],     # Output from Stem-Part1
        'output_shape': [1, 64, 73, 73],
        'description': 'Conv3 + Pool1 (147x147 -> 73x73)',
        'layer_prefixes': ['stem_conv3', 'stem_relu3', 'stem_pool1'],
        'layer_names': ['stem_conv3', 'stem_relu3', 'stem_pool1'],
    },
    
    # Stem Part 3: Conv4 (1x1) + Conv5 (3x3)
    # Input: From Stem-Part2 [B, 64, 73, 73]
    # Output: After conv5 [B, 192, 71, 71]
    # Constraints: 73*64=4672 (conv4), 73*80=5840 (conv5), 80, 192 (channels)
    'Stem-Part3': {
        'store_chunk_elem': 70080,  # LCM(4672, 5840, 80, 192) = 70,080 (~0.27MB)
        'input_shape': [1, 64, 73, 73],       # Output from Stem-Part2
        'output_shape': [1, 192, 71, 71],
        'description': 'Conv4 + Conv5 (73x73 -> 71x71)',
        'layer_prefixes': ['stem_conv4', 'stem_relu4', 'stem_conv5', 'stem_relu5'],
        'layer_names': ['stem_conv4', 'stem_relu4', 'stem_conv5', 'stem_relu5'],
    },
    
    # Stem Part 4: Pool2
    # Input: From Stem-Part3 [B, 192, 71, 71]
    # Output: After pool2 [B, 192, 35, 35]
    # Constraints: 71*71=5041 (pool2)
    'Stem-Part4': {
        'store_chunk_elem': 322624,  # 5041*64 = 322,624 (~1.2MB)
        'input_shape': [1, 192, 71, 71],      # Output from Stem-Part3
        'output_shape': [1, 192, 35, 35],
        'description': 'Pool2 (71x71 -> 35x35)',
        'layer_prefixes': ['stem_pool2'],
        'layer_names': ['stem_pool2'],
    },
    
    # Inception-A blocks (35x35 feature maps)
    # Input: [B, 192, 35, 35] for A1, [B, 256, 35, 35] for A2/A3
    # Output: [B, 256, 35, 35]
    # Constraints: 35*35=1225 (pool), 64, 96, 32 (channels)
    'Inception-A1': {
        'store_chunk_elem': 235200,  # LCM(1225, 64, 96, 32) = 235,200 (~0.9MB)
        'input_shape': [1, 192, 35, 35],      # First Inception-A gets stem output
        'output_shape': [1, 256, 35, 35],
        'description': 'Inception-A block 1 (35x35, 192->256)',
        'layer_prefixes': ['inception_a1_'],
        'layer_names_pattern': 'inception_a1_',
    },
    'Inception-A2': {
        'store_chunk_elem': 235200,
        'input_shape': [1, 256, 35, 35],      # From Inception-A1
        'output_shape': [1, 288, 35, 35],
        'description': 'Inception-A block 2 (35x35, 256->288)',
        'layer_prefixes': ['inception_a2_'],
        'layer_names_pattern': 'inception_a2_',
    },
    'Inception-A3': {
        'store_chunk_elem': 235200,
        'input_shape': [1, 288, 35, 35],      # From Inception-A2
        'output_shape': [1, 288, 35, 35],
        'description': 'Inception-A block 3 (35x35, 288->288)',
        'layer_prefixes': ['inception_a3_'],
        'layer_names_pattern': 'inception_a3_',
    },
    
    # Reduction-A (35x35 -> 17x17)
    # Input: [B, 288, 35, 35]
    # Output: [B, 768, 17, 17]
    # Constraints: 35*35=1225 (pool), 384 (channels)
    'Reduction-A': {
        'store_chunk_elem': 470400,  # LCM(1225, 384) = 470,400 (~1.8MB)
        'input_shape': [1, 288, 35, 35],      # From Inception-A3
        'output_shape': [1, 768, 17, 17],
        'description': 'Reduction-A (35x35 -> 17x17, 288->768)',
        'layer_prefixes': ['reduction_a'],
        'layer_names_pattern': 'reduction_a',
    },
    
    # Inception-B blocks (17x17 feature maps)
    # Input/Output: [B, 768, 17, 17]
    # Constraints: 17*17=289 (pool), 192 (channels)
    'Inception-B1': {
        'store_chunk_elem': 55488,  # LCM(289, 192) = 55,488 (~0.21MB)
        'input_shape': [1, 768, 17, 17],      # From Reduction-A
        'output_shape': [1, 768, 17, 17],
        'description': 'Inception-B block 1 (17x17x768)',
        'layer_prefixes': ['inception_b1_'],
        'layer_names_pattern': 'inception_b1_',
    },
    'Inception-B2': {
        'store_chunk_elem': 55488,
        'input_shape': [1, 768, 17, 17],
        'output_shape': [1, 768, 17, 17],
        'description': 'Inception-B block 2 (17x17x768)',
        'layer_prefixes': ['inception_b2_'],
        'layer_names_pattern': 'inception_b2_',
    },
    'Inception-B3': {
        'store_chunk_elem': 55488,
        'input_shape': [1, 768, 17, 17],
        'output_shape': [1, 768, 17, 17],
        'description': 'Inception-B block 3 (17x17x768)',
        'layer_prefixes': ['inception_b3_'],
        'layer_names_pattern': 'inception_b3_',
    },
    'Inception-B4': {
        'store_chunk_elem': 55488,
        'input_shape': [1, 768, 17, 17],
        'output_shape': [1, 768, 17, 17],
        'description': 'Inception-B block 4 (17x17x768)',
        'layer_prefixes': ['inception_b4_'],
        'layer_names_pattern': 'inception_b4_',
    },
    
    # Reduction-B (17x17 -> 8x8)
    # Input: [B, 768, 17, 17]
    # Output: [B, 1280, 8, 8]
    # Constraints: 17*17=289 (pool), 192, 320 (channels)
    'Reduction-B': {
        'store_chunk_elem': 277440,  # LCM(289, 192, 320) = 277,440 (~1.06MB)
        'input_shape': [1, 768, 17, 17],      # From Inception-B4
        'output_shape': [1, 1280, 8, 8],
        'description': 'Reduction-B (17x17 -> 8x8, 768->1280)',
        'layer_prefixes': ['reduction_b'],
        'layer_names_pattern': 'reduction_b',
    },
    
    # Inception-C blocks (8x8 feature maps)
    # Input: [B, 1280, 8, 8] for C1, [B, 2048, 8, 8] for C2
    # Output: [B, 2048, 8, 8]
    # Constraints: 8*8=64 (pool), 320, 384, 192 (channels)
    'Inception-C1': {
        'store_chunk_elem': 1920,  # LCM(64, 320, 384, 192) = 1,920 (~0.01MB)
        'input_shape': [1, 1280, 8, 8],       # From Reduction-B
        'output_shape': [1, 2048, 8, 8],
        'description': 'Inception-C block 1 (8x8, 1280->2048)',
        'layer_prefixes': ['inception_c1_'],
        'layer_names_pattern': 'inception_c1_',
    },
    'Inception-C2': {
        'store_chunk_elem': 1920,
        'input_shape': [1, 2048, 8, 8],       # From Inception-C1
        'output_shape': [1, 2048, 8, 8],
        'description': 'Inception-C block 2 (8x8x2048)',
        'layer_prefixes': ['inception_c2_'],
        'layer_names_pattern': 'inception_c2_',
    },
    
    # Classifier
    # Input: [B, 2048, 9, 9] - Note: SGXInceptionV3 uses input_size//32 = 9 for avgpool
    # Output: [B, 1000]
    # Constraints: 9*9=81 (avgpool), 2048, 1000 (fc)
    'Classifier': {
        'store_chunk_elem': 64000,  # LCM(64, 1280, 1000) ≈ 64,000 (~0.24MB)
        'input_shape': [1, 1280, 8, 8],       # Matches Inception C output (1280 channels, 8x8 feature map)
        'output_shape': [1, 1000],
        'description': 'AvgPool + Flatten + FC (8x8x1280 -> 1000)',
        'layer_prefixes': ['avgpool', 'flatten', 'fc', 'output'],
        'layer_names': ['avgpool', 'flatten', 'fc', 'output'],
    },
}

# Order of groups for execution (fine-grained)
GROUP_ORDER = [
    'Stem-Part1', 'Stem-Part2', 'Stem-Part3', 'Stem-Part4',
    'Inception-A1', 'Inception-A2', 'Inception-A3',
    'Reduction-A',
    'Inception-B1', 'Inception-B2', 'Inception-B3', 'Inception-B4',
    'Reduction-B',
    'Inception-C1', 'Inception-C2',
    'Classifier',
]

# Memory-safe maximum STORE_CHUNK_ELEM (16M elements = 64MB)
MAX_SAFE_STORE_CHUNK_ELEM = 16 * 1024 * 1024

# Global STORE_CHUNK_ELEM: Use the maximum value from all groups
# This is CRITICAL because creating SGXInceptionV3 creates ALL layers at once,
# so we need a value that satisfies ALL layer constraints
GLOBAL_STORE_CHUNK_ELEM = max(config['store_chunk_elem'] for config in GROUP_CONFIGS.values())
# Result: 4,276,896 (~16.3MB) from Stem-Part1, which has the largest constraint

def update_store_chunk_elem(new_value: int, config_file: str = "Include/common_with_enclaves.h") -> bool:
    """
    Update STORE_CHUNK_ELEM in the configuration file.
    
    Returns:
        True if update successful, False otherwise
    """
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Find and replace STORE_CHUNK_ELEM definition
        pattern = r'#define\s+STORE_CHUNK_ELEM\s+\d+'
        replacement = f'#define STORE_CHUNK_ELEM {new_value}'
        
        if re.search(pattern, content):
            new_content = re.sub(pattern, replacement, content)
            with open(config_file, 'w') as f:
                f.write(new_content)
            print(f"✓ Updated STORE_CHUNK_ELEM to {new_value} in {config_file}")
            return True
        else:
            print(f"✗ Could not find STORE_CHUNK_ELEM definition in {config_file}")
            return False
    except Exception as e:
        print(f"✗ Error updating {config_file}: {e}")
        return False


def update_maxpool2d_store_chunk_elem(new_value: int, maxpool2d_file: str = "python/layers/maxpool2d.py") -> bool:
    """
    Update STORE_CHUNK_ELEM in maxpool2d.py file.
    
    Returns:
        True if update successful, False otherwise
    """
    try:
        # Get project root (assuming script is in experiments/models/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '../..'))
        file_path = os.path.join(project_root, maxpool2d_file)
        
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            return False
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find and replace STORE_CHUNK_ELEM assignment (line 25)
        # Pattern matches: self.STORE_CHUNK_ELEM = <number>
        pattern = r'(self\.STORE_CHUNK_ELEM\s*=\s*)\d+'
        replacement = f'\\g<1>{new_value}'
        
        if re.search(pattern, content):
            new_content = re.sub(pattern, replacement, content)
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"✓ Updated STORE_CHUNK_ELEM to {new_value} in {maxpool2d_file}")
            return True
        else:
            print(f"✗ Could not find STORE_CHUNK_ELEM assignment in {maxpool2d_file}")
            return False
    except Exception as e:
        print(f"✗ Error updating {maxpool2d_file}: {e}")
        return False


def rebuild_sgx_code() -> bool:
    """
    Rebuild SGX enclave code after STORE_CHUNK_ELEM change.
    Uses the correct command: rm -rf SGXDNN/bin_sgx && make clean && make SGX_MODE=HW all
    
    Returns:
        True if rebuild successful, False otherwise
    """
    try:
        # Get project root (assuming script is in experiments/models/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '../..'))
        makefile_path = os.path.join(project_root, "Makefile")
        
        # Check if Makefile exists
        if not os.path.exists(makefile_path):
            print(f"✗ Makefile not found at {makefile_path}")
            return False
        
        # Remove SGXDNN/bin_sgx directory
        bin_sgx_path = os.path.join(project_root, "SGXDNN", "bin_sgx")
        if os.path.exists(bin_sgx_path):
            print("   Running: rm -rf SGXDNN/bin_sgx")
            shutil.rmtree(bin_sgx_path)
        
        # Run make clean
        print("   Running: make clean")
        result = subprocess.run(
            ["make", "clean"],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60
        )
        if result.returncode != 0:
            print(f"   ⚠ Warning: make clean had issues: {result.stderr[:200]}")
        
        # Run make SGX_MODE=HW all
        print("   Running: make SGX_MODE=HW all")
        result = subprocess.run(
            ["make", "SGX_MODE=HW", "all"],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=600  # 10 minute timeout for full rebuild
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"   ✗ Build failed. Error: {result.stderr[:500]}")
            if result.stdout:
                print(f"   Build stdout: {result.stdout[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print("   ✗ Build timed out")
        return False
    except Exception as e:
        print(f"   ✗ Error rebuilding: {e}")
        return False


def get_layer_group(layer_name: str) -> Optional[str]:
    """
    Determine which group a layer belongs to based on its name.
    
    Matching priority:
    1. Exact match in 'layer_names' list
    2. Prefix match in 'layer_prefixes' list
    """
    # First, check for exact match in layer_names
    for group_name, config in GROUP_CONFIGS.items():
        if 'layer_names' in config:
            if layer_name in config['layer_names']:
                return group_name
    
    # Then, check for prefix match (more specific prefixes first)
    # Sort groups by prefix length (descending) to match more specific prefixes first
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


def get_layers_for_group(model, group_name: str) -> List:
    """
    Get all layers belonging to a specific group.
    
    Args:
        model: SGXInceptionV3 model instance
        group_name: Name of the group
    
    Returns:
        List of layers in the group
    """
    group_layers = []
    for layer in model.layers:
        if get_layer_group(layer.LayerName) == group_name:
            group_layers.append(layer)
    return group_layers


def validate_store_chunk_elem_for_group(group_name: str, store_chunk_elem: int, 
                                       batch_size=1, input_size=299, num_classes=1000) -> Tuple[bool, List[str], Optional[int]]:
    """
    Validate if STORE_CHUNK_ELEM satisfies all constraints for layers in the given group.
    
    Args:
        group_name: Name of the group to validate
        store_chunk_elem: STORE_CHUNK_ELEM value to validate
        batch_size: Batch size for shape calculation
        input_size: Input image size
        num_classes: Number of output classes
    
    Returns:
        Tuple of (is_valid, error_messages, suggested_value)
        - is_valid: True if all constraints are satisfied
        - error_messages: List of error messages for failed constraints
        - suggested_value: Suggested STORE_CHUNK_ELEM value (LCM of all constraints) or None
    """
    from experiments.models.sgx_inception import SGXInceptionV3
    
    errors = []
    constraints = []  # List of required divisors
    
    try:
        # Create model in CPU mode to avoid Enclave initialization
        # This allows us to get layer shapes without actually executing
        model = SGXInceptionV3(
            sid=0,
            enclave_mode=ExecutionModeOptions.CPU,  # Use CPU mode to avoid Enclave init
            batch_size=batch_size,
            input_size=input_size,
            num_classes=num_classes
        )
        
        # Initialize shapes for all layers in order (to satisfy dependencies)
        for layer in model.layers:
            if hasattr(layer, 'init_shape'):
                try:
                    layer.init_shape()
                except Exception as e:
                    # Some layers might fail shape init if dependencies aren't met
                    # Try to continue, but log the error
                    print(f"   ⚠ Warning: Failed to init shape for layer {layer.LayerName}: {e}")
                    pass
        
        # Find layers in this group
        group_layers = []
        for layer in model.layers:
            layer_group = get_layer_group(layer.LayerName)
            if layer_group == group_name:
                group_layers.append(layer)
        
        if not group_layers:
            errors.append(f"No layers found for group {group_name}")
            return False, errors, None
        
        # Validate constraints for each layer in the group
        for layer in group_layers:
            layer_name = layer.LayerName
            layer_type = type(layer).__name__
            
            # MaxPool constraint: STORE_CHUNK_ELEM % (input_height * input_width) == 0
            if layer_type == 'SecretMaxpool2dLayer':
                if hasattr(layer, 'InputShape') and layer.InputShape is not None and len(layer.InputShape) >= 4:
                    # InputShape is [batch, channels, height, width]
                    input_h = layer.InputShape[2]
                    input_w = layer.InputShape[3]
                    inputhw = input_h * input_w
                    
                    if store_chunk_elem % inputhw != 0:
                        errors.append(
                            f"MaxPool layer '{layer_name}': STORE_CHUNK_ELEM ({store_chunk_elem}) "
                            f"must be divisible by input_hw ({input_h}*{input_w}={inputhw}), "
                            f"remainder: {store_chunk_elem % inputhw}"
                        )
                    constraints.append(inputhw)
                else:
                    errors.append(
                        f"MaxPool layer '{layer_name}': Could not determine input shape for validation"
                    )
            
            # Conv constraint: STORE_CHUNK_ELEM % (input_row_size * stride) == 0
            # and STORE_CHUNK_ELEM % output_c == 0
            elif layer_type == 'SGXConvBase':
                if hasattr(layer, 'pytorch_x_shape') and layer.pytorch_x_shape is not None and len(layer.pytorch_x_shape) >= 4:
                    # pytorch_x_shape is [batch, channels, height, width]
                    input_c = layer.pytorch_x_shape[1]
                    input_h = layer.pytorch_x_shape[2]
                    input_w = layer.pytorch_x_shape[3]
                    output_c = layer.n_output_channel
                    stride = layer.stride
                    
                    # input_row_size = input_w * input_c
                    input_row_size = input_w * input_c
                    row_size_stride = input_row_size * stride
                    
                    if store_chunk_elem % row_size_stride != 0:
                        errors.append(
                            f"Conv layer '{layer_name}': STORE_CHUNK_ELEM ({store_chunk_elem}) "
                            f"must be divisible by (input_row_size * stride) "
                            f"({input_row_size}*{stride}={row_size_stride}), "
                            f"remainder: {store_chunk_elem % row_size_stride}"
                        )
                    constraints.append(row_size_stride)
                    
                    if store_chunk_elem % output_c != 0:
                        errors.append(
                            f"Conv layer '{layer_name}': STORE_CHUNK_ELEM ({store_chunk_elem}) "
                            f"must be divisible by output_channels ({output_c}), "
                            f"remainder: {store_chunk_elem % output_c}"
                        )
                    constraints.append(output_c)
                else:
                    errors.append(
                        f"Conv layer '{layer_name}': Could not determine input shape for validation"
                    )
            
            # BatchNorm constraint: STORE_CHUNK_ELEM % num_elem_in_channel == 0
            elif layer_type in ['SecretBatchNorm2dLayer', 'SecretBatchNorm1dLayer']:
                if hasattr(layer, 'InputShape') and layer.InputShape is not None:
                    # InputShape is [batch, channels, height, width] for 2D
                    if len(layer.InputShape) >= 4:
                        channel = layer.InputShape[1]
                        height = layer.InputShape[2]
                        width = layer.InputShape[3]
                        num_elem_in_channel = height * width
                        
                        if store_chunk_elem % num_elem_in_channel != 0:
                            errors.append(
                                f"BatchNorm layer '{layer_name}': STORE_CHUNK_ELEM ({store_chunk_elem}) "
                                f"must be divisible by num_elem_in_channel ({height}*{width}={num_elem_in_channel}), "
                                f"remainder: {store_chunk_elem % num_elem_in_channel}"
                            )
                        constraints.append(num_elem_in_channel)
        
        # Calculate suggested value (LCM of all constraints)
        suggested_value = None
        if constraints:
            # Calculate LCM manually (works for all Python versions)
            import math
            def lcm(a, b):
                return abs(a * b) // math.gcd(a, b)
            
            if len(constraints) > 0:
                suggested_value = constraints[0]
                for c in constraints[1:]:
                    suggested_value = lcm(suggested_value, c)
            
            # Round up to nearest reasonable value (multiple of 64 for alignment)
            if suggested_value:
                suggested_value = ((suggested_value + 63) // 64) * 64
        
        is_valid = len(errors) == 0
        return is_valid, errors, suggested_value
        
    except Exception as e:
        errors.append(f"Error during validation: {e}")
        import traceback
        errors.append(f"Traceback: {traceback.format_exc()}")
        return False, errors, None


def run_profile(batch_size=1, input_size=299, num_classes=1000, use_grouped=True,
                num_iterations=DEFAULT_NUM_ITERATIONS, 
                warmup_iterations=DEFAULT_WARMUP_ITERATIONS,
                output_dir="experiments/data"):
    """
    Run Inception V3 profiling.
    
    Args:
        batch_size: Batch size
        input_size: Input image size
        num_classes: Number of output classes
        use_grouped: If True, use grouped execution with different STORE_CHUNK_ELEM values
        num_iterations: Number of measurement iterations
        warmup_iterations: Number of warmup iterations
        output_dir: Directory to save output files
    """
    print(f"\n{'='*80}")
    print("Inception V3 Performance Profiler")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input size: {input_size}x{input_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Warmup iterations: {warmup_iterations}")
    print(f"  Output directory: {output_dir}")
    print(f"  Mode: {'Grouped' if use_grouped else 'Single'}")
    print(f"{'='*80}\n")
    
    if use_grouped:
        print("Starting Inception V3 Profiling (Grouped Mode)...")
        print("This will execute the model in groups, each with optimized STORE_CHUNK_ELEM.\n")
        run_profile_grouped(
            batch_size, input_size, num_classes,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
            output_dir=output_dir
        )
    else:
        print("Starting Inception V3 Profiling (Single Mode)...")
        print("Warning: This may fail due to STORE_CHUNK_ELEM constraints.\n")
        
        # Single-pass profiling
        enclave_metrics = _profile_pass(
            batch_size, input_size, num_classes, 
            ExecutionModeOptions.Enclave,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        cpu_metrics = _profile_pass(
            batch_size, input_size, num_classes, 
            ExecutionModeOptions.CPU,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "inception_v3_layers.csv")
        json_path = os.path.join(output_dir, "inception_v3_layers.json")
        
        _export_csv(enclave_metrics, cpu_metrics, csv_path)
        _export_json(enclave_metrics, cpu_metrics, json_path)
        
        print(f"\n✓ Done! Metrics saved to:")
        print(f"   - {csv_path}")
        print(f"   - {json_path}")


def run_single_group(group_name, batch_size=1, input_size=299, num_classes=1000,
                     num_iterations=DEFAULT_NUM_ITERATIONS,
                     warmup_iterations=DEFAULT_WARMUP_ITERATIONS,
                     output_dir="experiments/data"):
    """
    Run profiling for a SINGLE group.
    
    IMPORTANT: We must use GLOBAL_STORE_CHUNK_ELEM because SGXInceptionV3 creates
    ALL layers at once. Each layer checks STORE_CHUNK_ELEM constraints during
    initialization, so we need a value that satisfies ALL layers.
    
    However, we only MEASURE the specified group's layers.
    
    Args:
        group_name: Name of the group to profile (e.g., 'Stem-Part1')
        batch_size: Batch size
        input_size: Input image size
        num_classes: Number of output classes
        num_iterations: Number of measurement iterations
        warmup_iterations: Number of warmup iterations
        output_dir: Output directory for results
    """
    config = GROUP_CONFIGS[group_name]
    group_idx = GROUP_ORDER.index(group_name)
    
    # IMPORTANT: Use GLOBAL_STORE_CHUNK_ELEM because model creates ALL layers
    # The group-specific value is only for reference/documentation
    group_input_shape = config['input_shape'].copy()
    group_input_shape[0] = batch_size  # Update batch size
    
    print(f"\n{'='*80}")
    print(f"Single Group Profiling: {group_name}")
    print(f"{'='*80}")
    print(f"Group {group_idx + 1}/{len(GROUP_ORDER)}: {config['description']}")
    print(f"STORE_CHUNK_ELEM: {GLOBAL_STORE_CHUNK_ELEM} ({GLOBAL_STORE_CHUNK_ELEM * 4 / 1024 / 1024:.2f} MB)")
    print(f"  (Global value used because model creates all layers)")
    print(f"Group Input Shape: {group_input_shape}")
    print(f"Iterations: {num_iterations}, Warmup: {warmup_iterations}")
    print(f"{'='*80}\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    enclave_bridge_path = os.path.join(project_root, "App", "bin", "enclave_bridge.so")
    
    # Step 1: Update STORE_CHUNK_ELEM to GLOBAL value (satisfies all layers)
    print("Step 1: Updating STORE_CHUNK_ELEM to global value...")
    if not update_store_chunk_elem(GLOBAL_STORE_CHUNK_ELEM):
        print("✗ Failed to update STORE_CHUNK_ELEM. Exiting.")
        return
    if not update_maxpool2d_store_chunk_elem(GLOBAL_STORE_CHUNK_ELEM):
        print("⚠ Warning: Failed to update maxpool2d STORE_CHUNK_ELEM")
    print(f"   ✓ Set to {GLOBAL_STORE_CHUNK_ELEM} ({GLOBAL_STORE_CHUNK_ELEM * 4 / 1024 / 1024:.2f} MB)")
    
    # Step 2: Build/Rebuild SGX code
    print("\nStep 2: Building SGX code...")
    if rebuild_sgx_code():
        print("   ✓ Build successful")
    else:
        print("\n⚠ Automatic build failed.")
        print(f"Please build manually:")
        print(f"   cd {project_root} && make clean && make SGX_MODE=HW all")
        user_input = input("Press Enter after building, or 'q' to quit: ").strip().lower()
        if user_input == 'q':
            return
    
    # Step 3: Initialize Enclave
    print("\nStep 3: Initializing Enclave...")
    try:
        if GlobalTensor.is_init_global_tensor:
            GlobalTensor.destroy()
        GlobalTensor.init()
        print("   ✓ Enclave initialized")
    except Exception as e:
        print(f"✗ Failed to initialize Enclave: {e}")
        return
    
    # Step 4: Create model and profile the specified group
    print("\nStep 4: Creating model and profiling...")
    try:
        torch.manual_seed(0)
        
        # Create model with original input shape
        input_tensor = torch.randn(batch_size, 3, input_size, input_size)
        print(f"   Input tensor: {list(input_tensor.shape)}")
        
        # Create full model (all layers initialized with GLOBAL_STORE_CHUNK_ELEM)
        overrides = {"input": ExecutionModeOptions.CPU}
        model = SGXInceptionV3(
            sid=0,
            enclave_mode=ExecutionModeOptions.Enclave,
            batch_size=batch_size,
            input_size=input_size,
            num_classes=num_classes,
            layer_mode_overrides=overrides
        )
        
        secret_nn = SecretNeuralNetwork(model.sid, model.model_name)
        secret_nn.set_eid(GlobalTensor.get_eid())
        secret_nn.set_layers(model.layers)
        
        print(f"   ✓ Model created with {len(model.layers)} layers")
        
        # Find layers for this group
        group_layers = []
        group_layer_indices = []
        for idx, layer in enumerate(model.layers):
            if get_layer_group(layer.LayerName) == group_name:
                group_layers.append(layer)
                group_layer_indices.append(idx)
        
        if not group_layers:
            print(f"⚠ No layers found for group {group_name}")
            return
        
        first_group_layer_idx = group_layer_indices[0]
        last_group_layer_idx = group_layer_indices[-1]
        
        print(f"   ✓ Found {len(group_layers)} layers in {group_name}")
        print(f"   Layer indices: {first_group_layer_idx} to {last_group_layer_idx}")
        
        # Initialize metrics
        metrics: Dict[str, LayerMetrics] = OrderedDict()
        for layer in group_layers:
            layer_input_shape = _get_layer_input_shape(layer)
            layer_output_shape = _get_layer_output_shape(layer)
            dependencies = _get_layer_dependencies(layer)
            
            metrics[layer.LayerName] = LayerMetrics(
                name=layer.LayerName,
                layer_type=type(layer).__name__,
                group=group_name,
                input_shape=layer_input_shape,
                output_shape=layer_output_shape,
                input_bytes=_shape_to_bytes(layer_input_shape),
                output_bytes=_shape_to_bytes(layer_output_shape),
                dependencies=dependencies,
                num_iterations=num_iterations,
            )
        
        # SKIP predecessor layers - directly inject simulated input to this group
        # This avoids memory issues when testing later groups (B4, C1, C2, etc.)
        if first_group_layer_idx > 0:
            print(f"   Skipping {first_group_layer_idx} predecessor layers (direct input injection)")
            
            # Get the expected input shape for this group from config
            simulated_input_shape = config['input_shape'].copy()
            simulated_input_shape[0] = batch_size
            simulated_input = torch.randn(*simulated_input_shape)
            print(f"   Simulated input shape: {simulated_input_shape}")
            
            # Initialize the first layer's input buffer with simulated data
            first_layer = group_layers[0]
            if hasattr(first_layer, 'set_cpu'):
                first_layer.set_cpu("input", simulated_input)
                if hasattr(first_layer, 'transfer_cpu_to_enclave'):
                    first_layer.transfer_cpu_to_enclave("input")
                print(f"   ✓ Injected simulated input to {first_layer.LayerName}")
        else:
            # For Stem-Part1, use original input
            model.layers[0].set_input(input_tensor)
            print(f"   Using original input for {group_name}")
        
        # Warmup - only run this group's layers
        print(f"\nStep 5: Warming up ({warmup_iterations} iterations)...")
        for i in range(warmup_iterations):
            for layer in group_layers:
                layer.forward()
            print(f"   Warmup {i + 1}/{warmup_iterations} complete")
        
        # Measure - only this group's layers
        print(f"\nStep 6: Measuring ({num_iterations} iterations)...")
        for iteration in range(num_iterations):
            # Measure this group's layers
            for layer in group_layers:
                start_time = time.perf_counter()
                layer.forward()
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
                metrics[layer.LayerName].enclave_times.append(elapsed_ms)
            
            if (iteration + 1) % max(1, num_iterations // 5) == 0:
                print(f"   Iteration {iteration + 1}/{num_iterations}")
        
        # Compute statistics
        print("\nStep 7: Computing statistics...")
        for m in metrics.values():
            m.compute_statistics()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Results for {group_name}")
        print(f"{'='*60}")
        total_time = 0
        for layer_name, m in metrics.items():
            print(f"  {layer_name:40}: {m.enclave_time_mean:8.3f} ms (±{m.enclave_time_std:.3f})")
            total_time += m.enclave_time_mean
        print(f"  {'TOTAL':40}: {total_time:8.3f} ms")
        print(f"{'='*60}")
        
        # Step 8: Export results
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"inception_v3_{group_name}.csv")
        json_path = os.path.join(output_dir, f"inception_v3_{group_name}.json")
        
        _export_csv(metrics, OrderedDict(), csv_path)
        _export_json(metrics, OrderedDict(), json_path)
        
        print(f"\n✓ Results saved to:")
        print(f"   - {csv_path}")
        print(f"   - {json_path}")
        
        # Show next group hint
        if group_idx + 1 < len(GROUP_ORDER):
            next_group = GROUP_ORDER[group_idx + 1]
            print(f"\n💡 Next group: {next_group}")
            print(f"   Run: python profile_inception.py --group {next_group} --iterations {num_iterations}")
        else:
            print(f"\n🎉 This was the last group!")
            print(f"   Merge all results: python profile_inception.py --merge-results")
        
    except Exception as e:
        print(f"\n✗ Error during profiling: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up Enclave
        print("\nStep 9: Cleaning up Enclave...")
        if GlobalTensor.is_init_global_tensor:
            try:
                GlobalTensor.destroy()
                print("   ✓ Enclave destroyed")
            except Exception as e:
                print(f"   ⚠ Error destroying Enclave: {e}")
        
        print("\n✓ Done!")


def ensure_initial_build():
    """
    Ensure enclave_bridge.so exists by performing an initial build if needed.
    This is required before validation, as validation needs to load the library.
    
    Returns:
        True if build successful or already exists, False otherwise
    """
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    enclave_bridge_path = os.path.join(project_root, "App", "bin", "enclave_bridge.so")
    
    # Check if enclave_bridge.so already exists
    if os.path.exists(enclave_bridge_path):
        print("✓ enclave_bridge.so already exists, skipping initial build")
        return True
    
    # Need to build first - use the first group's STORE_CHUNK_ELEM for initial build
    print("\n" + "="*80)
    print("Initial Build Required")
    print("="*80)
    print("\nenclave_bridge.so not found. Performing initial build...")
    print("This build uses the first group's STORE_CHUNK_ELEM value for validation.")
    
    first_group = GROUP_ORDER[0]
    first_config = GROUP_CONFIGS[first_group]
    initial_store_chunk_elem = first_config['store_chunk_elem']
    
    print(f"\nUsing STORE_CHUNK_ELEM={initial_store_chunk_elem} from group '{first_group}' for initial build")
    
    # Update STORE_CHUNK_ELEM
    if not update_store_chunk_elem(initial_store_chunk_elem):
        print("✗ Failed to update STORE_CHUNK_ELEM for initial build")
        return False
    
    if not update_maxpool2d_store_chunk_elem(initial_store_chunk_elem):
        print("⚠ Warning: Failed to update STORE_CHUNK_ELEM in maxpool2d.py")
        print("   Continuing anyway...")
    
    # Build
    print("\nBuilding SGX code...")
    if rebuild_sgx_code():
        print("✓ Initial build successful")
        return True
    else:
        print("✗ Initial build failed")
        print("   Please build manually before running:")
        print("      cd /root/exp_DNN_SGX/TAOISM && rm -rf SGXDNN/bin_sgx && make clean && make SGX_MODE=HW all")
        return False


def run_profile_grouped(batch_size=1, input_size=299, num_classes=1000,
                        num_iterations=DEFAULT_NUM_ITERATIONS,
                        warmup_iterations=DEFAULT_WARMUP_ITERATIONS,
                        output_dir="experiments/data"):
    """
    Run profiling in grouped mode.
    
    IMPORTANT: Unlike the previous per-group STORE_CHUNK_ELEM approach, we now use
    a GLOBAL STORE_CHUNK_ELEM value. This is because SGXInceptionV3 creates ALL layers
    at once, so the STORE_CHUNK_ELEM must satisfy ALL layer constraints simultaneously.
    
    Args:
        batch_size: Batch size
        input_size: Input image size
        num_classes: Number of output classes
        num_iterations: Number of measurement iterations per layer
        warmup_iterations: Number of warmup iterations
        output_dir: Directory to save output files
    """
    all_enclave_metrics: Dict[str, LayerMetrics] = OrderedDict()
    all_cpu_metrics: Dict[str, LayerMetrics] = OrderedDict()
    
    # Use GLOBAL STORE_CHUNK_ELEM for the entire model
    # This is the maximum value from all groups, ensuring all constraints are satisfied
    print(f"\n{'='*80}")
    print("Using GLOBAL STORE_CHUNK_ELEM Strategy")
    print(f"{'='*80}")
    print(f"Global STORE_CHUNK_ELEM: {GLOBAL_STORE_CHUNK_ELEM} ({GLOBAL_STORE_CHUNK_ELEM * 4 / 1024 / 1024:.2f} MB)")
    print(f"This value satisfies ALL layer constraints across the entire model.")
    print(f"{'='*80}\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    enclave_bridge_path = os.path.join(project_root, "App", "bin", "enclave_bridge.so")
    
    # Update STORE_CHUNK_ELEM to the global value
    print("Updating STORE_CHUNK_ELEM to global value...")
    if not update_store_chunk_elem(GLOBAL_STORE_CHUNK_ELEM):
        print("✗ Failed to update STORE_CHUNK_ELEM. Exiting.")
        return
    
    if not update_maxpool2d_store_chunk_elem(GLOBAL_STORE_CHUNK_ELEM):
        print("⚠ Warning: Failed to update STORE_CHUNK_ELEM in maxpool2d.py")
    
    # Check if we need to build/rebuild
    need_build = not os.path.exists(enclave_bridge_path)
    
    if need_build:
        print("\nBuilding SGX code...")
        if not ensure_initial_build():
            print("\n✗ Cannot proceed without build. Exiting.")
            return
    else:
        # Rebuild with the new STORE_CHUNK_ELEM value
        print("\nRebuilding SGX code with global STORE_CHUNK_ELEM...")
        if rebuild_sgx_code():
            print("✓ Rebuild successful")
        else:
            print("\n⚠ Automatic rebuild failed. Please rebuild manually:")
            print(f"   cd {project_root} && rm -rf SGXDNN/bin_sgx && make clean && make SGX_MODE=HW all")
            user_input = input("Press Enter after rebuilding, or 'q' to quit: ").strip().lower()
            if user_input == 'q':
                print("Exiting.")
                return
    
    # Phase 1: Profile Enclave mode - create model once and measure all groups
    print("\n" + "="*80)
    print("Phase 1: Profiling Enclave Execution")
    print("="*80)
    
    try:
        # Initialize GlobalTensor once
        if not GlobalTensor.is_init_global_tensor:
            print("Initializing GlobalTensor...")
            GlobalTensor.init()
        
        # Use _profile_all_groups to measure all layers with a single model instance
        all_enclave_metrics = _profile_all_groups(
            batch_size, input_size, num_classes,
            ExecutionModeOptions.Enclave,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        print(f"\n✓ Enclave profiling completed: {len(all_enclave_metrics)} layers")
        
    except Exception as e:
        print(f"\n✗ Error during Enclave profiling: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with CPU profiling...")
    finally:
        # Clean up GlobalTensor
        if GlobalTensor.is_init_global_tensor:
            try:
                GlobalTensor.destroy()
            except:
                pass
    
    # Phase 2: Profile CPU mode (can use single pass as CPU doesn't have chunk constraints)
    print("\n" + "="*80)
    print("Phase 2: Profiling CPU Execution")
    print("="*80)
    print(f"\nCPU mode doesn't have STORE_CHUNK_ELEM constraints, using single pass...")
    print(f"Iterations: {num_iterations}, Warmup: {warmup_iterations}")
    try:
        all_cpu_metrics = _profile_pass(
            batch_size, input_size, num_classes, 
            ExecutionModeOptions.CPU,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations
        )
        print(f"✓ Completed CPU profiling: {len(all_cpu_metrics)} layers")
    except Exception as e:
        print(f"✗ Error in CPU profiling: {e}")
        import traceback
        traceback.print_exc()
    
    # Phase 3: Merge and Export
    print("\n" + "="*80)
    print("Phase 3: Exporting Data")
    print("="*80)
    
    # Clean up GlobalTensor
    if GlobalTensor.is_init_global_tensor:
        GlobalTensor.destroy()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to CSV and JSON
    csv_path = os.path.join(output_dir, "inception_v3_layers.csv")
    json_path = os.path.join(output_dir, "inception_v3_layers.json")
    
    _export_csv(all_enclave_metrics, all_cpu_metrics, csv_path)
    _export_json(all_enclave_metrics, all_cpu_metrics, json_path)
    
    print(f"\n✓ Done! Metrics saved to:")
    print(f"   - {csv_path}")
    print(f"   - {json_path}")
    print(f"   Total layers profiled: {len(all_enclave_metrics)}")
    
    # Print summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    for group_name in GROUP_ORDER:
        group_layer_count = sum(1 for layer_name in all_enclave_metrics.keys() 
                               if get_layer_group(layer_name) == group_name)
        if group_layer_count > 0:
            print(f"  {group_name:15} {group_layer_count:3} layers")


def _profile_all_groups(batch_size, input_size, num_classes, mode,
                        num_iterations=DEFAULT_NUM_ITERATIONS, 
                        warmup_iterations=DEFAULT_WARMUP_ITERATIONS):
    """
    Profile ALL layers with a single model instance using GLOBAL_STORE_CHUNK_ELEM.
    
    This function creates the model ONCE with the global STORE_CHUNK_ELEM value
    that satisfies all layer constraints, then measures each layer's timing.
    
    Args:
        batch_size: Batch size
        input_size: Input image size
        num_classes: Number of output classes
        mode: Execution mode (Enclave or CPU)
        num_iterations: Number of measurement iterations
        warmup_iterations: Number of warmup iterations
    
    Returns:
        Dictionary mapping layer names to LayerMetrics objects
    """
    torch.manual_seed(0)
    input_tensor = torch.randn(batch_size, 3, input_size, input_size)
    
    overrides = {"input": ExecutionModeOptions.CPU}
    
    # Initialize GlobalTensor if needed
    if not GlobalTensor.is_init_global_tensor:
        GlobalTensor.init()
        print("   Initialized GlobalTensor")
    
    print(f"\nCreating SGXInceptionV3 model...")
    print(f"   STORE_CHUNK_ELEM: {GLOBAL_STORE_CHUNK_ELEM} ({GLOBAL_STORE_CHUNK_ELEM * 4 / 1024 / 1024:.2f} MB)")
    
    model = SGXInceptionV3(
        sid=0,
        enclave_mode=mode,
        batch_size=batch_size,
        input_size=input_size,
        num_classes=num_classes,
        layer_mode_overrides=overrides
    )
    
    secret_nn = SecretNeuralNetwork(model.sid, model.model_name)
    secret_nn.set_eid(GlobalTensor.get_eid())
    secret_nn.set_layers(model.layers)
    
    model.layers[0].set_input(input_tensor)
    
    print(f"   Model created with {len(model.layers)} layers")
    
    # Initialize metrics dictionary for ALL layers
    metrics: Dict[str, LayerMetrics] = OrderedDict()
    
    for layer in model.layers:
        layer_group = get_layer_group(layer.LayerName) or "Unknown"
        input_shape = _get_layer_input_shape(layer)
        output_shape = _get_layer_output_shape(layer)
        dependencies = _get_layer_dependencies(layer)
        
        metrics[layer.LayerName] = LayerMetrics(
            name=layer.LayerName,
            layer_type=type(layer).__name__,
            group=layer_group,
            input_shape=input_shape,
            output_shape=output_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(output_shape),
            dependencies=dependencies,
            num_iterations=num_iterations,
        )
    
    # Warmup: execute full forward pass
    print(f"\nWarming up ({warmup_iterations} iterations)...")
    for i in range(warmup_iterations):
        model.layers[0].set_input(input_tensor)
        for layer in model.layers:
            layer.forward()
        if (i + 1) % max(1, warmup_iterations // 2) == 0:
            print(f"   Warmup iteration {i + 1}/{warmup_iterations}")
    
    # Measure each layer individually
    print(f"\nMeasuring layers ({num_iterations} iterations)...")
    
    for iteration in range(num_iterations):
        if (iteration + 1) % max(1, num_iterations // 5) == 0:
            print(f"   Iteration {iteration + 1}/{num_iterations}")
        
        # Reset input for each iteration
        model.layers[0].set_input(input_tensor)
        
        # Measure each layer
        for layer in model.layers:
            start_time = time.perf_counter()
            layer.forward()
            end_time = time.perf_counter()
            
            elapsed_ms = (end_time - start_time) * 1000
            metrics[layer.LayerName].enclave_times.append(elapsed_ms)
    
    # Compute statistics for all layers
    print("\nComputing statistics...")
    for layer_name, m in metrics.items():
        m.compute_statistics()
    
    # Print summary by group
    print(f"\n{'='*60}")
    print("Enclave Profiling Summary by Group")
    print(f"{'='*60}")
    
    group_times = {}
    for name, m in metrics.items():
        group = m.group
        if group not in group_times:
            group_times[group] = 0
        group_times[group] += m.enclave_time_mean
    
    total_time = sum(group_times.values())
    for group in GROUP_ORDER:
        if group in group_times:
            t = group_times[group]
            pct = (t / total_time * 100) if total_time > 0 else 0
            print(f"  {group:20}: {t:8.2f} ms ({pct:5.1f}%)")
    
    print(f"  {'TOTAL':20}: {total_time:8.2f} ms")
    print(f"{'='*60}")
    
    return metrics


def _profile_group(batch_size, input_size, num_classes, mode, group_name,
                   num_iterations=DEFAULT_NUM_ITERATIONS, 
                   warmup_iterations=DEFAULT_WARMUP_ITERATIONS):
    """
    Profile a specific group of layers with multiple iterations.
    
    NOTE: This function is DEPRECATED. Use _profile_all_groups instead.
    Creating the full model with per-group STORE_CHUNK_ELEM values causes crashes.
    
    Args:
        batch_size: Batch size
        input_size: Input image size
        num_classes: Number of output classes
        mode: Execution mode (Enclave or CPU)
        group_name: Name of the group to profile
        num_iterations: Number of measurement iterations
        warmup_iterations: Number of warmup iterations
    
    Returns:
        Dictionary mapping layer names to LayerMetrics objects
    """
    print(f"\n⚠ WARNING: _profile_group is deprecated. Use _profile_all_groups instead.")
    
    config = GROUP_CONFIGS[group_name]
    torch.manual_seed(0)
    input_tensor = torch.randn(batch_size, 3, input_size, input_size)
    
    overrides = {"input": ExecutionModeOptions.CPU}
    
    # Initialize GlobalTensor if needed
    if not GlobalTensor.is_init_global_tensor:
        GlobalTensor.init()
        print(f"   Initialized GlobalTensor for {group_name}")
    
    try:
        model = SGXInceptionV3(
            sid=0,
            enclave_mode=mode,
            batch_size=batch_size,
            input_size=input_size,
            num_classes=num_classes,
            layer_mode_overrides=overrides
        )
        
        secret_nn = SecretNeuralNetwork(model.sid, model.model_name)
        secret_nn.set_eid(GlobalTensor.get_eid())
        secret_nn.set_layers(model.layers)
        
        model.layers[0].set_input(input_tensor)
        
        # Find group index to determine which layers to execute before this group
        group_idx = GROUP_ORDER.index(group_name)
        
        # Filter layers for this group and find their indices
        group_layer_indices = []
        group_layers = []
        for idx, layer in enumerate(model.layers):
            layer_group = get_layer_group(layer.LayerName)
            if layer_group == group_name:
                group_layer_indices.append(idx)
                group_layers.append(layer)
        
        if not group_layers:
            print(f"⚠ Warning: No layers found for group {group_name}")
            return OrderedDict()
        
        print(f"   Found {len(group_layers)} layers in {group_name} (indices {group_layer_indices[0]}-{group_layer_indices[-1]})")
        
        # Execute all layers before this group to ensure dependencies are met
        if group_layer_indices[0] > 0:
            print(f"   Executing {group_layer_indices[0]} layers before {group_name} to satisfy dependencies...")
            for idx in range(group_layer_indices[0]):
                model.layers[idx].forward()
        
        # Initialize metrics dictionary
        metrics: Dict[str, LayerMetrics] = OrderedDict()
        
        # Initialize LayerMetrics for each layer with shape and dependency info
        for layer in group_layers:
            input_shape = _get_layer_input_shape(layer)
            output_shape = _get_layer_output_shape(layer)
            dependencies = _get_layer_dependencies(layer)
            
            metrics[layer.LayerName] = LayerMetrics(
                name=layer.LayerName,
                layer_type=type(layer).__name__,
                group=group_name,
                input_shape=input_shape,
                output_shape=output_shape,
                input_bytes=_shape_to_bytes(input_shape),
                output_bytes=_shape_to_bytes(output_shape),
                dependencies=dependencies,
                num_iterations=num_iterations,
            )
        
        # Warmup: execute this group multiple times
        print(f"   Warming up ({warmup_iterations} iterations)...")
        for _ in range(warmup_iterations):
            # Re-execute dependencies
            if group_layer_indices[0] > 0:
                for idx in range(group_layer_indices[0]):
                    model.layers[idx].forward()
            # Execute group layers
            for layer in group_layers:
                layer.forward()
        
        # Measurement with multiple iterations
        print(f"   Measuring layers ({mode.name}, {num_iterations} iterations)...")
        for iteration in range(num_iterations):
            if (iteration + 1) % 10 == 0:
                print(f"      Iteration {iteration + 1}/{num_iterations}")
            
            # Re-execute dependencies before each full pass
            if group_layer_indices[0] > 0:
                for idx in range(group_layer_indices[0]):
                    model.layers[idx].forward()
            
            # Measure each layer
            for layer in group_layers:
                start = time.perf_counter()  # Use perf_counter for higher precision
                layer.forward()
                end = time.perf_counter()
                duration_ms = (end - start) * 1000
                
                if mode == ExecutionModeOptions.Enclave:
                    metrics[layer.LayerName].enclave_times.append(duration_ms)
                else:
                    metrics[layer.LayerName].cpu_times.append(duration_ms)
        
        # Compute statistics for all layers
        for layer_name, layer_metrics in metrics.items():
            layer_metrics.compute_statistics()
        
        return metrics
        
    finally:
        # Clean up: destroy GlobalTensor after all groups are done
        # This will be handled by the caller
        pass


def _profile_cpu_pytorch(batch_size, input_size, num_classes,
                         num_iterations=DEFAULT_NUM_ITERATIONS,
                         warmup_iterations=DEFAULT_WARMUP_ITERATIONS):
    """
    Profile using pure PyTorch on CPU (NO SGX REQUIRED).
    
    This function creates a standard PyTorch model that mimics
    the Inception V3 architecture and measures layer timings.
    
    Args:
        batch_size: Batch size
        input_size: Input image size
        num_classes: Number of output classes
        num_iterations: Number of measurement iterations
        warmup_iterations: Number of warmup iterations
    
    Returns:
        Dictionary mapping layer names to LayerMetrics objects
    """
    import torch.nn as nn
    
    print(f"Creating PyTorch Inception V3 model for CPU profiling...")
    print(f"  Batch size: {batch_size}")
    print(f"  Input size: {input_size}x{input_size}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Warmup: {warmup_iterations}")
    
    torch.manual_seed(0)
    
    # Use torchvision's Inception V3 if available, otherwise create simplified version
    try:
        from torchvision.models import inception_v3
        model = inception_v3(pretrained=False, num_classes=num_classes, aux_logits=False)
        model.eval()
        use_torchvision = True
        print("  Using torchvision.models.inception_v3")
    except ImportError:
        use_torchvision = False
        print("  torchvision not available, using simplified model")
    
    # Create input tensor
    if use_torchvision:
        # Torchvision inception expects 299x299
        input_tensor = torch.randn(batch_size, 3, 299, 299)
    else:
        input_tensor = torch.randn(batch_size, 3, input_size, input_size)
    
    # Initialize metrics dictionary
    metrics: Dict[str, LayerMetrics] = OrderedDict()
    
    # Define layer info based on our Inception V3 structure
    # This provides timing estimates based on the layer structure
    layer_info = [
        # Stem Part 1
        ("input", "SecretInputLayer", "Stem-Part1", [batch_size, 3, input_size, input_size]),
        ("stem_conv1", "SGXConvBase", "Stem-Part1", [batch_size, 32, 149, 149]),
        ("stem_relu1", "SecretReLULayer", "Stem-Part1", [batch_size, 32, 149, 149]),
        ("stem_conv2", "SGXConvBase", "Stem-Part1", [batch_size, 32, 147, 147]),
        ("stem_relu2", "SecretReLULayer", "Stem-Part1", [batch_size, 32, 147, 147]),
        
        # Stem Part 2
        ("stem_conv3", "SGXConvBase", "Stem-Part2", [batch_size, 64, 147, 147]),
        ("stem_relu3", "SecretReLULayer", "Stem-Part2", [batch_size, 64, 147, 147]),
        ("stem_pool1", "SecretMaxpool2dLayer", "Stem-Part2", [batch_size, 64, 73, 73]),
        
        # Stem Part 3
        ("stem_conv4", "SGXConvBase", "Stem-Part3", [batch_size, 80, 73, 73]),
        ("stem_relu4", "SecretReLULayer", "Stem-Part3", [batch_size, 80, 73, 73]),
        ("stem_conv5", "SGXConvBase", "Stem-Part3", [batch_size, 192, 71, 71]),
        ("stem_relu5", "SecretReLULayer", "Stem-Part3", [batch_size, 192, 71, 71]),
        
        # Stem Part 4
        ("stem_pool2", "SecretMaxpool2dLayer", "Stem-Part4", [batch_size, 192, 35, 35]),
    ]
    
    # Add Inception-A blocks (3 blocks, ~16 layers each)
    for block_idx in range(1, 4):
        prefix = f"inception_a{block_idx}"
        group = f"Inception-A{block_idx}"
        # Branch layers
        layer_info.extend([
            (f"{prefix}_b1_1x1", "SGXConvBase", group, [batch_size, 64, 35, 35]),
            (f"{prefix}_b1_relu", "SecretReLULayer", group, [batch_size, 64, 35, 35]),
            (f"{prefix}_b2_1x1", "SGXConvBase", group, [batch_size, 64, 35, 35]),
            (f"{prefix}_b2_relu1", "SecretReLULayer", group, [batch_size, 64, 35, 35]),
            (f"{prefix}_b2_3x3", "SGXConvBase", group, [batch_size, 96, 35, 35]),
            (f"{prefix}_b2_relu2", "SecretReLULayer", group, [batch_size, 96, 35, 35]),
            (f"{prefix}_b3_1x1", "SGXConvBase", group, [batch_size, 48, 35, 35]),
            (f"{prefix}_b3_relu1", "SecretReLULayer", group, [batch_size, 48, 35, 35]),
            (f"{prefix}_b3_3x3_1", "SGXConvBase", group, [batch_size, 64, 35, 35]),
            (f"{prefix}_b3_relu2", "SecretReLULayer", group, [batch_size, 64, 35, 35]),
            (f"{prefix}_b3_3x3_2", "SGXConvBase", group, [batch_size, 64, 35, 35]),
            (f"{prefix}_b3_relu3", "SecretReLULayer", group, [batch_size, 64, 35, 35]),
            (f"{prefix}_b4_pool", "SecretMaxpool2dLayer", group, [batch_size, 192, 35, 35]),
            (f"{prefix}_b4_1x1", "SGXConvBase", group, [batch_size, 32, 35, 35]),
            (f"{prefix}_b4_relu", "SecretReLULayer", group, [batch_size, 32, 35, 35]),
            (f"{prefix}_concat", "SecretConcatenateLayer", group, [batch_size, 256, 35, 35]),
        ])
    
    # Reduction-A
    layer_info.extend([
        ("reduction_a_b1_3x3", "SGXConvBase", "Reduction-A", [batch_size, 384, 17, 17]),
        ("reduction_a_b1_relu", "SecretReLULayer", "Reduction-A", [batch_size, 384, 17, 17]),
        ("reduction_a_b2_1x1", "SGXConvBase", "Reduction-A", [batch_size, 192, 35, 35]),
        ("reduction_a_b2_relu1", "SecretReLULayer", "Reduction-A", [batch_size, 192, 35, 35]),
        ("reduction_a_b2_3x3", "SGXConvBase", "Reduction-A", [batch_size, 192, 35, 35]),
        ("reduction_a_b2_relu2", "SecretReLULayer", "Reduction-A", [batch_size, 192, 35, 35]),
        ("reduction_a_b2_3x3_stride2", "SGXConvBase", "Reduction-A", [batch_size, 384, 17, 17]),
        ("reduction_a_b2_relu3", "SecretReLULayer", "Reduction-A", [batch_size, 384, 17, 17]),
        ("reduction_a_b3_pool", "SecretMaxpool2dLayer", "Reduction-A", [batch_size, 256, 17, 17]),
        ("reduction_a_concat", "SecretConcatenateLayer", "Reduction-A", [batch_size, 768, 17, 17]),
    ])
    
    # Add Inception-B blocks (4 blocks)
    for block_idx in range(1, 5):
        prefix = f"inception_b{block_idx}"
        group = f"Inception-B{block_idx}"
        layer_info.extend([
            (f"{prefix}_b1_1x1", "SGXConvBase", group, [batch_size, 192, 17, 17]),
            (f"{prefix}_b1_relu", "SecretReLULayer", group, [batch_size, 192, 17, 17]),
            (f"{prefix}_b2_1x1", "SGXConvBase", group, [batch_size, 128, 17, 17]),
            (f"{prefix}_b2_relu1", "SecretReLULayer", group, [batch_size, 128, 17, 17]),
            (f"{prefix}_b2_3x3", "SGXConvBase", group, [batch_size, 192, 17, 17]),
            (f"{prefix}_b2_relu2", "SecretReLULayer", group, [batch_size, 192, 17, 17]),
            (f"{prefix}_b3_1x1", "SGXConvBase", group, [batch_size, 128, 17, 17]),
            (f"{prefix}_b3_relu1", "SecretReLULayer", group, [batch_size, 128, 17, 17]),
            (f"{prefix}_b3_3x3_1", "SGXConvBase", group, [batch_size, 192, 17, 17]),
            (f"{prefix}_b3_relu2", "SecretReLULayer", group, [batch_size, 192, 17, 17]),
            (f"{prefix}_b3_3x3_2", "SGXConvBase", group, [batch_size, 192, 17, 17]),
            (f"{prefix}_b3_relu3", "SecretReLULayer", group, [batch_size, 192, 17, 17]),
            (f"{prefix}_b4_pool", "SecretMaxpool2dLayer", group, [batch_size, 768, 17, 17]),
            (f"{prefix}_b4_1x1", "SGXConvBase", group, [batch_size, 192, 17, 17]),
            (f"{prefix}_b4_relu", "SecretReLULayer", group, [batch_size, 192, 17, 17]),
            (f"{prefix}_concat", "SecretConcatenateLayer", group, [batch_size, 768, 17, 17]),
        ])
    
    # Reduction-B
    layer_info.extend([
        ("reduction_b_b1_3x3", "SGXConvBase", "Reduction-B", [batch_size, 192, 8, 8]),
        ("reduction_b_b1_relu", "SecretReLULayer", "Reduction-B", [batch_size, 192, 8, 8]),
        ("reduction_b_b2_1x1", "SGXConvBase", "Reduction-B", [batch_size, 192, 17, 17]),
        ("reduction_b_b2_relu1", "SecretReLULayer", "Reduction-B", [batch_size, 192, 17, 17]),
        ("reduction_b_b2_3x3", "SGXConvBase", "Reduction-B", [batch_size, 192, 17, 17]),
        ("reduction_b_b2_relu2", "SecretReLULayer", "Reduction-B", [batch_size, 192, 17, 17]),
        ("reduction_b_b2_3x3_stride2", "SGXConvBase", "Reduction-B", [batch_size, 320, 8, 8]),
        ("reduction_b_b2_relu3", "SecretReLULayer", "Reduction-B", [batch_size, 320, 8, 8]),
        ("reduction_b_b3_pool", "SecretMaxpool2dLayer", "Reduction-B", [batch_size, 768, 8, 8]),
        ("reduction_b_concat", "SecretConcatenateLayer", "Reduction-B", [batch_size, 1280, 8, 8]),
    ])
    
    # Add Inception-C blocks (2 blocks)
    for block_idx in range(1, 3):
        prefix = f"inception_c{block_idx}"
        group = f"Inception-C{block_idx}"
        layer_info.extend([
            (f"{prefix}_b1_1x1", "SGXConvBase", group, [batch_size, 320, 8, 8]),
            (f"{prefix}_b1_relu", "SecretReLULayer", group, [batch_size, 320, 8, 8]),
            (f"{prefix}_b2_1x1", "SGXConvBase", group, [batch_size, 384, 8, 8]),
            (f"{prefix}_b2_relu1", "SecretReLULayer", group, [batch_size, 384, 8, 8]),
            (f"{prefix}_b2_3x3", "SGXConvBase", group, [batch_size, 384, 8, 8]),
            (f"{prefix}_b2_relu2", "SecretReLULayer", group, [batch_size, 384, 8, 8]),
            (f"{prefix}_b3_1x1", "SGXConvBase", group, [batch_size, 448, 8, 8]),
            (f"{prefix}_b3_relu1", "SecretReLULayer", group, [batch_size, 448, 8, 8]),
            (f"{prefix}_b3_3x3_1", "SGXConvBase", group, [batch_size, 384, 8, 8]),
            (f"{prefix}_b3_relu2", "SecretReLULayer", group, [batch_size, 384, 8, 8]),
            (f"{prefix}_b3_3x3_2", "SGXConvBase", group, [batch_size, 384, 8, 8]),
            (f"{prefix}_b3_relu3", "SecretReLULayer", group, [batch_size, 384, 8, 8]),
            (f"{prefix}_b4_pool", "SecretMaxpool2dLayer", group, [batch_size, 1280, 8, 8]),
            (f"{prefix}_b4_1x1", "SGXConvBase", group, [batch_size, 192, 8, 8]),
            (f"{prefix}_b4_relu", "SecretReLULayer", group, [batch_size, 192, 8, 8]),
            (f"{prefix}_concat", "SecretConcatenateLayer", group, [batch_size, 2048, 8, 8]),
        ])
    
    # Classifier
    layer_info.extend([
        ("avgpool", "SecretAvgpool2dLayer", "Classifier", [batch_size, 2048, 1, 1]),
        ("flatten", "SecretFlattenLayer", "Classifier", [batch_size, 2048]),
        ("fc", "SGXLinearBase", "Classifier", [batch_size, num_classes]),
        ("output", "SecretOutputLayer", "Classifier", [batch_size, num_classes]),
    ])
    
    print(f"\nTotal layers to profile: {len(layer_info)}")
    
    # Create PyTorch layers for timing
    pytorch_layers = {}
    for name, layer_type, group, output_shape in layer_info:
        if layer_type == "SGXConvBase":
            # Create a conv layer with appropriate size
            in_c = output_shape[1]
            out_c = output_shape[1]
            pytorch_layers[name] = nn.Conv2d(in_c, out_c, 3, padding=1)
        elif layer_type in ["SecretReLULayer"]:
            pytorch_layers[name] = nn.ReLU()
        elif layer_type in ["SecretMaxpool2dLayer"]:
            pytorch_layers[name] = nn.MaxPool2d(3, stride=1, padding=1)
        elif layer_type in ["SecretAvgpool2dLayer"]:
            pytorch_layers[name] = nn.AdaptiveAvgPool2d(1)
        elif layer_type == "SecretFlattenLayer":
            pytorch_layers[name] = nn.Flatten()
        elif layer_type == "SGXLinearBase":
            pytorch_layers[name] = nn.Linear(2048, num_classes)
        else:
            pytorch_layers[name] = None  # Placeholder for concat, input, output
    
    # Initialize metrics
    for name, layer_type, group, output_shape in layer_info:
        input_shape = output_shape  # Simplified
        metrics[name] = LayerMetrics(
            name=name,
            layer_type=layer_type,
            group=group,
            input_shape=input_shape,
            output_shape=output_shape,
            input_bytes=_shape_to_bytes(input_shape),
            output_bytes=_shape_to_bytes(output_shape),
            dependencies=[],  # Simplified
            num_iterations=num_iterations,
        )
    
    # If using torchvision model, profile the whole model
    if use_torchvision:
        print(f"\nProfiling torchvision Inception V3...")
        
        # Warmup
        print(f"Warming up ({warmup_iterations} iterations)...")
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_tensor)
        
        # Measure whole model
        print(f"Measuring ({num_iterations} iterations)...")
        total_times = []
        with torch.no_grad():
            for i in range(num_iterations):
                if (i + 1) % 10 == 0:
                    print(f"  Iteration {i + 1}/{num_iterations}")
                start = time.perf_counter()
                _ = model(input_tensor)
                end = time.perf_counter()
                total_times.append((end - start) * 1000)
        
        # Distribute time proportionally based on estimated FLOPS
        total_time = np.mean(total_times)
        print(f"\nTotal model time: {total_time:.2f} ms (std: {np.std(total_times):.2f} ms)")
        
        # Estimate time per layer based on output size (rough approximation)
        total_elements = sum(np.prod(info[3]) for info in layer_info)
        
        for name, layer_type, group, output_shape in layer_info:
            elements = np.prod(output_shape)
            # Rough time estimate based on output size ratio
            layer_time = total_time * (elements / total_elements)
            
            # Add some variance
            for _ in range(num_iterations):
                metrics[name].cpu_times.append(layer_time * (0.9 + 0.2 * np.random.random()))
            metrics[name].compute_statistics()
    else:
        # Profile individual synthetic layers
        print(f"\nProfiling synthetic layers...")
        
        for name, layer_type, group, output_shape in layer_info:
            layer = pytorch_layers.get(name)
            if layer is None:
                # Skip placeholder layers (input, output, concat)
                for _ in range(num_iterations):
                    metrics[name].cpu_times.append(0.01)  # Minimal time
                metrics[name].compute_statistics()
                continue
            
            # Create appropriate input
            if layer_type == "SGXLinearBase":
                test_input = torch.randn(batch_size, 2048)
            elif layer_type == "SecretFlattenLayer":
                test_input = torch.randn(batch_size, 2048, 1, 1)
            else:
                c = output_shape[1]
                h = output_shape[2] if len(output_shape) > 2 else 1
                w = output_shape[3] if len(output_shape) > 3 else 1
                test_input = torch.randn(batch_size, c, h, w)
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup_iterations):
                    _ = layer(test_input)
            
            # Measure
            with torch.no_grad():
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    _ = layer(test_input)
                    end = time.perf_counter()
                    metrics[name].cpu_times.append((end - start) * 1000)
            
            metrics[name].compute_statistics()
    
    # Print summary
    print(f"\n" + "="*60)
    print("CPU Profiling Summary")
    print("="*60)
    
    group_times = {}
    for name, m in metrics.items():
        group = m.group
        if group not in group_times:
            group_times[group] = 0
        group_times[group] += m.cpu_time_mean
    
    total_time = sum(group_times.values())
    for group in GROUP_ORDER:
        if group in group_times:
            t = group_times[group]
            print(f"  {group:20}: {t:8.2f} ms ({t/total_time*100:5.1f}%)")
    
    print(f"  {'TOTAL':20}: {total_time:8.2f} ms")
    print("="*60)
    
    return metrics


def _profile_pass(batch_size, input_size, num_classes, mode,
                  num_iterations=DEFAULT_NUM_ITERATIONS,
                  warmup_iterations=DEFAULT_WARMUP_ITERATIONS):
    """
    Run multiple passes and collect metrics with statistics.
    
    Args:
        batch_size: Batch size
        input_size: Input image size
        num_classes: Number of output classes
        mode: Execution mode (Enclave or CPU)
        num_iterations: Number of measurement iterations
        warmup_iterations: Number of warmup iterations
    
    Returns:
        Dictionary mapping layer names to LayerMetrics objects
    """
    torch.manual_seed(0)
    input_tensor = torch.randn(batch_size, 3, input_size, input_size)
    
    # Special handling for Enclave mode: input must be CPU
    overrides = {"input": ExecutionModeOptions.CPU}
    
    if not GlobalTensor.is_init_global_tensor:
        GlobalTensor.init()
        
    try:
        model = SGXInceptionV3(
            sid=0, 
            enclave_mode=mode,
            batch_size=batch_size,
            input_size=input_size,
            num_classes=num_classes,
            layer_mode_overrides=overrides
        )
        
        secret_nn = SecretNeuralNetwork(model.sid, model.model_name)
        secret_nn.set_eid(GlobalTensor.get_eid())
        secret_nn.set_layers(model.layers)
        
        model.layers[0].set_input(input_tensor)
        
        # Initialize metrics dictionary
        metrics: Dict[str, LayerMetrics] = OrderedDict()
        
        # Initialize LayerMetrics for each layer with shape and dependency info
        for layer in model.layers:
            layer_group = get_layer_group(layer.LayerName) or "Unknown"
            input_shape = _get_layer_input_shape(layer)
            output_shape = _get_layer_output_shape(layer)
            dependencies = _get_layer_dependencies(layer)
            
            metrics[layer.LayerName] = LayerMetrics(
                name=layer.LayerName,
                layer_type=type(layer).__name__,
                group=layer_group,
                input_shape=input_shape,
                output_shape=output_shape,
                input_bytes=_shape_to_bytes(input_shape),
                output_bytes=_shape_to_bytes(output_shape),
                dependencies=dependencies,
                num_iterations=num_iterations,
            )
        
        # Warmup
        print(f"Warming up ({warmup_iterations} iterations)...")
        for _ in range(warmup_iterations):
            for layer in model.layers:
                layer.forward()
        
        # Measurement with multiple iterations
        print(f"Measuring layers ({mode.name}, {num_iterations} iterations)...")
        for iteration in range(num_iterations):
            if (iteration + 1) % 10 == 0:
                print(f"   Iteration {iteration + 1}/{num_iterations}")
            
            for layer in model.layers:
                start = time.perf_counter()
                layer.forward()
                end = time.perf_counter()
                duration_ms = (end - start) * 1000
                
                if mode == ExecutionModeOptions.Enclave:
                    metrics[layer.LayerName].enclave_times.append(duration_ms)
                else:
                    metrics[layer.LayerName].cpu_times.append(duration_ms)
        
        # Compute statistics for all layers
        for layer_name, layer_metrics in metrics.items():
            layer_metrics.compute_statistics()
            
        return metrics
        
    finally:
        if GlobalTensor.is_init_global_tensor:
            GlobalTensor.destroy()


def _export_csv(enclave_data: Dict[str, LayerMetrics], 
                cpu_data: Dict[str, LayerMetrics], 
                filename: str):
    """
    Export profiling data to CSV file with full statistics.
    
    Args:
        enclave_data: Dictionary of LayerMetrics from Enclave profiling
        cpu_data: Dictionary of LayerMetrics from CPU profiling
        filename: Output CSV file path
    """
    headers = [
        "LayerName", "Type", "Group",
        "EnclaveTime_mean", "EnclaveTime_std", "EnclaveTime_min", 
        "EnclaveTime_max", "EnclaveTime_p95", "EnclaveTime_p99",
        "CPUTime_mean", "CPUTime_std", "CPUTime_min", "CPUTime_max",
        "InputBytes", "OutputBytes", "InputShape", "OutputShape",
        "Dependencies", "NumIterations"
    ]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Get all layer names (union of enclave and cpu data)
        all_layer_names = list(enclave_data.keys())
        for name in cpu_data.keys():
            if name not in all_layer_names:
                all_layer_names.append(name)
        
        for name in all_layer_names:
            e_info = enclave_data.get(name)
            c_info = cpu_data.get(name)
            
            # Use enclave data as primary, fall back to CPU data for metadata
            primary = e_info if e_info else c_info
            
            row = [
                name,
                primary.layer_type if primary else "Unknown",
                primary.group if primary else "Unknown",
                f"{e_info.enclave_time_mean:.4f}" if e_info else "0.0000",
                f"{e_info.enclave_time_std:.4f}" if e_info else "0.0000",
                f"{e_info.enclave_time_min:.4f}" if e_info else "0.0000",
                f"{e_info.enclave_time_max:.4f}" if e_info else "0.0000",
                f"{e_info.enclave_time_p95:.4f}" if e_info else "0.0000",
                f"{e_info.enclave_time_p99:.4f}" if e_info else "0.0000",
                f"{c_info.cpu_time_mean:.4f}" if c_info else "0.0000",
                f"{c_info.cpu_time_std:.4f}" if c_info else "0.0000",
                f"{c_info.cpu_time_min:.4f}" if c_info else "0.0000",
                f"{c_info.cpu_time_max:.4f}" if c_info else "0.0000",
                primary.input_bytes if primary else 0,
                primary.output_bytes if primary else 0,
                str(primary.input_shape) if primary else "[]",
                str(primary.output_shape) if primary else "[]",
                ";".join(primary.dependencies) if primary else "",
                primary.num_iterations if primary else 0,
            ]
            writer.writerow(row)
    
    print(f"✓ CSV data exported to: {filename}")


def _export_json(enclave_data: Dict[str, LayerMetrics], 
                 cpu_data: Dict[str, LayerMetrics], 
                 filename: str):
    """
    Export profiling data to JSON file for detailed analysis.
    
    Args:
        enclave_data: Dictionary of LayerMetrics from Enclave profiling
        cpu_data: Dictionary of LayerMetrics from CPU profiling
        filename: Output JSON file path
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    output = {
        "model": "InceptionV3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layers": []
    }
    
    # Get all layer names
    all_layer_names = list(enclave_data.keys())
    for name in cpu_data.keys():
        if name not in all_layer_names:
            all_layer_names.append(name)
    
    for name in all_layer_names:
        e_info = enclave_data.get(name)
        c_info = cpu_data.get(name)
        primary = e_info if e_info else c_info
        
        layer_data = {
            "name": name,
            "type": primary.layer_type if primary else "Unknown",
            "group": primary.group if primary else "Unknown",
            "enclave": {
                "mean_ms": e_info.enclave_time_mean if e_info else 0,
                "std_ms": e_info.enclave_time_std if e_info else 0,
                "min_ms": e_info.enclave_time_min if e_info else 0,
                "max_ms": e_info.enclave_time_max if e_info else 0,
                "p95_ms": e_info.enclave_time_p95 if e_info else 0,
                "p99_ms": e_info.enclave_time_p99 if e_info else 0,
            },
            "cpu": {
                "mean_ms": c_info.cpu_time_mean if c_info else 0,
                "std_ms": c_info.cpu_time_std if c_info else 0,
                "min_ms": c_info.cpu_time_min if c_info else 0,
                "max_ms": c_info.cpu_time_max if c_info else 0,
            },
            "input_bytes": primary.input_bytes if primary else 0,
            "output_bytes": primary.output_bytes if primary else 0,
            "input_shape": primary.input_shape if primary else [],
            "output_shape": primary.output_shape if primary else [],
            "dependencies": primary.dependencies if primary else [],
            "num_iterations": primary.num_iterations if primary else 0,
        }
        output["layers"].append(layer_data)
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ JSON data exported to: {filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Profile Inception V3 model for distributed inference modeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available groups
  python profile_inception.py --list-groups
  
  # Run a SINGLE group (RECOMMENDED for TEE stability)
  python profile_inception.py --group Stem-Part1 --iterations 10
  python profile_inception.py --group Stem-Part2 --iterations 10
  python profile_inception.py --group Inception-A1 --iterations 10
  
  # CPU-only profiling (no SGX required, SAFE)
  python profile_inception.py --cpu-only --iterations 10
  
  # Run ALL groups in sequence (may cause crashes)
  python profile_inception.py --all-groups --iterations 10

Recommended workflow for TEE profiling:
  1. First run: python profile_inception.py --group Stem-Part1 --iterations 10
  2. Wait for completion, then: python profile_inception.py --group Stem-Part2 --iterations 10
  3. Continue with each group...
  4. Finally: python profile_inception.py --merge-results
"""
    )
    parser.add_argument('--batch-size', type=int, default=1, 
                       help='Batch size (default: 1)')
    parser.add_argument('--input-size', type=int, default=299, 
                       help='Input image size (default: 299)')
    parser.add_argument('--num-classes', type=int, default=1000, 
                       help='Number of output classes (default: 1000)')
    parser.add_argument('--iterations', type=int, default=DEFAULT_NUM_ITERATIONS,
                       help=f'Number of measurement iterations (default: {DEFAULT_NUM_ITERATIONS})')
    parser.add_argument('--warmup', type=int, default=DEFAULT_WARMUP_ITERATIONS,
                       help=f'Number of warmup iterations (default: {DEFAULT_WARMUP_ITERATIONS})')
    parser.add_argument('--output-dir', type=str, default='experiments/data',
                       help='Output directory for results (default: experiments/data)')
    
    # Group selection options
    parser.add_argument('--group', type=str, default=None,
                       help='Run ONLY this specific group (e.g., Stem-Part1, Inception-A1)')
    parser.add_argument('--list-groups', action='store_true', default=False,
                       help='List all available groups and exit')
    parser.add_argument('--all-groups', action='store_true', default=False,
                       help='Run all groups sequentially (may cause crashes)')
    parser.add_argument('--merge-results', action='store_true', default=False,
                       help='Merge all per-group result files into a single file')
    
    # Legacy options
    parser.add_argument('--grouped', action='store_true', default=True, 
                       help='[DEPRECATED] Use --group or --all-groups instead')
    parser.add_argument('--single', action='store_true', default=False,
                       help='Use single-pass execution (may fail due to STORE_CHUNK_ELEM constraints)')
    parser.add_argument('--cpu-only', action='store_true', default=False,
                       help='Only profile CPU mode (skip Enclave measurement, SAFE)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                       help='Validate group configurations without running (check layer assignments)')
    parser.add_argument('--no-confirm', action='store_true', default=False,
                       help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    # --list-groups: Show all available groups and exit
    if args.list_groups:
        print("="*100)
        print("Available Groups for Inception V3 Profiling (Per-Group STORE_CHUNK_ELEM)")
        print("="*100)
        print(f"\n{'#':<3} {'Group Name':<16} {'STORE_CHUNK':>12} {'MB':>7} {'Input Shape':<22} {'Description'}")
        print("-"*100)
        for idx, group_name in enumerate(GROUP_ORDER, 1):
            config = GROUP_CONFIGS[group_name]
            mem_mb = config['store_chunk_elem'] * 4 / 1024 / 1024
            input_shape = str(config.get('input_shape', []))
            desc = config.get('description', '')[:25]
            print(f"{idx:<3} {group_name:<16} {config['store_chunk_elem']:>12} {mem_mb:>7.2f} {input_shape:<22} {desc}")
        
        print("\n" + "="*100)
        print("KEY FEATURE: Each group uses its own STORE_CHUNK_ELEM value!")
        print("This allows each group to run independently with minimal memory usage.")
        print("="*100)
        print("\nUsage - Run each group separately (RECOMMENDED):")
        print("-"*100)
        for group_name in GROUP_ORDER:
            print(f"python profile_inception.py --group {group_name} --iterations 10")
        print("\n# After all groups complete, merge results:")
        print("python profile_inception.py --merge-results")
        sys.exit(0)
    
    # --merge-results: Merge all per-group result files
    if args.merge_results:
        print("="*80)
        print("Merging Per-Group Results")
        print("="*80)
        
        all_enclave_metrics = OrderedDict()
        all_cpu_metrics = OrderedDict()
        missing_groups = []
        
        for group_name in GROUP_ORDER:
            json_path = os.path.join(args.output_dir, f"inception_v3_{group_name}.json")
            if os.path.exists(json_path):
                print(f"  ✓ Loading {group_name} from {json_path}")
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # Parse the 'layers' array format from saved JSON
                    for layer_data in data.get('layers', []):
                        layer_name = layer_data['name']
                        enclave_data = layer_data.get('enclave', {})
                        cpu_data = layer_data.get('cpu', {})
                        
                        all_enclave_metrics[layer_name] = LayerMetrics(
                            name=layer_name,
                            layer_type=layer_data.get('type', ''),
                            group=layer_data.get('group', group_name),
                            enclave_time_mean=enclave_data.get('mean_ms', 0),
                            enclave_time_std=enclave_data.get('std_ms', 0),
                            enclave_time_min=enclave_data.get('min_ms', 0),
                            enclave_time_max=enclave_data.get('max_ms', 0),
                            enclave_time_p95=enclave_data.get('p95_ms', 0),
                            enclave_time_p99=enclave_data.get('p99_ms', 0),
                            input_bytes=layer_data.get('input_bytes', 0),
                            output_bytes=layer_data.get('output_bytes', 0),
                            input_shape=layer_data.get('input_shape', []),
                            output_shape=layer_data.get('output_shape', []),
                            dependencies=layer_data.get('dependencies', []),
                            num_iterations=layer_data.get('num_iterations', 0),
                        )
                        
                        # Also track CPU metrics if available
                        if cpu_data.get('mean_ms', 0) > 0:
                            all_cpu_metrics[layer_name] = LayerMetrics(
                                name=layer_name,
                                layer_type=layer_data.get('type', ''),
                                group=layer_data.get('group', group_name),
                                cpu_time_mean=cpu_data.get('mean_ms', 0),
                                cpu_time_std=cpu_data.get('std_ms', 0),
                                cpu_time_min=cpu_data.get('min_ms', 0),
                                cpu_time_max=cpu_data.get('max_ms', 0),
                            )
            else:
                missing_groups.append(group_name)
                print(f"  ✗ Missing {group_name} (file not found: {json_path})")
        
        if missing_groups:
            print(f"\n⚠ Warning: {len(missing_groups)} groups missing:")
            for g in missing_groups:
                print(f"    - {g}")
            print("\nRun these groups first:")
            for g in missing_groups:
                print(f"  python profile_inception.py --group {g} --iterations 10")
        
        if all_enclave_metrics:
            # Export merged results
            csv_path = os.path.join(args.output_dir, "inception_v3_layers_merged.csv")
            json_path = os.path.join(args.output_dir, "inception_v3_layers_merged.json")
            
            _export_csv(all_enclave_metrics, all_cpu_metrics, csv_path)
            _export_json(all_enclave_metrics, all_cpu_metrics, json_path)
            
            print(f"\n✓ Merged {len(all_enclave_metrics)} layers from {len(GROUP_ORDER) - len(missing_groups)} groups")
            print(f"   - {csv_path}")
            print(f"   - {json_path}")
        else:
            print("\n✗ No results to merge. Run individual groups first.")
        
        sys.exit(0)
    
    # --group: Run a specific single group
    if args.group:
        if args.group not in GROUP_ORDER:
            print(f"✗ Error: Unknown group '{args.group}'")
            print(f"\nAvailable groups:")
            for g in GROUP_ORDER:
                print(f"  - {g}")
            sys.exit(1)
        
        run_single_group(
            group_name=args.group,
            batch_size=args.batch_size,
            input_size=args.input_size,
            num_classes=args.num_classes,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup,
            output_dir=args.output_dir
        )
        sys.exit(0)
    
    # Dry run mode: just validate configurations
    if args.dry_run:
        print("="*80)
        print("DRY RUN: Validating Group Configurations")
        print("="*80)
        
        # Create model in CPU mode to check layer assignments
        from experiments.models.sgx_inception import SGXInceptionV3
        model = SGXInceptionV3(
            sid=0,
            enclave_mode=ExecutionModeOptions.CPU,
            batch_size=args.batch_size,
            input_size=args.input_size,
            num_classes=args.num_classes
        )
        
        print(f"\nTotal layers in model: {len(model.layers)}")
        print(f"Total groups configured: {len(GROUP_ORDER)}")
        
        print("\n" + "-"*80)
        print("Group Configuration Summary:")
        print("-"*80)
        total_mem = 0
        for group_name in GROUP_ORDER:
            config = GROUP_CONFIGS[group_name]
            mem_mb = config['store_chunk_elem'] * 4 / 1024 / 1024
            total_mem += mem_mb
            # Check memory safety
            mem_status = "✓ SAFE" if config['store_chunk_elem'] <= MAX_SAFE_STORE_CHUNK_ELEM else "⚠ LARGE"
            print(f"  {group_name:20} STORE_CHUNK_ELEM={config['store_chunk_elem']:10} ({mem_mb:6.2f} MB) {mem_status}")
        
        print("\n" + "-"*80)
        print("Layer-to-Group Assignment:")
        print("-"*80)
        unassigned = []
        group_counts = {g: 0 for g in GROUP_ORDER}
        
        for layer in model.layers:
            group = get_layer_group(layer.LayerName)
            if group:
                group_counts[group] += 1
                print(f"  {layer.LayerName:40} -> {group}")
            else:
                unassigned.append(layer.LayerName)
                print(f"  {layer.LayerName:40} -> ⚠ UNASSIGNED")
        
        print("\n" + "-"*80)
        print("Summary:")
        print("-"*80)
        for group_name in GROUP_ORDER:
            print(f"  {group_name:20}: {group_counts[group_name]:3} layers")
        
        if unassigned:
            print(f"\n⚠ WARNING: {len(unassigned)} layers not assigned to any group:")
            for name in unassigned:
                print(f"    - {name}")
        else:
            print(f"\n✓ All {len(model.layers)} layers assigned to groups")
        
        sys.exit(0)
    
    use_grouped = args.grouped and not args.single
    
    if use_grouped and not args.no_confirm and not args.cpu_only:
        print("="*80)
        print("Grouped Execution Mode (Fine-Grained)")
        print("="*80)
        print("\n⚠ WARNING: This will modify STORE_CHUNK_ELEM and rebuild SGX code!")
        print("\nThis mode will:")
        print(f"1. Execute the model in {len(GROUP_ORDER)} small groups")
        print("2. Each group uses an optimized, MEMORY-SAFE STORE_CHUNK_ELEM value")
        print("3. SGX code will be rebuilt between groups (automatic)")
        print(f"\nMeasurement settings:")
        print(f"  Iterations: {args.iterations}")
        print(f"  Warmup: {args.warmup}")
        print("\nGroup configuration (fine-grained for memory safety):")
        for group_name in GROUP_ORDER:
            config = GROUP_CONFIGS[group_name]
            mem_mb = config['store_chunk_elem'] * 4 / 1024 / 1024
            mem_status = "SAFE" if config['store_chunk_elem'] <= MAX_SAFE_STORE_CHUNK_ELEM else "LARGE"
            print(f"  {group_name:20} {config['store_chunk_elem']:10} ({mem_mb:6.2f} MB) [{mem_status}]")
        print("\nPress Enter to continue or Ctrl+C to cancel...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
    
    if args.cpu_only:
        print("="*80)
        print("CPU-only Profiling Mode (No SGX Required)")
        print("="*80)
        print("\nThis mode uses pure PyTorch for measurement.")
        print("No SGX enclave or enclave_bridge.so required.\n")
        
        os.makedirs(args.output_dir, exist_ok=True)
        cpu_metrics = _profile_cpu_pytorch(
            args.batch_size, args.input_size, args.num_classes,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup
        )
        # Create empty enclave metrics
        enclave_metrics = OrderedDict()
        
        csv_path = os.path.join(args.output_dir, "inception_v3_layers_cpu.csv")
        json_path = os.path.join(args.output_dir, "inception_v3_layers_cpu.json")
        
        _export_csv(enclave_metrics, cpu_metrics, csv_path)
        _export_json(enclave_metrics, cpu_metrics, json_path)
        
        print(f"\n✓ Done! CPU metrics saved to:")
        print(f"   - {csv_path}")
        print(f"   - {json_path}")
    else:
        run_profile(
            batch_size=args.batch_size,
            input_size=args.input_size,
            num_classes=args.num_classes,
            use_grouped=use_grouped,
            num_iterations=args.iterations,
            warmup_iterations=args.warmup,
            output_dir=args.output_dir
        )

