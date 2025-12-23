"""
Inception V3 Performance Profiler for Distributed Inference Modeling.

This script measures:
1. Execution time of each layer in Enclave mode.
2. Execution time of each layer in CPU mode.
3. Output tensor size of each layer (for communication cost modeling).

Output: inception_metrics.csv
Format: LayerName, Type, EnclaveTime(ms), CPUTime(ms), OutputBytes

Grouped Execution Mode:
Due to STORE_CHUNK_ELEM constraints, the model is executed in groups,
each group using a different STORE_CHUNK_ELEM value optimized for its layers.
"""

import sys
import time
import csv
import numpy as np
import torch
import os
import subprocess
import re
import shutil
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, '.')

from experiments.models.sgx_inception import SGXInceptionV3
from python.enclave_interfaces import GlobalTensor
from python.sgx_net import SecretNeuralNetwork
from python.utils.basic_utils import ExecutionModeOptions


def _ensure_torch():
    return torch

# Group configurations: each group has optimized STORE_CHUNK_ELEM value
GROUP_CONFIGS = {
    'Stem': {
        'store_chunk_elem': 130560500,
        'description': 'Input + Stem (299x299 -> 35x35x192)',
        'layer_prefixes': ['input', 'stem_'],
    },
    'Inception-A': {
        'store_chunk_elem': 940800,
        'description': '3x Inception-A modules (35x35, 192->256)',
        'layer_prefixes': ['inception_a'],
    },
    'Reduction-A': {
        'store_chunk_elem': 134175475,
        'description': 'Reduction-A (35x35 -> 17x17, 256->768)',
        'layer_prefixes': ['reduction_a'],
    },
    'Inception-B': {
        'store_chunk_elem': 221952,
        'description': '4x Inception-B modules (17x17, 768)',
        'layer_prefixes': ['inception_b'],
    },
    'Reduction-B': {
        'store_chunk_elem': 1109760,
        'description': 'Reduction-B (17x17 -> 8x8, 768->1280)',
        'layer_prefixes': ['reduction_b'],
    },
    'Inception-C': {
        'store_chunk_elem': 30720,
        'description': '2x Inception-C modules (8x8, 1280->2048)',
        'layer_prefixes': ['inception_c'],
    },
    'Classifier': {
        'store_chunk_elem': 256000,
        'description': 'AvgPool + Flatten + FC + Output',
        'layer_prefixes': ['avgpool', 'flatten', 'fc', 'output'],
    },
}

# Order of groups for execution
GROUP_ORDER = ['Stem', 'Inception-A', 'Reduction-A', 'Inception-B', 'Reduction-B', 'Inception-C', 'Classifier']

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
    """Determine which group a layer belongs to based on its name."""
    for group_name, config in GROUP_CONFIGS.items():
        for prefix in config['layer_prefixes']:
            if layer_name.startswith(prefix):
                return group_name
    return None


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


def run_profile(batch_size=1, input_size=299, num_classes=1000, use_grouped=True):
    """
    Run Inception V3 profiling.
    
    Args:
        batch_size: Batch size
        input_size: Input image size
        num_classes: Number of output classes
        use_grouped: If True, use grouped execution with different STORE_CHUNK_ELEM values
    """
    if use_grouped:
        print("Starting Inception V3 Profiling (Grouped Mode)...")
        print("This will execute the model in groups, each with optimized STORE_CHUNK_ELEM.\n")
        run_profile_grouped(batch_size, input_size, num_classes)
    else:
        print("Starting Inception V3 Profiling (Single Mode)...")
        print("Warning: This may fail due to STORE_CHUNK_ELEM constraints.\n")
        
        # Original single-pass profiling
        enclave_metrics = _profile_pass(batch_size, input_size, num_classes, ExecutionModeOptions.Enclave)
        cpu_metrics = _profile_pass(batch_size, input_size, num_classes, ExecutionModeOptions.CPU)
        _export_csv(enclave_metrics, cpu_metrics, "inception_metrics.csv")
        print("Done! Metrics saved to inception_metrics.csv")


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


def run_profile_grouped(batch_size=1, input_size=299, num_classes=1000):
    """
    Run profiling in grouped mode: execute model in groups with different STORE_CHUNK_ELEM values.
    """
    all_enclave_metrics = OrderedDict()
    all_cpu_metrics = OrderedDict()
    
    # Track if initial build was performed and what STORE_CHUNK_ELEM was used
    initial_build_performed = False
    initial_store_chunk_elem = None
    
    # Ensure initial build exists before validation
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    enclave_bridge_path = os.path.join(project_root, "App", "bin", "enclave_bridge.so")
    
    if not os.path.exists(enclave_bridge_path):
        if not ensure_initial_build():
            print("\n✗ Cannot proceed without initial build. Exiting.")
            return
        initial_build_performed = True
        initial_store_chunk_elem = GROUP_CONFIGS[GROUP_ORDER[0]]['store_chunk_elem']
    
    # Phase 1: Profile Enclave mode for each group
    print("\n" + "="*80)
    print("Phase 1: Profiling Enclave Execution (Grouped)")
    print("="*80)
    
    for group_idx, group_name in enumerate(GROUP_ORDER, 1):
        config = GROUP_CONFIGS[group_name]
        print(f"\n--- Group {group_idx}/{len(GROUP_ORDER)}: {group_name} ---")
        print(f"Description: {config['description']}")
        print(f"STORE_CHUNK_ELEM: {config['store_chunk_elem']} ({config['store_chunk_elem'] * 4 / 1024 / 1024:.2f} MB)")
        
        # Validate STORE_CHUNK_ELEM before updating
        print("   Validating STORE_CHUNK_ELEM constraints...")
        is_valid, errors, suggested_value = validate_store_chunk_elem_for_group(
            group_name, config['store_chunk_elem'], batch_size, input_size, num_classes
        )
        
        if not is_valid:
            print(f"✗ STORE_CHUNK_ELEM validation failed for {group_name}:")
            for error in errors:
                print(f"     - {error}")
            if suggested_value:
                print(f"   💡 Suggested STORE_CHUNK_ELEM: {suggested_value} ({suggested_value * 4 / 1024 / 1024:.2f} MB)")
            print(f"\n   ⚠ Skipping group {group_name} due to constraint violations.")
            print("   Please update GROUP_CONFIGS with a valid STORE_CHUNK_ELEM value.")
            continue
        
        print("   ✓ STORE_CHUNK_ELEM validation passed")
        
        # Update STORE_CHUNK_ELEM in Include/common_with_enclaves.h
        if not update_store_chunk_elem(config['store_chunk_elem']):
            print(f"✗ Failed to update STORE_CHUNK_ELEM for {group_name}. Skipping...")
            continue
        
        # Update STORE_CHUNK_ELEM in python/layers/maxpool2d.py
        if not update_maxpool2d_store_chunk_elem(config['store_chunk_elem']):
            print(f"⚠ Warning: Failed to update STORE_CHUNK_ELEM in maxpool2d.py for {group_name}")
            print("   Continuing anyway...")
        
        # Check if rebuild is needed
        # Skip rebuild if this is the first group and we just did initial build with same value
        skip_rebuild = False
        if initial_build_performed and group_idx == 1 and config['store_chunk_elem'] == initial_store_chunk_elem:
            print("\n   ℹ Skipping rebuild: Initial build already used this STORE_CHUNK_ELEM value")
            skip_rebuild = True
        
        # Rebuild (try automatic, fallback to manual)
        if not skip_rebuild:
            print("\n⚠ IMPORTANT: Rebuilding SGX code after STORE_CHUNK_ELEM change...")
            if rebuild_sgx_code():
                print("   ✓ Automatic rebuild successful")
            else:
                print("\n   ⚠ Automatic rebuild failed. Please rebuild manually:")
                print("      cd /root/exp_DNN_SGX/TAOISM && rm -rf SGXDNN/bin_sgx && make clean && make SGX_MODE=HW all")
                user_input = input("   Press Enter after rebuilding, or 's' to skip this group: ").strip().lower()
                if user_input == 's':
                    print("   Skipping this group...")
                    continue
            
            # IMPORTANT: Destroy old GlobalTensor and reinitialize after rebuild
            # The old Enclave is invalid after recompilation, so we must create a new one
            if GlobalTensor.is_init_global_tensor:
                print("   Destroying old GlobalTensor (Enclave invalid after recompilation)...")
                GlobalTensor.destroy()
        else:
            # Even if we skip rebuild, we still need to ensure GlobalTensor is initialized
            # (it might not be initialized yet if initial build just completed)
            if not GlobalTensor.is_init_global_tensor:
                print("   Initializing GlobalTensor...")
                GlobalTensor.init()
        
        # Profile this group
        try:
            group_metrics = _profile_group(
                batch_size, input_size, num_classes,
                ExecutionModeOptions.Enclave,
                group_name
            )
            all_enclave_metrics.update(group_metrics)
            print(f"✓ Completed {group_name}: {len(group_metrics)} layers")
        except Exception as e:
            print(f"✗ Error profiling {group_name}: {e}")
            print("   Continuing with next group...")
            # Clean up GlobalTensor on error to ensure clean state for next group
            if GlobalTensor.is_init_global_tensor:
                try:
                    GlobalTensor.destroy()
                except:
                    pass
    
    # Phase 2: Profile CPU mode (can use single pass as CPU doesn't have chunk constraints)
    print("\n" + "="*80)
    print("Phase 2: Profiling CPU Execution")
    print("="*80)
    print("\nCPU mode doesn't have STORE_CHUNK_ELEM constraints, using single pass...")
    try:
        all_cpu_metrics = _profile_pass(batch_size, input_size, num_classes, ExecutionModeOptions.CPU)
        print(f"✓ Completed CPU profiling: {len(all_cpu_metrics)} layers")
    except Exception as e:
        print(f"✗ Error in CPU profiling: {e}")
    
    # Phase 3: Merge and Export
    print("\n" + "="*80)
    print("Phase 3: Exporting Data")
    print("="*80)
    
    # Clean up GlobalTensor
    if GlobalTensor.is_init_global_tensor:
        GlobalTensor.destroy()
    
    _export_csv(all_enclave_metrics, all_cpu_metrics, "inception_metrics.csv")
    print(f"\n✓ Done! Metrics saved to inception_metrics.csv")
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


def _profile_group(batch_size, input_size, num_classes, mode, group_name):
    """
    Profile a specific group of layers.
    
    Args:
        batch_size: Batch size
        input_size: Input image size
        num_classes: Number of output classes
        mode: Execution mode (Enclave or CPU)
        group_name: Name of the group to profile
    
    Returns:
        Dictionary mapping layer names to metrics
    """
    config = GROUP_CONFIGS[group_name]
    torch.manual_seed(0)
    input_tensor = torch.randn(batch_size, 3, input_size, input_size)
    
    overrides = {"input": ExecutionModeOptions.CPU}
    
    # Initialize GlobalTensor if needed
    # Note: This should be called after rebuild, so GlobalTensor should be fresh
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
        
        metrics = OrderedDict()
        
        # Warmup: execute this group multiple times
        print("   Warming up...")
        for _ in range(3):
            # Re-execute dependencies
            if group_layer_indices[0] > 0:
                for idx in range(group_layer_indices[0]):
                    model.layers[idx].forward()
            # Execute group layers
            for layer in group_layers:
                layer.forward()
        
        # Measurement
        print(f"   Measuring layers ({mode.name})...")
        for layer in group_layers:
            # Re-execute dependencies before each measurement to ensure consistency
            if group_layer_indices[0] > 0:
                for idx in range(group_layer_indices[0]):
                    model.layers[idx].forward()
            
            start = time.time()
            layer.forward()
            end = time.time()
            duration_ms = (end - start) * 1000
            
            if hasattr(layer, "get_output_shape"):
                shape = layer.get_output_shape()
                num_elements = np.prod(shape)
                size_bytes = num_elements * 4
            else:
                size_bytes = 0
            
            metrics[layer.LayerName] = {
                "type": type(layer).__name__,
                "time_ms": duration_ms,
                "size_bytes": size_bytes
            }
        
        return metrics
        
    finally:
        # Clean up: destroy GlobalTensor after all groups are done
        # This will be handled by the caller
        pass


def _profile_pass(batch_size, input_size, num_classes, mode):
    """Run a single pass (either Enclave or CPU) and collect metrics."""
    torch.manual_seed(0)
    input_tensor = torch.randn(batch_size, 3, input_size, input_size)
    
    # Special handling for Enclave mode: input must be CPU
    # But for profiling, we want to measure compute.
    # So we set everything to 'mode', except input which is always CPU.
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
        
        metrics = OrderedDict()
        
        # Warmup
        print("Warming up...")
        for _ in range(3):
            for layer in model.layers:
                layer.forward()
                
        # Measurement
        print(f"Measuring layers ({mode.name})...")
        for layer in model.layers:
            # Measure time
            start = time.time()
            layer.forward()
            if mode == ExecutionModeOptions.Enclave:
                # For Enclave, we need to ensure execution is finished if async
                # But current implementation is blocking for forward.
                # However, we might need to fetch data to CPU to stop the clock accurately?
                # No, layer.forward() includes the call return.
                pass
            end = time.time()
            duration_ms = (end - start) * 1000
            
            # Measure output size
            # We need to peek at the shape.
            # shape is usually available in layer.OutputShape or we can check the tensor.
            # Since we are profiling, let's get the shape from metadata.
            if hasattr(layer, "get_output_shape"):
                shape = layer.get_output_shape()
                # Size in bytes = num_elements * 4 (float32)
                # shape is list. product(shape) * 4
                num_elements = np.prod(shape)
                size_bytes = num_elements * 4
            else:
                size_bytes = 0
                
            metrics[layer.LayerName] = {
                "type": type(layer).__name__,
                "time_ms": duration_ms,
                "size_bytes": size_bytes
            }
            # print(f"  {layer.LayerName}: {duration_ms:.3f} ms")
            
        return metrics
        
    finally:
        if GlobalTensor.is_init_global_tensor:
            GlobalTensor.destroy()


def _export_csv(enclave_data, cpu_data, filename):
    headers = ["LayerName", "Type", "EnclaveTime(ms)", "CPUTime(ms)", "OutputBytes"]
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for name, e_info in enclave_data.items():
            c_info = cpu_data.get(name, {"time_ms": 0})
            
            row = [
                name,
                e_info["type"],
                f"{e_info['time_ms']:.4f}",
                f"{c_info['time_ms']:.4f}",
                e_info["size_bytes"]
            ]
            writer.writerow(row)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Profile Inception V3 model')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--input-size', type=int, default=299, help='Input image size')
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of output classes')
    parser.add_argument('--grouped', action='store_true', default=True, 
                       help='Use grouped execution with different STORE_CHUNK_ELEM (default: True)')
    parser.add_argument('--single', action='store_true', default=False,
                       help='Use single-pass execution (may fail due to STORE_CHUNK_ELEM constraints)')
    
    args = parser.parse_args()
    
    use_grouped = args.grouped and not args.single
    
    if use_grouped:
        print("="*80)
        print("Grouped Execution Mode")
        print("="*80)
        print("\nThis mode will:")
        print("1. Execute the model in 7 groups")
        print("2. Each group uses an optimized STORE_CHUNK_ELEM value")
        print("3. You will need to rebuild SGX code between groups")
        print("\nGroup configuration:")
        for group_name in GROUP_ORDER:
            config = GROUP_CONFIGS[group_name]
            print(f"  {group_name:15} STORE_CHUNK_ELEM={config['store_chunk_elem']:12} ({config['store_chunk_elem'] * 4 / 1024 / 1024:6.2f} MB)")
        print("\nPress Enter to continue or Ctrl+C to cancel...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
    
    run_profile(
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_classes=args.num_classes,
        use_grouped=use_grouped
    )

