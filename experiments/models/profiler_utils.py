"""
Profiler Utilities for TEE/SGX Layer Memory Analysis.

This module provides common utilities for profiling neural network layers
in both CPU and TEE (Trusted Execution Environment) modes.

Key Components:
1. Memory analysis constants (SGX encryption overhead)
2. LayerMetrics dataclass with memory fields
3. Memory calculation functions for different layer types

Supported Layer Types:
- Conv2d (SGXConvBase)
- Linear (SGXLinearBase)
- LayerNorm (SecretLayerNormLayer) - Transformer specific
- BatchNorm (SecretBatchNorm2dLayer)
- Softmax, GELU, ReLU, MatMul, Add, Pool - No weights

Reference: profile_inception.py for original implementation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


# =============================================================================
# Memory Analysis Constants (from Include/crypto_common.h)
# =============================================================================

# SGX AES-GCM encryption metadata sizes
SGX_AESGCM_IV_SIZE = 12       # IV (Initialization Vector): 12 bytes
SGX_AESGCM_MAC_SIZE = 16      # TAG (Message Authentication Code): 16 bytes
SGX_AES_GCM_STRUCT_SIZE = 32  # Approximate size of sgx_aes_gcm_data_t struct

# Per-chunk encryption overhead: IV + TAG + struct metadata
CHUNK_ENCRYPTION_OVERHEAD = SGX_AESGCM_IV_SIZE + SGX_AESGCM_MAC_SIZE + SGX_AES_GCM_STRUCT_SIZE  # ~60 bytes

# Default STORE_CHUNK_ELEM (from Include/common_with_enclaves.h)
DEFAULT_STORE_CHUNK_ELEM = 4276896  # ~16.3MB per chunk in float32

# Thread pool size affects ChunkPool pre-allocation
THREAD_POOL_SIZE = 4


# =============================================================================
# LayerMetrics Dataclass
# =============================================================================

@dataclass
class LayerMetrics:
    """
    Data class to store layer profiling metrics including memory analysis.
    
    Compatible with both CPU and Enclave profiling modes.
    """
    name: str
    layer_type: str
    group: str
    execution_mode: str = 'CPU'  # 'CPU' or 'Enclave'
    
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
    
    # Enclave runtime breakdown (ms) per iteration
    enclave_get_ms: List[float] = field(default_factory=list)
    enclave_get2_ms: List[float] = field(default_factory=list)
    enclave_compute_ms: List[float] = field(default_factory=list)
    enclave_store_ms: List[float] = field(default_factory=list)
    
    # Number of iterations used
    num_iterations: int = 0
    
    # ===== Memory Analysis Fields (TEE vs CPU) =====
    # CPU Mode Memory (theoretical, in bytes)
    cpu_memory_bytes: int = 0           # Total CPU memory: input + output + weight + bias
    
    # TEE Mode Memory (theoretical, in bytes)
    tee_memory_bytes: int = 0           # Total TEE tensor memory (same as CPU tensor data)
    tee_encryption_overhead: int = 0    # AES-GCM encryption metadata overhead
    tee_total_memory_bytes: int = 0     # TEE total = tensor + encryption overhead
    
    # Memory breakdown (in bytes)
    weight_bytes: int = 0               # Weight tensor size
    bias_bytes: int = 0                 # Bias tensor size
    activation_bytes: int = 0           # Input + Output tensor size
    
    # TEE-specific metadata
    num_chunks: int = 0                 # Number of STORE_CHUNK for this layer
    chunk_metadata_bytes: int = 0       # Per-chunk encryption metadata (IV + TAG)

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
            'execution_mode': self.execution_mode,
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
            # Memory analysis fields
            'cpu_memory_bytes': self.cpu_memory_bytes,
            'tee_memory_bytes': self.tee_memory_bytes,
            'tee_encryption_overhead': self.tee_encryption_overhead,
            'tee_total_memory_bytes': self.tee_total_memory_bytes,
            'weight_bytes': self.weight_bytes,
            'bias_bytes': self.bias_bytes,
            'activation_bytes': self.activation_bytes,
            'num_chunks': self.num_chunks,
            'chunk_metadata_bytes': self.chunk_metadata_bytes,
        }


# =============================================================================
# Memory Calculation Functions
# =============================================================================

def shape_to_bytes(shape: List[int], bytes_per_elem: int = 4) -> int:
    """
    Convert tensor shape to size in bytes.
    
    Args:
        shape: Tensor shape, e.g., [batch, channels, height, width]
        bytes_per_elem: Bytes per element (default 4 for float32)
    
    Returns:
        Size in bytes
    """
    if not shape:
        return 0
    return int(np.prod(shape)) * bytes_per_elem


def estimate_weight_bias_from_type(
    layer_type: str,
    input_shape: List[int],
    output_shape: List[int],
    kernel_size: int = 3
) -> Tuple[int, int]:
    """
    Estimate weight and bias sizes (in bytes) from layer type and shapes.
    
    Supported layer types:
    - Conv2d (SGXConvBase): weight = [out_ch, in_ch, kH, kW], bias = [out_ch]
    - Linear (SGXLinearBase): weight = [out_features, in_features], bias = [out_features]
    - LayerNorm: weight = [normalized_dim], bias = [normalized_dim]
    - BatchNorm: weight = [num_features]*2, bias = [num_features]*2
    - Softmax, GELU, ReLU, MatMul, Add, Pool: No weights (returns 0, 0)
    
    Args:
        layer_type: Layer type string (e.g., 'Linear', 'Conv2d', 'LayerNorm')
        input_shape: Input tensor shape
        output_shape: Output tensor shape
        kernel_size: Kernel size for Conv layers (default 3)
    
    Returns:
        Tuple of (weight_bytes, bias_bytes)
    """
    bytes_per_elem = 4  # float32
    weight_bytes = 0
    bias_bytes = 0
    
    if not input_shape or not output_shape:
        return 0, 0
    
    # Normalize layer_type for matching
    layer_type_lower = layer_type.lower()
    
    # Conv layers: [out_ch, in_ch, kH, kW], [out_ch]
    if 'conv' in layer_type_lower or 'sgxconv' in layer_type_lower:
        # Conv: [B, C_in, H, W] -> [B, C_out, H', W']
        if len(input_shape) >= 4 and len(output_shape) >= 4:
            in_channels = input_shape[1]
            out_channels = output_shape[1]
            weight_bytes = out_channels * in_channels * kernel_size * kernel_size * bytes_per_elem
            bias_bytes = out_channels * bytes_per_elem
    
    # Linear layers: [out_features, in_features], [out_features]
    elif 'linear' in layer_type_lower or 'sgxlinear' in layer_type_lower:
        # Linear: [..., in_features] -> [..., out_features]
        in_features = input_shape[-1]
        out_features = output_shape[-1]
        weight_bytes = out_features * in_features * bytes_per_elem
        bias_bytes = out_features * bytes_per_elem
    
    # LayerNorm: [normalized_dim], [normalized_dim]
    elif 'layernorm' in layer_type_lower or 'layer_norm' in layer_type_lower:
        # LayerNorm normalizes over the last dimension(s)
        normalized_dim = input_shape[-1]
        weight_bytes = normalized_dim * bytes_per_elem  # gamma
        bias_bytes = normalized_dim * bytes_per_elem    # beta
    
    # BatchNorm: [num_features]*2, [num_features]*2
    # (gamma, running_mean) and (beta, running_var)
    elif 'batchnorm' in layer_type_lower or 'batch_norm' in layer_type_lower:
        if len(input_shape) >= 2:
            num_features = input_shape[1]
            # gamma + running_mean, beta + running_var
            weight_bytes = num_features * 2 * bytes_per_elem
            bias_bytes = num_features * 2 * bytes_per_elem
    
    # No-weight layers: Softmax, GELU, ReLU, MatMul, Add, Pool, Concat, Flatten
    # Return 0, 0 by default
    
    return weight_bytes, bias_bytes


def calc_layer_memory_from_shapes(
    layer_type: str,
    input_shape: List[int],
    output_shape: List[int],
    weight_shape: Optional[List[int]] = None,
    bias_shape: Optional[List[int]] = None,
    kernel_size: int = 3,
    store_chunk_elem: int = DEFAULT_STORE_CHUNK_ELEM
) -> Dict[str, int]:
    """
    Calculate memory footprint for a layer in both CPU and TEE modes.
    
    This function calculates:
    - CPU memory: input + output + weight + bias tensors
    - TEE memory: Same tensor data + encryption overhead per chunk
    
    Args:
        layer_type: Layer type string
        input_shape: Input tensor shape
        output_shape: Output tensor shape
        weight_shape: Optional explicit weight shape
        bias_shape: Optional explicit bias shape
        kernel_size: Kernel size for Conv layers (default 3)
        store_chunk_elem: STORE_CHUNK_ELEM value for chunk calculation
    
    Returns:
        Dictionary with memory breakdown:
        - cpu_memory_bytes: Total CPU memory
        - tee_memory_bytes: TEE tensor memory (same as CPU)
        - tee_encryption_overhead: Encryption metadata overhead
        - tee_total_memory_bytes: Total TEE memory
        - weight_bytes: Weight tensor size
        - bias_bytes: Bias tensor size
        - activation_bytes: Input + Output tensor size
        - num_chunks: Number of chunks for this layer
        - chunk_metadata_bytes: Per-chunk encryption metadata
    """
    bytes_per_elem = 4  # float32
    
    # Calculate activation (input + output) memory
    input_bytes = shape_to_bytes(input_shape)
    output_bytes = shape_to_bytes(output_shape)
    activation_bytes = input_bytes + output_bytes
    
    # Calculate weight and bias memory
    if weight_shape is not None:
        weight_bytes = shape_to_bytes(weight_shape)
    else:
        weight_bytes = 0
    
    if bias_shape is not None:
        bias_bytes = shape_to_bytes(bias_shape)
    else:
        bias_bytes = 0
    
    # If weight/bias not explicitly provided, estimate from layer type
    if weight_bytes == 0 and bias_bytes == 0:
        weight_bytes, bias_bytes = estimate_weight_bias_from_type(
            layer_type, input_shape, output_shape, kernel_size
        )
    
    # Total tensor memory (same for CPU and TEE tensor data)
    cpu_memory_bytes = activation_bytes + weight_bytes + bias_bytes
    tee_memory_bytes = cpu_memory_bytes
    
    # Calculate TEE encryption overhead
    # Each chunk of data gets encrypted with IV + TAG metadata
    total_elements = 0
    if input_shape:
        total_elements += int(np.prod(input_shape))
    if output_shape:
        total_elements += int(np.prod(output_shape))
    
    # Add weight and bias elements
    if weight_shape:
        total_elements += int(np.prod(weight_shape))
    elif weight_bytes > 0:
        total_elements += weight_bytes // bytes_per_elem
    
    if bias_shape:
        total_elements += int(np.prod(bias_shape))
    elif bias_bytes > 0:
        total_elements += bias_bytes // bytes_per_elem
    
    # Number of chunks = ceil(total_elements / store_chunk_elem)
    num_chunks = max(1, int(np.ceil(total_elements / store_chunk_elem))) if total_elements > 0 else 0
    
    # Per-chunk metadata overhead
    chunk_metadata_bytes = CHUNK_ENCRYPTION_OVERHEAD
    tee_encryption_overhead = num_chunks * chunk_metadata_bytes
    
    # Total TEE memory
    tee_total_memory_bytes = tee_memory_bytes + tee_encryption_overhead
    
    return {
        'cpu_memory_bytes': cpu_memory_bytes,
        'tee_memory_bytes': tee_memory_bytes,
        'tee_encryption_overhead': tee_encryption_overhead,
        'tee_total_memory_bytes': tee_total_memory_bytes,
        'weight_bytes': weight_bytes,
        'bias_bytes': bias_bytes,
        'activation_bytes': activation_bytes,
        'num_chunks': num_chunks,
        'chunk_metadata_bytes': chunk_metadata_bytes,
    }


def calc_chunkpool_overhead(store_chunk_elem: int = DEFAULT_STORE_CHUNK_ELEM) -> int:
    """
    Calculate ChunkPool pre-allocated memory (shared across all layers).
    
    Formula: THREAD_POOL_SIZE * 2 * STORE_CHUNK_ELEM * 4 bytes
    
    Args:
        store_chunk_elem: STORE_CHUNK_ELEM value
    
    Returns:
        ChunkPool overhead in bytes (~30MB for default)
    """
    return THREAD_POOL_SIZE * 2 * store_chunk_elem * 4


def print_memory_summary(metrics_dict: Dict[str, 'LayerMetrics'], title: str = "Memory Summary"):
    """
    Print a summary of memory usage for all layers.
    
    Args:
        metrics_dict: Dictionary of LayerMetrics
        title: Title for the summary
    """
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")
    
    total_cpu = 0
    total_tee = 0
    total_overhead = 0
    total_chunks = 0
    
    print(f"\n{'Layer Name':<40} {'CPU Memory':>12} {'TEE Memory':>12} {'Overhead':>10} {'Chunks':>8}")
    print("-" * 84)
    
    for name, m in metrics_dict.items():
        cpu_kb = m.cpu_memory_bytes / 1024
        tee_kb = m.tee_total_memory_bytes / 1024
        overhead_b = m.tee_encryption_overhead
        chunks = m.num_chunks
        
        print(f"{name:<40} {cpu_kb:>10.1f}KB {tee_kb:>10.1f}KB {overhead_b:>8}B {chunks:>8}")
        
        total_cpu += m.cpu_memory_bytes
        total_tee += m.tee_total_memory_bytes
        total_overhead += m.tee_encryption_overhead
        total_chunks += m.num_chunks
    
    print("-" * 84)
    print(f"{'TOTAL':<40} {total_cpu/1024/1024:>10.2f}MB {total_tee/1024/1024:>10.2f}MB "
          f"{total_overhead:>8}B {total_chunks:>8}")
    
    # TEE overhead summary
    print(f"\n--- TEE Overhead Analysis ---")
    print(f"Total Encryption Overhead: {total_overhead/1024:.2f} KB ({100*total_overhead/max(total_tee,1):.2f}% of TEE memory)")
    print(f"Total Chunks: {total_chunks}")
    print(f"Average Overhead per Chunk: {CHUNK_ENCRYPTION_OVERHEAD} bytes")
    
    # ChunkPool overhead (shared)
    chunkpool_overhead = calc_chunkpool_overhead()
    print(f"\nChunkPool Shared Overhead: {chunkpool_overhead/(1024*1024):.2f} MB (amortized across all layers)")
    print(f"Effective TEE Total (with ChunkPool): {(total_tee + chunkpool_overhead)/(1024*1024):.2f} MB")


# =============================================================================
# CSV Field Names (for consistent output format)
# =============================================================================

# Standard CSV field names matching Inception output format
CSV_FIELDNAMES = [
    'name', 'type', 'group', 'execution_mode',
    'enclave_time_mean', 'enclave_time_std', 'enclave_time_min', 'enclave_time_max',
    'enclave_time_p95', 'enclave_time_p99',
    'cpu_time_mean', 'cpu_time_std', 'cpu_time_min', 'cpu_time_max',
    'input_bytes', 'output_bytes',
    'input_shape', 'output_shape',
    'dependencies', 'num_iterations',
    # Memory analysis columns
    'cpu_memory_bytes', 'tee_memory_bytes', 'tee_encryption_overhead',
    'tee_total_memory_bytes', 'weight_bytes', 'bias_bytes',
    'activation_bytes', 'num_chunks', 'chunk_metadata_bytes'
]

# Alias for legacy column names (matching inception_v3_layers_merged.csv)
CSV_FIELDNAMES_LEGACY = [
    'LayerName', 'Type', 'Group',
    'EnclaveTime_mean', 'EnclaveTime_std', 'EnclaveTime_min', 'EnclaveTime_max',
    'EnclaveTime_p95', 'EnclaveTime_p99',
    'CPUTime_mean', 'CPUTime_std', 'CPUTime_min', 'CPUTime_max',
    'InputBytes', 'OutputBytes', 'InputShape', 'OutputShape',
    'Dependencies', 'NumIterations',
    'CPU_Memory_Bytes', 'TEE_Memory_Bytes', 'TEE_Encryption_Overhead',
    'TEE_Total_Memory_Bytes', 'Weight_Bytes', 'Bias_Bytes',
    'Activation_Bytes', 'Num_Chunks', 'Chunk_Metadata_Bytes'
]


def metrics_to_csv_row(metrics: LayerMetrics, legacy_format: bool = False) -> Dict[str, Any]:
    """
    Convert LayerMetrics to a CSV row dictionary.
    
    Args:
        metrics: LayerMetrics object
        legacy_format: If True, use legacy column names (for Inception compatibility)
    
    Returns:
        Dictionary suitable for csv.DictWriter
    """
    d = metrics.to_dict()
    
    if legacy_format:
        return {
            'LayerName': d['name'],
            'Type': d['type'],
            'Group': d['group'],
            'EnclaveTime_mean': d['enclave_time_mean'],
            'EnclaveTime_std': d['enclave_time_std'],
            'EnclaveTime_min': d['enclave_time_min'],
            'EnclaveTime_max': d['enclave_time_max'],
            'EnclaveTime_p95': d['enclave_time_p95'],
            'EnclaveTime_p99': d['enclave_time_p99'],
            'CPUTime_mean': d['cpu_time_mean'],
            'CPUTime_std': d['cpu_time_std'],
            'CPUTime_min': d['cpu_time_min'],
            'CPUTime_max': d['cpu_time_max'],
            'InputBytes': d['input_bytes'],
            'OutputBytes': d['output_bytes'],
            'InputShape': str(d['input_shape']),
            'OutputShape': str(d['output_shape']),
            'Dependencies': str(d['dependencies']),
            'NumIterations': d['num_iterations'],
            'CPU_Memory_Bytes': d['cpu_memory_bytes'],
            'TEE_Memory_Bytes': d['tee_memory_bytes'],
            'TEE_Encryption_Overhead': d['tee_encryption_overhead'],
            'TEE_Total_Memory_Bytes': d['tee_total_memory_bytes'],
            'Weight_Bytes': d['weight_bytes'],
            'Bias_Bytes': d['bias_bytes'],
            'Activation_Bytes': d['activation_bytes'],
            'Num_Chunks': d['num_chunks'],
            'Chunk_Metadata_Bytes': d['chunk_metadata_bytes'],
        }
    else:
        # Convert shapes and dependencies to strings for CSV
        d['input_shape'] = str(d['input_shape'])
        d['output_shape'] = str(d['output_shape'])
        d['dependencies'] = str(d['dependencies'])
        return d


# =============================================================================
# Layer Dependency Inference for Transformer Models
# =============================================================================

# Transformer block component ordering (standard architecture)
# This defines the data flow within a single Transformer block
TRANSFORMER_COMPONENT_ORDER = [
    'norm1',           # Pre-LayerNorm for attention
    'attn_qkv_proj',   # QKV projection (Linear)
    'attn_q_proj',     # Q projection (for separate Q/K/V)
    'attn_k_proj',     # K projection
    'attn_v_proj',     # V projection
    'attn_qk_matmul',  # Q @ K^T
    'attn_softmax',    # Softmax(QK^T / sqrt(d))
    'attn_v_matmul',   # Attn @ V
    'attn_out_proj',   # Output projection
    'norm2',           # Pre-LayerNorm for FFN
    'ffn_fc1',         # FFN first layer
    'ffn_intermediate',  # Alternative name for fc1
    'ffn_gelu',        # GELU activation
    'ffn_act',         # Alternative name for activation
    'ffn_fc2',         # FFN second layer
    'ffn_output',      # Alternative name for fc2
]

# Component dependency map within a block
# Maps each component to its immediate predecessor(s) within the same block
COMPONENT_DEPENDENCY_MAP = {
    'norm1': None,  # Depends on previous block's output
    'attn_qkv_proj': ['norm1'],
    'attn_q_proj': ['norm1'],
    'attn_k_proj': ['norm1'],
    'attn_v_proj': ['norm1'],
    'attn_qk_matmul': ['attn_qkv_proj'],  # or ['attn_q_proj', 'attn_k_proj']
    'attn_softmax': ['attn_qk_matmul'],
    'attn_v_matmul': ['attn_softmax', 'attn_qkv_proj'],  # Needs both softmax output and V
    'attn_out_proj': ['attn_v_matmul'],
    'norm2': ['attn_out_proj'],
    'ffn_fc1': ['norm2'],
    'ffn_intermediate': ['norm2'],
    'ffn_gelu': ['ffn_fc1', 'ffn_intermediate'],
    'ffn_act': ['ffn_fc1', 'ffn_intermediate'],
    'ffn_fc2': ['ffn_gelu', 'ffn_act'],
    'ffn_output': ['ffn_gelu', 'ffn_act'],
}

# Special layer patterns for different model architectures
SPECIAL_LAYER_PATTERNS = {
    # ViT / BERT / DistilBERT / TinyBERT / ALBERT
    'patch_embed': [],  # First layer, no dependencies
    'embedding': [],    # First layer for NLP
    'embed': [],        # Alternative
    'head_norm': None,  # Depends on last block
    'classifier': ['head_norm'],
    'pooler': None,     # Depends on last block
    'cls_head': ['pooler', 'head_norm'],
    
    # Swin Transformer specific
    'patch_merging': None,  # Depends on previous stage
    'downsample': None,
}


def _extract_block_info(layer_name: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Extract block prefix, block index, and component name from layer name.
    
    Supported patterns:
    - block{i}_{component}: ViT, BERT, DistilBERT, TinyBERT, ALBERT
    - layer{i}_{component}: Alternative BERT naming
    - stage{s}_block{b}_{component}: Swin Transformer
    
    Returns:
        Tuple of (block_prefix, block_index, component_name)
        Returns (None, None, None) if pattern doesn't match
    """
    import re
    
    # Pattern 1: block{i}_{component} (ViT, BERT, etc.)
    match = re.match(r'^(block)(\d+)_(.+)$', layer_name)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    
    # Pattern 2: layer{i}_{component} (Alternative BERT)
    match = re.match(r'^(layer)(\d+)_(.+)$', layer_name)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    
    # Pattern 3: stage{s}_block{b}_{component} (Swin)
    match = re.match(r'^(stage\d+_block)(\d+)_(.+)$', layer_name)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    
    # Pattern 4: stage{s}_{component} (Swin stage-level layers)
    match = re.match(r'^(stage)(\d+)_(.+)$', layer_name)
    if match:
        return match.group(1), int(match.group(2)), match.group(3)
    
    return None, None, None


def _find_layer_in_list(layer_names: List[str], pattern: str) -> Optional[str]:
    """Find a layer name that ends with the given pattern."""
    for name in layer_names:
        if name.endswith(pattern):
            return name
    return None


def infer_layer_dependencies(
    layer_name: str,
    all_layer_names: List[str],
    model_type: str = 'auto'
) -> List[str]:
    """
    Infer layer dependencies based on layer name and known Transformer topology.
    
    This function analyzes layer names to determine data flow dependencies
    within Transformer architectures. It supports:
    - ViT (Vision Transformer)
    - BERT and variants (DistilBERT, TinyBERT, ALBERT)
    - Swin Transformer
    
    Args:
        layer_name: Name of the current layer
        all_layer_names: List of all layer names in order of profiling
        model_type: Model type hint ('vit', 'bert', 'swin', or 'auto')
    
    Returns:
        List of predecessor layer names that this layer depends on
    
    Example:
        >>> layers = ['patch_embed', 'block0_norm1', 'block0_attn_qkv_proj', ...]
        >>> infer_layer_dependencies('block0_attn_qkv_proj', layers)
        ['block0_norm1']
        >>> infer_layer_dependencies('block1_norm1', layers)
        ['block0_ffn_fc2']
    """
    dependencies = []
    
    # Get the index of current layer
    if layer_name not in all_layer_names:
        return []
    
    current_idx = all_layer_names.index(layer_name)
    
    # Check for special layer patterns first
    for pattern, deps in SPECIAL_LAYER_PATTERNS.items():
        if layer_name == pattern or layer_name.endswith(f'_{pattern}') or layer_name.startswith(pattern):
            if deps is None:
                # Find the last block's output (ffn_fc2 or ffn_output)
                for prev_name in reversed(all_layer_names[:current_idx]):
                    if 'ffn_fc2' in prev_name or 'ffn_output' in prev_name:
                        return [prev_name]
                # Fallback: return previous layer
                if current_idx > 0:
                    return [all_layer_names[current_idx - 1]]
                return []
            elif deps == []:
                return []  # First layer, no dependencies
            else:
                # Find matching dependency layers
                for dep in deps:
                    found = _find_layer_in_list(all_layer_names[:current_idx], dep)
                    if found:
                        dependencies.append(found)
                return dependencies if dependencies else (
                    [all_layer_names[current_idx - 1]] if current_idx > 0 else []
                )
    
    # Extract block information
    block_prefix, block_idx, component = _extract_block_info(layer_name)
    
    if block_prefix is None or block_idx is None or component is None:
        # Unknown pattern, depend on previous layer
        if current_idx > 0:
            return [all_layer_names[current_idx - 1]]
        return []
    
    # Check component dependency map
    deps_templates = COMPONENT_DEPENDENCY_MAP.get(component)
    
    if deps_templates is None:
        # First component in block (norm1) - depends on previous block's output
        if block_idx == 0:
            # First block depends on patch embedding or embedding layer
            for prev_name in reversed(all_layer_names[:current_idx]):
                if 'embed' in prev_name.lower() or 'patch' in prev_name.lower():
                    return [prev_name]
            # Fallback
            if current_idx > 0:
                return [all_layer_names[current_idx - 1]]
            return []
        else:
            # Depends on previous block's final FFN layer
            prev_block_prefix = f'{block_prefix}{block_idx - 1}'
            for suffix in ['_ffn_fc2', '_ffn_output']:
                prev_layer = f'{prev_block_prefix}{suffix}'
                if prev_layer in all_layer_names:
                    return [prev_layer]
            # Fallback: find any layer from previous block
            for prev_name in reversed(all_layer_names[:current_idx]):
                if prev_name.startswith(prev_block_prefix):
                    return [prev_name]
            # Last resort
            if current_idx > 0:
                return [all_layer_names[current_idx - 1]]
            return []
    
    # Build dependencies from template
    current_block_prefix = f'{block_prefix}{block_idx}'
    for dep_component in deps_templates:
        dep_layer = f'{current_block_prefix}_{dep_component}'
        if dep_layer in all_layer_names:
            dependencies.append(dep_layer)
    
    # If no explicit dependencies found, check for alternative component names
    if not dependencies:
        # Try to find any matching component in current block
        for dep_component in deps_templates:
            for prev_name in reversed(all_layer_names[:current_idx]):
                if prev_name.startswith(current_block_prefix) and dep_component in prev_name:
                    dependencies.append(prev_name)
                    break
            if dependencies:
                break
    
    # Fallback to previous layer if still no dependencies
    if not dependencies and current_idx > 0:
        dependencies = [all_layer_names[current_idx - 1]]
    
    return dependencies


def infer_all_dependencies(layer_names: List[str], model_type: str = 'auto') -> Dict[str, List[str]]:
    """
    Infer dependencies for all layers in a model.
    
    Args:
        layer_names: List of all layer names in execution order
        model_type: Model type hint ('vit', 'bert', 'swin', or 'auto')
    
    Returns:
        Dictionary mapping layer names to their dependencies
    """
    deps_dict = {}
    for name in layer_names:
        deps_dict[name] = infer_layer_dependencies(name, layer_names, model_type)
    return deps_dict
