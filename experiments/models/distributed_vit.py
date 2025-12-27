"""
Distributed Inference for Vision Transformer with Flexible Partitioning.

This script demonstrates distributed inference on ViT where different components
can be assigned to run in Enclave or on CPU, following the same pattern as
distributed_resnet.py.

Supported Partitioning Strategies:
1. Layer Pipeline: Early blocks in Enclave, later blocks on CPU
2. Head Parallel: Split attention heads across Enclave and CPU
3. FFN Parallel: Split FFN hidden dimension across workers

Key Differences from CNN Distributed Inference:
- Transformer blocks are more uniform than CNN layers
- Attention computation is O(N^2) in sequence length
- Head parallelism is natural in Transformers (unlike CNNs)
- No spatial downsampling between blocks

Example Usage:
    python -m experiments.models.distributed_vit --strategy layer_pipeline
    python -m experiments.models.distributed_vit --strategy head_parallel
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None

import sys
sys.path.insert(0, '.')

from experiments.models.sgx_vit import (
    SGXVisionTransformer,
    ViTSmallConfig,
    ViTTinyConfig,
    create_vit_small,
    create_vit_tiny,
)


# ==============================================================================
# Execution Mode (mirrors existing framework)
# ==============================================================================

class ExecutionMode(Enum):
    """Execution environment for a layer/component."""
    ENCLAVE = "enclave"
    CPU = "cpu"


@dataclass
class PartitionAssignment:
    """Assignment of model components to execution modes."""
    layer_modes: Dict[str, ExecutionMode]
    head_modes: Optional[Dict[int, ExecutionMode]] = None  # For head parallelism
    ffn_modes: Optional[Dict[int, ExecutionMode]] = None   # For FFN parallelism


# ==============================================================================
# Timing Utilities
# ==============================================================================

def _format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")


def _time_layer(
    partition_label: str,
    layer_name: str,
    fn: Callable[[], Any],
    timings: Optional[Dict[str, float]] = None,
    verbose: bool = True
) -> Any:
    start_ts = time.time()
    if verbose:
        print(f"{partition_label} {layer_name} start @ {_format_timestamp(start_ts)}")
    result = fn()
    end_ts = time.time()
    duration_ms = (end_ts - start_ts) * 1000
    if verbose:
        print(f"{partition_label} {layer_name} end @ {_format_timestamp(end_ts)} (+{duration_ms:.3f} ms)")
    if timings is not None:
        timings[layer_name] = duration_ms
    return result


# ==============================================================================
# Strategy 1: Layer-wise Pipeline Parallelism
# ==============================================================================

class LayerPipelineExecutor:
    """
    Execute ViT with layer-wise pipeline parallelism.
    
    Similar to distributed_resnet.py pattern:
    - Split transformer blocks into two groups
    - Group A runs in "Enclave" (simulated with threading)
    - Group B runs in "CPU"
    - Communication happens at the split point
    """
    
    def __init__(
        self,
        model: SGXVisionTransformer,
        split_point: int,  # Block index where split occurs
        verbose: bool = True
    ):
        self.model = model
        self.split_point = split_point
        self.verbose = verbose
        self.num_blocks = model.config.num_layers
        
        # Queues for inter-partition communication
        self.split_queue: queue.Queue = queue.Queue()
        
        # Timing records
        self.enclave_timings: Dict[str, float] = {}
        self.cpu_timings: Dict[str, float] = {}
    
    def _run_enclave_partition(self, x: torch.Tensor) -> None:
        """Execute first partition (blocks 0 to split_point-1)."""
        label = "[ENCLAVE]"
        
        # Patch embedding always in enclave
        x = _time_layer(label, "patch_embed", lambda: self.model.patch_embed(x),
                       self.enclave_timings, self.verbose)
        
        # First group of blocks
        for i in range(self.split_point):
            x = _time_layer(label, f"block_{i}", lambda i=i: self.model.blocks[i](x),
                           self.enclave_timings, self.verbose)
        
        # Send to CPU partition via queue
        self.split_queue.put(x.clone())
    
    def _run_cpu_partition(self) -> torch.Tensor:
        """Execute second partition (blocks split_point to end)."""
        label = "[CPU]"
        
        # Wait for input from enclave partition
        x = self.split_queue.get()
        
        # Second group of blocks
        for i in range(self.split_point, self.num_blocks):
            x = _time_layer(label, f"block_{i}", lambda i=i: self.model.blocks[i](x),
                           self.cpu_timings, self.verbose)
        
        # Final norm and head
        x = _time_layer(label, "norm", lambda: self.model.norm(x),
                       self.cpu_timings, self.verbose)
        
        cls_token = x[:, 0]
        output = _time_layer(label, "head", lambda: self.model.head(cls_token),
                            self.cpu_timings, self.verbose)
        
        return output
    
    def run(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run distributed inference.
        
        Returns:
            output: Model output tensor
            stats: Timing and performance statistics
        """
        overall_start = time.time()
        
        # Start enclave partition in background thread
        enclave_thread = threading.Thread(target=self._run_enclave_partition, args=(x,))
        enclave_thread.start()
        
        # Run CPU partition (waits for enclave output)
        with torch.no_grad():
            output = self._run_cpu_partition()
        
        # Wait for enclave to complete
        enclave_thread.join()
        
        overall_end = time.time()
        total_latency_ms = (overall_end - overall_start) * 1000
        
        # Compute statistics
        enclave_total = sum(self.enclave_timings.values())
        cpu_total = sum(self.cpu_timings.values())
        
        # Find critical path
        # In pipeline, the critical path is the longer partition + communication
        # But since they overlap, total ≈ max(enclave, cpu) + overhead
        
        stats = {
            'total_latency_ms': total_latency_ms,
            'enclave_total_ms': enclave_total,
            'cpu_total_ms': cpu_total,
            'enclave_layers': list(range(self.split_point)),
            'cpu_layers': list(range(self.split_point, self.num_blocks)),
            'speedup_vs_sequential': (enclave_total + cpu_total) / total_latency_ms,
            'enclave_timings': dict(self.enclave_timings),
            'cpu_timings': dict(self.cpu_timings),
        }
        
        return output, stats


# ==============================================================================
# Strategy 2: Attention Head Parallelism
# ==============================================================================

class HeadParallelExecutor:
    """
    Execute ViT with attention head parallelism.
    
    For each transformer block:
    - Split attention heads between Enclave and CPU
    - Both compute their assigned heads in parallel
    - Merge results before FFN
    
    Trade-offs:
    - More parallelism opportunities per block
    - Higher communication overhead (sync after every attention)
    - Memory efficient (each partition holds fewer heads)
    """
    
    def __init__(
        self,
        model: SGXVisionTransformer,
        enclave_heads: List[int],
        cpu_heads: List[int],
        verbose: bool = True
    ):
        self.model = model
        self.enclave_heads = enclave_heads
        self.cpu_heads = cpu_heads
        self.verbose = verbose
        
        # Queues for each block's attention synchronization
        self.head_queues: Dict[int, queue.Queue] = {
            i: queue.Queue() for i in range(model.config.num_layers)
        }
        
        self.enclave_timings: Dict[str, float] = {}
        self.cpu_timings: Dict[str, float] = {}
    
    def _compute_heads_enclave(
        self,
        block_idx: int,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Compute enclave-assigned heads for one block."""
        block = self.model.blocks[block_idx]
        norm1_out = block.norm1(x)
        
        # Compute only assigned heads
        out_partial = block.attn.forward_heads_subset(norm1_out, self.enclave_heads)
        return out_partial
    
    def _compute_heads_cpu(
        self,
        block_idx: int,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Compute CPU-assigned heads for one block."""
        block = self.model.blocks[block_idx]
        norm1_out = block.norm1(x)
        
        # Compute only assigned heads
        out_partial = block.attn.forward_heads_subset(norm1_out, self.cpu_heads)
        return out_partial
    
    def run_block_parallel(
        self,
        block_idx: int,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Run one transformer block with head parallelism.
        
        1. Both partitions compute their heads in parallel
        2. Merge attention outputs
        3. One partition runs FFN (or can also be parallelized)
        """
        block = self.model.blocks[block_idx]
        label_enc = f"[ENCLAVE B{block_idx}]"
        label_cpu = f"[CPU B{block_idx}]"
        
        # Parallel head computation
        enclave_result = [None]
        cpu_result = [None]
        
        def compute_enclave():
            enclave_result[0] = _time_layer(
                label_enc, "attn_heads",
                lambda: self._compute_heads_enclave(block_idx, x),
                self.enclave_timings, self.verbose
            )
        
        def compute_cpu():
            cpu_result[0] = _time_layer(
                label_cpu, "attn_heads",
                lambda: self._compute_heads_cpu(block_idx, x),
                self.cpu_timings, self.verbose
            )
        
        # Run in parallel
        enc_thread = threading.Thread(target=compute_enclave)
        cpu_thread = threading.Thread(target=compute_cpu)
        
        enc_thread.start()
        cpu_thread.start()
        
        enc_thread.join()
        cpu_thread.join()
        
        # Merge attention outputs
        merged_attn = block.attn.merge_head_outputs(
            [enclave_result[0], cpu_result[0]],
            block.attn.proj.weight,
            block.attn.proj.bias
        )
        
        # Residual connection
        x = x + merged_attn
        
        # FFN (run on one partition for simplicity)
        norm2_out = block.norm2(x)
        ffn_out = _time_layer(
            label_cpu, "ffn",
            lambda: block.ffn(norm2_out),
            self.cpu_timings, self.verbose
        )
        x = x + ffn_out
        
        return x
    
    def run(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Run full inference with head parallelism."""
        overall_start = time.time()
        
        with torch.no_grad():
            # Patch embedding
            x = _time_layer("[ENCLAVE]", "patch_embed",
                           lambda: self.model.patch_embed(x),
                           self.enclave_timings, self.verbose)
            
            # Transformer blocks with head parallelism
            for i in range(self.model.config.num_layers):
                x = self.run_block_parallel(i, x)
            
            # Final layers
            x = _time_layer("[CPU]", "norm", lambda: self.model.norm(x),
                           self.cpu_timings, self.verbose)
            
            cls_token = x[:, 0]
            output = _time_layer("[CPU]", "head", lambda: self.model.head(cls_token),
                                self.cpu_timings, self.verbose)
        
        overall_end = time.time()
        total_latency_ms = (overall_end - overall_start) * 1000
        
        stats = {
            'total_latency_ms': total_latency_ms,
            'enclave_heads': self.enclave_heads,
            'cpu_heads': self.cpu_heads,
            'enclave_timings': dict(self.enclave_timings),
            'cpu_timings': dict(self.cpu_timings),
        }
        
        return output, stats


# ==============================================================================
# Baseline: Sequential Execution
# ==============================================================================

def run_sequential(
    model: SGXVisionTransformer,
    x: torch.Tensor,
    verbose: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run standard sequential inference for baseline comparison."""
    timings = {}
    
    overall_start = time.time()
    
    with torch.no_grad():
        output = model.forward_layer_by_layer(x, lambda name, ms: timings.update({name: ms}))
    
    overall_end = time.time()
    total_latency_ms = (overall_end - overall_start) * 1000
    
    stats = {
        'total_latency_ms': total_latency_ms,
        'layer_timings': timings,
    }
    
    if verbose:
        print(f"\n[Sequential] Total: {total_latency_ms:.2f} ms")
    
    return output, stats


# ==============================================================================
# Benchmark Runner
# ==============================================================================

def run_benchmark(
    model_size: str = 'tiny',
    strategies: List[str] = None,
    num_runs: int = 10,
    warmup_runs: int = 3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run benchmark comparing different partitioning strategies.
    
    Args:
        model_size: 'tiny' or 'small'
        strategies: List of strategies to test
        num_runs: Number of timed runs
        warmup_runs: Number of warmup runs
        verbose: Print per-layer timings
        
    Returns:
        Benchmark results with timing statistics
    """
    if strategies is None:
        strategies = ['sequential', 'layer_pipeline', 'head_parallel']
    
    # Create model
    if model_size == 'tiny':
        model = create_vit_tiny(num_classes=10)
    else:
        model = create_vit_small(num_classes=10)
    
    model.eval()
    
    # Create input
    x = torch.randn(1, 3, model.config.img_size, model.config.img_size)
    
    print("=" * 70)
    print(f"ViT Distributed Inference Benchmark")
    print(f"Model: ViT-{model_size.capitalize()}")
    print(f"Input: {tuple(x.shape)}")
    print(f"Num Blocks: {model.config.num_layers}")
    print(f"Num Heads: {model.config.num_heads}")
    print("=" * 70)
    
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*30} {strategy.upper()} {'='*30}")
        
        latencies = []
        
        for run_idx in range(warmup_runs + num_runs):
            is_warmup = run_idx < warmup_runs
            
            if strategy == 'sequential':
                output, stats = run_sequential(model, x, verbose=False)
                
            elif strategy == 'layer_pipeline':
                split_point = model.config.num_layers // 2
                executor = LayerPipelineExecutor(model, split_point, verbose=False)
                output, stats = executor.run(x)
                
            elif strategy == 'head_parallel':
                mid = model.config.num_heads // 2
                enclave_heads = list(range(mid))
                cpu_heads = list(range(mid, model.config.num_heads))
                executor = HeadParallelExecutor(model, enclave_heads, cpu_heads, verbose=False)
                output, stats = executor.run(x)
            
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            if not is_warmup:
                latencies.append(stats['total_latency_ms'])
                if verbose:
                    print(f"  Run {run_idx - warmup_runs + 1}: {stats['total_latency_ms']:.2f} ms")
        
        import numpy as np
        results[strategy] = {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'all_latencies': latencies
        }
        
        print(f"  Mean: {results[strategy]['mean_ms']:.2f} ms")
        print(f"  Std:  {results[strategy]['std_ms']:.2f} ms")
    
    # Compare against baseline
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    baseline = results.get('sequential', results[strategies[0]])['mean_ms']
    
    print(f"{'Strategy':<25} {'Mean (ms)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for strategy in strategies:
        mean = results[strategy]['mean_ms']
        speedup = baseline / mean
        print(f"{strategy:<25} {mean:<15.2f} {speedup:<10.2f}x")
    
    return results


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Distributed Vision Transformer Inference"
    )
    parser.add_argument(
        '--model', type=str, default='tiny',
        choices=['tiny', 'small'],
        help='Model size'
    )
    parser.add_argument(
        '--strategy', type=str, default='all',
        choices=['sequential', 'layer_pipeline', 'head_parallel', 'all'],
        help='Partitioning strategy'
    )
    parser.add_argument(
        '--num-runs', type=int, default=10,
        help='Number of timed runs'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print detailed timing'
    )
    
    args = parser.parse_args()
    
    if args.strategy == 'all':
        strategies = ['sequential', 'layer_pipeline', 'head_parallel']
    else:
        strategies = [args.strategy]
    
    run_benchmark(
        model_size=args.model,
        strategies=strategies,
        num_runs=args.num_runs,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()

