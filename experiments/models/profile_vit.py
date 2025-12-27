"""
Vision Transformer Performance Profiler for Distributed Inference Modeling.

This script measures:
1. Execution time of each transformer component (attention, FFN, layer norm)
2. Memory footprint per component
3. Component dependencies for DAG construction
4. Parallelization opportunities analysis

Output: experiments/data/vit_layers.csv

Design Philosophy:
- Mirror profile_inception.py patterns for consistency
- Focus on Transformer-specific decomposition (heads, FFN partitions)
- Provide actionable data for partitioning decisions
"""

from __future__ import annotations

import sys
import time
import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import argparse

import numpy as np

sys.path.insert(0, '.')

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None

from experiments.models.sgx_vit import (
    SGXVisionTransformer,
    ViTConfig,
    ViTTinyConfig,
    ViTSmallConfig,
    create_vit_tiny,
    create_vit_small,
)


# ==============================================================================
# Profiling Data Structures
# ==============================================================================

@dataclass
class ComponentMetrics:
    """Metrics for a single component (layer/head/FFN partition)."""
    name: str
    component_type: str
    parent_block: Optional[str] = None
    
    # Timing statistics (ms)
    time_mean: float = 0.0
    time_std: float = 0.0
    time_min: float = 0.0
    time_max: float = 0.0
    time_p95: float = 0.0
    time_p99: float = 0.0
    
    # Memory (bytes)
    input_bytes: int = 0
    output_bytes: int = 0
    param_bytes: int = 0
    activation_bytes: int = 0
    
    # Shape info
    input_shape: List[int] = field(default_factory=list)
    output_shape: List[int] = field(default_factory=list)
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    
    # Parallelization potential
    is_parallelizable: bool = False
    parallel_units: int = 1  # e.g., num_heads for attention
    
    # Raw data
    all_times: List[float] = field(default_factory=list)
    num_iterations: int = 0
    
    def compute_statistics(self):
        """Compute statistics from raw timing data."""
        if self.all_times:
            times = np.array(self.all_times)
            self.time_mean = float(np.mean(times))
            self.time_std = float(np.std(times))
            self.time_min = float(np.min(times))
            self.time_max = float(np.max(times))
            self.time_p95 = float(np.percentile(times, 95))
            self.time_p99 = float(np.percentile(times, 99))
            self.num_iterations = len(times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'type': self.component_type,
            'parent_block': self.parent_block,
            'time_mean_ms': self.time_mean,
            'time_std_ms': self.time_std,
            'time_min_ms': self.time_min,
            'time_max_ms': self.time_max,
            'time_p95_ms': self.time_p95,
            'time_p99_ms': self.time_p99,
            'input_bytes': self.input_bytes,
            'output_bytes': self.output_bytes,
            'param_bytes': self.param_bytes,
            'activation_bytes': self.activation_bytes,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'dependencies': self.dependencies,
            'is_parallelizable': self.is_parallelizable,
            'parallel_units': self.parallel_units,
            'num_iterations': self.num_iterations,
        }


# ==============================================================================
# Profiler Implementation
# ==============================================================================

class ViTProfiler:
    """
    Comprehensive profiler for Vision Transformer.
    
    Profiles at multiple granularities:
    1. Block level: Each transformer block as a unit
    2. Component level: Attention, FFN, LayerNorm separately
    3. Head level: Individual attention heads
    4. FFN partition level: Split FFN computation
    """
    
    def __init__(
        self,
        model: SGXVisionTransformer,
        num_iterations: int = 30,
        warmup_iterations: int = 5,
        batch_size: int = 1,
    ):
        self.model = model
        self.config = model.config
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
        self.batch_size = batch_size
        
        self.model.eval()
        
        # Results storage
        self.metrics: Dict[str, ComponentMetrics] = {}
    
    def _create_input(self) -> torch.Tensor:
        """Create random input tensor."""
        return torch.randn(
            self.batch_size,
            self.config.in_channels,
            self.config.img_size,
            self.config.img_size
        )
    
    def _time_component(
        self,
        name: str,
        fn: Callable[[], torch.Tensor],
        num_iters: int,
        warmup_iters: int
    ) -> List[float]:
        """Time a component over multiple iterations."""
        times = []
        
        for i in range(warmup_iters + num_iters):
            start = time.perf_counter()
            with torch.no_grad():
                _ = fn()
            end = time.perf_counter()
            
            if i >= warmup_iters:
                times.append((end - start) * 1000)  # ms
        
        return times
    
    def profile_block_level(self) -> Dict[str, ComponentMetrics]:
        """Profile each transformer block as a unit."""
        print("Profiling at block level...")
        
        x = self._create_input()
        
        # Patch embedding
        metrics = ComponentMetrics(
            name='patch_embed',
            component_type='PatchEmbed',
            dependencies=['input'],
            is_parallelizable=False,
            parallel_units=1
        )
        
        with torch.no_grad():
            patch_out = self.model.patch_embed(x)
        
        metrics.input_shape = list(x.shape)
        metrics.output_shape = list(patch_out.shape)
        metrics.input_bytes = x.numel() * 4
        metrics.output_bytes = patch_out.numel() * 4
        metrics.param_bytes = sum(p.numel() for p in self.model.patch_embed.parameters()) * 4
        
        metrics.all_times = self._time_component(
            'patch_embed',
            lambda: self.model.patch_embed(x),
            self.num_iterations,
            self.warmup_iterations
        )
        metrics.compute_statistics()
        self.metrics['patch_embed'] = metrics
        
        # Transformer blocks
        current_input = patch_out
        prev_layer = 'patch_embed'
        
        for i, block in enumerate(self.model.blocks):
            name = f'block_{i}'
            metrics = ComponentMetrics(
                name=name,
                component_type='TransformerBlock',
                parent_block=name,
                dependencies=[prev_layer],
                is_parallelizable=True,  # Heads and FFN can be parallelized
                parallel_units=self.config.num_heads
            )
            
            metrics.input_shape = list(current_input.shape)
            metrics.output_shape = list(current_input.shape)  # Same shape
            metrics.input_bytes = current_input.numel() * 4
            metrics.output_bytes = current_input.numel() * 4
            metrics.param_bytes = sum(p.numel() for p in block.parameters()) * 4
            
            # Estimate activation memory (QKV + attention matrix + FFN hidden)
            N = self.config.num_patches + 1
            D = self.config.embed_dim
            H = self.config.mlp_hidden_dim
            metrics.activation_bytes = self.batch_size * (
                N * D * 3 +  # QKV
                self.config.num_heads * N * N +  # Attention matrices
                N * H  # FFN hidden
            ) * 4
            
            block_input = current_input.clone()
            metrics.all_times = self._time_component(
                name,
                lambda: block(block_input),
                self.num_iterations,
                self.warmup_iterations
            )
            metrics.compute_statistics()
            
            with torch.no_grad():
                current_input = block(current_input)
            
            self.metrics[name] = metrics
            prev_layer = name
            
            print(f"  Block {i}: {metrics.time_mean:.2f} ± {metrics.time_std:.2f} ms")
        
        # Final norm
        metrics = ComponentMetrics(
            name='norm',
            component_type='LayerNorm',
            dependencies=[prev_layer],
            is_parallelizable=False
        )
        norm_input = current_input.clone()
        metrics.all_times = self._time_component(
            'norm',
            lambda: self.model.norm(norm_input),
            self.num_iterations,
            self.warmup_iterations
        )
        metrics.compute_statistics()
        self.metrics['norm'] = metrics
        
        # Classification head
        with torch.no_grad():
            normed = self.model.norm(current_input)
            cls_token = normed[:, 0]
        
        metrics = ComponentMetrics(
            name='head',
            component_type='Linear',
            dependencies=['norm'],
            is_parallelizable=False
        )
        head_input = cls_token.clone()
        metrics.all_times = self._time_component(
            'head',
            lambda: self.model.head(head_input),
            self.num_iterations,
            self.warmup_iterations
        )
        metrics.compute_statistics()
        self.metrics['head'] = metrics
        
        return self.metrics
    
    def profile_component_level(self) -> Dict[str, ComponentMetrics]:
        """Profile attention, FFN, LayerNorm separately within each block."""
        print("Profiling at component level...")
        
        x = self._create_input()
        
        with torch.no_grad():
            current = self.model.patch_embed(x)
        
        for i, block in enumerate(self.model.blocks):
            block_prefix = f'block_{i}'
            
            # LayerNorm 1
            norm1_in = current.clone()
            metrics = ComponentMetrics(
                name=f'{block_prefix}_norm1',
                component_type='LayerNorm',
                parent_block=block_prefix,
            )
            metrics.all_times = self._time_component(
                f'{block_prefix}_norm1',
                lambda: block.norm1(norm1_in),
                self.num_iterations,
                self.warmup_iterations
            )
            metrics.compute_statistics()
            self.metrics[f'{block_prefix}_norm1'] = metrics
            
            with torch.no_grad():
                norm1_out = block.norm1(current)
            
            # Attention
            attn_in = norm1_out.clone()
            metrics = ComponentMetrics(
                name=f'{block_prefix}_attn',
                component_type='MultiHeadAttention',
                parent_block=block_prefix,
                is_parallelizable=True,
                parallel_units=self.config.num_heads
            )
            metrics.all_times = self._time_component(
                f'{block_prefix}_attn',
                lambda: block.attn(attn_in),
                self.num_iterations,
                self.warmup_iterations
            )
            metrics.compute_statistics()
            self.metrics[f'{block_prefix}_attn'] = metrics
            
            with torch.no_grad():
                attn_out = block.attn(norm1_out)
                current = current + attn_out
                norm2_out = block.norm2(current)
            
            # LayerNorm 2
            norm2_in = current.clone()
            metrics = ComponentMetrics(
                name=f'{block_prefix}_norm2',
                component_type='LayerNorm',
                parent_block=block_prefix,
            )
            metrics.all_times = self._time_component(
                f'{block_prefix}_norm2',
                lambda: block.norm2(norm2_in),
                self.num_iterations,
                self.warmup_iterations
            )
            metrics.compute_statistics()
            self.metrics[f'{block_prefix}_norm2'] = metrics
            
            # FFN
            ffn_in = norm2_out.clone()
            metrics = ComponentMetrics(
                name=f'{block_prefix}_ffn',
                component_type='FFN',
                parent_block=block_prefix,
                is_parallelizable=True,
                parallel_units=2  # Can split hidden dim in 2
            )
            metrics.all_times = self._time_component(
                f'{block_prefix}_ffn',
                lambda: block.ffn(ffn_in),
                self.num_iterations,
                self.warmup_iterations
            )
            metrics.compute_statistics()
            self.metrics[f'{block_prefix}_ffn'] = metrics
            
            with torch.no_grad():
                ffn_out = block.ffn(norm2_out)
                current = current + ffn_out
            
            print(f"  Block {i}: attn={self.metrics[f'{block_prefix}_attn'].time_mean:.2f}ms, "
                  f"ffn={self.metrics[f'{block_prefix}_ffn'].time_mean:.2f}ms")
        
        return self.metrics
    
    def profile_head_level(self) -> Dict[str, ComponentMetrics]:
        """Profile individual attention heads."""
        print("Profiling at head level...")
        
        x = self._create_input()
        
        with torch.no_grad():
            current = self.model.patch_embed(x)
        
        for block_idx, block in enumerate(self.model.blocks):
            block_prefix = f'block_{block_idx}'
            
            with torch.no_grad():
                norm1_out = block.norm1(current)
            
            # Profile each head
            for head_idx in range(self.config.num_heads):
                name = f'{block_prefix}_head_{head_idx}'
                head_in = norm1_out.clone()
                
                metrics = ComponentMetrics(
                    name=name,
                    component_type='AttentionHead',
                    parent_block=block_prefix,
                    is_parallelizable=False,  # Individual head is atomic
                )
                
                metrics.all_times = self._time_component(
                    name,
                    lambda h=head_idx: block.attn.forward_heads_subset(head_in, [h]),
                    self.num_iterations,
                    self.warmup_iterations
                )
                metrics.compute_statistics()
                self.metrics[name] = metrics
            
            # Move to next block
            with torch.no_grad():
                current = block(current)
            
            # Print summary for this block
            head_times = [
                self.metrics[f'{block_prefix}_head_{h}'].time_mean
                for h in range(self.config.num_heads)
            ]
            print(f"  Block {block_idx}: heads mean={np.mean(head_times):.2f}ms, "
                  f"std={np.std(head_times):.2f}ms")
        
        return self.metrics
    
    def profile_all_levels(self) -> Dict[str, ComponentMetrics]:
        """Run all profiling levels."""
        self.profile_block_level()
        self.profile_component_level()
        self.profile_head_level()
        return self.metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        block_metrics = [m for k, m in self.metrics.items() if k.startswith('block_') and '_' not in k[6:]]
        
        block_times = [m.time_mean for m in block_metrics]
        
        summary = {
            'model_config': {
                'num_layers': self.config.num_layers,
                'num_heads': self.config.num_heads,
                'embed_dim': self.config.embed_dim,
                'num_patches': self.config.num_patches,
            },
            'total_inference_time_ms': sum(block_times) + 
                self.metrics.get('patch_embed', ComponentMetrics('', '')).time_mean +
                self.metrics.get('norm', ComponentMetrics('', '')).time_mean +
                self.metrics.get('head', ComponentMetrics('', '')).time_mean,
            'block_time_stats': {
                'mean': np.mean(block_times) if block_times else 0,
                'std': np.std(block_times) if block_times else 0,
                'min': np.min(block_times) if block_times else 0,
                'max': np.max(block_times) if block_times else 0,
            },
            'parallelization_potential': {
                'num_heads': self.config.num_heads,
                'attention_time_fraction': self._compute_attention_fraction(),
                'ffn_time_fraction': self._compute_ffn_fraction(),
            }
        }
        
        return summary
    
    def _compute_attention_fraction(self) -> float:
        """Compute fraction of time spent in attention."""
        attn_time = sum(
            m.time_mean for k, m in self.metrics.items() 
            if '_attn' in k
        )
        total_time = sum(m.time_mean for m in self.metrics.values())
        return attn_time / total_time if total_time > 0 else 0
    
    def _compute_ffn_fraction(self) -> float:
        """Compute fraction of time spent in FFN."""
        ffn_time = sum(
            m.time_mean for k, m in self.metrics.items() 
            if '_ffn' in k
        )
        total_time = sum(m.time_mean for m in self.metrics.values())
        return ffn_time / total_time if total_time > 0 else 0
    
    def save_csv(self, filepath: str):
        """Save metrics to CSV file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = None
            
            for name, metrics in self.metrics.items():
                row = metrics.to_dict()
                
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    writer.writeheader()
                
                writer.writerow(row)
        
        print(f"Saved metrics to {filepath}")
    
    def save_json(self, filepath: str):
        """Save metrics and summary to JSON file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        data = {
            'metrics': {k: v.to_dict() for k, v in self.metrics.items()},
            'summary': self.get_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved metrics to {filepath}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Profile Vision Transformer")
    parser.add_argument('--model', type=str, default='tiny', choices=['tiny', 'small'])
    parser.add_argument('--level', type=str, default='block',
                       choices=['block', 'component', 'head', 'all'])
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--output', type=str, default='experiments/data/vit_profile.json')
    
    args = parser.parse_args()
    
    # Create model
    if args.model == 'tiny':
        model = create_vit_tiny(num_classes=10)
    else:
        model = create_vit_small(num_classes=10)
    
    print("=" * 70)
    print("Vision Transformer Profiler")
    print("=" * 70)
    print(f"Model: ViT-{args.model.capitalize()}")
    print(f"Layers: {model.config.num_layers}")
    print(f"Heads: {model.config.num_heads}")
    print(f"Embed Dim: {model.config.embed_dim}")
    print(f"Profiling Level: {args.level}")
    print(f"Iterations: {args.iterations} (warmup: {args.warmup})")
    print("=" * 70)
    
    # Run profiler
    profiler = ViTProfiler(
        model=model,
        num_iterations=args.iterations,
        warmup_iterations=args.warmup
    )
    
    if args.level == 'block':
        profiler.profile_block_level()
    elif args.level == 'component':
        profiler.profile_component_level()
    elif args.level == 'head':
        profiler.profile_head_level()
    else:
        profiler.profile_all_levels()
    
    # Print summary
    summary = profiler.get_summary()
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total Inference Time: {summary['total_inference_time_ms']:.2f} ms")
    print(f"Block Time Mean: {summary['block_time_stats']['mean']:.2f} ms")
    print(f"Attention Time Fraction: {summary['parallelization_potential']['attention_time_fraction']*100:.1f}%")
    print(f"FFN Time Fraction: {summary['parallelization_potential']['ffn_time_fraction']*100:.1f}%")
    
    # Save results
    profiler.save_json(args.output)
    
    csv_path = args.output.replace('.json', '.csv')
    profiler.save_csv(csv_path)


if __name__ == '__main__':
    main()

