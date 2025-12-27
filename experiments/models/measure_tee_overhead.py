#!/usr/bin/env python3
"""
TEE Overhead Measurement Script

Measures various TEE-related overheads for distributed inference simulation:
1. Enclave initialization time
2. Enclave destruction time  
3. CPU <-> Enclave data transfer time
4. Memory allocation in Enclave

These measurements are essential for accurate distributed TEE inference simulation.
"""

import os
import sys
import time
import json
import argparse
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

import torch
import numpy as np


@dataclass
class OverheadMetrics:
    """Container for overhead measurements"""
    name: str
    num_iterations: int = 0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    measurements: List[float] = field(default_factory=list)
    
    def compute_stats(self):
        """Compute statistics from measurements"""
        if not self.measurements:
            return
        self.num_iterations = len(self.measurements)
        self.mean_ms = statistics.mean(self.measurements)
        self.std_ms = statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0
        self.min_ms = min(self.measurements)
        self.max_ms = max(self.measurements)
        sorted_m = sorted(self.measurements)
        p95_idx = int(0.95 * len(sorted_m))
        p99_idx = int(0.99 * len(sorted_m))
        self.p95_ms = sorted_m[min(p95_idx, len(sorted_m)-1)]
        self.p99_ms = sorted_m[min(p99_idx, len(sorted_m)-1)]


def measure_enclave_init_destroy(num_iterations: int = 10) -> Dict[str, OverheadMetrics]:
    """
    Measure Enclave initialization and destruction overhead.
    
    This simulates the cost of starting/stopping TEE contexts for distributed inference.
    """
    from python.enclave_interfaces import GlobalTensor
    
    init_metrics = OverheadMetrics(name="enclave_init")
    destroy_metrics = OverheadMetrics(name="enclave_destroy")
    
    print(f"\n{'='*60}")
    print("Measuring Enclave Init/Destroy Overhead")
    print(f"{'='*60}")
    print(f"Iterations: {num_iterations}")
    
    for i in range(num_iterations):
        # Measure init
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        GlobalTensor.init()
        end = time.perf_counter()
        init_time_ms = (end - start) * 1000
        init_metrics.measurements.append(init_time_ms)
        
        # Measure destroy
        start = time.perf_counter()
        GlobalTensor.destroy()
        end = time.perf_counter()
        destroy_time_ms = (end - start) * 1000
        destroy_metrics.measurements.append(destroy_time_ms)
        
        print(f"  Iteration {i+1}/{num_iterations}: init={init_time_ms:.2f}ms, destroy={destroy_time_ms:.2f}ms")
    
    init_metrics.compute_stats()
    destroy_metrics.compute_stats()
    
    print(f"\nEnclave Init:    mean={init_metrics.mean_ms:.2f}ms, std={init_metrics.std_ms:.2f}ms")
    print(f"Enclave Destroy: mean={destroy_metrics.mean_ms:.2f}ms, std={destroy_metrics.std_ms:.2f}ms")
    
    return {
        "enclave_init": init_metrics,
        "enclave_destroy": destroy_metrics
    }


def measure_data_transfer(num_iterations: int = 100, 
                          data_sizes: List[int] = None) -> Dict[str, List[OverheadMetrics]]:
    """
    Measure CPU <-> Enclave data transfer overhead for various tensor sizes.
    
    This measures the cost of moving data between untrusted and trusted memory.
    """
    from python.enclave_interfaces import GlobalTensor
    
    if data_sizes is None:
        # Representative sizes from Inception V3 layers (in bytes)
        data_sizes = [
            4000,       # FC output: 1000 * 4 bytes
            5120,       # Classifier avgpool output: 1280 * 4
            49152,      # 192 * 8 * 8 * 4
            98304,      # 384 * 8 * 8 * 4
            221952,     # 192 * 17 * 17 * 4
            327680,     # 1280 * 8 * 8 * 4
            443904,     # 384 * 17 * 17 * 4
            887808,     # 768 * 17 * 17 * 4
            940800,     # 192 * 35 * 35 * 4
            1254400,    # 256 * 35 * 35 * 4
            2765952,    # 32 * 147 * 147 * 4
            5531904,    # 64 * 147 * 147 * 4
        ]
    
    print(f"\n{'='*60}")
    print("Measuring CPU <-> Enclave Data Transfer Overhead")
    print(f"{'='*60}")
    print(f"Iterations per size: {num_iterations}")
    print(f"Data sizes to test: {len(data_sizes)}")
    
    # Initialize Enclave
    print("\nInitializing Enclave...")
    GlobalTensor.init()
    enclave_interface = GlobalTensor.EnclaveInterface
    
    cpu_to_enclave_results = []
    enclave_to_cpu_results = []
    
    try:
        for size_bytes in data_sizes:
            num_floats = size_bytes // 4
            size_kb = size_bytes / 1024
            size_mb = size_bytes / 1024 / 1024
            
            print(f"\n  Testing {size_bytes} bytes ({size_kb:.1f} KB / {size_mb:.3f} MB)...")
            
            # Create test tensor
            tensor = torch.randn(num_floats, dtype=torch.float32)
            tensor_name = f"test_tensor_{size_bytes}"
            output_tensor = torch.zeros(num_floats, dtype=torch.float32)
            
            # First, initialize the tensor in enclave (required before get)
            enclave_interface.init_enclave_tensor(tensor_name, list(tensor.shape))
            
            # Metrics
            c2e_metrics = OverheadMetrics(name=f"cpu_to_enclave_{size_bytes}B")
            e2c_metrics = OverheadMetrics(name=f"enclave_to_cpu_{size_bytes}B")
            
            # Warmup
            for _ in range(5):
                enclave_interface.set_tensor(tensor_name, tensor)
                enclave_interface.get_tensor(tensor_name, output_tensor)
            
            # Measure CPU -> Enclave
            for _ in range(num_iterations):
                start = time.perf_counter()
                enclave_interface.set_tensor(tensor_name, tensor)
                end = time.perf_counter()
                c2e_metrics.measurements.append((end - start) * 1000)
            
            # Measure Enclave -> CPU
            for _ in range(num_iterations):
                start = time.perf_counter()
                enclave_interface.get_tensor(tensor_name, output_tensor)
                end = time.perf_counter()
                e2c_metrics.measurements.append((end - start) * 1000)
            
            c2e_metrics.compute_stats()
            e2c_metrics.compute_stats()
            
            # Add size info
            c2e_metrics.size_bytes = size_bytes
            e2c_metrics.size_bytes = size_bytes
            
            # Calculate bandwidth
            c2e_bandwidth_mbps = (size_bytes / 1024 / 1024) / (c2e_metrics.mean_ms / 1000) if c2e_metrics.mean_ms > 0 else 0
            e2c_bandwidth_mbps = (size_bytes / 1024 / 1024) / (e2c_metrics.mean_ms / 1000) if e2c_metrics.mean_ms > 0 else 0
            
            c2e_metrics.bandwidth_mbps = c2e_bandwidth_mbps
            e2c_metrics.bandwidth_mbps = e2c_bandwidth_mbps
            
            cpu_to_enclave_results.append(c2e_metrics)
            enclave_to_cpu_results.append(e2c_metrics)
            
            print(f"    CPU→Enclave: {c2e_metrics.mean_ms:.4f}ms (±{c2e_metrics.std_ms:.4f}ms), {c2e_bandwidth_mbps:.1f} MB/s")
            print(f"    Enclave→CPU: {e2c_metrics.mean_ms:.4f}ms (±{e2c_metrics.std_ms:.4f}ms), {e2c_bandwidth_mbps:.1f} MB/s")
    
    finally:
        print("\nDestroying Enclave...")
        GlobalTensor.destroy()
    
    return {
        "cpu_to_enclave": cpu_to_enclave_results,
        "enclave_to_cpu": enclave_to_cpu_results
    }


def measure_memory_allocation(num_iterations: int = 50,
                              tensor_sizes: List[int] = None) -> List[OverheadMetrics]:
    """
    Measure memory allocation overhead in Enclave.
    """
    from python.enclave_interfaces import GlobalTensor
    
    if tensor_sizes is None:
        tensor_sizes = [1000, 10000, 100000, 1000000, 5000000]
    
    print(f"\n{'='*60}")
    print("Measuring Enclave Memory Allocation Overhead")
    print(f"{'='*60}")
    
    GlobalTensor.init()
    eid = GlobalTensor.get_eid()
    
    results = []
    
    try:
        for size in tensor_sizes:
            size_mb = size * 4 / 1024 / 1024
            print(f"\n  Testing allocation of {size} floats ({size_mb:.2f} MB)...")
            
            metrics = OverheadMetrics(name=f"alloc_{size}")
            
            for i in range(num_iterations):
                tensor = torch.randn(size, dtype=torch.float32)
                tensor_name = f"alloc_test_{i}"
                
                start = time.perf_counter()
                GlobalTensor.set_tensor_with_name(eid, tensor_name, tensor)
                end = time.perf_counter()
                
                metrics.measurements.append((end - start) * 1000)
            
            metrics.compute_stats()
            metrics.size_floats = size
            metrics.size_mb = size_mb
            results.append(metrics)
            
            print(f"    Allocation: {metrics.mean_ms:.4f}ms (±{metrics.std_ms:.4f}ms)")
    
    finally:
        GlobalTensor.destroy()
    
    return results


def estimate_network_transfer_time(data_bytes: int, 
                                   bandwidth_mbps: float = 1000,  # 1 Gbps default
                                   latency_ms: float = 0.1) -> float:
    """
    Estimate network transfer time for inter-node communication.
    
    Args:
        data_bytes: Size of data to transfer
        bandwidth_mbps: Network bandwidth in Mbps (default 1 Gbps)
        latency_ms: Network latency in milliseconds
    
    Returns:
        Estimated transfer time in milliseconds
    """
    # Convert Mbps to bytes per ms
    bandwidth_bytes_per_ms = (bandwidth_mbps * 1000000) / 8 / 1000
    
    transfer_time_ms = data_bytes / bandwidth_bytes_per_ms
    total_time_ms = latency_ms + transfer_time_ms
    
    return total_time_ms


def export_results(results: Dict, output_path: str):
    """Export measurement results to JSON"""
    
    def serialize(obj):
        if isinstance(obj, OverheadMetrics):
            d = asdict(obj)
            # Remove raw measurements to save space
            d.pop('measurements', None)
            return d
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "measurements": {}
    }
    
    for key, value in results.items():
        if isinstance(value, dict):
            output["measurements"][key] = {
                k: serialize(v) if isinstance(v, OverheadMetrics) 
                   else [serialize(x) for x in v] if isinstance(v, list)
                   else v
                for k, v in value.items()
            }
        elif isinstance(value, list):
            output["measurements"][key] = [serialize(v) for v in value]
        elif isinstance(value, OverheadMetrics):
            output["measurements"][key] = serialize(value)
        else:
            output["measurements"][key] = value
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results exported to: {output_path}")


def print_summary(results: Dict):
    """Print summary of all measurements"""
    
    print("\n" + "="*80)
    print("TEE OVERHEAD MEASUREMENT SUMMARY")
    print("="*80)
    
    # Enclave init/destroy
    if "init_destroy" in results:
        init = results["init_destroy"]["enclave_init"]
        destroy = results["init_destroy"]["enclave_destroy"]
        print(f"\n1. Enclave Lifecycle Overhead:")
        print(f"   Initialization: {init.mean_ms:.2f} ms (±{init.std_ms:.2f})")
        print(f"   Destruction:    {destroy.mean_ms:.2f} ms (±{destroy.std_ms:.2f})")
        print(f"   Total (init+destroy): {init.mean_ms + destroy.mean_ms:.2f} ms")
    
    # Data transfer
    if "data_transfer" in results:
        print(f"\n2. Data Transfer Overhead (CPU ↔ Enclave):")
        print(f"   {'Size (KB)':<12} {'CPU→Enclave':<18} {'Enclave→CPU':<18} {'Bandwidth (MB/s)'}")
        print(f"   {'-'*60}")
        
        c2e = results["data_transfer"]["cpu_to_enclave"]
        e2c = results["data_transfer"]["enclave_to_cpu"]
        
        for i in range(len(c2e)):
            size_kb = c2e[i].size_bytes / 1024
            print(f"   {size_kb:<12.1f} {c2e[i].mean_ms:<18.4f} {e2c[i].mean_ms:<18.4f} {c2e[i].bandwidth_mbps:.1f}")
    
    # Network estimation
    print(f"\n3. Network Transfer Estimation (for distributed TEE):")
    print(f"   Assuming 1 Gbps network, 0.1ms latency:")
    sample_sizes = [4000, 327680, 1254400, 5531904]
    for size in sample_sizes:
        time_ms = estimate_network_transfer_time(size)
        print(f"   {size/1024:.1f} KB: {time_ms:.4f} ms")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Measure TEE overhead for distributed inference simulation"
    )
    parser.add_argument('--init-destroy', action='store_true', default=True,
                       help='Measure Enclave init/destroy overhead')
    parser.add_argument('--data-transfer', action='store_true', default=True,
                       help='Measure CPU<->Enclave data transfer overhead')
    parser.add_argument('--memory-alloc', action='store_true', default=False,
                       help='Measure Enclave memory allocation overhead')
    parser.add_argument('--all', action='store_true',
                       help='Run all measurements')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations for init/destroy measurement')
    parser.add_argument('--transfer-iterations', type=int, default=100,
                       help='Number of iterations for transfer measurement')
    parser.add_argument('--output', type=str, default='experiments/data/tee_overhead.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    if args.all:
        args.init_destroy = True
        args.data_transfer = True
        args.memory_alloc = True
    
    results = {}
    
    print("="*80)
    print("TEE Overhead Measurement Tool")
    print("="*80)
    print(f"Output: {args.output}")
    
    try:
        # Measure init/destroy overhead
        if args.init_destroy:
            results["init_destroy"] = measure_enclave_init_destroy(args.iterations)
        
        # Measure data transfer overhead
        if args.data_transfer:
            results["data_transfer"] = measure_data_transfer(args.transfer_iterations)
        
        # Measure memory allocation overhead
        if args.memory_alloc:
            results["memory_alloc"] = measure_memory_allocation()
        
        # Print summary
        print_summary(results)
        
        # Export results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        export_results(results, args.output)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

