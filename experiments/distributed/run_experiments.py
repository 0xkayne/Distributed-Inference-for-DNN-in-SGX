"""
Run Distributed Inference Experiments

This script runs comprehensive experiments to analyze the speedup
potential of distributed TEE inference for Inception-like models.

Experiments:
1. Speedup vs Number of Nodes
2. Speedup vs Network Bandwidth
3. Scheduler Comparison
4. Communication Cost Analysis

Output:
- experiments/data/distributed/: CSV and JSON results
- experiments/figures/distributed/: Visualization plots
"""

import sys
import os
import json
import csv
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

sys.path.insert(0, '.')

from experiments.distributed.dag_model import InceptionDAG, LayerNode
from experiments.distributed.cost_model import CostModel, CostConfig
from experiments.distributed.simulator import DistributedSimulator
from experiments.distributed.scheduler import (
    get_all_schedulers,
    ASAPScheduler,
    HEFTScheduler,
    CriticalPathFirstScheduler,
    LoadBalancingScheduler,
)


# Default experiment parameters
DEFAULT_NODE_COUNTS = [1, 2, 4, 8, 16, 32]
DEFAULT_BANDWIDTHS = [100, 1000, 10000]  # Mbps
DEFAULT_SCHEDULERS = ['ASAP', 'HEFT', 'CriticalPathFirst', 'LoadBalancing']
DEFAULT_OUTPUT_DIR = 'experiments/data/distributed'
DEFAULT_FIGURES_DIR = 'experiments/figures/distributed'


def create_sample_dag() -> InceptionDAG:
    """
    Create a sample DAG representing Inception V3 structure.
    
    This is used when no measurement data is available.
    The timing values are estimated based on typical layer execution times.
    """
    dag = InceptionDAG("InceptionV3_Sample")
    
    # Stem layers (input to first inception block)
    stem_layers = [
        ("input", "SecretInputLayer", 0.1, 1*3*299*299*4),
        ("stem_conv1", "SGXConvBase", 5.0, 1*32*149*149*4),
        ("stem_relu1", "SecretReLULayer", 0.5, 1*32*149*149*4),
        ("stem_conv2", "SGXConvBase", 4.0, 1*32*147*147*4),
        ("stem_relu2", "SecretReLULayer", 0.5, 1*32*147*147*4),
        ("stem_conv3", "SGXConvBase", 4.0, 1*64*147*147*4),
        ("stem_relu3", "SecretReLULayer", 0.5, 1*64*147*147*4),
        ("stem_pool1", "SecretMaxpool2dLayer", 1.0, 1*64*73*73*4),
        ("stem_conv4", "SGXConvBase", 2.0, 1*80*73*73*4),
        ("stem_relu4", "SecretReLULayer", 0.3, 1*80*73*73*4),
        ("stem_conv5", "SGXConvBase", 3.0, 1*192*71*71*4),
        ("stem_relu5", "SecretReLULayer", 0.3, 1*192*71*71*4),
        ("stem_pool2", "SecretMaxpool2dLayer", 0.8, 1*192*35*35*4),
    ]
    
    prev_layer = None
    for name, ltype, exec_time, output_bytes in stem_layers:
        node = LayerNode(
            name=name,
            layer_type=ltype,
            group="Stem",
            exec_time_enclave=exec_time,
            exec_time_cpu=exec_time * 0.8,
            output_bytes=output_bytes,
        )
        if prev_layer:
            node.predecessors.append(prev_layer)
        dag.add_node(node)
        prev_layer = name
    
    # Add Inception blocks (simplified)
    inception_configs = [
        # (name_prefix, num_blocks, input_channels, group)
        ("inception_a", 3, 192, "Inception-A"),
        ("inception_b", 4, 768, "Inception-B"),
        ("inception_c", 2, 1280, "Inception-C"),
    ]
    
    last_output = "stem_pool2"
    
    for prefix, num_blocks, in_channels, group in inception_configs:
        for block_idx in range(1, num_blocks + 1):
            block_name = f"{prefix}{block_idx}"
            
            # Each inception block has 4 parallel branches
            branch_outputs = []
            
            for branch_idx in range(1, 5):
                branch_name = f"{block_name}_b{branch_idx}"
                
                # Simplified: each branch is one layer
                branch_exec_time = 2.0 + np.random.uniform(0, 2)
                branch_output = in_channels // 4 * 35 * 35 * 4
                
                branch_node = LayerNode(
                    name=branch_name,
                    layer_type="SGXConvBase",
                    group=group,
                    exec_time_enclave=branch_exec_time,
                    exec_time_cpu=branch_exec_time * 0.8,
                    output_bytes=branch_output,
                    predecessors=[last_output],
                )
                dag.add_node(branch_node)
                branch_outputs.append(branch_name)
            
            # Concatenate layer
            concat_name = f"{block_name}_concat"
            concat_node = LayerNode(
                name=concat_name,
                layer_type="SecretConcatenateLayer",
                group=group,
                exec_time_enclave=0.5,
                exec_time_cpu=0.3,
                output_bytes=in_channels * 35 * 35 * 4,
                predecessors=branch_outputs,
            )
            dag.add_node(concat_node)
            last_output = concat_name
        
        # Add reduction layer between inception stages (except after last)
        if group != "Inception-C":
            reduction_name = f"reduction_{group[-1].lower()}"
            reduction_node = LayerNode(
                name=reduction_name,
                layer_type="SGXConvBase",
                group=group.replace("Inception", "Reduction"),
                exec_time_enclave=5.0,
                exec_time_cpu=4.0,
                output_bytes=in_channels * 2 * 17 * 17 * 4,
                predecessors=[last_output],
            )
            dag.add_node(reduction_node)
            last_output = reduction_name
    
    # Classifier layers
    classifier_layers = [
        ("avgpool", "SecretAvgpool2dLayer", 0.5, 1*2048*1*1*4),
        ("flatten", "SecretFlattenLayer", 0.1, 1*2048*4),
        ("fc", "SGXLinearBase", 2.0, 1*1000*4),
        ("output", "SecretOutputLayer", 0.1, 1*1000*4),
    ]
    
    for name, ltype, exec_time, output_bytes in classifier_layers:
        node = LayerNode(
            name=name,
            layer_type=ltype,
            group="Classifier",
            exec_time_enclave=exec_time,
            exec_time_cpu=exec_time * 0.8,
            output_bytes=output_bytes,
            predecessors=[last_output],
        )
        dag.add_node(node)
        last_output = name
    
    # Build successor lists
    for name, node in dag.nodes.items():
        for pred_name in node.predecessors:
            if pred_name in dag.nodes:
                dag.nodes[pred_name].add_successor(name)
    
    return dag


def load_dag_from_data(data_path: str) -> Optional[InceptionDAG]:
    """Load DAG from measurement data file."""
    if not os.path.exists(data_path):
        return None
    
    if data_path.endswith('.json'):
        return InceptionDAG.build_from_json(data_path)
    elif data_path.endswith('.csv'):
        return InceptionDAG.build_from_csv(data_path)
    
    return None


def run_node_scaling_experiment(
    dag: InceptionDAG,
    node_counts: List[int],
    bandwidth_mbps: float,
    schedulers: List[str],
    output_dir: str
) -> Dict[str, Any]:
    """
    Experiment 1: Speedup vs Number of Nodes
    
    Measures how speedup changes with increasing number of nodes
    for different scheduling strategies.
    """
    print("\n" + "="*60)
    print("Experiment 1: Speedup vs Number of Nodes")
    print("="*60)
    
    results = {
        'experiment': 'node_scaling',
        'bandwidth_mbps': bandwidth_mbps,
        'timestamp': datetime.now().isoformat(),
        'data': []
    }
    
    config = CostConfig(
        network_bandwidth_mbps=bandwidth_mbps,
        enclave_init_ms=100,
    )
    
    all_schedulers = get_all_schedulers(dag)
    
    for num_nodes in node_counts:
        print(f"\nNodes: {num_nodes}")
        
        for scheduler_name in schedulers:
            if scheduler_name not in all_schedulers:
                continue
            
            strategy = all_schedulers[scheduler_name]
            sim = DistributedSimulator(dag, num_nodes, config)
            result = sim.simulate(strategy.get_scheduler())
            
            entry = {
                'num_nodes': num_nodes,
                'scheduler': scheduler_name,
                'makespan_ms': result['makespan_ms'],
                'serial_time_ms': result['serial_time_ms'],
                'speedup': result['speedup'],
                'efficiency': result['efficiency'],
                'avg_utilization': result['avg_utilization'],
                'total_compute_ms': result['total_compute_ms'],
                'total_transfer_ms': result['total_transfer_ms'],
            }
            results['data'].append(entry)
            
            print(f"  {scheduler_name}: Speedup={result['speedup']:.2f}x, "
                  f"Efficiency={result['efficiency']:.1%}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'node_scaling.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    return results


def run_bandwidth_experiment(
    dag: InceptionDAG,
    num_nodes: int,
    bandwidths: List[float],
    scheduler_name: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Experiment 2: Speedup vs Network Bandwidth
    
    Measures how speedup changes with different network bandwidths.
    """
    print("\n" + "="*60)
    print("Experiment 2: Speedup vs Network Bandwidth")
    print("="*60)
    
    results = {
        'experiment': 'bandwidth_scaling',
        'num_nodes': num_nodes,
        'scheduler': scheduler_name,
        'timestamp': datetime.now().isoformat(),
        'data': []
    }
    
    all_schedulers = get_all_schedulers(dag)
    strategy = all_schedulers.get(scheduler_name, ASAPScheduler())
    
    for bandwidth in bandwidths:
        config = CostConfig(
            network_bandwidth_mbps=bandwidth,
            enclave_init_ms=100,
        )
        
        sim = DistributedSimulator(dag, num_nodes, config)
        result = sim.simulate(strategy.get_scheduler())
        
        entry = {
            'bandwidth_mbps': bandwidth,
            'makespan_ms': result['makespan_ms'],
            'serial_time_ms': result['serial_time_ms'],
            'speedup': result['speedup'],
            'efficiency': result['efficiency'],
            'total_compute_ms': result['total_compute_ms'],
            'total_transfer_ms': result['total_transfer_ms'],
            'transfer_ratio': result['total_transfer_ms'] / result['makespan_ms'] if result['makespan_ms'] > 0 else 0,
        }
        results['data'].append(entry)
        
        print(f"Bandwidth {bandwidth} Mbps: Speedup={result['speedup']:.2f}x, "
              f"Transfer ratio={entry['transfer_ratio']:.1%}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'bandwidth_scaling.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    return results


def run_scheduler_comparison(
    dag: InceptionDAG,
    num_nodes: int,
    bandwidth_mbps: float,
    output_dir: str
) -> Dict[str, Any]:
    """
    Experiment 3: Scheduler Comparison
    
    Compares all scheduling strategies for a fixed configuration.
    """
    print("\n" + "="*60)
    print("Experiment 3: Scheduler Comparison")
    print("="*60)
    
    results = {
        'experiment': 'scheduler_comparison',
        'num_nodes': num_nodes,
        'bandwidth_mbps': bandwidth_mbps,
        'timestamp': datetime.now().isoformat(),
        'data': []
    }
    
    config = CostConfig(
        network_bandwidth_mbps=bandwidth_mbps,
        enclave_init_ms=100,
    )
    
    all_schedulers = get_all_schedulers(dag)
    
    for scheduler_name, strategy in all_schedulers.items():
        sim = DistributedSimulator(dag, num_nodes, config)
        result = sim.simulate(strategy.get_scheduler())
        
        entry = {
            'scheduler': scheduler_name,
            'makespan_ms': result['makespan_ms'],
            'serial_time_ms': result['serial_time_ms'],
            'speedup': result['speedup'],
            'efficiency': result['efficiency'],
            'avg_utilization': result['avg_utilization'],
            'node_utilizations': result['node_utilizations'],
        }
        results['data'].append(entry)
        
        print(f"{scheduler_name:20} Speedup={result['speedup']:.2f}x, "
              f"Efficiency={result['efficiency']:.1%}, "
              f"Utilization={result['avg_utilization']:.1%}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'scheduler_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    return results


def run_all_experiments(
    dag: InceptionDAG,
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> Dict[str, Any]:
    """Run all experiments and collect results."""
    print("\n" + "="*60)
    print("Running All Distributed Inference Experiments")
    print("="*60)
    
    all_results = {
        'model': dag.model_name,
        'num_layers': len(dag.nodes),
        'timestamp': datetime.now().isoformat(),
        'experiments': {}
    }
    
    # Get DAG statistics
    stats = dag.get_parallelism_stats()
    all_results['dag_stats'] = stats
    print(f"\nDAG Statistics:")
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Levels: {stats['num_levels']}")
    print(f"  Max parallelism: {stats['max_parallelism']}")
    print(f"  Theoretical speedup: {stats['theoretical_speedup']:.2f}x")
    
    # Experiment 1: Node scaling
    all_results['experiments']['node_scaling'] = run_node_scaling_experiment(
        dag, DEFAULT_NODE_COUNTS, 1000, DEFAULT_SCHEDULERS, output_dir
    )
    
    # Experiment 2: Bandwidth scaling
    all_results['experiments']['bandwidth_scaling'] = run_bandwidth_experiment(
        dag, 8, DEFAULT_BANDWIDTHS, 'HEFT', output_dir
    )
    
    # Experiment 3: Scheduler comparison
    all_results['experiments']['scheduler_comparison'] = run_scheduler_comparison(
        dag, 8, 1000, output_dir
    )
    
    # Save combined results
    combined_path = os.path.join(output_dir, 'all_experiments.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run distributed inference experiments'
    )
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='Path to measurement data file (CSV or JSON)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--use-sample', action='store_true',
        help='Use sample DAG instead of measurement data'
    )
    parser.add_argument(
        '--experiment', type=str, default='all',
        choices=['all', 'node_scaling', 'bandwidth', 'scheduler'],
        help='Which experiment to run'
    )
    parser.add_argument(
        '--nodes', type=int, nargs='+', default=DEFAULT_NODE_COUNTS,
        help='Number of nodes to test'
    )
    parser.add_argument(
        '--bandwidth', type=float, default=1000,
        help='Network bandwidth in Mbps'
    )
    
    args = parser.parse_args()
    
    # Load or create DAG
    dag = None
    if args.data_path and not args.use_sample:
        dag = load_dag_from_data(args.data_path)
        if dag:
            print(f"Loaded DAG from {args.data_path}")
    
    if dag is None:
        print("Using sample DAG (no measurement data available)")
        dag = create_sample_dag()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments
    if args.experiment == 'all':
        run_all_experiments(dag, args.output_dir)
    elif args.experiment == 'node_scaling':
        run_node_scaling_experiment(
            dag, args.nodes, args.bandwidth, DEFAULT_SCHEDULERS, args.output_dir
        )
    elif args.experiment == 'bandwidth':
        run_bandwidth_experiment(
            dag, args.nodes[0] if args.nodes else 8, DEFAULT_BANDWIDTHS, 'HEFT', args.output_dir
        )
    elif args.experiment == 'scheduler':
        run_scheduler_comparison(
            dag, args.nodes[0] if args.nodes else 8, args.bandwidth, args.output_dir
        )


if __name__ == '__main__':
    main()

