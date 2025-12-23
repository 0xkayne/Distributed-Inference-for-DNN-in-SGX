"""
Visualization Tools for Distributed Inference Analysis

This module provides visualization tools for analyzing and presenting
the results of distributed TEE inference experiments.

Visualizations:
1. DAG Structure Graph
2. Speedup vs Nodes Curve
3. Scheduler Comparison Bar Chart
4. Gantt Chart for Schedule Visualization
5. Node Utilization Heatmap
6. Communication vs Computation Breakdown
"""

import sys
import os
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

sys.path.insert(0, '.')

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization disabled.")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from experiments.distributed.dag_model import InceptionDAG


# Style configuration
COLORS = {
    'ASAP': '#1f77b4',
    'HEFT': '#ff7f0e',
    'CriticalPathFirst': '#2ca02c',
    'LoadBalancing': '#d62728',
    'RoundRobin': '#9467bd',
    'MinCommunication': '#8c564b',
}

LAYER_COLORS = {
    'SGXConvBase': '#4CAF50',
    'SecretReLULayer': '#2196F3',
    'SecretMaxpool2dLayer': '#FF9800',
    'SecretAvgpool2dLayer': '#FF9800',
    'SecretConcatenateLayer': '#9C27B0',
    'SGXLinearBase': '#E91E63',
    'SecretInputLayer': '#607D8B',
    'SecretOutputLayer': '#795548',
    'SecretFlattenLayer': '#00BCD4',
}


def ensure_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")


def plot_dag_structure(
    dag: InceptionDAG,
    output_path: str,
    figsize: Tuple[int, int] = (20, 16),
    show_labels: bool = True
) -> None:
    """
    Plot the DAG structure as a network graph.
    
    Args:
        dag: InceptionDAG to visualize
        output_path: Path to save the figure
        figsize: Figure size
        show_labels: Whether to show node labels
    """
    ensure_matplotlib()
    
    if not HAS_NETWORKX:
        print("Warning: networkx not available. Skipping DAG visualization.")
        return
    
    # Create networkx graph
    G = nx.DiGraph()
    
    for name, node in dag.nodes.items():
        G.add_node(name, 
                   layer_type=node.layer_type,
                   group=node.group,
                   exec_time=node.exec_time_enclave)
        
        for succ in node.successors:
            G.add_edge(name, succ)
    
    # Compute layout
    try:
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
    except:
        # Fallback to spring layout
        pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color nodes by layer type
    node_colors = []
    for name in G.nodes():
        layer_type = dag.nodes[name].layer_type
        color = LAYER_COLORS.get(layer_type, '#CCCCCC')
        node_colors.append(color)
    
    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=300, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=10, alpha=0.6, ax=ax)
    
    if show_labels:
        # Simplified labels
        labels = {name: name.split('_')[-1][:8] for name in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)
    
    # Add legend
    legend_handles = []
    for layer_type, color in LAYER_COLORS.items():
        if any(dag.nodes[n].layer_type == layer_type for n in dag.nodes):
            patch = mpatches.Patch(color=color, label=layer_type)
            legend_handles.append(patch)
    
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8)
    
    ax.set_title(f'{dag.model_name} DAG Structure\n({len(dag.nodes)} layers)', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ DAG structure saved to {output_path}")


def plot_speedup_vs_nodes(
    results: Dict[str, Any],
    output_path: str,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot speedup as a function of number of nodes.
    
    Args:
        results: Results from node_scaling experiment
        output_path: Path to save the figure
        figsize: Figure size
    """
    ensure_matplotlib()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    data = results.get('data', [])
    
    # Group by scheduler
    schedulers = {}
    for entry in data:
        scheduler = entry['scheduler']
        if scheduler not in schedulers:
            schedulers[scheduler] = {'nodes': [], 'speedup': [], 'efficiency': []}
        schedulers[scheduler]['nodes'].append(entry['num_nodes'])
        schedulers[scheduler]['speedup'].append(entry['speedup'])
        schedulers[scheduler]['efficiency'].append(entry['efficiency'])
    
    # Plot speedup
    for scheduler, values in schedulers.items():
        color = COLORS.get(scheduler, '#666666')
        ax1.plot(values['nodes'], values['speedup'], 
                marker='o', label=scheduler, color=color, linewidth=2)
    
    # Add ideal speedup line
    max_nodes = max(entry['num_nodes'] for entry in data) if data else 32
    ax1.plot([1, max_nodes], [1, max_nodes], 
            'k--', alpha=0.5, label='Ideal (linear)')
    
    ax1.set_xlabel('Number of TEE Nodes')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup vs Number of Nodes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Plot efficiency
    for scheduler, values in schedulers.items():
        color = COLORS.get(scheduler, '#666666')
        ax2.plot(values['nodes'], [e * 100 for e in values['efficiency']], 
                marker='s', label=scheduler, color=color, linewidth=2)
    
    ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Ideal (100%)')
    ax2.set_xlabel('Number of TEE Nodes')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_title('Parallel Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Speedup plot saved to {output_path}")


def plot_bandwidth_impact(
    results: Dict[str, Any],
    output_path: str,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot the impact of network bandwidth on speedup.
    
    Args:
        results: Results from bandwidth_scaling experiment
        output_path: Path to save the figure
        figsize: Figure size
    """
    ensure_matplotlib()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    data = results.get('data', [])
    
    bandwidths = [entry['bandwidth_mbps'] for entry in data]
    speedups = [entry['speedup'] for entry in data]
    transfer_ratios = [entry.get('transfer_ratio', 0) * 100 for entry in data]
    
    # Speedup vs bandwidth
    ax1.plot(bandwidths, speedups, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Network Bandwidth (Mbps)')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup vs Network Bandwidth')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Transfer ratio vs bandwidth
    ax2.bar(range(len(bandwidths)), transfer_ratios, color='coral', alpha=0.7)
    ax2.set_xticks(range(len(bandwidths)))
    ax2.set_xticklabels([f'{b}' for b in bandwidths])
    ax2.set_xlabel('Network Bandwidth (Mbps)')
    ax2.set_ylabel('Communication Overhead (%)')
    ax2.set_title('Communication as % of Makespan')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Bandwidth impact plot saved to {output_path}")


def plot_scheduler_comparison(
    results: Dict[str, Any],
    output_path: str,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Plot scheduler comparison as bar charts.
    
    Args:
        results: Results from scheduler_comparison experiment
        output_path: Path to save the figure
        figsize: Figure size
    """
    ensure_matplotlib()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    data = results.get('data', [])
    
    schedulers = [entry['scheduler'] for entry in data]
    speedups = [entry['speedup'] for entry in data]
    efficiencies = [entry['efficiency'] * 100 for entry in data]
    utilizations = [entry['avg_utilization'] * 100 for entry in data]
    
    colors = [COLORS.get(s, '#666666') for s in schedulers]
    x = range(len(schedulers))
    
    # Speedup
    ax1.bar(x, speedups, color=colors, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(schedulers, rotation=45, ha='right')
    ax1.set_ylabel('Speedup')
    ax1.set_title('Speedup by Scheduler')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Efficiency
    ax2.bar(x, efficiencies, color=colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(schedulers, rotation=45, ha='right')
    ax2.set_ylabel('Efficiency (%)')
    ax2.set_title('Parallel Efficiency')
    ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Utilization
    ax3.bar(x, utilizations, color=colors, alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(schedulers, rotation=45, ha='right')
    ax3.set_ylabel('Avg Utilization (%)')
    ax3.set_title('Node Utilization')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Scheduler comparison saved to {output_path}")


def plot_gantt_chart(
    schedule: List[Dict[str, Any]],
    num_nodes: int,
    output_path: str,
    figsize: Tuple[int, int] = (14, 6),
    max_tasks: int = 50
) -> None:
    """
    Plot Gantt chart of the execution schedule.
    
    Args:
        schedule: List of task execution records
        num_nodes: Number of nodes
        output_path: Path to save the figure
        figsize: Figure size
        max_tasks: Maximum tasks to show (for readability)
    """
    ensure_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Limit tasks for readability
    if len(schedule) > max_tasks:
        print(f"  Showing first {max_tasks} of {len(schedule)} tasks")
        schedule = schedule[:max_tasks]
    
    # Plot tasks
    for task in schedule:
        node_id = task['node_id']
        start = task['start_time']
        duration = task['end_time'] - task['start_time']
        
        layer_type = task.get('type', 'Unknown')
        color = LAYER_COLORS.get(layer_type, '#CCCCCC')
        
        ax.barh(node_id, duration, left=start, height=0.6, 
               color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Configure axes
    ax.set_yticks(range(num_nodes))
    ax.set_yticklabels([f'Node {i}' for i in range(num_nodes)])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('TEE Node')
    ax.set_title('Execution Schedule (Gantt Chart)')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    legend_handles = []
    for layer_type, color in LAYER_COLORS.items():
        if any(task.get('type') == layer_type for task in schedule):
            patch = mpatches.Patch(color=color, label=layer_type)
            legend_handles.append(patch)
    
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Gantt chart saved to {output_path}")


def plot_node_utilization_heatmap(
    results: Dict[str, Any],
    output_path: str,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot node utilization as a heatmap across different configurations.
    
    Args:
        results: Results from node_scaling experiment
        output_path: Path to save the figure
        figsize: Figure size
    """
    ensure_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    data = results.get('data', [])
    
    # Get unique node counts and schedulers
    node_counts = sorted(set(entry['num_nodes'] for entry in data))
    schedulers = sorted(set(entry['scheduler'] for entry in data))
    
    # Build utilization matrix
    matrix = np.zeros((len(schedulers), len(node_counts)))
    
    for entry in data:
        i = schedulers.index(entry['scheduler'])
        j = node_counts.index(entry['num_nodes'])
        matrix[i, j] = entry.get('avg_utilization', 0) * 100
    
    # Plot heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Configure axes
    ax.set_xticks(range(len(node_counts)))
    ax.set_xticklabels(node_counts)
    ax.set_yticks(range(len(schedulers)))
    ax.set_yticklabels(schedulers)
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Scheduler')
    ax.set_title('Node Utilization (%)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Utilization (%)')
    
    # Add value annotations
    for i in range(len(schedulers)):
        for j in range(len(node_counts)):
            ax.text(j, i, f'{matrix[i, j]:.0f}%', 
                   ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Utilization heatmap saved to {output_path}")


def generate_all_visualizations(
    results_dir: str,
    output_dir: str,
    dag: Optional[InceptionDAG] = None
) -> None:
    """
    Generate all visualizations from experiment results.
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save visualizations
        dag: Optional DAG for structure visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    # DAG structure
    if dag:
        plot_dag_structure(dag, os.path.join(output_dir, 'dag_structure.png'))
    
    # Node scaling
    node_scaling_path = os.path.join(results_dir, 'node_scaling.json')
    if os.path.exists(node_scaling_path):
        with open(node_scaling_path, 'r') as f:
            results = json.load(f)
        plot_speedup_vs_nodes(results, os.path.join(output_dir, 'speedup_vs_nodes.png'))
        plot_node_utilization_heatmap(results, os.path.join(output_dir, 'utilization_heatmap.png'))
    
    # Bandwidth scaling
    bandwidth_path = os.path.join(results_dir, 'bandwidth_scaling.json')
    if os.path.exists(bandwidth_path):
        with open(bandwidth_path, 'r') as f:
            results = json.load(f)
        plot_bandwidth_impact(results, os.path.join(output_dir, 'bandwidth_impact.png'))
    
    # Scheduler comparison
    scheduler_path = os.path.join(results_dir, 'scheduler_comparison.json')
    if os.path.exists(scheduler_path):
        with open(scheduler_path, 'r') as f:
            results = json.load(f)
        plot_scheduler_comparison(results, os.path.join(output_dir, 'scheduler_comparison.png'))
    
    print(f"\n✓ All visualizations saved to {output_dir}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualizations')
    parser.add_argument(
        '--results-dir', type=str, default='experiments/data/distributed',
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--output-dir', type=str, default='experiments/figures/distributed',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--dag-json', type=str, default=None,
        help='Path to DAG JSON file for structure visualization'
    )
    
    args = parser.parse_args()
    
    # Load DAG if provided
    dag = None
    if args.dag_json and os.path.exists(args.dag_json):
        dag = InceptionDAG.build_from_json(args.dag_json)
    
    generate_all_visualizations(args.results_dir, args.output_dir, dag)


if __name__ == '__main__':
    main()

