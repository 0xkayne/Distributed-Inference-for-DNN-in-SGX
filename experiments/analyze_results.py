#!/usr/bin/env python3
"""
Analyze and visualize measurement results
分析和可视化测量结果
"""

import sys
sys.path.insert(0, '.')

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ResultAnalyzer:
    """Analyze measurement results"""
    
    def __init__(self, data_dir='experiments/data', output_dir='experiments/figures'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_json(self, filename):
        """Load JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found")
            return None
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def list_data_files(self):
        """List all data files"""
        files = list(Path(self.data_dir).glob('*.json'))
        return [f.name for f in files]
    
    def analyze_computation_cost(self, model_name):
        """
        Analyze computation cost for a model
        """
        print(f"\n{'='*60}")
        print(f"Analyzing Computation Cost: {model_name}")
        print(f"{'='*60}\n")
        
        # Load aggregated data
        filename = f'computation_cost_{model_name}_aggregated.json'
        data = self.load_json(filename)
        
        if not data:
            print(f"No data found for {model_name}")
            return
        
        # Extract data for plotting
        devices = list(data.get('devices', {}).keys())
        
        if not devices:
            print("No device data found")
            return
        
        # Plot layer-wise computation time
        fig, axes = plt.subplots(1, len(devices), figsize=(6*len(devices), 6))
        if len(devices) == 1:
            axes = [axes]
        
        for idx, device in enumerate(devices):
            device_data = data['devices'][device]
            
            # Get first batch size data
            batch_keys = [k for k in device_data.keys() if k.startswith('batch_')]
            if not batch_keys:
                continue
            
            batch_data = device_data[batch_keys[0]]
            layers = batch_data.get('layers', [])
            
            if not layers:
                continue
            
            # Extract layer names and times
            layer_names = [l.get('name', f"L{l.get('index', i)}") 
                          for i, l in enumerate(layers)]
            layer_times = [l.get('mean_ms', 0) for l in layers]
            
            # Plot
            ax = axes[idx]
            ax.bar(range(len(layer_times)), layer_times)
            ax.set_xlabel('Layer Index')
            ax.set_ylabel('Time (ms)')
            ax.set_title(f'{model_name} - {device}')
            ax.grid(True, alpha=0.3)
            
            # Add total time
            total_time = sum(layer_times)
            ax.text(0.02, 0.98, f'Total: {total_time:.2f}ms',
                   transform=ax.transAxes, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 
                                   f'{model_name}_computation_layerwise.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
        
        # Print summary
        print(f"\nSummary for {model_name}:")
        for device in devices:
            device_data = data['devices'][device]
            batch_keys = [k for k in device_data.keys() if k.startswith('batch_')]
            for batch_key in batch_keys:
                batch_data = device_data[batch_key]
                summary = batch_data.get('summary', {})
                print(f"  {device} {batch_key}:")
                print(f"    Total time: {summary.get('total_time_ms', 0):.2f}ms")
                print(f"    Total params: {summary.get('total_params', 0):,}")
                print(f"    Total memory: {summary.get('total_memory_mb', 0):.2f}MB")
    
    def analyze_communication_cost(self, model_name):
        """
        Analyze communication cost for a model
        """
        print(f"\n{'='*60}")
        print(f"Analyzing Communication Cost: {model_name}")
        print(f"{'='*60}\n")
        
        filename = f'communication_cost_{model_name}.json'
        data = self.load_json(filename)
        
        if not data:
            return
        
        layers = data.get('layers', [])
        if not layers:
            print("No layer data found")
            return
        
        # Extract data
        layer_indices = [l['layer_index'] for l in layers]
        data_sizes_mb = [l['output_size_mb'] for l in layers]
        
        # Get bandwidth keys
        if layers:
            bw_keys = list(layers[0].get('transfer_times', {}).keys())
        else:
            bw_keys = []
        
        # Plot 1: Data size per layer
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.bar(layer_indices, data_sizes_mb)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Data Size (MB)')
        ax1.set_title(f'{model_name} - Layer Output Size')
        ax1.grid(True, alpha=0.3)
        
        total_data = sum(data_sizes_mb)
        ax1.text(0.02, 0.98, f'Total: {total_data:.2f}MB',
                transform=ax1.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Communication cost for different bandwidths
        for bw_key in bw_keys:
            comm_costs = [l['total_comm_cost'][bw_key] for l in layers]
            ax2.plot(layer_indices, comm_costs, marker='o', label=bw_key)
        
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Communication Cost (ms)')
        ax2.set_title(f'{model_name} - Communication Cost vs Bandwidth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 
                                   f'{model_name}_communication.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
        
        # Print summary
        summary = data.get('summary', {})
        print(f"\nSummary for {model_name}:")
        print(f"  Total data transferred: {summary.get('total_data_mb', 0):.2f}MB")
        for bw_key, cost in summary.get('total_comm_cost', {}).items():
            print(f"  Total comm cost ({bw_key}): {cost:.2f}ms")
    
    def analyze_security_overhead(self, model_name):
        """
        Analyze security overhead for a model
        """
        print(f"\n{'='*60}")
        print(f"Analyzing Security Overhead: {model_name}")
        print(f"{'='*60}\n")
        
        filename = f'security_cost_{model_name}.json'
        data = self.load_json(filename)
        
        if not data:
            return
        
        if 'error' in data:
            print(f"Error in data: {data['error']}")
            return
        
        layers = data.get('layers', [])
        if not layers:
            print("No layer data found")
            return
        
        # Extract data
        layer_names = [l.get('layer_name', f"L{l['layer_index']}") for l in layers]
        cpu_times = [l['cpu_time_ms'] for l in layers]
        enclave_times = [l['enclave_time_ms'] for l in layers]
        overhead_percents = [l['overhead_percent'] for l in layers]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: CPU vs Enclave time
        x = np.arange(len(layer_names))
        width = 0.35
        
        ax1.bar(x - width/2, cpu_times, width, label='CPU', alpha=0.8)
        ax1.bar(x + width/2, enclave_times, width, label='Enclave', alpha=0.8)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title(f'{model_name} - CPU vs Enclave Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Overhead percentage
        ax2.bar(x, overhead_percents, color='coral', alpha=0.8)
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Overhead (%)')
        ax2.set_title(f'{model_name} - Security Overhead')
        ax2.axhline(y=np.mean(overhead_percents), color='r', linestyle='--', 
                   label=f'Average: {np.mean(overhead_percents):.1f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 
                                   f'{model_name}_security_overhead.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
        
        # Print summary
        summary = data.get('summary', {})
        print(f"\nSummary for {model_name}:")
        print(f"  CPU total: {summary.get('cpu_total_ms', 0):.2f}ms")
        print(f"  Enclave total: {summary.get('enclave_total_ms', 0):.2f}ms")
        print(f"  Total overhead: {summary.get('total_overhead_ms', 0):.2f}ms " +
              f"({summary.get('total_overhead_percent', 0):.1f}%)")
        print(f"  Average overhead: {summary.get('avg_overhead_percent', 0):.1f}%")
    
    def analyze_all(self, model_name):
        """Analyze all measurement types for a model"""
        self.analyze_computation_cost(model_name)
        self.analyze_communication_cost(model_name)
        self.analyze_security_overhead(model_name)


def main():
    parser = argparse.ArgumentParser(description='Analyze measurement results')
    parser.add_argument('--model', type=str, default='NiN',
                       help='Model name to analyze')
    parser.add_argument('--type', type=str, default='all',
                       choices=['all', 'computation', 'communication', 'security'],
                       help='Type of analysis')
    parser.add_argument('--list', action='store_true',
                       help='List all available data files')
    
    args = parser.parse_args()
    
    analyzer = ResultAnalyzer()
    
    if args.list:
        print("\nAvailable data files:")
        files = analyzer.list_data_files()
        for f in files:
            print(f"  - {f}")
        return
    
    print(f"\nAnalyzing {args.model}...")
    
    if args.type == 'all':
        analyzer.analyze_all(args.model)
    elif args.type == 'computation':
        analyzer.analyze_computation_cost(args.model)
    elif args.type == 'communication':
        analyzer.analyze_communication_cost(args.model)
    elif args.type == 'security':
        analyzer.analyze_security_overhead(args.model)
    
    print(f"\n✓ Analysis complete. Figures saved to {analyzer.output_dir}/")


if __name__ == '__main__':
    main()

