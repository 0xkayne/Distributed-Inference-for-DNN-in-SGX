"""
DAG Model for Neural Network Layer Dependencies

This module provides data structures for representing neural network
models as Directed Acyclic Graphs (DAGs) for distributed inference analysis.

Classes:
- LayerNode: Represents a single layer in the DAG
- InceptionDAG: Complete DAG representation of an Inception-like model
"""

import sys
import csv
import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, deque
import numpy as np

sys.path.insert(0, '.')


@dataclass
class LayerNode:
    """
    Represents a single layer node in the DAG.
    
    Attributes:
        name: Unique layer name
        layer_type: Type of layer (e.g., 'SGXConvBase', 'SecretReLULayer')
        group: Group name for grouped execution
        
        exec_time_enclave: Execution time in Enclave mode (ms)
        exec_time_cpu: Execution time in CPU mode (ms)
        
        input_bytes: Input tensor size in bytes
        output_bytes: Output tensor size in bytes
        input_shape: Input tensor shape
        output_shape: Output tensor shape
        
        predecessors: List of predecessor layer names
        successors: List of successor layer names
        
        depth: Depth in the DAG (distance from input)
        critical_path_length: Length of critical path through this node
        is_on_critical_path: Whether this node is on the critical path
    """
    name: str
    layer_type: str = ""
    group: str = ""
    
    # Timing (in milliseconds)
    exec_time_enclave: float = 0.0
    exec_time_cpu: float = 0.0
    exec_time_enclave_std: float = 0.0
    exec_time_cpu_std: float = 0.0
    
    # Data sizes (in bytes)
    input_bytes: int = 0
    output_bytes: int = 0
    input_shape: List[int] = field(default_factory=list)
    output_shape: List[int] = field(default_factory=list)
    
    # Graph structure
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    
    # Analysis results (computed)
    depth: int = 0
    earliest_start: float = 0.0  # Earliest possible start time
    latest_start: float = float('inf')  # Latest possible start time
    critical_path_length: float = 0.0
    is_on_critical_path: bool = False
    
    def add_predecessor(self, pred_name: str):
        """Add a predecessor to this node."""
        if pred_name not in self.predecessors:
            self.predecessors.append(pred_name)
    
    def add_successor(self, succ_name: str):
        """Add a successor to this node."""
        if succ_name not in self.successors:
            self.successors.append(succ_name)
    
    def is_source(self) -> bool:
        """Check if this is a source node (no predecessors)."""
        return len(self.predecessors) == 0
    
    def is_sink(self) -> bool:
        """Check if this is a sink node (no successors)."""
        return len(self.successors) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'type': self.layer_type,
            'group': self.group,
            'exec_time_enclave': self.exec_time_enclave,
            'exec_time_cpu': self.exec_time_cpu,
            'exec_time_enclave_std': self.exec_time_enclave_std,
            'exec_time_cpu_std': self.exec_time_cpu_std,
            'input_bytes': self.input_bytes,
            'output_bytes': self.output_bytes,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'predecessors': self.predecessors,
            'successors': self.successors,
            'depth': self.depth,
            'earliest_start': self.earliest_start,
            'latest_start': self.latest_start,
            'critical_path_length': self.critical_path_length,
            'is_on_critical_path': self.is_on_critical_path,
        }


class InceptionDAG:
    """
    DAG representation of an Inception-like neural network model.
    
    This class provides methods to:
    - Build DAG from model definition or profiling data
    - Analyze graph structure (topological sort, critical path)
    - Find parallel execution opportunities
    - Compute scheduling bounds
    """
    
    def __init__(self, model_name: str = "InceptionV3"):
        self.model_name = model_name
        self.nodes: Dict[str, LayerNode] = {}
        self._topo_order: List[str] = []
        self._topo_order_valid = False
        self._critical_path: List[str] = []
        self._total_compute_time: float = 0.0
    
    def add_node(self, node: LayerNode) -> None:
        """Add a layer node to the DAG."""
        self.nodes[node.name] = node
        self._topo_order_valid = False
    
    def add_edge(self, from_name: str, to_name: str) -> None:
        """Add an edge from one layer to another."""
        if from_name in self.nodes and to_name in self.nodes:
            self.nodes[from_name].add_successor(to_name)
            self.nodes[to_name].add_predecessor(from_name)
            self._topo_order_valid = False
    
    def get_node(self, name: str) -> Optional[LayerNode]:
        """Get a node by name."""
        return self.nodes.get(name)
    
    def get_source_nodes(self) -> List[LayerNode]:
        """Get all source nodes (no predecessors)."""
        return [node for node in self.nodes.values() if node.is_source()]
    
    def get_sink_nodes(self) -> List[LayerNode]:
        """Get all sink nodes (no successors)."""
        return [node for node in self.nodes.values() if node.is_sink()]
    
    def topological_sort(self) -> List[str]:
        """
        Compute topological ordering of nodes.
        Returns list of layer names in topological order.
        """
        if self._topo_order_valid:
            return self._topo_order
        
        in_degree = {name: len(node.predecessors) for name, node in self.nodes.items()}
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            name = queue.popleft()
            result.append(name)
            
            for succ_name in self.nodes[name].successors:
                in_degree[succ_name] -= 1
                if in_degree[succ_name] == 0:
                    queue.append(succ_name)
        
        if len(result) != len(self.nodes):
            raise ValueError("DAG contains a cycle!")
        
        self._topo_order = result
        self._topo_order_valid = True
        return result
    
    def compute_depths(self) -> None:
        """Compute depth (level) for each node."""
        topo_order = self.topological_sort()
        
        for name in topo_order:
            node = self.nodes[name]
            if node.is_source():
                node.depth = 0
            else:
                max_pred_depth = max(
                    self.nodes[pred].depth for pred in node.predecessors
                )
                node.depth = max_pred_depth + 1
    
    def compute_earliest_start_times(self, use_enclave: bool = True) -> None:
        """
        Compute earliest possible start time for each node.
        
        Args:
            use_enclave: If True, use enclave execution times; otherwise use CPU times
        """
        topo_order = self.topological_sort()
        
        for name in topo_order:
            node = self.nodes[name]
            exec_time = node.exec_time_enclave if use_enclave else node.exec_time_cpu
            
            if node.is_source():
                node.earliest_start = 0.0
            else:
                # Earliest start = max(predecessor finish times)
                max_pred_finish = 0.0
                for pred_name in node.predecessors:
                    pred = self.nodes[pred_name]
                    pred_exec = pred.exec_time_enclave if use_enclave else pred.exec_time_cpu
                    pred_finish = pred.earliest_start + pred_exec
                    max_pred_finish = max(max_pred_finish, pred_finish)
                node.earliest_start = max_pred_finish
    
    def compute_latest_start_times(self, deadline: float, use_enclave: bool = True) -> None:
        """
        Compute latest possible start time for each node given a deadline.
        
        Args:
            deadline: Total execution time deadline
            use_enclave: If True, use enclave execution times; otherwise use CPU times
        """
        topo_order = self.topological_sort()
        
        # Process in reverse topological order
        for name in reversed(topo_order):
            node = self.nodes[name]
            exec_time = node.exec_time_enclave if use_enclave else node.exec_time_cpu
            
            if node.is_sink():
                node.latest_start = deadline - exec_time
            else:
                # Latest start = min(successor latest start) - this node's exec time
                min_succ_start = float('inf')
                for succ_name in node.successors:
                    succ = self.nodes[succ_name]
                    min_succ_start = min(min_succ_start, succ.latest_start)
                node.latest_start = min_succ_start - exec_time
    
    def compute_critical_path(self, use_enclave: bool = True) -> Tuple[List[str], float]:
        """
        Compute the critical path through the DAG.
        
        Args:
            use_enclave: If True, use enclave execution times; otherwise use CPU times
        
        Returns:
            Tuple of (critical path as list of node names, critical path length in ms)
        """
        self.compute_earliest_start_times(use_enclave)
        
        # Find the makespan (total execution time)
        makespan = 0.0
        for name, node in self.nodes.items():
            exec_time = node.exec_time_enclave if use_enclave else node.exec_time_cpu
            finish_time = node.earliest_start + exec_time
            makespan = max(makespan, finish_time)
        
        self.compute_latest_start_times(makespan, use_enclave)
        
        # Mark critical path nodes (where earliest_start == latest_start)
        tolerance = 1e-6
        for node in self.nodes.values():
            node.is_on_critical_path = abs(node.earliest_start - node.latest_start) < tolerance
        
        # Build critical path by following edges from source to sink
        critical_path = []
        
        # Find critical source
        for node in self.get_source_nodes():
            if node.is_on_critical_path:
                current = node.name
                break
        else:
            # Fallback: use node with earliest_start = 0
            for name, node in self.nodes.items():
                if node.earliest_start == 0 and node.is_on_critical_path:
                    current = name
                    break
        
        # Follow critical path
        while current:
            critical_path.append(current)
            node = self.nodes[current]
            
            # Find next node on critical path
            next_node = None
            for succ_name in node.successors:
                if self.nodes[succ_name].is_on_critical_path:
                    next_node = succ_name
                    break
            current = next_node
        
        self._critical_path = critical_path
        return critical_path, makespan
    
    def get_parallel_groups(self) -> List[List[str]]:
        """
        Group nodes that can execute in parallel (same depth level).
        
        Returns:
            List of lists, where each inner list contains node names
            that can execute in parallel.
        """
        self.compute_depths()
        
        depth_groups: Dict[int, List[str]] = defaultdict(list)
        for name, node in self.nodes.items():
            depth_groups[node.depth].append(name)
        
        # Sort by depth
        max_depth = max(depth_groups.keys()) if depth_groups else 0
        return [depth_groups[d] for d in range(max_depth + 1)]
    
    def get_total_serial_time(self, use_enclave: bool = True) -> float:
        """
        Get total serial execution time (sum of all layer times).
        
        Args:
            use_enclave: If True, use enclave times; otherwise use CPU times
        
        Returns:
            Total serial execution time in ms
        """
        total = 0.0
        for node in self.nodes.values():
            if use_enclave:
                total += node.exec_time_enclave
            else:
                total += node.exec_time_cpu
        return total
    
    def get_theoretical_speedup(self, use_enclave: bool = True) -> float:
        """
        Compute theoretical maximum speedup (serial time / critical path time).
        
        This represents the upper bound on parallelization benefit.
        
        Args:
            use_enclave: If True, use enclave times; otherwise use CPU times
        
        Returns:
            Theoretical maximum speedup
        """
        serial_time = self.get_total_serial_time(use_enclave)
        _, critical_path_time = self.compute_critical_path(use_enclave)
        
        if critical_path_time > 0:
            return serial_time / critical_path_time
        return 1.0
    
    def get_parallelism_stats(self) -> Dict[str, Any]:
        """
        Get statistics about parallelism opportunities in the DAG.
        
        Returns:
            Dictionary with parallelism statistics
        """
        parallel_groups = self.get_parallel_groups()
        _, critical_path_time = self.compute_critical_path(use_enclave=True)
        serial_time = self.get_total_serial_time(use_enclave=True)
        
        group_sizes = [len(g) for g in parallel_groups]
        
        return {
            'num_nodes': len(self.nodes),
            'num_levels': len(parallel_groups),
            'max_parallelism': max(group_sizes) if group_sizes else 0,
            'avg_parallelism': np.mean(group_sizes) if group_sizes else 0,
            'critical_path_length': len(self._critical_path),
            'critical_path_time_ms': critical_path_time,
            'serial_time_ms': serial_time,
            'theoretical_speedup': serial_time / critical_path_time if critical_path_time > 0 else 1.0,
            'parallel_groups': parallel_groups,
        }
    
    @classmethod
    def build_from_csv(cls, csv_path: str, model_name: str = "InceptionV3") -> 'InceptionDAG':
        """
        Build DAG from profiling CSV file.
        
        Args:
            csv_path: Path to the profiling CSV file
            model_name: Name of the model
        
        Returns:
            InceptionDAG instance
        """
        dag = cls(model_name)
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Parse dependencies
                deps = []
                if row.get('Dependencies'):
                    deps = [d.strip() for d in row['Dependencies'].split(';') if d.strip()]
                
                # Parse shapes
                input_shape = []
                output_shape = []
                try:
                    if row.get('InputShape'):
                        input_shape = eval(row['InputShape'])
                    if row.get('OutputShape'):
                        output_shape = eval(row['OutputShape'])
                except:
                    pass
                
                node = LayerNode(
                    name=row['LayerName'],
                    layer_type=row.get('Type', ''),
                    group=row.get('Group', ''),
                    exec_time_enclave=float(row.get('EnclaveTime_mean', 0)),
                    exec_time_enclave_std=float(row.get('EnclaveTime_std', 0)),
                    exec_time_cpu=float(row.get('CPUTime_mean', 0)),
                    exec_time_cpu_std=float(row.get('CPUTime_std', 0)),
                    input_bytes=int(row.get('InputBytes', 0)),
                    output_bytes=int(row.get('OutputBytes', 0)),
                    input_shape=input_shape,
                    output_shape=output_shape,
                    predecessors=deps,
                )
                dag.add_node(node)
        
        # Build successor lists from predecessor lists
        for name, node in dag.nodes.items():
            for pred_name in node.predecessors:
                if pred_name in dag.nodes:
                    dag.nodes[pred_name].add_successor(name)
        
        return dag
    
    @classmethod
    def build_from_json(cls, json_path: str) -> 'InceptionDAG':
        """
        Build DAG from profiling JSON file.
        
        Args:
            json_path: Path to the profiling JSON file
        
        Returns:
            InceptionDAG instance
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        model_name = data.get('model', 'InceptionV3')
        dag = cls(model_name)
        
        for layer_data in data.get('layers', []):
            node = LayerNode(
                name=layer_data['name'],
                layer_type=layer_data.get('type', ''),
                group=layer_data.get('group', ''),
                exec_time_enclave=layer_data.get('enclave', {}).get('mean_ms', 0),
                exec_time_enclave_std=layer_data.get('enclave', {}).get('std_ms', 0),
                exec_time_cpu=layer_data.get('cpu', {}).get('mean_ms', 0),
                exec_time_cpu_std=layer_data.get('cpu', {}).get('std_ms', 0),
                input_bytes=layer_data.get('input_bytes', 0),
                output_bytes=layer_data.get('output_bytes', 0),
                input_shape=layer_data.get('input_shape', []),
                output_shape=layer_data.get('output_shape', []),
                predecessors=layer_data.get('dependencies', []),
            )
            dag.add_node(node)
        
        # Build successor lists from predecessor lists
        for name, node in dag.nodes.items():
            for pred_name in node.predecessors:
                if pred_name in dag.nodes:
                    dag.nodes[pred_name].add_successor(name)
        
        return dag
    
    @classmethod
    def build_from_model(cls, model) -> 'InceptionDAG':
        """
        Build DAG directly from an SGXInceptionV3 model instance.
        
        Args:
            model: SGXInceptionV3 model instance
        
        Returns:
            InceptionDAG instance
        """
        dag = cls(getattr(model, 'model_name', 'InceptionV3'))
        
        # Build nodes from model layers
        for layer in model.layers:
            layer_name = layer.LayerName
            layer_type = type(layer).__name__
            
            # Determine group
            group = "Unknown"
            from experiments.models.profile_inception import get_layer_group
            detected_group = get_layer_group(layer_name)
            if detected_group:
                group = detected_group
            
            # Get shapes
            input_shape = []
            output_shape = []
            if hasattr(layer, 'InputShape') and layer.InputShape:
                input_shape = list(layer.InputShape)
            if hasattr(layer, 'get_output_shape'):
                try:
                    output_shape = list(layer.get_output_shape())
                except:
                    pass
            
            # Get dependencies
            predecessors = []
            if hasattr(layer, 'PrevLayer') and layer.PrevLayer is not None:
                if isinstance(layer.PrevLayer, list):
                    predecessors = [l.LayerName for l in layer.PrevLayer if hasattr(l, 'LayerName')]
                elif hasattr(layer.PrevLayer, 'LayerName'):
                    predecessors = [layer.PrevLayer.LayerName]
            
            node = LayerNode(
                name=layer_name,
                layer_type=layer_type,
                group=group,
                input_shape=input_shape,
                output_shape=output_shape,
                predecessors=predecessors,
            )
            dag.add_node(node)
        
        # Build successor lists
        for name, node in dag.nodes.items():
            for pred_name in node.predecessors:
                if pred_name in dag.nodes:
                    dag.nodes[pred_name].add_successor(name)
        
        return dag
    
    def to_json(self, filepath: str) -> None:
        """Save DAG to JSON file."""
        data = {
            'model': self.model_name,
            'num_nodes': len(self.nodes),
            'nodes': [node.to_dict() for node in self.nodes.values()],
            'statistics': self.get_parallelism_stats(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self) -> None:
        """Print a summary of the DAG."""
        stats = self.get_parallelism_stats()
        
        print(f"\n{'='*60}")
        print(f"DAG Summary: {self.model_name}")
        print(f"{'='*60}")
        print(f"Total nodes: {stats['num_nodes']}")
        print(f"Depth levels: {stats['num_levels']}")
        print(f"Max parallelism: {stats['max_parallelism']}")
        print(f"Avg parallelism: {stats['avg_parallelism']:.2f}")
        print(f"Critical path length: {stats['critical_path_length']} nodes")
        print(f"Critical path time: {stats['critical_path_time_ms']:.2f} ms")
        print(f"Serial execution time: {stats['serial_time_ms']:.2f} ms")
        print(f"Theoretical speedup: {stats['theoretical_speedup']:.2f}x")
        print(f"{'='*60}")
        
        # Print parallel groups
        print("\nParallel execution levels:")
        for i, group in enumerate(stats['parallel_groups']):
            print(f"  Level {i}: {len(group)} nodes")
            if len(group) <= 5:
                for name in group:
                    node = self.nodes[name]
                    print(f"    - {name} ({node.exec_time_enclave:.2f}ms)")
            else:
                print(f"    - {group[0]} ... {group[-1]}")


# Test function
def test_dag():
    """Test DAG functionality with a simple example."""
    print("Testing DAG Model...")
    
    dag = InceptionDAG("TestModel")
    
    # Create simple diamond DAG: A -> B, C -> D
    #     A
    #    / \
    #   B   C
    #    \ /
    #     D
    
    dag.add_node(LayerNode(name="A", layer_type="Input", exec_time_enclave=1.0))
    dag.add_node(LayerNode(name="B", layer_type="Conv", exec_time_enclave=3.0))
    dag.add_node(LayerNode(name="C", layer_type="Conv", exec_time_enclave=2.0))
    dag.add_node(LayerNode(name="D", layer_type="Concat", exec_time_enclave=1.0))
    
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")
    
    # Test topological sort
    topo = dag.topological_sort()
    print(f"Topological order: {topo}")
    assert topo[0] == "A" and topo[-1] == "D"
    
    # Test critical path
    cp, cp_time = dag.compute_critical_path()
    print(f"Critical path: {cp}")
    print(f"Critical path time: {cp_time} ms")
    assert cp_time == 5.0  # A(1) + B(3) + D(1) = 5
    
    # Test parallel groups
    groups = dag.get_parallel_groups()
    print(f"Parallel groups: {groups}")
    assert len(groups) == 3  # Level 0: A, Level 1: B,C, Level 2: D
    
    # Print summary
    dag.print_summary()
    
    print("\n✓ All DAG tests passed!")


if __name__ == "__main__":
    test_dag()

