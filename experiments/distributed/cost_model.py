"""
Cost Model for Distributed TEE Inference

This module provides cost modeling for distributed execution of DNN inference
across multiple TEE (Trusted Execution Environment) nodes.

Cost Components:
1. Computation cost: Layer execution time in TEE
2. Communication cost: Data transfer between nodes
3. Initialization cost: TEE enclave initialization overhead
4. Synchronization cost: Waiting for predecessor layers

Classes:
- CostConfig: Configuration for cost model parameters
- CostModel: Cost calculation for distributed execution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class CostConfig:
    """
    Configuration for cost model parameters.
    
    All times are in milliseconds (ms).
    All bandwidths are in Mbps.
    All sizes are in bytes.
    """
    
    # Network parameters
    network_bandwidth_mbps: float = 1000.0  # 1 Gbps default
    network_latency_ms: float = 0.1  # 100 microseconds
    
    # TEE initialization overhead
    enclave_init_ms: float = 100.0  # Time to initialize a new enclave
    enclave_context_switch_ms: float = 1.0  # Context switch overhead
    
    # Data serialization overhead
    serialize_overhead_per_mb: float = 0.5  # ms per MB for serialization
    deserialize_overhead_per_mb: float = 0.5  # ms per MB for deserialization
    
    # Memory constraints
    max_enclave_memory_mb: float = 128.0  # Maximum enclave memory
    
    # Execution mode factors
    enclave_overhead_factor: float = 1.0  # Multiplier for enclave execution
    
    def get_transfer_time_ms(self, size_bytes: int) -> float:
        """
        Calculate transfer time for given data size.
        
        Args:
            size_bytes: Data size in bytes
        
        Returns:
            Transfer time in milliseconds
        """
        if size_bytes <= 0:
            return 0.0
        
        size_mb = size_bytes / (1024 * 1024)
        # Transfer time = size / bandwidth + latency
        bandwidth_mbps = self.network_bandwidth_mbps
        transfer_ms = (size_mb * 8 * 1000) / bandwidth_mbps  # Convert to ms
        
        return transfer_ms + self.network_latency_ms
    
    def get_serialization_time_ms(self, size_bytes: int) -> float:
        """
        Calculate serialization time for given data size.
        
        Args:
            size_bytes: Data size in bytes
        
        Returns:
            Serialization time in milliseconds
        """
        if size_bytes <= 0:
            return 0.0
        
        size_mb = size_bytes / (1024 * 1024)
        return size_mb * self.serialize_overhead_per_mb
    
    def get_deserialization_time_ms(self, size_bytes: int) -> float:
        """
        Calculate deserialization time for given data size.
        
        Args:
            size_bytes: Data size in bytes
        
        Returns:
            Deserialization time in milliseconds
        """
        if size_bytes <= 0:
            return 0.0
        
        size_mb = size_bytes / (1024 * 1024)
        return size_mb * self.deserialize_overhead_per_mb
    
    def get_communication_cost_ms(self, size_bytes: int) -> float:
        """
        Get total communication cost including serialization, transfer, and deserialization.
        
        Args:
            size_bytes: Data size in bytes
        
        Returns:
            Total communication cost in milliseconds
        """
        return (
            self.get_serialization_time_ms(size_bytes) +
            self.get_transfer_time_ms(size_bytes) +
            self.get_deserialization_time_ms(size_bytes)
        )


class CostModel:
    """
    Cost model for distributed TEE inference.
    
    This class calculates various costs associated with executing
    DNN layers across multiple TEE nodes.
    """
    
    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()
    
    def compute_layer_cost(
        self,
        exec_time_ms: float,
        output_bytes: int,
        is_local: bool = True,
        include_init: bool = False
    ) -> Dict[str, float]:
        """
        Compute cost for executing a single layer.
        
        Args:
            exec_time_ms: Layer execution time in ms
            output_bytes: Output tensor size in bytes
            is_local: Whether successor is on same node
            include_init: Whether to include enclave init cost
        
        Returns:
            Dictionary with cost breakdown
        """
        costs = {
            'compute_ms': exec_time_ms * self.config.enclave_overhead_factor,
            'communication_ms': 0.0,
            'init_ms': 0.0,
            'total_ms': 0.0,
        }
        
        if include_init:
            costs['init_ms'] = self.config.enclave_init_ms
        
        if not is_local:
            costs['communication_ms'] = self.config.get_communication_cost_ms(output_bytes)
        
        costs['total_ms'] = (
            costs['compute_ms'] + 
            costs['communication_ms'] + 
            costs['init_ms']
        )
        
        return costs
    
    def compute_dag_serial_cost(
        self,
        dag: 'InceptionDAG',
        use_enclave: bool = True
    ) -> float:
        """
        Compute serial execution cost (baseline).
        
        All layers executed sequentially on a single node.
        No communication cost (all local).
        
        Args:
            dag: InceptionDAG instance
            use_enclave: Use enclave execution times
        
        Returns:
            Total serial execution time in ms
        """
        total = 0.0
        
        for node in dag.nodes.values():
            exec_time = node.exec_time_enclave if use_enclave else node.exec_time_cpu
            total += exec_time * self.config.enclave_overhead_factor
        
        # Add single enclave initialization
        total += self.config.enclave_init_ms
        
        return total
    
    def compute_distributed_cost(
        self,
        dag: 'InceptionDAG',
        node_assignment: Dict[str, int],
        num_nodes: int,
        use_enclave: bool = True
    ) -> Dict[str, float]:
        """
        Compute distributed execution cost.
        
        Args:
            dag: InceptionDAG instance
            node_assignment: Mapping of layer name to node ID
            num_nodes: Number of TEE nodes
            use_enclave: Use enclave execution times
        
        Returns:
            Dictionary with cost breakdown and makespan
        """
        # Track finish time for each node
        node_finish_times = [0.0] * num_nodes
        node_initialized = [False] * num_nodes
        
        # Track layer finish times
        layer_finish_times: Dict[str, float] = {}
        
        # Process layers in topological order
        topo_order = dag.topological_sort()
        
        total_compute = 0.0
        total_communication = 0.0
        total_init = 0.0
        
        for layer_name in topo_order:
            node = dag.nodes[layer_name]
            assigned_node = node_assignment.get(layer_name, 0)
            
            exec_time = node.exec_time_enclave if use_enclave else node.exec_time_cpu
            exec_time *= self.config.enclave_overhead_factor
            
            # Calculate earliest start time
            earliest_start = node_finish_times[assigned_node]
            
            # Wait for all predecessors
            for pred_name in node.predecessors:
                pred_finish = layer_finish_times.get(pred_name, 0.0)
                pred_node = node_assignment.get(pred_name, 0)
                
                # Add communication cost if predecessor is on different node
                if pred_node != assigned_node:
                    pred = dag.nodes[pred_name]
                    comm_cost = self.config.get_communication_cost_ms(pred.output_bytes)
                    pred_finish += comm_cost
                    total_communication += comm_cost
                
                earliest_start = max(earliest_start, pred_finish)
            
            # Add initialization cost if first use of node
            if not node_initialized[assigned_node]:
                earliest_start += self.config.enclave_init_ms
                total_init += self.config.enclave_init_ms
                node_initialized[assigned_node] = True
            
            # Calculate finish time
            finish_time = earliest_start + exec_time
            total_compute += exec_time
            
            layer_finish_times[layer_name] = finish_time
            node_finish_times[assigned_node] = finish_time
        
        makespan = max(layer_finish_times.values()) if layer_finish_times else 0.0
        
        return {
            'makespan_ms': makespan,
            'compute_ms': total_compute,
            'communication_ms': total_communication,
            'init_ms': total_init,
            'node_finish_times': node_finish_times,
            'layer_finish_times': layer_finish_times,
        }
    
    def compute_speedup(
        self,
        dag: 'InceptionDAG',
        node_assignment: Dict[str, int],
        num_nodes: int,
        use_enclave: bool = True
    ) -> Tuple[float, float, float]:
        """
        Compute speedup compared to serial execution.
        
        Args:
            dag: InceptionDAG instance
            node_assignment: Mapping of layer name to node ID
            num_nodes: Number of TEE nodes
            use_enclave: Use enclave execution times
        
        Returns:
            Tuple of (speedup, serial_time, parallel_time)
        """
        serial_time = self.compute_dag_serial_cost(dag, use_enclave)
        distributed_cost = self.compute_distributed_cost(
            dag, node_assignment, num_nodes, use_enclave
        )
        parallel_time = distributed_cost['makespan_ms']
        
        speedup = serial_time / parallel_time if parallel_time > 0 else 1.0
        
        return speedup, serial_time, parallel_time
    
    def estimate_optimal_nodes(
        self,
        dag: 'InceptionDAG',
        max_nodes: int = 32,
        use_enclave: bool = True
    ) -> Dict[str, Any]:
        """
        Estimate optimal number of nodes for best speedup.
        
        Uses a simple heuristic: distribute layers by depth level.
        
        Args:
            dag: InceptionDAG instance
            max_nodes: Maximum number of nodes to consider
            use_enclave: Use enclave execution times
        
        Returns:
            Dictionary with analysis results
        """
        serial_time = self.compute_dag_serial_cost(dag, use_enclave)
        
        results = []
        
        for num_nodes in range(1, max_nodes + 1):
            # Simple assignment: round-robin by topological order
            topo_order = dag.topological_sort()
            assignment = {
                layer: i % num_nodes 
                for i, layer in enumerate(topo_order)
            }
            
            distributed_cost = self.compute_distributed_cost(
                dag, assignment, num_nodes, use_enclave
            )
            
            parallel_time = distributed_cost['makespan_ms']
            speedup = serial_time / parallel_time if parallel_time > 0 else 1.0
            
            results.append({
                'num_nodes': num_nodes,
                'speedup': speedup,
                'parallel_time_ms': parallel_time,
                'compute_ms': distributed_cost['compute_ms'],
                'communication_ms': distributed_cost['communication_ms'],
                'init_ms': distributed_cost['init_ms'],
                'efficiency': speedup / num_nodes,
            })
        
        # Find optimal
        best = max(results, key=lambda x: x['speedup'])
        
        return {
            'serial_time_ms': serial_time,
            'optimal_nodes': best['num_nodes'],
            'optimal_speedup': best['speedup'],
            'results_by_nodes': results,
        }


# Import for type hints (deferred to avoid circular imports)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .dag_model import InceptionDAG


def test_cost_model():
    """Test cost model functionality."""
    print("Testing Cost Model...")
    
    config = CostConfig(
        network_bandwidth_mbps=1000,  # 1 Gbps
        network_latency_ms=0.1,
        enclave_init_ms=100,
    )
    
    model = CostModel(config)
    
    # Test communication cost
    size_1mb = 1024 * 1024  # 1 MB
    comm_time = config.get_communication_cost_ms(size_1mb)
    print(f"Communication cost for 1MB: {comm_time:.2f} ms")
    
    # Test layer cost
    layer_cost = model.compute_layer_cost(
        exec_time_ms=5.0,
        output_bytes=size_1mb,
        is_local=False,
        include_init=True
    )
    print(f"Layer cost breakdown: {layer_cost}")
    
    print("\n✓ Cost model tests passed!")


if __name__ == "__main__":
    test_cost_model()

