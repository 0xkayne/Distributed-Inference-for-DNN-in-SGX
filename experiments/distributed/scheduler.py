"""
Scheduling Strategies for Distributed TEE Inference

This module provides various scheduling strategies for distributing
DNN inference tasks across multiple TEE nodes.

Strategies:
- ASAP (As Soon As Possible): Greedy scheduling
- Critical Path First: Prioritize critical path tasks
- Load Balancing: Balance load across nodes
- HEFT: Heterogeneous Earliest Finish Time
"""

from typing import Dict, List, Optional, Tuple, Callable, TYPE_CHECKING
from collections import defaultdict
import numpy as np

if TYPE_CHECKING:
    from .simulator import DistributedSimulator
    from .dag_model import InceptionDAG


class SchedulingStrategy:
    """Base class for scheduling strategies."""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
    
    def get_scheduler(self) -> Callable[['DistributedSimulator'], Optional[Tuple[str, int]]]:
        """Return the scheduler function for use with DistributedSimulator."""
        raise NotImplementedError
    
    def __str__(self) -> str:
        return self.name


class ASAPScheduler(SchedulingStrategy):
    """
    As Soon As Possible (ASAP) Scheduler.
    
    Greedy strategy that schedules tasks as soon as their dependencies
    are met and a node is available. Tasks are assigned to the first
    available node.
    """
    
    def __init__(self):
        super().__init__("ASAP")
    
    def get_scheduler(self) -> Callable[['DistributedSimulator'], Optional[Tuple[str, int]]]:
        def schedule(sim: 'DistributedSimulator') -> Optional[Tuple[str, int]]:
            ready = sim.get_ready_tasks()
            available = sim.get_available_nodes()
            
            if ready and available:
                # Assign first ready task to first available node
                return (ready[0], available[0])
            return None
        
        return schedule


class RoundRobinScheduler(SchedulingStrategy):
    """
    Round Robin Scheduler.
    
    Distributes tasks evenly across nodes in a round-robin fashion.
    """
    
    def __init__(self):
        super().__init__("RoundRobin")
        self._next_node = 0
    
    def get_scheduler(self) -> Callable[['DistributedSimulator'], Optional[Tuple[str, int]]]:
        def schedule(sim: 'DistributedSimulator') -> Optional[Tuple[str, int]]:
            ready = sim.get_ready_tasks()
            
            if not ready:
                return None
            
            # Find next available node starting from _next_node
            for _ in range(sim.num_nodes):
                node_id = self._next_node % sim.num_nodes
                self._next_node = (self._next_node + 1) % sim.num_nodes
                
                node = sim.nodes[node_id]
                if not node.is_busy or node.available_at <= sim.current_time:
                    return (ready[0], node_id)
            
            return None
        
        return schedule


class CriticalPathFirstScheduler(SchedulingStrategy):
    """
    Critical Path First Scheduler.
    
    Prioritizes tasks on the critical path to minimize makespan.
    Non-critical tasks are scheduled to balance load.
    """
    
    def __init__(self, dag: 'InceptionDAG'):
        super().__init__("CriticalPathFirst")
        self.dag = dag
        self._critical_tasks = set()
        self._initialize()
    
    def _initialize(self):
        """Pre-compute critical path information."""
        _, _ = self.dag.compute_critical_path(use_enclave=True)
        self._critical_tasks = {
            name for name, node in self.dag.nodes.items()
            if node.is_on_critical_path
        }
    
    def get_scheduler(self) -> Callable[['DistributedSimulator'], Optional[Tuple[str, int]]]:
        def schedule(sim: 'DistributedSimulator') -> Optional[Tuple[str, int]]:
            ready = sim.get_ready_tasks()
            available = sim.get_available_nodes()
            
            if not ready or not available:
                return None
            
            # Sort ready tasks: critical path tasks first, then by execution time
            def task_priority(task_name: str) -> Tuple[int, float]:
                is_critical = 0 if task_name in self._critical_tasks else 1
                exec_time = self.dag.nodes[task_name].exec_time_enclave
                return (is_critical, -exec_time)  # Negative for descending order
            
            ready_sorted = sorted(ready, key=task_priority)
            task = ready_sorted[0]
            
            # Assign to node with earliest availability
            best_node = min(available, key=lambda n: sim.nodes[n].available_at)
            
            return (task, best_node)
        
        return schedule


class LoadBalancingScheduler(SchedulingStrategy):
    """
    Load Balancing Scheduler.
    
    Attempts to balance the computational load across all nodes
    by considering both current load and task execution times.
    """
    
    def __init__(self, dag: 'InceptionDAG'):
        super().__init__("LoadBalancing")
        self.dag = dag
    
    def get_scheduler(self) -> Callable[['DistributedSimulator'], Optional[Tuple[str, int]]]:
        def schedule(sim: 'DistributedSimulator') -> Optional[Tuple[str, int]]:
            ready = sim.get_ready_tasks()
            
            if not ready:
                return None
            
            # Calculate current load on each node
            node_loads = []
            for node in sim.nodes:
                load = node.total_compute_time
                if node.is_busy and node.current_task:
                    # Add remaining time of current task
                    remaining = node.available_at - sim.current_time
                    load += max(0, remaining)
                node_loads.append(load)
            
            # Select task with longest execution time from ready tasks
            ready_sorted = sorted(
                ready,
                key=lambda t: self.dag.nodes[t].exec_time_enclave,
                reverse=True
            )
            task = ready_sorted[0]
            
            # Assign to least loaded available node
            available = sim.get_available_nodes()
            if available:
                best_node = min(available, key=lambda n: node_loads[n])
                return (task, best_node)
            
            return None
        
        return schedule


class HEFTScheduler(SchedulingStrategy):
    """
    Heterogeneous Earliest Finish Time (HEFT) Scheduler.
    
    A list scheduling algorithm that orders tasks by upward rank
    and assigns each task to the processor that minimizes its
    earliest finish time.
    
    Reference: Topcuoglu, H., Hariri, S., & Wu, M. Y. (2002).
    """
    
    def __init__(self, dag: 'InceptionDAG'):
        super().__init__("HEFT")
        self.dag = dag
        self._upward_ranks: Dict[str, float] = {}
        self._task_order: List[str] = []
        self._next_task_idx = 0
        self._initialize()
    
    def _initialize(self):
        """Pre-compute upward ranks and task ordering."""
        # Compute upward rank for each task
        # rank_u(n) = w(n) + max(c(n,m) + rank_u(m)) for all successors m
        
        topo_order = self.dag.topological_sort()
        
        # Process in reverse topological order
        for task_name in reversed(topo_order):
            node = self.dag.nodes[task_name]
            exec_time = node.exec_time_enclave
            
            if not node.successors:
                # Exit node: rank = execution time
                self._upward_ranks[task_name] = exec_time
            else:
                # rank = exec_time + max(comm + successor_rank)
                max_successor_cost = 0.0
                for succ_name in node.successors:
                    # Assume average communication cost
                    comm_cost = node.output_bytes / (1024 * 1024) * 8  # Rough estimate
                    succ_cost = comm_cost + self._upward_ranks[succ_name]
                    max_successor_cost = max(max_successor_cost, succ_cost)
                
                self._upward_ranks[task_name] = exec_time + max_successor_cost
        
        # Sort tasks by descending upward rank
        self._task_order = sorted(
            topo_order,
            key=lambda t: self._upward_ranks[t],
            reverse=True
        )
    
    def get_scheduler(self) -> Callable[['DistributedSimulator'], Optional[Tuple[str, int]]]:
        def schedule(sim: 'DistributedSimulator') -> Optional[Tuple[str, int]]:
            ready = sim.get_ready_tasks()
            
            if not ready:
                return None
            
            # Find highest priority ready task according to HEFT ordering
            for task_name in self._task_order:
                if task_name in ready:
                    # Find processor with earliest finish time
                    best_node = None
                    best_eft = float('inf')
                    
                    for node_id in range(sim.num_nodes):
                        node = sim.nodes[node_id]
                        
                        # Calculate earliest start time on this node
                        est = max(sim.current_time, node.available_at)
                        
                        # Add dependency constraints
                        layer = sim.dag.nodes[task_name]
                        for pred_name in layer.predecessors:
                            if pred_name in sim.task_executions:
                                pred_exec = sim.task_executions[pred_name]
                                pred_finish = pred_exec.end_time
                                
                                # Add communication if on different node
                                if sim.task_assignment.get(pred_name) != node_id:
                                    pred_layer = sim.dag.nodes[pred_name]
                                    comm = sim.cost_config.get_communication_cost_ms(
                                        pred_layer.output_bytes
                                    )
                                    pred_finish += comm
                                
                                est = max(est, pred_finish)
                        
                        # Add init cost if node not initialized
                        if not node.is_initialized:
                            est += sim.cost_config.enclave_init_ms
                        
                        # Calculate EFT
                        exec_time = layer.exec_time_enclave
                        eft = est + exec_time
                        
                        if eft < best_eft:
                            best_eft = eft
                            best_node = node_id
                    
                    if best_node is not None:
                        return (task_name, best_node)
            
            return None
        
        return schedule


class MinCommunicationScheduler(SchedulingStrategy):
    """
    Minimum Communication Scheduler.
    
    Tries to minimize communication overhead by grouping related
    tasks on the same node when possible.
    """
    
    def __init__(self, dag: 'InceptionDAG'):
        super().__init__("MinCommunication")
        self.dag = dag
    
    def get_scheduler(self) -> Callable[['DistributedSimulator'], Optional[Tuple[str, int]]]:
        def schedule(sim: 'DistributedSimulator') -> Optional[Tuple[str, int]]:
            ready = sim.get_ready_tasks()
            available = sim.get_available_nodes()
            
            if not ready or not available:
                return None
            
            # For each ready task, find the best node considering communication
            best_task = None
            best_node = None
            best_score = float('inf')
            
            for task_name in ready:
                layer = sim.dag.nodes[task_name]
                
                for node_id in available:
                    # Calculate communication cost
                    comm_cost = 0.0
                    for pred_name in layer.predecessors:
                        pred_node = sim.task_assignment.get(pred_name)
                        if pred_node is not None and pred_node != node_id:
                            pred_layer = sim.dag.nodes[pred_name]
                            comm_cost += sim.cost_config.get_communication_cost_ms(
                                pred_layer.output_bytes
                            )
                    
                    # Score = communication cost + wait time
                    node = sim.nodes[node_id]
                    wait_time = max(0, node.available_at - sim.current_time)
                    score = comm_cost + wait_time
                    
                    if score < best_score:
                        best_score = score
                        best_task = task_name
                        best_node = node_id
            
            if best_task and best_node is not None:
                return (best_task, best_node)
            return None
        
        return schedule


def get_all_schedulers(dag: 'InceptionDAG') -> Dict[str, SchedulingStrategy]:
    """
    Get all available scheduling strategies.
    
    Args:
        dag: InceptionDAG for strategies that need it
    
    Returns:
        Dictionary mapping strategy names to strategy objects
    """
    return {
        'ASAP': ASAPScheduler(),
        'RoundRobin': RoundRobinScheduler(),
        'CriticalPathFirst': CriticalPathFirstScheduler(dag),
        'LoadBalancing': LoadBalancingScheduler(dag),
        'HEFT': HEFTScheduler(dag),
        'MinCommunication': MinCommunicationScheduler(dag),
    }


def compare_schedulers(
    dag: 'InceptionDAG',
    num_nodes: int,
    cost_config=None
) -> Dict[str, Dict]:
    """
    Compare all scheduling strategies on a given DAG.
    
    Args:
        dag: InceptionDAG to schedule
        num_nodes: Number of nodes
        cost_config: Optional cost configuration
    
    Returns:
        Dictionary mapping strategy names to results
    """
    from .simulator import DistributedSimulator
    from .cost_model import CostConfig
    
    config = cost_config or CostConfig()
    schedulers = get_all_schedulers(dag)
    results = {}
    
    for name, strategy in schedulers.items():
        sim = DistributedSimulator(dag, num_nodes, config)
        result = sim.simulate(strategy.get_scheduler())
        results[name] = result
    
    return results


def test_schedulers():
    """Test all scheduling strategies."""
    print("Testing Scheduling Strategies...")
    
    from .dag_model import InceptionDAG, LayerNode
    from .simulator import DistributedSimulator
    from .cost_model import CostConfig
    
    # Create test DAG
    dag = InceptionDAG("TestModel")
    dag.add_node(LayerNode(name="A", layer_type="Input", exec_time_enclave=1.0, output_bytes=1000))
    dag.add_node(LayerNode(name="B", layer_type="Conv", exec_time_enclave=3.0, output_bytes=2000))
    dag.add_node(LayerNode(name="C", layer_type="Conv", exec_time_enclave=2.0, output_bytes=2000))
    dag.add_node(LayerNode(name="D", layer_type="Concat", exec_time_enclave=1.0, output_bytes=4000))
    
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")
    
    config = CostConfig(
        network_bandwidth_mbps=1000,
        enclave_init_ms=10,
    )
    
    print("\nComparing schedulers with 2 nodes:")
    results = compare_schedulers(dag, num_nodes=2, cost_config=config)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Makespan: {result['makespan_ms']:.2f} ms")
        print(f"  Speedup: {result['speedup']:.2f}x")
        print(f"  Efficiency: {result['efficiency']:.2%}")
    
    print("\n✓ Scheduler tests passed!")


if __name__ == "__main__":
    test_schedulers()

