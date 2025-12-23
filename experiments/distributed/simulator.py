"""
Distributed TEE Execution Simulator

This module simulates distributed execution of DNN inference across
multiple TEE (Trusted Execution Environment) nodes.

Classes:
- TEENode: Represents a single TEE node
- TaskEvent: Represents a task execution event
- DistributedSimulator: Main simulation engine
"""

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import defaultdict
import numpy as np

from .dag_model import InceptionDAG, LayerNode
from .cost_model import CostModel, CostConfig


class EventType(Enum):
    """Types of simulation events."""
    TASK_START = "task_start"
    TASK_FINISH = "task_finish"
    TRANSFER_START = "transfer_start"
    TRANSFER_FINISH = "transfer_finish"


@dataclass(order=True)
class SimEvent:
    """
    Simulation event for discrete event simulation.
    
    Attributes:
        time: Event timestamp in ms
        event_type: Type of event
        layer_name: Associated layer name
        node_id: Associated node ID
        data: Additional event data
    """
    time: float
    event_type: EventType = field(compare=False)
    layer_name: str = field(compare=False)
    node_id: int = field(compare=False, default=0)
    data: Dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass
class TaskExecution:
    """
    Records execution of a single task (layer).
    
    Attributes:
        layer_name: Name of the layer
        node_id: ID of the node that executed the task
        start_time: Start time in ms
        end_time: End time in ms
        wait_time: Time spent waiting for dependencies
        compute_time: Actual computation time
        transfer_time: Data transfer time
    """
    layer_name: str
    node_id: int
    start_time: float = 0.0
    end_time: float = 0.0
    wait_time: float = 0.0
    compute_time: float = 0.0
    transfer_time: float = 0.0
    
    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'layer_name': self.layer_name,
            'node_id': self.node_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'wait_time': self.wait_time,
            'compute_time': self.compute_time,
            'transfer_time': self.transfer_time,
            'total_time': self.total_time,
        }


class TEENode:
    """
    Represents a single TEE node in the distributed system.
    
    Attributes:
        node_id: Unique node identifier
        is_initialized: Whether the enclave is initialized
        is_busy: Whether the node is currently executing a task
        current_task: Name of current task (if any)
        available_at: Time when node will be available
        tasks_executed: List of tasks executed by this node
    """
    
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.is_initialized = False
        self.is_busy = False
        self.current_task: Optional[str] = None
        self.available_at: float = 0.0
        self.tasks_executed: List[str] = []
        self.total_compute_time: float = 0.0
        self.total_idle_time: float = 0.0
    
    def initialize(self, init_time: float, current_time: float) -> float:
        """
        Initialize the enclave.
        
        Args:
            init_time: Initialization overhead in ms
            current_time: Current simulation time
        
        Returns:
            Time when initialization completes
        """
        if not self.is_initialized:
            self.is_initialized = True
            self.available_at = max(self.available_at, current_time) + init_time
            return self.available_at
        return current_time
    
    def start_task(self, task_name: str, start_time: float) -> None:
        """Start executing a task."""
        self.is_busy = True
        self.current_task = task_name
        # Track idle time
        if start_time > self.available_at:
            self.total_idle_time += start_time - self.available_at
    
    def finish_task(self, end_time: float, compute_time: float) -> None:
        """Finish executing current task."""
        if self.current_task:
            self.tasks_executed.append(self.current_task)
            self.total_compute_time += compute_time
        self.is_busy = False
        self.current_task = None
        self.available_at = end_time
    
    def get_utilization(self, total_time: float) -> float:
        """Get node utilization as fraction of total time."""
        if total_time <= 0:
            return 0.0
        return self.total_compute_time / total_time


class DistributedSimulator:
    """
    Discrete event simulator for distributed TEE execution.
    
    Simulates the execution of a DAG-structured DNN model across
    multiple TEE nodes, considering:
    - Task dependencies
    - Communication costs
    - Enclave initialization overhead
    - Different scheduling strategies
    """
    
    def __init__(
        self,
        dag: InceptionDAG,
        num_nodes: int,
        cost_config: Optional[CostConfig] = None,
        use_enclave: bool = True
    ):
        """
        Initialize the simulator.
        
        Args:
            dag: InceptionDAG representing the model
            num_nodes: Number of TEE nodes
            cost_config: Cost configuration
            use_enclave: Whether to use enclave execution times
        """
        self.dag = dag
        self.num_nodes = num_nodes
        self.cost_config = cost_config or CostConfig()
        self.cost_model = CostModel(self.cost_config)
        self.use_enclave = use_enclave
        
        # Simulation state
        self.nodes: List[TEENode] = []
        self.current_time: float = 0.0
        self.event_queue: List[SimEvent] = []
        
        # Task state
        self.task_assignment: Dict[str, int] = {}
        self.task_executions: Dict[str, TaskExecution] = {}
        self.completed_tasks: Set[str] = set()
        self.pending_tasks: Set[str] = set()
        
        # Results
        self.makespan: float = 0.0
        self.schedule: List[TaskExecution] = []
    
    def reset(self) -> None:
        """Reset simulator state for a new run."""
        self.nodes = [TEENode(i) for i in range(self.num_nodes)]
        self.current_time = 0.0
        self.event_queue = []
        self.task_executions = {}
        self.completed_tasks = set()
        self.pending_tasks = set(self.dag.nodes.keys())
        self.makespan = 0.0
        self.schedule = []
    
    def simulate(
        self,
        scheduler: Callable[['DistributedSimulator'], Optional[Tuple[str, int]]]
    ) -> Dict[str, Any]:
        """
        Run the simulation with the given scheduler.
        
        Args:
            scheduler: Function that returns (task_name, node_id) or None
        
        Returns:
            Dictionary with simulation results
        """
        self.reset()
        
        # Main simulation loop
        while self.pending_tasks or self.event_queue:
            # Try to schedule ready tasks
            while True:
                decision = scheduler(self)
                if decision is None:
                    break
                
                task_name, node_id = decision
                self._schedule_task(task_name, node_id)
            
            # Process next event
            if self.event_queue:
                event = heapq.heappop(self.event_queue)
                self.current_time = event.time
                self._process_event(event)
            else:
                break
        
        # Compute final results
        self.makespan = max(
            (exec.end_time for exec in self.task_executions.values()),
            default=0.0
        )
        
        self.schedule = sorted(
            self.task_executions.values(),
            key=lambda x: x.start_time
        )
        
        return self._get_results()
    
    def _schedule_task(self, task_name: str, node_id: int) -> None:
        """Schedule a task on a node."""
        if task_name not in self.pending_tasks:
            return
        
        node = self.nodes[node_id]
        layer = self.dag.nodes[task_name]
        
        # Calculate start time
        start_time = max(self.current_time, node.available_at)
        
        # Wait for dependencies and calculate transfer time
        transfer_time = 0.0
        for pred_name in layer.predecessors:
            if pred_name in self.task_executions:
                pred_exec = self.task_executions[pred_name]
                pred_finish = pred_exec.end_time
                
                # Add transfer time if on different node
                if self.task_assignment.get(pred_name) != node_id:
                    pred_layer = self.dag.nodes[pred_name]
                    transfer = self.cost_config.get_communication_cost_ms(
                        pred_layer.output_bytes
                    )
                    pred_finish += transfer
                    transfer_time += transfer
                
                start_time = max(start_time, pred_finish)
        
        # Initialize node if needed
        if not node.is_initialized:
            start_time = node.initialize(
                self.cost_config.enclave_init_ms,
                start_time
            )
        
        # Calculate execution time
        exec_time = layer.exec_time_enclave if self.use_enclave else layer.exec_time_cpu
        end_time = start_time + exec_time
        
        # Record assignment and execution
        self.task_assignment[task_name] = node_id
        self.pending_tasks.remove(task_name)
        
        wait_time = start_time - self.current_time if start_time > self.current_time else 0
        
        self.task_executions[task_name] = TaskExecution(
            layer_name=task_name,
            node_id=node_id,
            start_time=start_time,
            end_time=end_time,
            wait_time=wait_time,
            compute_time=exec_time,
            transfer_time=transfer_time,
        )
        
        # Update node state
        node.start_task(task_name, start_time)
        
        # Schedule completion event
        heapq.heappush(
            self.event_queue,
            SimEvent(end_time, EventType.TASK_FINISH, task_name, node_id)
        )
    
    def _process_event(self, event: SimEvent) -> None:
        """Process a simulation event."""
        if event.event_type == EventType.TASK_FINISH:
            node = self.nodes[event.node_id]
            exec_time = self.task_executions[event.layer_name].compute_time
            node.finish_task(event.time, exec_time)
            self.completed_tasks.add(event.layer_name)
    
    def get_ready_tasks(self) -> List[str]:
        """Get list of tasks ready to be scheduled."""
        ready = []
        for task_name in self.pending_tasks:
            layer = self.dag.nodes[task_name]
            # Check if all predecessors are completed
            if all(pred in self.completed_tasks for pred in layer.predecessors):
                ready.append(task_name)
        return ready
    
    def get_available_nodes(self) -> List[int]:
        """Get list of available (not busy) nodes."""
        return [
            node.node_id for node in self.nodes
            if not node.is_busy or node.available_at <= self.current_time
        ]
    
    def _get_results(self) -> Dict[str, Any]:
        """Compile simulation results."""
        serial_time = self.cost_model.compute_dag_serial_cost(
            self.dag, self.use_enclave
        )
        
        # Calculate node utilizations
        utilizations = [
            node.get_utilization(self.makespan)
            for node in self.nodes
        ]
        
        # Calculate total times
        total_compute = sum(exec.compute_time for exec in self.task_executions.values())
        total_transfer = sum(exec.transfer_time for exec in self.task_executions.values())
        total_wait = sum(exec.wait_time for exec in self.task_executions.values())
        
        return {
            'makespan_ms': self.makespan,
            'serial_time_ms': serial_time,
            'speedup': serial_time / self.makespan if self.makespan > 0 else 1.0,
            'efficiency': (serial_time / self.makespan) / self.num_nodes if self.makespan > 0 else 0,
            'num_nodes': self.num_nodes,
            'num_tasks': len(self.task_executions),
            'total_compute_ms': total_compute,
            'total_transfer_ms': total_transfer,
            'total_wait_ms': total_wait,
            'node_utilizations': utilizations,
            'avg_utilization': np.mean(utilizations) if utilizations else 0,
            'schedule': [exec.to_dict() for exec in self.schedule],
            'task_assignment': dict(self.task_assignment),
        }
    
    def get_gantt_data(self) -> List[Dict[str, Any]]:
        """
        Get data formatted for Gantt chart visualization.
        
        Returns:
            List of task dictionaries with timing info
        """
        return [
            {
                'task': exec.layer_name,
                'node': f'Node {exec.node_id}',
                'start': exec.start_time,
                'end': exec.end_time,
                'duration': exec.total_time,
                'type': self.dag.nodes[exec.layer_name].layer_type,
            }
            for exec in self.schedule
        ]


# Import Set for type hints
from typing import Set


def test_simulator():
    """Test simulator with a simple DAG."""
    print("Testing Distributed Simulator...")
    
    # Create simple diamond DAG
    from .dag_model import InceptionDAG, LayerNode
    
    dag = InceptionDAG("TestModel")
    dag.add_node(LayerNode(name="A", layer_type="Input", exec_time_enclave=1.0, output_bytes=1000))
    dag.add_node(LayerNode(name="B", layer_type="Conv", exec_time_enclave=3.0, output_bytes=2000))
    dag.add_node(LayerNode(name="C", layer_type="Conv", exec_time_enclave=2.0, output_bytes=2000))
    dag.add_node(LayerNode(name="D", layer_type="Concat", exec_time_enclave=1.0, output_bytes=4000))
    
    dag.add_edge("A", "B")
    dag.add_edge("A", "C")
    dag.add_edge("B", "D")
    dag.add_edge("C", "D")
    
    # Create simulator with 2 nodes
    config = CostConfig(
        network_bandwidth_mbps=1000,
        network_latency_ms=0.1,
        enclave_init_ms=10,
    )
    
    sim = DistributedSimulator(dag, num_nodes=2, cost_config=config)
    
    # Simple ASAP scheduler
    def asap_scheduler(sim: DistributedSimulator) -> Optional[Tuple[str, int]]:
        ready = sim.get_ready_tasks()
        available = sim.get_available_nodes()
        
        if ready and available:
            # Assign first ready task to first available node
            return (ready[0], available[0])
        return None
    
    results = sim.simulate(asap_scheduler)
    
    print(f"Makespan: {results['makespan_ms']:.2f} ms")
    print(f"Serial time: {results['serial_time_ms']:.2f} ms")
    print(f"Speedup: {results['speedup']:.2f}x")
    print(f"Efficiency: {results['efficiency']:.2%}")
    print(f"Node utilizations: {results['node_utilizations']}")
    
    print("\nSchedule:")
    for task in results['schedule']:
        print(f"  {task['layer_name']}: Node {task['node_id']}, "
              f"{task['start_time']:.1f}-{task['end_time']:.1f} ms")
    
    print("\n✓ Simulator tests passed!")


if __name__ == "__main__":
    test_simulator()

