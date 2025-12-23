"""
Performance Simulator for Distributed Inference on Inception V3.

This tool models the execution time of Inception V3 under arbitrary partitioning strategies.
It calculates both "Distributed Sequential" (Baseline) and "Distributed Parallel" (Optimized) latencies.

Methodology:
1. Graph Construction: Builds a DAG representing layer dependencies.
2. Profiling Data: Uses measured compute times (CPU/Enclave) and data sizes.
3. Strategy Mapping: Maps each layer to an execution mode (CPU/Enclave) based on user input.
4. Latency Calculation:
   - Distributed Sequential: Sum of (Compute + Communication) for all layers in topological order.
     Simulates a single thread moving between devices.
   - Distributed Parallel: Critical Path Analysis on the DAG.
     End(N) = max(End(P) + Comm(P,N)) + Compute(N).
     Simulates multiple devices working in parallel.

Usage:
    python experiments/models/analyze_inception_flexible.py [metrics_file]
"""

import csv
import sys
import copy
from collections import defaultdict, OrderedDict

# ==============================================================================
# 1. Graph Topology Definition
# ==============================================================================

class LayerNode:
    def __init__(self, name, layer_type, output_size=0):
        self.name = name
        self.type = layer_type
        self.parents = []
        self.children = []
        self.output_size = output_size
        
        # Metrics (to be filled from CSV)
        self.t_enclave = 0.0
        self.t_cpu = 0.0
        
        # Runtime state (assigned by strategy)
        self.mode = "Enclave" # Default
        
        # Simulation state
        self.end_time = 0.0

def build_inception_graph():
    """
    Manually reconstructs the InceptionV3 graph structure matching sgx_inception.py
    Returns: dict of name -> LayerNode
    """
    nodes = {}
    
    def add_node(name, type_name, parents=None):
        node = LayerNode(name, type_name)
        nodes[name] = node
        if parents:
            if isinstance(parents, str): parents = [parents]
            for p_name in parents:
                parent = nodes[p_name]
                node.parents.append(parent)
                parent.children.append(node)
        return node

    # --- Stem ---
    add_node("input", "SecretInputLayer")
    add_node("conv1", "SGXConvBase", "input")
    add_node("relu1", "SecretReLULayer", "conv1")
    add_node("pool1", "SecretMaxpool2dLayer", "relu1")
    
    # --- Inception Block 1 (inc1) ---
    prev = "pool1"
    prefix = "inc1"
    
    # Branch 1
    add_node(f"{prefix}_b1_1x1", "SGXConvBase", prev)
    add_node(f"{prefix}_b1_relu", "SecretReLULayer", f"{prefix}_b1_1x1")
    
    # Branch 2
    add_node(f"{prefix}_b2_1x1", "SGXConvBase", prev)
    add_node(f"{prefix}_b2_relu1", "SecretReLULayer", f"{prefix}_b2_1x1")
    add_node(f"{prefix}_b2_3x3", "SGXConvBase", f"{prefix}_b2_relu1")
    add_node(f"{prefix}_b2_relu2", "SecretReLULayer", f"{prefix}_b2_3x3")
    
    # Branch 3
    add_node(f"{prefix}_b3_1x1", "SGXConvBase", prev)
    add_node(f"{prefix}_b3_relu1", "SecretReLULayer", f"{prefix}_b3_1x1")
    add_node(f"{prefix}_b3_3x3_1", "SGXConvBase", f"{prefix}_b3_relu1")
    add_node(f"{prefix}_b3_relu2", "SecretReLULayer", f"{prefix}_b3_3x3_1")
    add_node(f"{prefix}_b3_3x3_2", "SGXConvBase", f"{prefix}_b3_relu2")
    add_node(f"{prefix}_b3_relu3", "SecretReLULayer", f"{prefix}_b3_3x3_2")
    
    # Branch 4
    add_node(f"{prefix}_b4_pool", "SecretMaxpool2dLayer", prev)
    add_node(f"{prefix}_b4_1x1", "SGXConvBase", f"{prefix}_b4_pool")
    add_node(f"{prefix}_b4_relu", "SecretReLULayer", f"{prefix}_b4_1x1")
    
    # Concat
    tails = [f"{prefix}_b1_relu", f"{prefix}_b2_relu2", f"{prefix}_b3_relu3", f"{prefix}_b4_relu"]
    add_node(f"{prefix}_concat", "SecretConcatenateLayer", tails)
    
    # --- Inception Block 2 (inc2) ---
    prev = f"{prefix}_concat"
    prefix = "inc2"
    
    # Branch 1
    add_node(f"{prefix}_b1_1x1", "SGXConvBase", prev)
    add_node(f"{prefix}_b1_relu", "SecretReLULayer", f"{prefix}_b1_1x1")
    
    # Branch 2
    add_node(f"{prefix}_b2_1x1", "SGXConvBase", prev)
    add_node(f"{prefix}_b2_relu1", "SecretReLULayer", f"{prefix}_b2_1x1")
    add_node(f"{prefix}_b2_3x3", "SGXConvBase", f"{prefix}_b2_relu1")
    add_node(f"{prefix}_b2_relu2", "SecretReLULayer", f"{prefix}_b2_3x3")
    
    # Branch 3
    add_node(f"{prefix}_b3_1x1", "SGXConvBase", prev)
    add_node(f"{prefix}_b3_relu1", "SecretReLULayer", f"{prefix}_b3_1x1")
    add_node(f"{prefix}_b3_3x3_1", "SGXConvBase", f"{prefix}_b3_relu1")
    add_node(f"{prefix}_b3_relu2", "SecretReLULayer", f"{prefix}_b3_3x3_1")
    add_node(f"{prefix}_b3_3x3_2", "SGXConvBase", f"{prefix}_b3_relu2")
    add_node(f"{prefix}_b3_relu3", "SecretReLULayer", f"{prefix}_b3_3x3_2")
    
    # Branch 4
    add_node(f"{prefix}_b4_pool", "SecretMaxpool2dLayer", prev)
    add_node(f"{prefix}_b4_1x1", "SGXConvBase", f"{prefix}_b4_pool")
    add_node(f"{prefix}_b4_relu", "SecretReLULayer", f"{prefix}_b4_1x1")
    
    # Concat
    tails = [f"{prefix}_b1_relu", f"{prefix}_b2_relu2", f"{prefix}_b3_relu3", f"{prefix}_b4_relu"]
    add_node(f"{prefix}_concat", "SecretConcatenateLayer", tails)
    
    # --- Classifier ---
    prev = f"{prefix}_concat"
    add_node("avgpool", "SecretAvgpool2dLayer", prev)
    add_node("flatten", "SecretFlattenLayer", "avgpool")
    add_node("fc", "SGXLinearBase", "flatten")
    add_node("output", "SecretOutputLayer", "fc")
    
    return nodes

def load_metrics(graph, csv_file):
    """Populate graph nodes with metrics from CSV."""
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["LayerName"]
            if name in graph:
                node = graph[name]
                node.t_enclave = float(row["EnclaveTime(ms)"])
                node.t_cpu = float(row["CPUTime(ms)"])
                node.output_size = int(row["OutputBytes"])
    return graph

# ==============================================================================
# 2. Strategy Definition Language
# ==============================================================================

def apply_strategy(graph, strategy_dict):
    """
    Applies a partitioning strategy to the graph.
    
    strategy_dict format:
    {
        "default": "Enclave",          # Default mode for all layers
        "layers": {                    # Specific overrides
            "conv1": "CPU",
            "inc1.*": "Enclave",       # Wildcard support (regex)
            "fc": "CPU"
        }
    }
    """
    import re
    
    default_mode = strategy_dict.get("default", "Enclave")
    
    # Initialize all to default
    for node in graph.values():
        node.mode = default_mode
        
    # Apply overrides
    overrides = strategy_dict.get("layers", {})
    
    # Process overrides. Order matters? No, dict is unordered.
    # We should probably process specific names first, then wildcards? 
    # Let's just process in order.
    
    for pattern, mode in overrides.items():
        regex = re.compile(f"^{pattern}$")
        for name, node in graph.items():
            if regex.match(name):
                node.mode = mode
                
    return graph

# ==============================================================================
# 3. Simulation Engine
# ==============================================================================

class InceptionSimulator:
    def __init__(self, graph, bandwidth_gbps, latency_ms):
        self.graph = graph
        # BW in bytes/ms
        # 1 Gbps = 10^9 bits/s = 125 MB/s = 125,000 bytes/ms
        self.bw = (bandwidth_gbps * 1e9 / 8) / 1000
        self.lat = latency_ms
        
    def comm_time(self, size):
        if size == 0: return 0
        return self.lat + (size / self.bw)
        
    def get_compute_time(self, node):
        return node.t_enclave if node.mode == "Enclave" else node.t_cpu

    def simulate_sequential(self):
        """
        Calculates "Distributed Sequential" latency.
        Assumption: A single thread executes layers one by one in topological order.
        Communication happens whenever execution mode changes between layers.
        Wait, for sequential, does it matter if multiple parents have different modes?
        Sequential execution implies a linear ordering of the DAG.
        Let's assume a valid topological sort.
        """
        # Topological sort
        visited = set()
        order = []
        
        def dfs(node):
            if node.name in visited: return
            visited.add(node.name)
            for p in node.parents:
                dfs(p)
            order.append(node)
            
        for node in self.graph.values():
            if not node.children: # End nodes
                dfs(node)
                
        total_time = 0.0
        comm_time = 0.0
        compute_time = 0.0
        
        # We need to track data availability location.
        # But for strictly sequential:
        # Exec Layer A (Enclave) -> Data at Enclave
        # Exec Layer B (CPU) (Parent=A) -> Need transfer Enclave->CPU
        
        # Since we execute sequentially, we just sum compute + necessary comms.
        # For each layer, we check its parents. If parent's data is not in current mode, add comm.
        
        for node in order:
            t_comp = self.get_compute_time(node)
            compute_time += t_comp
            
            # Check input communication
            for parent in node.parents:
                if parent.mode != node.mode:
                    # Need to transfer parent's output to current node's device
                    t_c = self.comm_time(parent.output_size)
                    comm_time += t_c
            
            total_time += t_comp + comm_time # Accumulate
            
        # Wait, comm_time should be summed per edge
        # Let's recalculate properly
        total_time = 0.0
        
        for node in order:
            t_comp = self.get_compute_time(node)
            t_comm_node = 0
            for parent in node.parents:
                if parent.mode != node.mode:
                    t_comm_node += self.comm_time(parent.output_size)
            
            total_time += t_comp + t_comm_node
            
        return total_time

    def simulate_parallel(self):
        """
        Calculates "Distributed Parallel" latency with resource constraints.
        Each device (CPU, Enclave) acts as a single serial executor.
        Tasks assigned to the same device must be serialized.
        """
        # Reset
        for node in self.graph.values():
            node.end_time = -1.0
            
        # Device availability tracking
        # Since DAG execution order is not strictly defined by layers list,
        # we need a scheduling policy. "Earliest Ready First" is a reasonable policy.
        
        # We use a discrete event simulation approach.
        # Events: TaskReady
        # Queue: Ready tasks
        
        ready_queue = []
        
        # Initialize ready queue with input
        # Input layer has no parents
        for node in self.graph.values():
            node.unsatisfied_deps = len(node.parents)
            if node.unsatisfied_deps == 0:
                ready_queue.append(node)
                node.ready_time = 0.0
                
        # Device clocks
        device_clock = {
            "Enclave": 0.0,
            "CPU": 0.0
        }
        
        completed_nodes = 0
        total_nodes = len(self.graph)
        
        # Simple simulation loop
        # In each step, pick a ready task.
        # Which one? The one that can start earliest on its assigned device.
        
        while completed_nodes < total_nodes:
            if not ready_queue:
                # Deadlock or logic error
                break
                
            # Pick best candidate
            best_node = None
            best_start_time = float('inf')
            
            for node in ready_queue:
                # Earliest start = max(DependencyReady, DeviceAvailable)
                start = max(node.ready_time, device_clock[node.mode])
                if start < best_start_time:
                    best_start_time = start
                    best_node = node
            
            # Execute
            node = best_node
            ready_queue.remove(node)
            
            duration = self.get_compute_time(node)
            finish_time = best_start_time + duration
            node.end_time = finish_time
            
            # Update device clock
            device_clock[node.mode] = finish_time
            completed_nodes += 1
            
            # Update children
            for child in node.children:
                child.unsatisfied_deps -= 1
                if child.unsatisfied_deps == 0:
                    # Calculate ready time
                    # Ready when all parents done + communication
                    max_arrival = 0.0
                    for p in child.parents:
                        comm = 0.0
                        if p.mode != child.mode:
                            comm = self.comm_time(p.output_size)
                        arrival = p.end_time + comm
                        max_arrival = max(max_arrival, arrival)
                    
                    child.ready_time = max_arrival
                    ready_queue.append(child)
                    
        return self.graph["output"].end_time

# ==============================================================================
# 4. Main Interface
# ==============================================================================

def run_analysis(metrics_file="inception_metrics.csv"):
    print(f"Loading metrics from {metrics_file}...")
    graph = build_inception_graph()
    load_metrics(graph, metrics_file)
    
    # Define Network Scenarios
    networks = [
        ("PCIe 3.0", 126.0, 0.01),
        ("10GbE", 10.0, 0.1),
    ]
    
    # Define Strategies
    # Strategy 1: All CPU
    strat_all_cpu = {"default": "CPU"}
    
    # Strategy 2: All Enclave (Baseline)
    strat_all_enclave = {"default": "Enclave", "layers": {"input": "CPU"}}
    
    # Strategy 3: Branch Parallelism (Enclave: b1,b2; CPU: b3,b4)
    # Matches our code logic
    strat_branch = {
        "default": "Enclave",
        "layers": {
            "input": "CPU",
            # Inc1 CPU branches
            "inc1_b3.*": "CPU", "inc1_b4.*": "CPU",
            # Inc2 CPU branches
            "inc2_b3.*": "CPU", "inc2_b4.*": "CPU",
            # Concat & Classifier
            ".*_concat": "CPU",
            "avgpool": "CPU", "flatten": "CPU", "fc": "CPU", "output": "CPU"
        }
    }
    
    # Strategy 4: Pipeline (Inc1 Enclave -> Inc2 CPU)
    strat_pipeline = {
        "default": "Enclave",
        "layers": {
            "input": "CPU",
            "inc2.*": "CPU",
            "avgpool": "CPU", "flatten": "CPU", "fc": "CPU", "output": "CPU"
        }
    }
    
    strategies = [
        ("All CPU", strat_all_cpu),
        ("All Enclave", strat_all_enclave),
        ("Branch Parallel", strat_branch),
        ("Pipeline Split", strat_pipeline),
    ]
    
    print("\n" + "="*80)
    print(f"{'Strategy':<20} {'Network':<10} {'Seq(ms)':<10} {'Par(ms)':<10} {'Speedup':<10} {'vs Seq'}")
    print("="*80)
    
    for s_name, s_dict in strategies:
        apply_strategy(graph, s_dict)
        
        for net_name, bw, lat in networks:
            sim = InceptionSimulator(graph, bw, lat)
            t_seq = sim.simulate_sequential()
            t_par = sim.simulate_parallel()
            
            speedup = t_seq / t_par if t_par > 0 else 0
            
            print(f"{s_name:<20} {net_name:<10} {t_seq:<10.2f} {t_par:<10.2f} {speedup:<10.2f}x")
            
    print("="*80)

if __name__ == "__main__":
    run_analysis()

