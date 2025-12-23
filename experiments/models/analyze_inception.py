"""
Performance Modeling for Distributed Inference.

Reads layer_metrics.csv and analyzes theoretical speedup for different partitioning strategies
under various network bandwidth assumptions.
"""

import csv
import sys

class NetworkModel:
    def __init__(self, bandwidth_gbps, latency_ms):
        self.bandwidth_bytes_per_ms = (bandwidth_gbps * 1e9 / 8) / 1000
        self.latency_ms = latency_ms
        
    def comm_time(self, size_bytes):
        if size_bytes == 0: return 0
        return self.latency_ms + (size_bytes / self.bandwidth_bytes_per_ms)

def analyze_inception_partition(metrics_file):
    # Load metrics
    layers = []
    with open(metrics_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            layers.append({
                "name": row["LayerName"],
                "type": row["Type"],
                "t_enclave": float(row["EnclaveTime(ms)"]),
                "t_cpu": float(row["CPUTime(ms)"]),
                "size": int(row["OutputBytes"])
            })
            
    print(f"Loaded {len(layers)} layers metrics.")
    
    # Simulation Scenarios
    scenarios = [
        {"name": "PCIe 3.0 x16", "bw": 126.0, "lat": 0.01}, # ~15.75 GB/s -> 126 Gbps
        {"name": "10GbE LAN",    "bw": 10.0,  "lat": 0.1},
        {"name": "1GbE LAN",     "bw": 1.0,   "lat": 0.2},
        {"name": "WiFi/WAN",     "bw": 0.1,   "lat": 10.0} # 100 Mbps
    ]
    
    # Inception Topology Definition (Simplified for logic)
    # We need to know which layers belong to which branch to calculate parallel time.
    # Structure from sgx_inception.py:
    # Stem: conv1, relu1, pool1
    # Inc1: 
    #   b1: b1_1x1, b1_relu
    #   b2: b2_1x1, b2_relu1, b2_3x3, b2_relu2
    #   b3: b3_1x1, b3_relu1, b3_3x3_1, b3_relu2, b3_3x3_2, b3_relu3
    #   b4: b4_pool, b4_1x1, b4_relu
    #   concat
    # Inc2: (same structure)
    # Classifier: avgpool, flatten, fc, output
    
    blocks = {
        "stem": ["conv1", "relu1", "pool1"],
        "inc1_b1": ["inc1_b1_1x1", "inc1_b1_relu"],
        "inc1_b2": ["inc1_b2_1x1", "inc1_b2_relu1", "inc1_b2_3x3", "inc1_b2_relu2"],
        "inc1_b3": ["inc1_b3_1x1", "inc1_b3_relu1", "inc1_b3_3x3_1", "inc1_b3_relu2", "inc1_b3_3x3_2", "inc1_b3_relu3"],
        "inc1_b4": ["inc1_b4_pool", "inc1_b4_1x1", "inc1_b4_relu"],
        "inc1_concat": ["inc1_concat"],
        
        "inc2_b1": ["inc2_b1_1x1", "inc2_b1_relu"],
        "inc2_b2": ["inc2_b2_1x1", "inc2_b2_relu1", "inc2_b2_3x3", "inc2_b2_relu2"],
        "inc2_b3": ["inc2_b3_1x1", "inc2_b3_relu1", "inc2_b3_3x3_1", "inc2_b3_relu2", "inc2_b3_3x3_2", "inc2_b3_relu3"],
        "inc2_b4": ["inc2_b4_pool", "inc2_b4_1x1", "inc2_b4_relu"],
        "inc2_concat": ["inc2_concat"],
        
        "classifier": ["avgpool", "flatten", "fc", "output"]
    }
    
    # Partition Strategy: 
    # Enclave: Branch 1 & 2 (Compute heavy convs)
    # CPU: Branch 3 & 4 (Also convs but let's split here)
    
    # Let's define a function to calculate total time given a network model
    def calc_pipeline_time(net):
        t_seq = sum(l["t_enclave"] for l in layers) # Baseline: All Enclave
        
        # Logic for Inc1 Block Parallelism
        # Input arrives at Inc1.
        # Split:
        #   Partition 1 (Enclave): b1 + b2
        #   Partition 2 (CPU): b3 + b4
        
        # Need to transfer input to Partition 2 (comm cost)
        # Then P1 and P2 run in parallel.
        # Then P2 sends output to P1 (concat happens in Enclave usually, or CPU)
        # Let's assume Concat is in Enclave.
        
        # Time breakdown:
        # T_stem (Enclave)
        
        # Block 1:
        # T_comm_in (Send Stem output to CPU)
        # T_p1 = T(b1) + T(b2) (Enclave compute)
        # T_p2 = T(b3) + T(b4) (CPU compute)
        # T_comm_out = T_comm(b3_out) + T_comm(b4_out) (Send CPU results back to Enclave)
        # T_block1 = max(T_p1, T_comm_in + T_p2 + T_comm_out) + T_concat
        
        # Block 2: similar...
        
        def get_layer_time(name, mode):
            l = next((x for x in layers if x["name"] == name), None)
            if not l: return 0
            return l["t_enclave"] if mode == "enclave" else l["t_cpu"]
            
        def get_layer_size(name):
            l = next((x for x in layers if x["name"] == name), None)
            return l["size"] if l else 0
            
        # Stem
        t_stem = sum(get_layer_time(n, "enclave") for n in blocks["stem"])
        stem_out_size = get_layer_size(blocks["stem"][-1])
        
        # Block 1
        # P1: b1, b2 (Enclave)
        t_p1 = sum(get_layer_time(n, "enclave") for n in blocks["inc1_b1"] + blocks["inc1_b2"])
        
        # P2: b3, b4 (CPU)
        t_p2 = sum(get_layer_time(n, "cpu") for n in blocks["inc1_b3"] + blocks["inc1_b4"])
        
        # Comm
        t_comm_in = net.comm_time(stem_out_size) # Stem out -> P2 input
        
        # P2 outputs -> P1 (Concat)
        b3_out_size = get_layer_size(blocks["inc1_b3"][-1])
        b4_out_size = get_layer_size(blocks["inc1_b4"][-1])
        t_comm_out = net.comm_time(b3_out_size) + net.comm_time(b4_out_size)
        
        t_block1 = max(t_p1, t_comm_in + t_p2 + t_comm_out) + get_layer_time("inc1_concat", "enclave")
        
        # Block 2 (Input is inc1_concat output)
        inc1_out_size = get_layer_size("inc1_concat")
        
        t_p1_2 = sum(get_layer_time(n, "enclave") for n in blocks["inc2_b1"] + blocks["inc2_b2"])
        t_p2_2 = sum(get_layer_time(n, "cpu") for n in blocks["inc2_b3"] + blocks["inc2_b4"])
        
        t_comm_in_2 = net.comm_time(inc1_out_size)
        
        b3_out_size_2 = get_layer_size(blocks["inc2_b3"][-1])
        b4_out_size_2 = get_layer_size(blocks["inc2_b4"][-1])
        t_comm_out_2 = net.comm_time(b3_out_size_2) + net.comm_time(b4_out_size_2)
        
        t_block2 = max(t_p1_2, t_comm_in_2 + t_p2_2 + t_comm_out_2) + get_layer_time("inc2_concat", "enclave")
        
        # Classifier (Enclave)
        t_cls = sum(get_layer_time(n, "enclave") for n in blocks["classifier"])
        
        t_total_dist = t_stem + t_block1 + t_block2 + t_cls
        
        return t_seq, t_total_dist, t_p1, t_p2, t_comm_in+t_comm_out

    print("\nAnalysis Results (Strategy: Branch Splitting)")
    print(f"{'Network':<15} {'Seq(ms)':<10} {'Dist(ms)':<10} {'Speedup':<10} {'Bottleneck'}")
    print("-" * 65)
    
    for sc in scenarios:
        net = NetworkModel(sc["bw"], sc["lat"])
        seq, dist, tp1, tp2, tcomm = calc_pipeline_time(net)
        
        speedup = seq / dist
        
        # Analyze bottleneck
        # In Block 1
        if tp1 > (tp2 + tcomm):
            bn = "Compute(Enclave)"
        elif tp2 > (tp1 + tcomm): # Unlikely with comm added to tp2 side
            bn = "Compute(CPU)"
        else:
            bn = "Communication"
            
        print(f"{sc['name']:<15} {seq:<10.2f} {dist:<10.2f} {speedup:<10.2f}x {bn}")

if __name__ == "__main__":
    import os
    if len(sys.argv) > 1:
        file = sys.argv[1]
    else:
        file = "inception_metrics.csv"
        
    if os.path.exists(file):
        analyze_inception_partition(file)
    else:
        print(f"Metrics file {file} not found. Run profile_inception.py first.")

