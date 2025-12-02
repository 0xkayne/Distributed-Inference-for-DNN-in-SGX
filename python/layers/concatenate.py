from python.layers.nonlinear import SecretNonlinearLayer
from python.tensor_loader import TensorLoader
from python.utils.timer_utils import NamedTimerInstance, VerboseLevel
from python.enclave_interfaces import GlobalTensor as gt
from python.utils.basic_utils import ExecutionModeOptions

import torch
from pdb import set_trace as st

class SecretConcatenateLayer(SecretNonlinearLayer):
    def __init__(
        self, sid, LayerName, EnclaveMode, link_prev=True, link_next=True,
        manually_register_prev=False, manually_register_next=False, dim=1
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)
        self.PrevLayer = []  # List of previous layers
        self.dim = dim
        # Enclave execution not yet supported for Concatenate, default to CPU
        if self.EnclaveMode == ExecutionModeOptions.Enclave:
            print(f"Warning: {LayerName} forced to CPU mode (Enclave concat not implemented)")
            self.EnclaveMode = ExecutionModeOptions.CPU

    def register_prev_layer(self, layer):
        if layer not in self.PrevLayer:
            self.PrevLayer.append(layer)

    def init_shape(self):
        if len(self.PrevLayer) == 0:
            raise ValueError(f"Concatenate layer {self.LayerName} has no input layers")
            
        # Validate input shapes match on non-concat dimensions
        base_shape = list(self.PrevLayer[0].get_output_shape())
        total_dim_size = 0
        
        for layer in self.PrevLayer:
            shape = list(layer.get_output_shape())
            if len(shape) != len(base_shape):
                raise ValueError(f"Input shapes rank mismatch: {base_shape} vs {shape}")
                
            for i, (d1, d2) in enumerate(zip(base_shape, shape)):
                if i != self.dim and d1 != d2:
                    raise ValueError(f"Input shapes mismatch at dim {i}: {base_shape} vs {shape}")
            
            total_dim_size += shape[self.dim]
            
        self.InputShape = None # Multi-input
        self.OutputShape = list(base_shape)
        self.OutputShape[self.dim] = total_dim_size
        self.HandleShape = self.OutputShape

    def init(self, start_enclave=True):
        TensorLoader.init(self, start_enclave)

    def get_output_shape(self):
        return self.OutputShape

    def generate_tensor_name_list(self, force=False):
        if not force and self.tensor_name_list:
            return
            
        NeededTensorNames = [("output", self.OutputShape, None)]
        
        # Generate input names: input, input1, input2...
        for idx, layer in enumerate(self.PrevLayer):
            if idx == 0:
                input_name = "input"
            else:
                input_name = f"input{idx}"
            
            NeededTensorNames.append(
                (input_name, layer.get_output_shape(), None)
            )
            
        self.tensor_name_list = NeededTensorNames

    def link_tensors(self):
        # Link inputs
        if self.link_prev:
            for idx, layer in enumerate(self.PrevLayer):
                if idx == 0:
                    input_name = "input"
                else:
                    input_name = f"input{idx}"
                
                # Link my inputX to prev layer's output
                gt.link_tags(self.get_tag(input_name, remap=False), layer.get_tag("output", remap=False))
                
        # Link output
        if self.link_next and self.NextLayer is not None:
            gt.link_tags(self.get_tag("output", remap=False), self.NextLayer.get_tag("input", remap=False))

    def forward(self):
        with NamedTimerInstance(f"S{self.sid}: {self.LayerName} Forward", verbose_level=VerboseLevel.LAYER):
            inputs = []
            
            # Fetch inputs from all previous layers
            for idx, layer in enumerate(self.PrevLayer):
                if idx == 0:
                    input_name = "input"
                else:
                    input_name = f"input{idx}"
                
                # Ensure data is on CPU (since we only support CPU concat for now)
                self.make_sure_cpu_is_latest(input_name)
                inputs.append(self.get_cpu(input_name))
            
            # Perform concatenation
            output = torch.cat(inputs, dim=self.dim)
            self.set_cpu("output", output)
            
            # If next layer expects data in Enclave/GPU, transfer it
            # But since this layer is CPU, next layer will pull from CPU via transfer_from_cpu
            
    def backward(self):
        # Not implemented for inference demo
        pass

    def print_connection_info(self):
        prev_names = [l.LayerName for l in self.PrevLayer]
        print(f"{self.LayerName:20} shape{self.OutputShape}{' ':30} inputs {prev_names} output {self.NextLayer.LayerName if self.NextLayer else 'None'}")


