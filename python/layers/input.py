from python.layers.nonlinear import SecretNonlinearLayer
from python.enclave_interfaces import GlobalTensor as gt
from python.utils.basic_utils import ExecutionModeOptions
from pdb import set_trace as st

class SecretInputLayer(SecretNonlinearLayer):
    shape = None

    def __init__(
        self, sid, LayerName, input_shape, EnclaveMode, link_prev=True, link_next=True, 
        manually_register_prev=False, manually_register_next=False
    ):
        super().__init__(sid, LayerName, EnclaveMode, link_prev, link_next, manually_register_prev, manually_register_next)
        self.shape = input_shape

    def link_tensors(self):
        gt.link_tags(self.get_tag("input", remap=False), self.get_tag("output", remap=False))
        super().link_tensors()

    def init_shape(self):
        return

    def generate_tensor_name_list(self, force=False):
        """Generate tensor list for InputLayer - input and output (linked)."""
        if not force and self.tensor_name_list:
            return
        
        # InputLayer needs input and output tensors (they will be linked)
        # Note: shape is set at __init__ time, not from PrevLayer
        NeededTensorNames = [
            ("input", self.shape, None),
            ("output", self.shape, None),
        ]
        self.tensor_name_list = NeededTensorNames

    def set_input(self, tensor):
        self.set_cpu("input", tensor)
        if self.EnclaveMode is ExecutionModeOptions.Enclave:
            # For InputLayer, input and output are linked (share same enclave storage)
            # But we must ensure BOTH tags are initialized before any SetTen/GetTen
            self.set_tensor("input", tensor)
            # Explicitly init output tag if not already done (link alone doesn't init)
            output_tag = self.get_tag("output", remap=True)
            from python.enclave_interfaces import GlobalTensor
            if output_tag not in GlobalTensor.IsInitEnclaveTensor:
                GlobalTensor.init_enclave_tensor(output_tag, list(tensor.shape))
        if self.EnclaveMode is ExecutionModeOptions.GPU:
            self.set_gpu("input", tensor)

    def get_output_shape(self):
        return self.shape

    def forward(self):
        return

    def backward(self):
        return

    def plain_forward(self):
        return

    def plain_backward(self):
        return

    def show_plain_error(self):
        return

    def print_connection_info(self):
        print(f"{self.LayerName:30} shape{self.shape} output {self.NextLayer.LayerName:30}")


