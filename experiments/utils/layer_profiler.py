"""
Layer Profiler - Measure performance metrics for individual layers
"""

import time
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional

from python.enclave_interfaces import GlobalTensor
from python.global_config import SecretConfig
from python.utils.timer_utils import NamedTimer, VerboseLevel


class LayerProfiler:
    """Profile performance of individual DNN layers"""
    
    def __init__(self, model, device='CPU'):
        """
        Args:
            model: The SGX model to profile
            device: Device type ('CPU', 'GPU', or 'Enclave')
        """
        self.model = model
        self.device = device
        self.layers = getattr(model, 'layers', [])
        self.layer_info = {}
        self.profiling_results = defaultdict(dict)
        self._initialized = False
        self._current_batch_size = None
        self._input_shape = None
        self._skip_layer_types = {'SecretInputLayer'}
    
    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _reset_layer_states(self):
        for layer in self.layers:
            layer.tensor_name_list = []
            layer.PrevLayer = None
            layer.NextLayer = None
            if hasattr(layer, 'PlainForwardResult'):
                layer.PlainForwardResult = None
            if hasattr(layer, 'PlainBackwardResult'):
                layer.PlainBackwardResult = None
    
    def _link_layers(self):
        for idx in range(len(self.layers) - 1):
            prev_layer = self.layers[idx]
            next_layer = self.layers[idx + 1]
            if not prev_layer.manually_register_next:
                prev_layer.register_next_layer(next_layer)
            if not next_layer.manually_register_prev:
                next_layer.register_prev_layer(prev_layer)
    
    def _set_layer_eids(self):
        try:
            eid = GlobalTensor.get_eid()
        except Exception:
            eid = None
        if eid is None:
            return
        for layer in self.layers:
            layer.set_eid(eid)
    
    def _initialize_network(self, batch_size: int):
        if self._initialized and self._current_batch_size == batch_size:
            return
        if not self.layers:
            raise ValueError("Model does not contain any layers to profile")
        
        GlobalTensor.init()
        if hasattr(self.model, 'batch_size'):
            self.model.batch_size = batch_size
        
        self._reset_layer_states()
        self._link_layers()
        self._set_layer_eids()
        
        for layer in self.layers:
            try:
                layer.init_shape()
            except Exception as exc:
                raise RuntimeError(f"Failed to init_shape for layer {layer.LayerName}") from exc
        
        for layer in self.layers:
            try:
                layer.link_tensors()
            except Exception as exc:
                raise RuntimeError(f"Failed to link tensors for layer {layer.LayerName}") from exc
        
        for layer in self.layers:
            if hasattr(layer, 'inference'):
                layer.inference = True
            try:
                layer.init(start_enclave=False)
            except Exception as exc:
                raise RuntimeError(f"Failed to init layer {layer.LayerName}") from exc
        
        self._input_shape = self._infer_input_shape(batch_size)
        if self.layers and self.layers[0].__class__.__name__ == 'SecretInputLayer':
            self.layers[0].shape = self._input_shape
        
        self._initialized = True
        self._current_batch_size = batch_size
    
    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _infer_input_shape(self, batch_size: int) -> List[int]:
        for layer in self.layers:
            shape = getattr(layer, 'pytorch_x_shape', None)
            if shape is not None:
                inferred = list(shape)
                inferred[0] = batch_size
                return inferred
        for layer in self.layers:
            n_input = getattr(layer, 'n_input_channel', None)
            img_hw = getattr(layer, 'img_hw', None)
            if n_input is not None and img_hw is not None:
                return [batch_size, n_input, img_hw, img_hw]
        if hasattr(self.model, 'input_size'):
            channels = getattr(self.model, 'input_channels', 3)
            return [batch_size, channels, self.model.input_size, self.model.input_size]
        raise ValueError("Unable to infer input tensor shape for profiling")
    
    def _prepare_input_tensor(self, batch_size: int):
        input_shape = self._input_shape or self._infer_input_shape(batch_size)
        tensor = torch.randn(*input_shape, dtype=SecretConfig.dtypeForCpuOp)
        if self.device == 'GPU' and torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor
    
    def _should_profile(self, layer) -> bool:
        layer_type = layer.__class__.__name__
        if layer_type in self._skip_layer_types:
            return False
        if layer_type == 'SecretOutputLayer' and getattr(layer, 'inference', False):
            # inference mode output layer performs no computation
            return False
        return True
    
    def _run_forward_pass(self, input_tensor, timings: Dict[int, List[float]], record: bool):
        if not self.layers:
            return
        # Load input
        self.layers[0].set_input(input_tensor)
        for idx, layer in enumerate(self.layers):
            should_time = self._should_profile(layer)
            start = None
            if should_time and record:
                if self.device == 'GPU' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
            try:
                with torch.no_grad():
                    layer.forward()
            except Exception as exc:
                raise RuntimeError(
                    f"Forward execution failed on layer {layer.LayerName} ({layer.__class__.__name__})"
                ) from exc
            if should_time and record and start is not None:
                if self.device == 'GPU' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000.0
                timings[idx].append(elapsed)
    
    def _collect_timings(self, batch_size: int, num_iterations: int, warmup: int = 5):
        layer_indices = [idx for idx, layer in enumerate(self.layers) if self._should_profile(layer)]
        timings = {idx: [] for idx in layer_indices}
        
        # Warmup runs
        for _ in range(warmup):
            input_tensor = self._prepare_input_tensor(batch_size)
            self._run_forward_pass(input_tensor, timings, record=False)
        
        # Timed runs
        for _ in range(num_iterations):
            input_tensor = self._prepare_input_tensor(batch_size)
            self._run_forward_pass(input_tensor, timings, record=True)
        
        return timings
    
    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def get_layer_info(self, layer, layer_idx):
        """Extract layer information"""
        info = {
            'index': layer_idx,
            'name': getattr(layer, 'LayerName', f'layer_{layer_idx}'),
            'type': layer.__class__.__name__,
        }
        
        # Get input/output shapes if available
        if hasattr(layer, 'pytorch_x_shape'):
            info['input_shape'] = layer.pytorch_x_shape
        if hasattr(layer, 'pytorch_y_shape'):
            info['output_shape'] = layer.pytorch_y_shape
            
        # Get parameter count
        param_count = 0
        if hasattr(layer, 'get_cpu'):
            try:
                if hasattr(layer, 'weight'):
                    weight = layer.get_cpu('weight')
                    if weight is not None:
                        param_count += weight.numel()
                if hasattr(layer, 'bias') and getattr(layer, 'bias', False):
                    bias = layer.get_cpu('bias')
                    if bias is not None:
                        param_count += bias.numel()
            except Exception:
                pass
        
        info['param_count'] = int(param_count)
        
        # Estimate memory footprint (parameters + activations)
        memory_bytes = param_count * 4  # float32
        if 'output_shape' in info and info['output_shape'] is not None:
            try:
                output_size = np.prod(info['output_shape'])
                if output_size is not None:
                    memory_bytes += output_size * 4
            except Exception:
                pass
        info['memory_bytes'] = int(memory_bytes)
        info['memory_mb'] = float(memory_bytes / (1024 * 1024))
        
        return info
    
    def profile_single_layer(self, layer, layer_idx, batch_size=1, num_iterations=100):
        """
        Profile a single layer by running the full network and returning the requested layer stats.
        """
        all_results = self.profile_all_layers(batch_size=batch_size, num_iterations=num_iterations)
        for result in all_results:
            if result.get('index') == layer_idx:
                return result
        return None
    
    def profile_all_layers(self, batch_size=1, num_iterations=100):
        """
        Profile all layers in the model
        
        Returns:
            List of profiling results for each layer
        """
        print(f"\nProfiling model on {self.device} (batch_size={batch_size})...")
        
        if not self.layers:
            print("Error: Model does not have 'layers' attribute")
            return []
        
        self._initialize_network(batch_size)
        named_timer = NamedTimer.get_instance()
        original_verbose = named_timer.verbose_level
        NamedTimer.set_verbose_level(VerboseLevel.RUN)
        try:
            timings = self._collect_timings(batch_size, num_iterations)
        finally:
            NamedTimer.set_verbose_level(original_verbose)
        
        results = []
        for idx in sorted(timings.keys()):
            layer = self.layers[idx]
            layer_times = timings[idx]
            if not layer_times:
                continue
            stats = np.array(layer_times)
            layer_info = self.get_layer_info(layer, idx)
            layer_info.update({
                'mean_ms': float(np.mean(stats)),
                'std_ms': float(np.std(stats)),
                'min_ms': float(np.min(stats)),
                'max_ms': float(np.max(stats)),
                'median_ms': float(np.median(stats)),
                'p95_ms': float(np.percentile(stats, 95)),
                'p99_ms': float(np.percentile(stats, 99)),
                'batch_size': batch_size,
                'device': self.device,
                'num_iterations': num_iterations,
            })
            results.append(layer_info)
        
        print(f"Profiled {len(results)} layers successfully\n")
        return results
    
    def get_model_summary(self, results):
        """
        Generate summary statistics for the entire model
        
        Args:
            results: List of layer profiling results
            
        Returns:
            Dict with model-level statistics
        """
        if not results:
            return {
                'total_layers': 0,
                'total_time_ms': 0,
                'total_params': 0,
                'total_memory_mb': 0,
                'avg_layer_time_ms': 0,
                'device': self.device,
            }
        
        total_time = sum(r.get('mean_ms', 0) for r in results)
        total_params = sum(r.get('param_count', 0) for r in results)
        total_memory = sum(r.get('memory_mb', 0) for r in results)
        
        return {
            'total_layers': len(results),
            'total_time_ms': total_time,
            'total_params': total_params,
            'total_memory_mb': total_memory,
            'avg_layer_time_ms': total_time / len(results) if results else 0,
            'device': self.device,
        }

