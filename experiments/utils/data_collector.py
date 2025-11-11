"""
Data Collector - Save and load measurement results
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any


class DataCollector:
    """Collect and manage measurement data"""
    
    def __init__(self, data_dir='experiments/data'):
        """
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save_json(self, data: Dict[str, Any], filename: str):
        """
        Save data to JSON file
        
        Args:
            data: Data to save
            filename: Output filename
        """
        filepath = os.path.join(self.data_dir, filename)
        
        # Add metadata
        data['_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Data saved to {filepath}")
    
    def load_json(self, filename: str) -> Dict[str, Any]:
        """
        Load data from JSON file
        
        Args:
            filename: Input filename
            
        Returns:
            Loaded data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return data
    
    def list_data_files(self, pattern: str = '*.json') -> List[str]:
        """
        List all data files matching pattern
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of matching filenames
        """
        import glob
        files = glob.glob(os.path.join(self.data_dir, pattern))
        return [os.path.basename(f) for f in files]
    
    def save_layer_results(self, model_name: str, device: str, 
                          results: List[Dict], cost_type: str):
        """
        Save layer profiling results
        
        Args:
            model_name: Name of the model
            device: Device type (CPU/GPU/Enclave)
            results: List of layer results
            cost_type: Type of cost (computation/communication/etc)
        """
        filename = f"{cost_type}_{model_name}_{device}.json"
        
        data = {
            'model': model_name,
            'device': device,
            'cost_type': cost_type,
            'num_layers': len(results),
            'layers': results
        }
        
        self.save_json(data, filename)
    
    def aggregate_results(self, model_name: str, cost_type: str,
                         devices: List[str] = ['CPU', 'GPU', 'Enclave']) -> Dict:
        """
        Aggregate results across devices
        
        Args:
            model_name: Name of the model
            cost_type: Type of cost
            devices: List of devices
            
        Returns:
            Aggregated data
        """
        aggregated = {
            'model': model_name,
            'cost_type': cost_type,
            'devices': {}
        }
        
        for device in devices:
            filename = f"{cost_type}_{model_name}_{device}.json"
            try:
                data = self.load_json(filename)
                aggregated['devices'][device] = data
            except FileNotFoundError:
                print(f"Warning: {filename} not found")
        
        return aggregated

