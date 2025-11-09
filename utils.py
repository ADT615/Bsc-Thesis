"""
Utilities module
Chứa các utility functions và helper functions
"""

import numpy as np
import json
import pickle
from datetime import datetime


def save_results(data, filename, format='json'):
    """
    Save results to file
    
    Args:
        data: Data to save
        filename: Output filename
        format: File format ('json', 'pickle', 'npy')
    """
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        json_data = convert_numpy_to_json(data)
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
    elif format == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'npy':
        np.save(filename, data)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Results saved to {filename}")


def load_results(filename, format='json'):
    """
    Load results from file
    
    Args:
        filename: Input filename
        format: File format ('json', 'pickle', 'npy')
        
    Returns:
        Loaded data
    """
    if format == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
        return convert_json_to_numpy(data)
    elif format == 'pickle':
        with open(filename, 'rb') as f:
            return pickle.load(f)
    elif format == 'npy':
        return np.load(filename, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported format: {format}")


def convert_numpy_to_json(obj):
    """
    Convert numpy arrays to JSON-serializable format
    
    Args:
        obj: Object containing numpy arrays
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.complex_):
        return {'real': float(obj.real), 'imag': float(obj.imag), '_type': 'complex'}
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json(item) for item in obj]
    else:
        return obj


def convert_json_to_numpy(obj):
    """
    Convert JSON data back to numpy arrays where appropriate
    
    Args:
        obj: JSON-loaded object
        
    Returns:
        Object with numpy arrays restored
    """
    if isinstance(obj, dict):
        if '_type' in obj and obj['_type'] == 'complex':
            return complex(obj['real'], obj['imag'])
        else:
            return {key: convert_json_to_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Try to convert to numpy array if all elements are numbers
        try:
            converted_list = [convert_json_to_numpy(item) for item in obj]
            if all(isinstance(x, (int, float, complex)) for x in converted_list):
                return np.array(converted_list)
            else:
                return converted_list
        except:
            return [convert_json_to_numpy(item) for item in obj]
    else:
        return obj


def create_timestamp():
    """
    Create timestamp string for file naming
    
    Returns:
        str: Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_statistics(data):
    """
    Calculate basic statistics for data
    
    Args:
        data: Input data array
        
    Returns:
        dict: Statistics dictionary
    """
    data = np.array(data)
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'size': data.size
    }


def validate_parameters(params, param_ranges):
    """
    Validate parameter values against specified ranges
    
    Args:
        params: Parameter dictionary
        param_ranges: Dictionary of parameter ranges
        
    Returns:
        bool: True if all parameters are valid
    """
    for param_name, value in params.items():
        if param_name in param_ranges:
            min_val, max_val = param_ranges[param_name]
            if not (min_val <= value <= max_val):
                print(f"Parameter {param_name} = {value} is outside range [{min_val}, {max_val}]")
                return False
    return True


def normalize_array(arr, method='minmax'):
    """
    Normalize array using specified method
    
    Args:
        arr: Input array
        method: Normalization method ('minmax', 'zscore', 'unit')
        
    Returns:
        ndarray: Normalized array
    """
    arr = np.array(arr)
    
    if method == 'minmax':
        min_val, max_val = np.min(arr), np.max(arr)
        if max_val == min_val:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean_val, std_val = np.mean(arr), np.std(arr)
        if std_val == 0:
            return np.zeros_like(arr)
        return (arr - mean_val) / std_val
    elif method == 'unit':
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr
        return arr / norm
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def print_system_info():
    """
    Print system and environment information
    """
    import sys
    import platform
    
    print("System Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    
    try:
        import qiskit
        print(f"Qiskit version: {qiskit.__version__}")
    except ImportError:
        print("Qiskit not available")
    
    try:
        import pennylane
        print(f"PennyLane version: {pennylane.__version__}")
    except ImportError:
        print("PennyLane not available")


def memory_usage():
    """
    Get current memory usage
    
    Returns:
        dict: Memory usage information
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
        'percent': process.memory_percent()
    }
