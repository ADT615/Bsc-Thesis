"""
Common optimization utilities for quantum algorithms
"""

import numpy as np
import qiskit
from qiskit.quantum_info import Statevector


def unitary_fidelity(U_target, U_approx):
    """
    Compute fidelity between target and approximate unitary matrices
    
    Args:
        U_target: Target unitary matrix
        U_approx: Approximate unitary matrix
        
    Returns:
        float: Fidelity value
    """
    # Ensure both matrices are complex
    U_target = np.array(U_target, dtype=complex)
    U_approx = np.array(U_approx, dtype=complex)
    
    # Compute trace of U_target^dagger * U_approx
    overlap = np.trace(np.conj(U_target.T) @ U_approx)
    
    # Fidelity is |trace|^2 / d^2 where d is dimension
    d = U_target.shape[0]
    fidelity = np.abs(overlap)**2 / (d**2)
    
    return fidelity


def calculate_cost_statistics(cost_histories_list):
    """
    Calculate comprehensive statistics from multiple cost histories
    
    Args:
        cost_histories_list: List of cost history arrays
        
    Returns:
        dict: Statistical analysis of cost histories
    """
    if not cost_histories_list:
        return {}
    
    # Flatten all cost values
    all_final_costs = []
    all_improvements = []
    convergence_steps = []
    
    for history in cost_histories_list:
        if len(history) > 0:
            final_cost = history[-1]
            initial_cost = history[0]
            improvement = initial_cost - final_cost
            
            all_final_costs.append(final_cost)
            all_improvements.append(improvement)
            
            # Find convergence point (when cost stops decreasing significantly)
            if len(history) > 10:
                recent_costs = history[-10:]
                cost_diff = np.diff(recent_costs)
                if np.all(np.abs(cost_diff) < 1e-6):
                    convergence_steps.append(len(history) - 10)
                else:
                    convergence_steps.append(len(history))
            else:
                convergence_steps.append(len(history))
    
    # Calculate statistics
    stats = {
        'num_runs': len(cost_histories_list),
        'mean_final_cost': np.mean(all_final_costs) if all_final_costs else 0,
        'std_final_cost': np.std(all_final_costs) if all_final_costs else 0,
        'min_final_cost': np.min(all_final_costs) if all_final_costs else 0,
        'max_final_cost': np.max(all_final_costs) if all_final_costs else 0,
        'mean_improvement': np.mean(all_improvements) if all_improvements else 0,
        'std_improvement': np.std(all_improvements) if all_improvements else 0,
        'mean_convergence_steps': np.mean(convergence_steps) if convergence_steps else 0,
        'std_convergence_steps': np.std(convergence_steps) if convergence_steps else 0,
        'mean_final_fidelity': np.mean([1.0 - c for c in all_final_costs]) if all_final_costs else 0,
        'best_fidelity': 1.0 - np.min(all_final_costs) if all_final_costs else 0,
        'worst_fidelity': 1.0 - np.max(all_final_costs) if all_final_costs else 0
    }
    
    return stats


def analyze_optimization_performance(results_dict, method_names=None):
    """
    Analyze and compare optimization performance across different methods
    
    Args:
        results_dict: Dictionary of results from different methods
        method_names: Custom names for methods (optional)
        
    Returns:
        dict: Performance analysis
    """
    if not results_dict:
        return {}
    
    if method_names is None:
        method_names = list(results_dict.keys())
    
    analysis = {
        'methods': method_names,
        'comparison': {},
        'rankings': {}
    }
    
    # Extract performance metrics for each method
    performance_data = {}
    for method, results in results_dict.items():
        if 'cost_histories' in results:
            stats = calculate_cost_statistics(results['cost_histories'])
            performance_data[method] = stats
    
    # Rank methods by different criteria
    if performance_data:
        methods = list(performance_data.keys())
        
        # Rank by final fidelity (higher is better)
        fidelity_ranking = sorted(methods, 
                                key=lambda m: performance_data[m].get('mean_final_fidelity', 0), 
                                reverse=True)
        
        # Rank by convergence speed (lower steps is better)
        speed_ranking = sorted(methods,
                             key=lambda m: performance_data[m].get('mean_convergence_steps', float('inf')))
        
        # Rank by consistency (lower std is better)
        consistency_ranking = sorted(methods,
                                   key=lambda m: performance_data[m].get('std_final_cost', float('inf')))
        
        analysis['rankings'] = {
            'by_fidelity': fidelity_ranking,
            'by_speed': speed_ranking,
            'by_consistency': consistency_ranking
        }
        
        analysis['comparison'] = performance_data
    
    return analysis


def calculate_expectation_value_robust(state_vector_flat, pauli_op_sparse):
    """
    Calculate expectation value robustly handling contiguous array issues
    
    Args:
        state_vector_flat: Flattened state vector
        pauli_op_sparse: Sparse Pauli operator
        
    Returns:
        float: Real part of expectation value
    """
    if pauli_op_sparse is None or state_vector_flat is None:
        return np.nan
    
    # Fix "not contiguous" error by creating contiguous array copy
    try:
        sv = Statevector(np.ascontiguousarray(state_vector_flat))
        exp_val = sv.expectation_value(pauli_op_sparse)
        return exp_val.real
    except Exception as e_exp:
        print(f"    Error calculating expectation value: {e_exp}")
        return np.nan


def save_cost_history(cost_history, filename):
    """
    Save cost history to text file
    
    Args:
        cost_history: List/array of cost values
        filename: Output filename
    """
    try:
        # Extract numeric values from cost history
        numeric_costs = [float(c) for c in cost_history]
        
        print(f"Successfully extracted {len(numeric_costs)} cost values.")
        print(f"Writing results to file '{filename}'...")
        
        with open(filename, "w") as f:
            # Write header
            f.write("# Optimization_Step   Cost_Value\n")
            
            # Write data
            for step, cost_value in enumerate(numeric_costs):
                f.write(f"{step + 1:<20} {cost_value:<.12f}\n")
                
        print(f"\nCompleted! Successfully saved cost history to '{filename}'.")
        
    except Exception as e:
        print(f"Unexpected error occurred: {e}")


class OptimizationTracker:
    """
    Base class to track optimization progress
    """
    def __init__(self):
        self.costs = []
        self.fidelities = []
        self.iteration = 0
    
    def callback(self, cost):
        """
        Callback function for optimization
        
        Args:
            cost: Current cost value
        """
        self.costs.append(cost)
        self.fidelities.append(1.0 - cost)
        self.iteration += 1
        
        if self.iteration % 10 == 0:
            print(f"Iteration {self.iteration}: Cost = {cost:.6f}, Fidelity = {1.0 - cost:.6f}")
    
    def get_results(self):
        """
        Get optimization results
        
        Returns:
            dict: Dictionary with costs and fidelities
        """
        return {
            'costs': np.array(self.costs),
            'fidelities': np.array(self.fidelities),
            'iterations': np.arange(len(self.costs))
        }
