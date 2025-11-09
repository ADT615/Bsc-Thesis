"""
VQE Optimization module
Chứa các hàm optimization cho Variational Quantum Eigensolver (VQE)
"""

import numpy as np
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import SLSQP, COBYLA, L_BFGS_B, SPSA, NELDER_MEAD
from qiskit_algorithms import VQE
from qiskit_algorithms.eigensolvers import NumPyEigensolver
from qiskit.quantum_info import Statevector

from config import OPTIMIZER_MAXITER


def run_vqe(qubit_jw_op, ansatz, optimizer_name='SLSQP', maxiter=None):
    """
    Run VQE optimization
    
    Args:
        qubit_jw_op: Qubit Hamiltonian operator
        ansatz: Quantum circuit ansatz
        optimizer_name: Name of optimizer ('SLSQP', 'COBYLA', 'L_BFGS_B', 'SPSA', 'NELDER_MEAD')
        maxiter: Maximum iterations (uses config default if None)
        
    Returns:
        VQE result object
    """
    estimator = Estimator()
    
    # Select optimizer
    if maxiter is None:
        maxiter = OPTIMIZER_MAXITER
        
    optimizer_map = {
        'SLSQP': SLSQP(maxiter=maxiter),
        'COBYLA': COBYLA(maxiter=maxiter),
        'L_BFGS_B': L_BFGS_B(maxiter=maxiter),
        'SPSA': SPSA(maxiter=maxiter),
        'NELDER_MEAD': NELDER_MEAD(maxiter=maxiter)
    }
    
    if optimizer_name not in optimizer_map:
        print(f"Warning: Unknown optimizer {optimizer_name}, using SLSQP")
        optimizer_name = 'SLSQP'
    
    optimizer = optimizer_map[optimizer_name]
    print(f"Using {optimizer_name} optimizer with maxiter={maxiter}")
    
    vqe = VQE(estimator, ansatz, optimizer)
    result = vqe.compute_minimum_eigenvalue(qubit_jw_op)
    
    return result


def get_vqe_ground_state(ansatz, optimal_parameters):
    """
    Get ground state from VQE result
    
    Args:
        ansatz: Quantum circuit ansatz
        optimal_parameters: Optimal parameters from VQE
        
    Returns:
        ndarray: Ground state vector
    """
    circuit = ansatz.assign_parameters(optimal_parameters)
    psi_0_vqe = np.array(Statevector(circuit).data)
    return psi_0_vqe


def get_exact_ground_state(qubit_jw_op):
    """
    Compute exact ground state using NumPy eigensolver for comparison
    
    Args:
        qubit_jw_op: Qubit hamiltonian operator
        
    Returns:
        exact_result: Exact eigenvalue result
    """
    numpy_solver = NumPyEigensolver()
    exact_result = numpy_solver.compute_eigenvalues(qubit_jw_op)
    return exact_result


def compare_vqe_with_exact(vqe_result, exact_result):
    """
    Compare VQE result with exact solution
    
    Args:
        vqe_result: VQE optimization result
        exact_result: Exact eigenvalue result
        
    Returns:
        dict: Comparison statistics
    """
    vqe_energy = vqe_result.optimal_value
    exact_energy = exact_result.eigenvalues[0].real
    energy_error = abs(vqe_energy - exact_energy)
    relative_error = energy_error / abs(exact_energy) if exact_energy != 0 else float('inf')
    
    comparison = {
        'vqe_energy': vqe_energy,
        'exact_energy': exact_energy,
        'energy_error': energy_error,
        'relative_error': relative_error,
        'function_evaluations': vqe_result.optimizer_evals,
        'optimal_parameters': vqe_result.optimal_parameters,
        'converged': hasattr(vqe_result, 'optimizer_result') and getattr(vqe_result.optimizer_result, 'success', True)
    }
    
    return comparison


def run_vqe_with_multiple_optimizers(qubit_jw_op, ansatz, optimizers=['SLSQP', 'COBYLA']):
    """
    Run VQE with multiple optimizers and compare results
    
    Args:
        qubit_jw_op: Qubit Hamiltonian operator
        ansatz: Quantum circuit ansatz
        optimizers: List of optimizer names to try
        
    Returns:
        dict: Results from all optimizers
    """
    results = {}
    
    # Get exact solution for comparison
    exact_result = get_exact_ground_state(qubit_jw_op)
    exact_energy = exact_result.eigenvalues[0].real
    
    print(f"Exact ground state energy: {exact_energy:.6f}")
    print("="*60)
    
    for optimizer_name in optimizers:
        print(f"\nRunning VQE with {optimizer_name}...")
        try:
            vqe_result = run_vqe(qubit_jw_op, ansatz, optimizer_name)
            comparison = compare_vqe_with_exact(vqe_result, exact_result)
            
            results[optimizer_name] = {
                'vqe_result': vqe_result,
                'comparison': comparison
            }
            
            print(f"  VQE energy: {comparison['vqe_energy']:.6f}")
            print(f"  Energy error: {comparison['energy_error']:.6f}")
            print(f"  Relative error: {comparison['relative_error']:.2e}")
            print(f"  Function evaluations: {comparison['function_evaluations']}")
            print(f"  Converged: {comparison['converged']}")
            
        except Exception as e:
            print(f"  Error with {optimizer_name}: {e}")
            results[optimizer_name] = {'error': str(e)}
    
    return results


class VQETracker:
    """
    Class to track VQE optimization progress
    """
    def __init__(self):
        self.energies = []
        self.iterations = []
        self.iteration_count = 0
        self.best_energy = float('inf')
        self.best_params = None
        
    def callback(self, eval_count, parameters, energy, stddev):
        """
        Callback function for VQE optimization tracking
        
        Args:
            eval_count: Number of function evaluations
            parameters: Current parameters
            energy: Current energy estimate
            stddev: Standard deviation of energy estimate
        """
        self.iteration_count += 1
        self.energies.append(energy)
        self.iterations.append(self.iteration_count)
        
        if energy < self.best_energy:
            self.best_energy = energy
            self.best_params = parameters.copy() if hasattr(parameters, 'copy') else list(parameters)
        
        if self.iteration_count % 10 == 0:
            print(f"Iteration {self.iteration_count}: Energy = {energy:.6f} ± {stddev:.6f}")
    
    def get_convergence_data(self):
        """
        Get convergence data
        
        Returns:
            dict: Convergence statistics
        """
        if not self.energies:
            return {}
            
        return {
            'energies': np.array(self.energies),
            'iterations': np.array(self.iterations),
            'best_energy': self.best_energy,
            'best_params': self.best_params,
            'total_iterations': self.iteration_count,
            'energy_variance': np.var(self.energies[-10:]) if len(self.energies) >= 10 else 0,
            'final_energy': self.energies[-1] if self.energies else None
        }


def run_vqe_with_tracking(qubit_jw_op, ansatz, optimizer_name='SLSQP'):
    """
    Run VQE with detailed tracking of optimization progress
    
    Args:
        qubit_jw_op: Qubit Hamiltonian operator
        ansatz: Quantum circuit ansatz
        optimizer_name: Optimizer name
        
    Returns:
        tuple: (vqe_result, tracker)
    """
    tracker = VQETracker()
    estimator = Estimator()
    
    # Setup optimizer
    optimizer_map = {
        'SLSQP': SLSQP(maxiter=OPTIMIZER_MAXITER),
        'COBYLA': COBYLA(maxiter=OPTIMIZER_MAXITER),
        'L_BFGS_B': L_BFGS_B(maxiter=OPTIMIZER_MAXITER)
    }
    
    optimizer = optimizer_map.get(optimizer_name, SLSQP(maxiter=OPTIMIZER_MAXITER))
    
    # Create VQE with callback
    vqe = VQE(estimator, ansatz, optimizer, callback=tracker.callback)
    
    print(f"Running VQE with tracking using {optimizer_name}...")
    result = vqe.compute_minimum_eigenvalue(qubit_jw_op)
    
    return result, tracker


def save_vqe_results(vqe_results, filename_prefix="vqe_results"):
    """
    Save VQE results to files
    
    Args:
        vqe_results: Dictionary of VQE results
        filename_prefix: Prefix for output files
    """
    from utils import save_results, create_timestamp
    
    timestamp = create_timestamp()
    
    # Prepare summary data
    summary_data = {}
    for optimizer_name, result_data in vqe_results.items():
        if 'comparison' in result_data:
            comp = result_data['comparison']
            summary_data[optimizer_name] = {
                'vqe_energy': comp['vqe_energy'],
                'exact_energy': comp['exact_energy'],
                'energy_error': comp['energy_error'],
                'relative_error': comp['relative_error'],
                'function_evaluations': comp['function_evaluations'],
                'converged': comp['converged']
            }
    
    # Save summary
    filename = f"{filename_prefix}_summary_{timestamp}.json"
    save_results(summary_data, filename)
    
    print(f"VQE results saved to {filename}")
    
    return filename


def analyze_vqe_performance(vqe_results):
    """
    Analyze VQE performance across different optimizers
    
    Args:
        vqe_results: Dictionary of VQE results from multiple optimizers
        
    Returns:
        dict: Performance analysis
    """
    if not vqe_results:
        return {}
    
    analysis = {
        'best_optimizer': None,
        'best_energy': float('inf'),
        'fastest_optimizer': None,
        'min_evaluations': float('inf'),
        'most_accurate': None,
        'min_error': float('inf'),
        'summary': {}
    }
    
    for optimizer_name, result_data in vqe_results.items():
        if 'comparison' not in result_data:
            continue
            
        comp = result_data['comparison']
        
        # Track best energy
        if comp['vqe_energy'] < analysis['best_energy']:
            analysis['best_energy'] = comp['vqe_energy']
            analysis['best_optimizer'] = optimizer_name
        
        # Track fastest (fewest evaluations)
        if comp['function_evaluations'] < analysis['min_evaluations']:
            analysis['min_evaluations'] = comp['function_evaluations']
            analysis['fastest_optimizer'] = optimizer_name
        
        # Track most accurate
        if comp['energy_error'] < analysis['min_error']:
            analysis['min_error'] = comp['energy_error']
            analysis['most_accurate'] = optimizer_name
        
        # Store summary
        analysis['summary'][optimizer_name] = {
            'energy': comp['vqe_energy'],
            'error': comp['energy_error'],
            'evaluations': comp['function_evaluations'],
            'converged': comp['converged']
        }
    
    return analysis
