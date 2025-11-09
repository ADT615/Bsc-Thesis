"""
VQC (Variational Quantum Compilation) Optimization module
Chứa các hàm optimization cho quantum compilation using VQA approach
"""

import numpy as np
import math
import pennylane as qml
import pennylane.numpy as pnp
from numpy.random import Generator, PCG64

import config


def vqa_cost(thetas, num_qubits, pauli_labels, num_layers, target_unitary):
    """
    VQA cost function for single target unitary - following original approach
    
    Args:
        thetas: Variational parameters
        num_qubits: Number of qubits
        pauli_labels: List of Pauli string labels
        num_layers: Number of ansatz layers
        target_unitary: Target unitary matrix
        
    Returns:
        float: Cost value (HST distance)
    """
    from ansatz import pennylane_ansatz_from_qiskit_pauli_evo
    
    # Get ansatz matrix using PennyLane
    ansatz_matrix = qml.matrix(pennylane_ansatz_from_qiskit_pauli_evo, wire_order=list(range(num_qubits)))(
        thetas, num_qubits, pauli_labels, num_layers
    )
    
    # Calculate HST (Hilbert-Schmidt Test) distance
    return calculate_hst_cost(ansatz_matrix, target_unitary)


def calculate_hst_cost(U_ansatz, U_target):
    """
    Calculate Hilbert-Schmidt Test (HST) cost function
    Following the original cost function from sync/cost.py
    
    Args:
        U_ansatz: Ansatz unitary matrix
        U_target: Target unitary matrix
        
    Returns:
        float: HST cost value
    """
    try:
        # Try to import from sync module if available
        from sync.cost import c_hst
        return c_hst(U_ansatz, U_target)
    except ImportError:
        # Fallback to local implementation
        d = U_target.shape[0]
        overlap = np.trace(np.conj(U_target.T) @ U_ansatz)
        return 1 - (1/(d**2)) * (np.abs(overlap))**2


def train_vqa_for_time(target_unitary, num_qubits, pauli_labels, num_layers, init_thetas, 
                      steps=None, learning_rate=None, error_threshold=None):
    """
    Train VQA for a single time point - following original approach
    
    Args:
        target_unitary: Target unitary matrix for this time
        num_qubits: Number of qubits
        pauli_labels: List of Pauli string labels
        num_layers: Number of ansatz layers
        init_thetas: Initial parameters
        steps: Number of optimization steps (uses config default if None)
        learning_rate: Learning rate for optimizer (uses config default if None)
        error_threshold: Convergence threshold (uses config default if None)
        
    Returns:
        tuple: (optimized_thetas, cost_history)
    """
    # Use config defaults if not specified
    if steps is None:
        steps = config.VQA_STEPS
    if learning_rate is None:
        learning_rate = config.VQA_LEARNING_RATE
    if error_threshold is None:
        error_threshold = config.VQA_ERROR_THRESHOLD
    
    # Initialize optimizer
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    thetas = init_thetas.copy()
    cost_history = []
    
    for n in range(steps):
        # Define cost function for this step
        cost_fn = lambda th: vqa_cost(th, num_qubits, pauli_labels, num_layers, target_unitary)
        
        # Optimization step
        thetas, prev_cost = opt.step_and_cost(cost_fn, thetas)
        
        # Log progress
        if (n + 1) % (steps // 2 if steps >= 2 else 1) == 0:
            print(f"    Opt. step {n+1}/{steps}, Cost: {prev_cost:.6f}")
        
        # Check convergence
        if prev_cost < error_threshold:
            print(f"    Reached error threshold at step {n+1}")
            # break  # Commented out to match original behavior
        
        cost_history.append(prev_cost)
    
    return thetas, cost_history


def run_vqa_time_evolution(target_unitaries_list, times, num_qubits, pauli_labels, 
                          num_layers=None, steps=None, learning_rate=None, error_threshold=None):
    """
    Run complete VQA time evolution simulation
    
    Args:
        target_unitaries_list: List of target unitary matrices
        times: Time points array
        num_qubits: Number of qubits
        pauli_labels: List of Pauli string labels
        num_layers: Number of ansatz layers (uses config default if None)
        steps: Optimization steps per time point (uses config default if None)
        learning_rate: Learning rate (uses config default if None)
        error_threshold: Convergence threshold (uses config default if None)
        
    Returns:
        dict: Results containing optimized unitaries and evolved states
    """
    # Use config defaults if not specified
    if num_layers is None:
        num_layers = config.VQA_NUM_LAYERS
    if steps is None:
        steps = config.VQA_STEPS
    if learning_rate is None:
        learning_rate = config.VQA_LEARNING_RATE
    if error_threshold is None:
        error_threshold = config.VQA_ERROR_THRESHOLD
    
    print("Starting VQA time evolution simulation with PennyLane")
    print(f"Configuration: layers={num_layers}, steps={steps}, lr={learning_rate}")
    
    # Initialize parameters
    num_thetas = len(pauli_labels) * num_layers
    rng = Generator(PCG64())
    init_thetas = pnp.array(2 * math.pi * rng.random(size=num_thetas), requires_grad=True)
    
    print(f"Number of parameters: {num_thetas}")
    print(f"Number of Pauli strings: {len(pauli_labels)}")
    
    optimized_unitaries = {}
    all_cost_histories = {}
    
    # VQA training for each time point
    for i, t in enumerate(times):
        print(f"\n--- Time t = {t:.4f} ({i+1}/{len(times)}) ---")
        target = target_unitaries_list[i]
        
        # Train VQA for this time point
        thetas, cost_history = train_vqa_for_time(
            target, num_qubits, pauli_labels, num_layers, init_thetas,
            steps=steps, learning_rate=learning_rate, error_threshold=error_threshold
        )
        
        # Get optimized unitary matrix
        from ansatz import pennylane_ansatz_from_qiskit_pauli_evo
        U_theta_t = qml.matrix(
            pennylane_ansatz_from_qiskit_pauli_evo, wire_order=range(num_qubits)
        )(thetas, num_qubits, pauli_labels, num_layers)
        
        # Store results
        optimized_unitaries[t] = U_theta_t.numpy() if hasattr(U_theta_t, 'numpy') else np.asarray(U_theta_t)
        all_cost_histories[t] = cost_history
        
        # Print final fidelity for this time point
        final_cost = cost_history[-1]
        final_fidelity = 1.0 - final_cost
        print(f"    Final cost: {final_cost:.6f}, Final fidelity: {final_fidelity:.6f}")
    
    return {
        'optimized_unitaries': optimized_unitaries,
        'cost_histories': all_cost_histories,
        'times': times,
        'final_thetas': thetas,
        'config': {
            'num_layers': num_layers,
            'steps': steps,
            'learning_rate': learning_rate,
            'error_threshold': error_threshold,
            'num_parameters': num_thetas
        }
    }


def evolve_states_with_vqa_unitaries(optimized_unitaries, initial_state, times):
    """
    Evolve quantum states using optimized VQA unitaries
    
    Args:
        optimized_unitaries: Dictionary of optimized unitary matrices
        initial_state: Initial quantum state vector
        times: Time points array
        
    Returns:
        dict: Evolved states at each time point
    """
    evolved_states = {}
    psi_0_col = initial_state.reshape(-1, 1)
    
    print(f"Evolving states for {len(times)} time points...")
    
    for t in times:
        if t in optimized_unitaries:
            U_t = optimized_unitaries[t]
            psi_t_col_approx = U_t @ psi_0_col
            psi_t_approx = psi_t_col_approx.flatten()
            evolved_states[t] = psi_t_approx
    
    print(f"Successfully evolved {len(evolved_states)} states.")
    return evolved_states


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
        from qiskit.quantum_info import Statevector
        sv = Statevector(np.ascontiguousarray(state_vector_flat))
        exp_val = sv.expectation_value(pauli_op_sparse)
        return exp_val.real
    except Exception as e_exp:
        print(f"    Error calculating expectation value: {e_exp}")
        return np.nan


def calculate_dipole_moments_from_states(evolved_states, dipole_operator, times):
    """
    Calculate dipole moments from evolved quantum states
    
    Args:
        evolved_states: Dictionary of evolved states at different times
        dipole_operator: Dipole moment operator
        times: Time points array
        
    Returns:
        tuple: (times_array, dipole_moments_array)
    """
    times_plot = []
    dipole_moments = []
    
    print("Calculating dipole moments from evolved states...")
    
    for t in sorted(times):
        if t in evolved_states:
            psi_t = evolved_states[t]
            if psi_t is not None:
                # Method 1: Direct matrix multiplication (for matrix operators)
                if hasattr(dipole_operator, 'to_matrix'):
                    dipole_matrix = dipole_operator.to_matrix()
                    exp_val = np.real(psi_t.conj().T @ dipole_matrix @ psi_t)
                # Method 2: Direct matrix multiplication (for numpy arrays)
                elif isinstance(dipole_operator, np.ndarray):
                    exp_val = np.real(psi_t.conj().T @ dipole_operator @ psi_t)
                # Method 3: Using Qiskit expectation value (for SparsePauliOp)
                else:
                    exp_val = calculate_expectation_value_robust(psi_t, dipole_operator)
                
                times_plot.append(t)
                dipole_moments.append(exp_val)
    
    print(f"Calculated dipole moments for {len(times_plot)} time points.")
    return np.array(times_plot), np.array(dipole_moments)


def save_cost_history(cost_history, filename):
    """
    Save cost history to text file - following original format
    
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


class VQCTracker:
    """
    Class to track VQC optimization progress across multiple time points
    """
    def __init__(self):
        self.fidelities_per_time = {}
        self.convergence_times = {}
        self.cost_histories = {}
        self.final_costs = {}
        
    def track_time_point(self, time_point, cost_history):
        """
        Track optimization progress for a specific time point
        
        Args:
            time_point: Time value
            cost_history: Cost history for this time point
        """
        self.cost_histories[time_point] = cost_history
        self.fidelities_per_time[time_point] = [1.0 - cost for cost in cost_history]
        self.final_costs[time_point] = cost_history[-1] if cost_history else float('inf')
        
        # Find convergence point (when cost stops decreasing significantly)
        if len(cost_history) > 10:
            cost_diff = np.diff(cost_history[-10:])
            if np.all(np.abs(cost_diff) < 1e-6):
                self.convergence_times[time_point] = len(cost_history) - 10
            else:
                self.convergence_times[time_point] = len(cost_history)
        else:
            self.convergence_times[time_point] = len(cost_history)
    
    def get_summary_stats(self):
        """
        Get summary statistics across all time points
        
        Returns:
            dict: Summary statistics
        """
        all_final_fidelities = []
        all_convergence_steps = []
        all_final_costs = []
        
        for t, fidelities in self.fidelities_per_time.items():
            if fidelities:
                all_final_fidelities.append(fidelities[-1])
                all_convergence_steps.append(self.convergence_times.get(t, len(fidelities)))
                all_final_costs.append(self.final_costs.get(t, float('inf')))
        
        return {
            'mean_final_fidelity': np.mean(all_final_fidelities) if all_final_fidelities else 0,
            'std_final_fidelity': np.std(all_final_fidelities) if all_final_fidelities else 0,
            'mean_final_cost': np.mean(all_final_costs) if all_final_costs else 0,
            'std_final_cost': np.std(all_final_costs) if all_final_costs else 0,
            'mean_convergence_steps': np.mean(all_convergence_steps) if all_convergence_steps else 0,
            'total_time_points': len(self.fidelities_per_time),
            'best_fidelity': max(all_final_fidelities) if all_final_fidelities else 0,
            'worst_fidelity': min(all_final_fidelities) if all_final_fidelities else 0,
            'final_fidelities': all_final_fidelities,
            'final_costs': all_final_costs,
            'convergence_steps': all_convergence_steps
        }
    
    def get_time_analysis(self):
        """
        Get time-specific analysis
        
        Returns:
            dict: Time analysis data
        """
        analysis = {
            'best_time': None,
            'worst_time': None,
            'fastest_convergence': None,
            'slowest_convergence': None
        }
        
        if self.final_costs:
            # Best and worst performing time points
            best_time = min(self.final_costs.keys(), key=lambda t: self.final_costs[t])
            worst_time = max(self.final_costs.keys(), key=lambda t: self.final_costs[t])
            
            analysis['best_time'] = {
                'time': best_time,
                'cost': self.final_costs[best_time],
                'fidelity': 1.0 - self.final_costs[best_time]
            }
            
            analysis['worst_time'] = {
                'time': worst_time,
                'cost': self.final_costs[worst_time],
                'fidelity': 1.0 - self.final_costs[worst_time]
            }
        
        if self.convergence_times:
            # Fastest and slowest convergence
            fastest_time = min(self.convergence_times.keys(), key=lambda t: self.convergence_times[t])
            slowest_time = max(self.convergence_times.keys(), key=lambda t: self.convergence_times[t])
            
            analysis['fastest_convergence'] = {
                'time': fastest_time,
                'steps': self.convergence_times[fastest_time]
            }
            
            analysis['slowest_convergence'] = {
                'time': slowest_time,
                'steps': self.convergence_times[slowest_time]
            }
        
        return analysis


def save_vqc_results(vqc_results, filename_prefix="vqc_results"):
    """
    Save VQC results to files
    
    Args:
        vqc_results: VQC results dictionary
        filename_prefix: Prefix for output files
    """
    from utils import save_results, create_timestamp
    
    timestamp = create_timestamp()
    
    # Prepare data for saving (exclude large matrices)
    save_data = {
        'config': vqc_results.get('config', {}),
        'times': vqc_results.get('times', []),
        'final_thetas': vqc_results.get('final_thetas', []),
        'cost_histories': vqc_results.get('cost_histories', {})
    }
    
    # Save main results
    filename = f"{filename_prefix}_{timestamp}.json"
    save_results(save_data, filename)
    
    # Save cost history from last time point
    if 'cost_histories' in vqc_results and vqc_results['cost_histories']:
        last_time = max(vqc_results['cost_histories'].keys())
        last_cost_history = vqc_results['cost_histories'][last_time]
        cost_filename = f"{filename_prefix}_cost_history_{timestamp}.txt"
        save_cost_history(last_cost_history, cost_filename)
    
    # Save unitaries separately (they might be large)
    if 'optimized_unitaries' in vqc_results:
        unitaries_filename = f"{filename_prefix}_unitaries_{timestamp}.npy"
        np.save(unitaries_filename, vqc_results['optimized_unitaries'])
        print(f"Optimized unitaries saved to {unitaries_filename}")
    
    print(f"VQC results saved with prefix: {filename_prefix}_{timestamp}")
    
    return filename
