"""
Optimization module - Common utilities
Chứa các utility functions chung cho optimization
"""

import numpy as np


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


def check_unitary(U, tolerance=1e-10):
    """
    Check if a matrix is unitary
    
    Args:
        U: Matrix to check
        tolerance: Numerical tolerance
        
    Returns:
        tuple: (is_unitary, error)
    """
    U = np.array(U, dtype=complex)
    U_dag = np.conj(U.T)
    identity = U @ U_dag
    error = np.max(np.abs(identity - np.eye(U.shape[0])))
    return error < tolerance, error


def compare_unitaries(U1, U2, labels=["U1", "U2"]):
    """
    Compare two unitary matrices
    
    Args:
        U1: First unitary matrix
        U2: Second unitary matrix
        labels: Labels for the matrices
        
    Returns:
        dict: Comparison results
    """
    fidelity = unitary_fidelity(U1, U2)
    
    # Check unitarity
    is_unitary_1, error_1 = check_unitary(U1)
    is_unitary_2, error_2 = check_unitary(U2)
    
    # Frobenius distance
    frobenius_dist = np.linalg.norm(U1 - U2, 'fro')
    
    comparison = {
        'fidelity': fidelity,
        'frobenius_distance': frobenius_dist,
        f'{labels[0]}_unitary': is_unitary_1,
        f'{labels[0]}_unitary_error': error_1,
        f'{labels[1]}_unitary': is_unitary_2,
        f'{labels[1]}_unitary_error': error_2,
        'spectral_norm_diff': np.linalg.norm(U1 - U2, 2)
    }
    
    return comparison


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
    
    def reset(self):
        """Reset tracker state"""
        self.costs = []
        self.fidelities = []
        self.iteration = 0
    
    def get_convergence_info(self, window_size=10, threshold=1e-6):
        """
        Get convergence information
        
        Args:
            window_size: Window size for checking convergence
            threshold: Convergence threshold
            
        Returns:
            dict: Convergence information
        """
        if len(self.costs) < window_size:
            return {'converged': False, 'convergence_step': None}
        
        # Check if cost has converged in the last window_size steps
        recent_costs = self.costs[-window_size:]
        cost_diff = np.diff(recent_costs)
        
        if np.all(np.abs(cost_diff) < threshold):
            convergence_step = len(self.costs) - window_size
            return {
                'converged': True,
                'convergence_step': convergence_step,
                'final_cost': self.costs[-1],
                'final_fidelity': self.fidelities[-1]
            }
        
        return {'converged': False, 'convergence_step': None}


def calculate_cost_statistics(cost_histories):
    """
    Calculate statistics from multiple cost histories
    
    Args:
        cost_histories: Dictionary of cost histories (time -> cost_list)
        
    Returns:
        dict: Cost statistics
    """
    if not cost_histories:
        return {}
    
    all_final_costs = []
    all_initial_costs = []
    all_improvements = []
    convergence_steps = []
    
    for time_point, costs in cost_histories.items():
        if costs:
            final_cost = costs[-1]
            initial_cost = costs[0]
            improvement = initial_cost - final_cost
            
            all_final_costs.append(final_cost)
            all_initial_costs.append(initial_cost)
            all_improvements.append(improvement)
            
            # Find convergence point
            if len(costs) > 10:
                cost_diff = np.diff(costs[-10:])
                if np.all(np.abs(cost_diff) < 1e-6):
                    convergence_steps.append(len(costs) - 10)
                else:
                    convergence_steps.append(len(costs))
            else:
                convergence_steps.append(len(costs))
    
    stats = {
        'num_time_points': len(cost_histories),
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
                      steps=300, learning_rate=0.01, error_threshold=1e-6):
    """
    Train VQA for a single time point - following original approach
    
    Args:
        target_unitary: Target unitary matrix for this time
        num_qubits: Number of qubits
        pauli_labels: List of Pauli string labels
        num_layers: Number of ansatz layers
        init_thetas: Initial parameters
        steps: Number of optimization steps
        learning_rate: Learning rate for optimizer
        error_threshold: Convergence threshold
        
    Returns:
        tuple: (optimized_thetas, cost_history)
    """
    import pennylane.numpy as pnp
    
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
                          num_layers=6, steps=300, learning_rate=0.01, error_threshold=1e-6):
    """
    Run complete VQA time evolution simulation
    
    Args:
        target_unitaries_list: List of target unitary matrices
        times: Time points array
        num_qubits: Number of qubits
        pauli_labels: List of Pauli string labels
        num_layers: Number of ansatz layers
        steps: Optimization steps per time point
        learning_rate: Learning rate
        error_threshold: Convergence threshold
        
    Returns:
        dict: Results containing optimized unitaries and evolved states
    """
    import pennylane.numpy as pnp
    import math
    from numpy.random import Generator, PCG64
    
    print("Starting VQA time evolution simulation with PennyLane")
    
    # Initialize parameters
    num_thetas = len(pauli_labels) * num_layers
    rng = Generator(PCG64())
    init_thetas = pnp.array(2 * math.pi * rng.random(size=num_thetas), requires_grad=True)
    
    optimized_unitaries = {}
    evolved_states = {}
    all_cost_histories = {}
    
    # VQA training for each time point
    for i, t in enumerate(times):
        print(f"\n--- Time t = {t:.4f} ---")
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
    
    return {
        'optimized_unitaries': optimized_unitaries,
        'evolved_states': evolved_states,
        'cost_histories': all_cost_histories,
        'times': times,
        'final_thetas': thetas
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
    
    for t in times:
        if t in optimized_unitaries:
            U_t = optimized_unitaries[t]
            psi_t_col_approx = U_t @ psi_0_col
            psi_t_approx = psi_t_col_approx.flatten()
            evolved_states[t] = psi_t_approx
    
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
    
    for t in sorted(times):
        if t in evolved_states:
            psi_t = evolved_states[t]
            if psi_t is not None:
                # Method 1: Direct matrix multiplication (for matrix operators)
                if hasattr(dipole_operator, 'to_matrix'):
                    dipole_matrix = dipole_operator.to_matrix()
                    exp_val = np.real(psi_t.conj().T @ dipole_matrix @ psi_t)
                # Method 2: Using Qiskit expectation value (for SparsePauliOp)
                else:
                    exp_val = calculate_expectation_value_robust(psi_t, dipole_operator)
                
                times_plot.append(t)
                dipole_moments.append(exp_val)
    
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


class OptimizationTracker:
    """
    Class to track optimization progress
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


class VQATracker(OptimizationTracker):
    """
    Extended optimization tracker for VQA with additional features
    """
    def __init__(self):
        super().__init__()
        self.fidelities_per_time = {}
        self.convergence_times = {}
        
    def track_time_point(self, time_point, cost_history):
        """
        Track optimization progress for a specific time point
        
        Args:
            time_point: Time value
            cost_history: Cost history for this time point
        """
        self.fidelities_per_time[time_point] = [1.0 - cost for cost in cost_history]
        
        # Find convergence point (when cost stops decreasing significantly)
        if len(cost_history) > 10:
            cost_diff = np.diff(cost_history[-10:])
            if np.all(np.abs(cost_diff) < 1e-6):
                self.convergence_times[time_point] = len(cost_history) - 10
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
        
        for t, fidelities in self.fidelities_per_time.items():
            if fidelities:
                all_final_fidelities.append(fidelities[-1])
                all_convergence_steps.append(self.convergence_times.get(t, len(fidelities)))
        
        return {
            'mean_final_fidelity': np.mean(all_final_fidelities) if all_final_fidelities else 0,
            'std_final_fidelity': np.std(all_final_fidelities) if all_final_fidelities else 0,
            'mean_convergence_steps': np.mean(all_convergence_steps) if all_convergence_steps else 0,
            'total_time_points': len(self.fidelities_per_time),
            'final_fidelities': all_final_fidelities,
            'convergence_steps': all_convergence_steps
        }
    """
    Class to track optimization progress
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
