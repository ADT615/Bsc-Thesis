"""
Main workflow module
Orchestrates the entire quantum simulation workflow
"""

from quantum_setup import (
    setup_molecular_problem, 
    setup_qubit_operators, 
    setup_ansatz, 
    get_exact_ground_state
)
from time_evolution import compute_target_unitaries, create_time_evolution_function
from ansatz import create_pauli_labels_from_operators
from optimization import OptimizationTracker, save_cost_history
from vqe_optimization import run_vqe, get_vqe_ground_state
from vqc_optimization import (
    run_vqa_time_evolution,
    evolve_states_with_vqa_unitaries,
    calculate_dipole_moments_from_states
)
from visualization import (
    plot_optimization_convergence, 
    plot_energy_convergence, 
    plot_electric_field,
    setup_matplotlib_style
)
from utils import save_results, create_timestamp, print_system_info
import config
import numpy as np


def run_complete_workflow():
    """
    Run the complete quantum simulation workflow
    
    Returns:
        dict: Results dictionary containing all computed data
    """
    print("="*60)
    print("QUANTUM SIMULATION WORKFLOW")
    print("="*60)
    
    # Setup matplotlib style
    setup_matplotlib_style()
    
    # Print system information
    print_system_info()
    print()
    
    # Step 1: Setup molecular problem
    print("Step 1: Setting up molecular problem...")
    problem, hamiltonian, dipole = setup_molecular_problem()
    print(f"Number of qubits: {problem.num_spin_orbitals}")
    print(f"Number of particles: {problem.num_particles}")
    print()
    
    # Step 2: Setup qubit operators
    print("Step 2: Setting up qubit operators...")
    qubit_ops = setup_qubit_operators(hamiltonian, dipole)
    H_0 = qubit_ops['H_0']
    H_static = qubit_ops['H_static']
    dipole_qubit = qubit_ops['dipole_qubit']
    dipole_matrix = qubit_ops['dipole_matrix']
    mapper = qubit_ops['mapper']
    print("Qubit operators setup complete.")
    print()
    
    # Step 3: Setup ansatz
    print("Step 3: Setting up ansatz...")
    ansatz = setup_ansatz(problem, mapper)
    print(f"Ansatz parameters: {ansatz.num_parameters}")
    print()
    
    # Step 4: Run VQE
    print("Step 4: Running VQE...")
    vqe_result = run_vqe(H_0, ansatz)
    print(f"VQE ground state energy: {vqe_result.optimal_value:.6f}")
    
    # Get exact result for comparison
    exact_result = get_exact_ground_state(H_0)
    exact_energy = exact_result.eigenvalues[0]
    print(f"Exact ground state energy: {exact_energy:.6f}")
    print(f"Energy error: {abs(vqe_result.optimal_value - exact_energy):.6f}")
    print()
    
    # Step 5: Get ground state wavefunction
    print("Step 5: Computing ground state wavefunction...")
    psi_0_vqe = get_vqe_ground_state(ansatz, vqe_result.optimal_parameters)
    print(f"Ground state norm: {np.linalg.norm(psi_0_vqe):.6f}")
    print()
    
    # Step 6: Compute target unitaries
    print("Step 6: Computing target unitaries for time evolution...")
    target_unitaries, times = compute_target_unitaries(H_static, dipole_matrix)
    print(f"Computed {len(target_unitaries)} target unitaries.")
    print()
    
    # Step 7: Create Pauli labels for ansatz
    print("Step 7: Creating Pauli labels...")
    pauli_labels = create_pauli_labels_from_operators(H_0, dipole_qubit)
    print(f"Number of unique Pauli strings: {len(pauli_labels)}")
    print()
    
    # Step 8: Create time evolution function
    print("Step 8: Creating time evolution function...")
    H_time = create_time_evolution_function(H_0, dipole_qubit)
    print("Time evolution function created.")
    print()
    
    # Prepare results
    results = {
        'vqe_energy': vqe_result.optimal_value,
        'exact_energy': exact_energy,
        'energy_error': abs(vqe_result.optimal_value - exact_energy),
        'optimal_parameters': vqe_result.optimal_parameters,
        'ground_state': psi_0_vqe,
        'target_unitaries': target_unitaries,
        'times': times,
        'pauli_labels': pauli_labels,
        'num_qubits': H_0.num_qubits,
        'num_parameters': ansatz.num_parameters,
        'problem_info': {
            'num_spin_orbitals': problem.num_spin_orbitals,
            'num_particles': problem.num_particles,
            'num_spatial_orbitals': problem.num_spatial_orbitals
        }
    }
    
    print("="*60)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return results


def save_workflow_results(results, prefix="quantum_sim"):
    """
    Save workflow results to files
    
    Args:
        results: Results dictionary from run_complete_workflow
        prefix: Filename prefix
    """
    timestamp = create_timestamp()
    filename = f"{prefix}_{timestamp}"
    
    # Save main results
    save_results(results, f"{filename}.json", format='json')
    
    # Save target unitaries separately (they might be large)
    np.save(f"{filename}_target_unitaries.npy", results['target_unitaries'])
    
    print(f"All results saved with prefix: {filename}")


def run_vqa_compilation_workflow():
    """
    Run VQA quantum compilation workflow
    
    Returns:
        dict: VQA compilation results
    """
    print("="*60)
    print("VQA QUANTUM COMPILATION WORKFLOW")
    print("="*60)
    
    # Setup matplotlib style
    setup_matplotlib_style()
    
    # Step 1: Setup molecular problem
    print("Step 1: Setting up molecular problem...")
    problem, hamiltonian, dipole = setup_molecular_problem()
    print(f"Number of qubits: {problem.num_spin_orbitals}")
    print()
    
    # Step 2: Setup qubit operators
    print("Step 2: Setting up qubit operators...")
    qubit_ops = setup_qubit_operators(hamiltonian, dipole)
    H_0 = qubit_ops['H_0']
    H_static = qubit_ops['H_static']
    dipole_qubit = qubit_ops['dipole_qubit']
    dipole_matrix = qubit_ops['dipole_matrix']
    mapper = qubit_ops['mapper']
    print("Qubit operators setup complete.")
    print()
    
    # Step 3: Run VQE for ground state
    print("Step 3: Running VQE for ground state...")
    ansatz = setup_ansatz(problem, mapper)
    vqe_result = run_vqe(H_0, ansatz)
    psi_0_vqe = get_vqe_ground_state(ansatz, vqe_result.optimal_parameters)
    print(f"VQE ground state energy: {vqe_result.optimal_value:.6f}")
    print()
    
    # Step 4: Compute target unitaries
    print("Step 4: Computing target unitaries...")
    target_unitaries, times = compute_target_unitaries(H_static, dipole_matrix)
    print(f"Computed {len(target_unitaries)} target unitaries.")
    print()
    
    # Step 5: Create Pauli labels for VQA
    print("Step 5: Creating Pauli labels for VQA...")
    pauli_labels = create_pauli_labels_from_operators(H_0, dipole_qubit)
    print(f"Number of unique Pauli strings: {len(pauli_labels)}")
    print()
    
    # Step 6: Run VQA time evolution
    print("Step 6: Running VQA quantum compilation...")
    vqa_results = run_vqa_time_evolution(
        target_unitaries, times, H_0.num_qubits, pauli_labels,
        num_layers=config.VQA_NUM_LAYERS,
        steps=config.VQA_STEPS,
        learning_rate=config.VQA_LEARNING_RATE,
        error_threshold=config.VQA_ERROR_THRESHOLD
    )
    print("VQA compilation completed.")
    print()
    
    # Step 7: Evolve states using VQA unitaries
    print("Step 7: Computing evolved states...")
    evolved_states = evolve_states_with_vqa_unitaries(
        vqa_results['optimized_unitaries'], psi_0_vqe, times
    )
    print(f"Computed evolved states for {len(evolved_states)} time points.")
    print()
    
    # Step 8: Calculate dipole moments
    print("Step 8: Calculating dipole moments...")
    times_plot, dipole_moments = calculate_dipole_moments_from_states(
        evolved_states, dipole_matrix, times
    )
    print(f"Calculated dipole moments for {len(times_plot)} time points.")
    print()
    
    # Prepare results
    results = {
        'vqe_energy': vqe_result.optimal_value,
        'optimal_parameters': vqe_result.optimal_parameters,
        'ground_state': psi_0_vqe,
        'target_unitaries': target_unitaries,
        'optimized_unitaries': vqa_results['optimized_unitaries'],
        'evolved_states': evolved_states,
        'cost_histories': vqa_results['cost_histories'],
        'times': times,
        'times_plot': times_plot,
        'dipole_moments': dipole_moments,
        'pauli_labels': pauli_labels,
        'num_qubits': H_0.num_qubits,
        'vqa_config': {
            'num_layers': config.VQA_NUM_LAYERS,
            'steps': config.VQA_STEPS,
            'learning_rate': config.VQA_LEARNING_RATE,
            'error_threshold': config.VQA_ERROR_THRESHOLD
        },
        'problem_info': {
            'num_spin_orbitals': problem.num_spin_orbitals,
            'num_particles': problem.num_particles,
            'num_spatial_orbitals': problem.num_spatial_orbitals
        }
    }
    
    print("="*60)
    print("VQA COMPILATION WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*60)
    
    return results


if __name__ == "__main__":
    # Run the complete workflow
    results = run_complete_workflow()
    
    # Save results
    save_workflow_results(results)
    
    # Run VQA compilation workflow
    print("\n" + "="*60)
    print("Starting VQA Compilation Workflow...")
    vqa_results = run_vqa_compilation_workflow()
    
    # Save VQA results
    save_workflow_results(vqa_results, prefix="vqa_compilation")
    
    # Save cost history from last time point
    if vqa_results['cost_histories']:
        last_time = max(vqa_results['cost_histories'].keys())
        last_cost_history = vqa_results['cost_histories'][last_time]
        save_cost_history(last_cost_history, "cost_thesis_perfect.txt")
    
    # Plot some basic visualizations
    from visualization import plot_electric_field, plot_comparison
    plot_electric_field(results['times'], "Electric Field Profile")
    
    # Plot comparison of dipole moments
    if 'dipole_moments' in vqa_results:
        plot_comparison(
            vqa_results['times_plot'],
            [vqa_results['dipole_moments']],
            ['VQA Compilation'],
            title="Dipole Moment from VQA Compilation",
            xlabel="Time (a.u.)",
            ylabel="Dipole moment (z) (a.u.)"
        )
