"""
Quantum Simulation Package

A modular package for quantum molecular simulation including:
- VQE ground state computation
- Time evolution with electric field
- Quantum circuit compilation
- Visualization tools

Modules:
    config: Configuration parameters
    quantum_setup: Molecular problem and operator setup
    ansatz: Quantum circuit ansatz definitions
    time_evolution: Time-dependent evolution calculations
    optimization: VQE and quantum compilation optimization
    visualization: Plotting and visualization tools
    utils: Utility functions and helpers
    main_workflow: Complete workflow orchestration

Example usage:
    from quantum_setup import setup_molecular_problem
    from vqe_optimization import run_vqe
    from visualization import plot_energy_convergence
    
    # Or run complete workflow
    from main_workflow import run_complete_workflow
    results = run_complete_workflow()
"""

__version__ = "1.0.0"
__author__ = "Quantum Simulation Team"

# Import main functions for convenience
from .quantum_setup import setup_molecular_problem, setup_qubit_operators, setup_ansatz
from .optimization import run_vqe, get_vqe_ground_state
from .time_evolution import compute_target_unitaries, E_field
from .visualization import plot_optimization_convergence, plot_energy_convergence
from .main_workflow import run_complete_workflow
from .utils import save_results, load_results, print_system_info

__all__ = [
    'setup_molecular_problem',
    'setup_qubit_operators', 
    'setup_ansatz',
    'run_vqe',
    'get_vqe_ground_state',
    'compute_target_unitaries',
    'E_field',
    'plot_optimization_convergence',
    'plot_energy_convergence',
    'run_complete_workflow',
    'save_results',
    'load_results',
    'print_system_info'
]
