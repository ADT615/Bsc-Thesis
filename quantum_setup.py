"""
Quantum setup module
Chứa các hàm để setup molecular hamiltonian, dipole operators, và các quantum operators
"""

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.properties import ElectronicDipoleMoment
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit_algorithms.eigensolvers import NumPyEigensolver
import numpy as np

from config import MOLECULE_GEOMETRY, BASIS_SET, CHARGE, SPIN, EXCITATIONS, REPS


def setup_molecular_problem():
    """
    Setup molecular problem using PySCF driver
    
    Returns:
        problem: Molecular problem object
        hamiltonian: Molecular hamiltonian
        dipole: Electronic dipole moment
    """
    driver = PySCFDriver(
        atom=MOLECULE_GEOMETRY,
        basis=BASIS_SET,
        charge=CHARGE,
        spin=SPIN,
        unit=DistanceUnit.ANGSTROM,
    )
    
    problem = driver.run()
    hamiltonian = problem.hamiltonian
    dipole = problem.properties.electronic_dipole_moment
    
    return problem, hamiltonian, dipole


def setup_qubit_operators(hamiltonian, dipole):
    """
    Map second quantized operators to qubit operators
    
    Args:
        hamiltonian: Molecular hamiltonian
        dipole: Electronic dipole moment
        
    Returns:
        dict: Dictionary containing all qubit operators and matrices
    """
    mapper = JordanWignerMapper()
    
    # Setup dipole operators
    dipole_ops = dipole.second_q_ops()
    x_dipole_op = dipole_ops["XDipole"]
    y_dipole_op = dipole_ops["YDipole"] 
    z_dipole_op = dipole_ops["ZDipole"]
    
    qubit_dipole_ops = {
        "XDipole": mapper.map(x_dipole_op),
        "YDipole": mapper.map(y_dipole_op),
        "ZDipole": mapper.map(z_dipole_op),
    }
    
    qubit_dipole_ops_matrix = {
        "XDipole": qubit_dipole_ops["XDipole"].to_matrix(),
        "YDipole": qubit_dipole_ops["YDipole"].to_matrix(),
        "ZDipole": qubit_dipole_ops["ZDipole"].to_matrix(),
    }
    
    # Combined dipole operator
    dipole_qubit = (qubit_dipole_ops["XDipole"] + 
                   qubit_dipole_ops["YDipole"] + 
                   qubit_dipole_ops["ZDipole"])
    dipole_matrix = (qubit_dipole_ops_matrix["XDipole"] + 
                    qubit_dipole_ops_matrix["YDipole"] + 
                    qubit_dipole_ops_matrix["ZDipole"])
    
    # Setup hamiltonian
    second_q_op = hamiltonian.second_q_op()
    qubit_jw_op = mapper.map(second_q_op)
    H_static = qubit_jw_op.to_matrix()
    
    return {
        'mapper': mapper,
        'H_0': qubit_jw_op,
        'H_static': H_static,
        'dipole_qubit': dipole_qubit,
        'dipole_matrix': dipole_matrix,
        'qubit_dipole_ops': qubit_dipole_ops,
        'qubit_dipole_ops_matrix': qubit_dipole_ops_matrix
    }


def setup_ansatz(problem, mapper):
    """
    Setup UCC ansatz circuit
    
    Args:
        problem: Molecular problem object
        mapper: Qubit mapper
        
    Returns:
        ansatz: UCC ansatz circuit
    """
    ansatz = UCC(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        excitations=EXCITATIONS,
        qubit_mapper=mapper,
        initial_state=HartreeFock(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
        ),
        reps=REPS,
    )
    return ansatz


def get_exact_ground_state(qubit_jw_op):
    """
    Compute exact ground state using NumPy eigensolver
    
    Args:
        qubit_jw_op: Qubit hamiltonian operator
        
    Returns:
        exact_result: Exact eigenvalue result
    """
    numpy_solver = NumPyEigensolver()
    exact_result = numpy_solver.compute_eigenvalues(qubit_jw_op)
    return exact_result


def Hamilton_SP(t, H_0, dipole_qubit, E0, Gamma):
    """
    Time-dependent Hamiltonian in SparsePauliOp format
    
    Args:
        t: Time
        H_0: Static hamiltonian
        dipole_qubit: Dipole operator
        E0: Field amplitude
        Gamma: Field width parameter
        
    Returns:
        H_total_q: Total time-dependent hamiltonian
    """
    t_float = float(t)
    E_t = (E0 / np.pi) * Gamma / (Gamma**2 + t_float**2)
    V_t = E_t * dipole_qubit
    H_total_q = H_0 + V_t
    return H_total_q
