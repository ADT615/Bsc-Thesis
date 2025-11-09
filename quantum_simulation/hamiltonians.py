"""
Hamiltonian construction and quantum operators module.

This module handles the creation of Hamiltonians, dipole operators,
and time-dependent operators for quantum simulation.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import numpy.typing as npt
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.properties import ElectronicDipoleMoment
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCC
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.eigensolvers import NumPyEigensolver
from qiskit.primitives import Estimator
from qiskit_algorithms.optimizers import SLSQP
from qiskit.quantum_info import Statevector

from .config import SimulationConfig, DEFAULT_CONFIG


class QuantumSystem:
    """
    Class to represent and manipulate quantum molecular systems.
    
    This class handles the creation of Hamiltonians, dipole operators,
    and the computation of ground states using VQE.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize quantum system with given configuration.
        
        Parameters:
        -----------
        config : SimulationConfig, optional
            Configuration parameters. Uses DEFAULT_CONFIG if None.
        """
        self.config = config or DEFAULT_CONFIG
        self.driver: Optional[PySCFDriver] = None
        self.problem = None
        self.mapper = JordanWignerMapper()
        
        # Operators
        self.static_hamiltonian: Optional[SparsePauliOp] = None
        self.dipole_operators: Optional[Dict[str, SparsePauliOp]] = None
        self.total_dipole_operator: Optional[SparsePauliOp] = None
        
        # Matrices for efficient computation
        self.static_hamiltonian_matrix: Optional[npt.NDArray[np.complex128]] = None
        self.dipole_matrix: Optional[npt.NDArray[np.complex128]] = None
        
        # Ground state
        self.ground_state: Optional[npt.NDArray[np.complex128]] = None
        self.ground_energy: Optional[float] = None
        
    def setup_molecular_system(self) -> None:
        """Set up the molecular system using PySCF driver."""
        try:
            self.driver = PySCFDriver(
                atom=self.config.atom_string,
                basis=self.config.basis,
                charge=self.config.charge,
                spin=self.config.spin,
                unit=self.config.unit,
            )
            self.problem = self.driver.run()
        except Exception as e:
            raise RuntimeError(f"Failed to setup molecular system: {e}")
    
    def build_operators(self) -> None:
        """Build Hamiltonian and dipole operators."""
        if self.problem is None:
            raise ValueError("Molecular system must be setup first")
        
        # Build static Hamiltonian
        hamiltonian = self.problem.hamiltonian
        second_q_op = hamiltonian.second_q_op()
        self.static_hamiltonian = self.mapper.map(second_q_op)
        
        # Build dipole operators
        dipole_moment: ElectronicDipoleMoment = self.problem.properties.electronic_dipole_moment
        dipole_ops = dipole_moment.second_q_ops()
        
        self.dipole_operators = {
            "XDipole": self.mapper.map(dipole_ops["XDipole"]),
            "YDipole": self.mapper.map(dipole_ops["YDipole"]),
            "ZDipole": self.mapper.map(dipole_ops["ZDipole"]),
        }
        
        # Total dipole operator (sum of all components)
        self.total_dipole_operator = (
            self.dipole_operators["XDipole"] + 
            self.dipole_operators["YDipole"] + 
            self.dipole_operators["ZDipole"]
        )
        
        # Convert to matrices for efficient computation
        self._build_matrices()
    
    def _build_matrices(self) -> None:
        """Convert operators to matrix form for efficient computation."""
        if self.static_hamiltonian is None or self.total_dipole_operator is None:
            raise ValueError("Operators must be built first")
        
        self.static_hamiltonian_matrix = self.static_hamiltonian.to_matrix()
        self.dipole_matrix = self.total_dipole_operator.to_matrix()
    
    def compute_ground_state_vqe(self) -> Tuple[npt.NDArray[np.complex128], float]:
        """
        Compute ground state using Variational Quantum Eigensolver (VQE).
        
        Returns:
        --------
        Tuple[np.ndarray, float]
            Ground state vector and ground state energy
        """
        if self.static_hamiltonian is None:
            raise ValueError("Hamiltonian must be built first")
        
        # Create ansatz
        ansatz = UCC(
            num_spatial_orbitals=self.problem.num_spatial_orbitals,
            num_particles=self.problem.num_particles,
            excitations='sd',
            qubit_mapper=self.mapper,
            initial_state=HartreeFock(
                num_spatial_orbitals=self.problem.num_spatial_orbitals,
                num_particles=self.problem.num_particles,
                qubit_mapper=self.mapper,
            ),
            reps=1,
        )
        
        # Setup VQE
        estimator = Estimator()
        optimizer = SLSQP(maxiter=200)
        vqe = VQE(estimator, ansatz, optimizer)
        
        # Compute ground state
        try:
            result = vqe.compute_minimum_eigenvalue(self.static_hamiltonian)
            circuit = ansatz.assign_parameters(result.optimal_parameters)
            self.ground_state = np.array(Statevector(circuit).data)
            self.ground_energy = result.eigenvalue.real
            
            return self.ground_state, self.ground_energy
        
        except Exception as e:
            raise RuntimeError(f"VQE computation failed: {e}")
    
    def compute_exact_ground_state(self) -> Tuple[npt.NDArray[np.complex128], float]:
        """
        Compute exact ground state using NumPy eigensolver.
        
        Returns:
        --------
        Tuple[np.ndarray, float]
            Ground state vector and ground state energy
        """
        if self.static_hamiltonian is None:
            raise ValueError("Hamiltonian must be built first")
        
        try:
            numpy_solver = NumPyEigensolver()
            exact_result = numpy_solver.compute_eigenvalues(self.static_hamiltonian)
            
            # Get ground state
            eigenvalues = exact_result.eigenvalues
            eigenstates = exact_result.eigenstates
            
            ground_idx = np.argmin(eigenvalues.real)
            self.ground_energy = eigenvalues[ground_idx].real
            self.ground_state = eigenstates[ground_idx].data
            
            return self.ground_state, self.ground_energy
        
        except Exception as e:
            raise RuntimeError(f"Exact computation failed: {e}")
    
    def electric_field(self, t: float) -> float:
        """
        Compute electric field at time t using Lorentzian pulse.
        
        Parameters:
        -----------
        t : float
            Time point
            
        Returns:
        --------
        float
            Electric field strength at time t
        """
        return (self.config.e0 / np.pi) * self.config.gamma / (self.config.gamma**2 + t**2)
    
    def time_dependent_hamiltonian_sparse(self, t: float) -> SparsePauliOp:
        """
        Compute time-dependent Hamiltonian H(t) = H₀ + E(t) * μ.
        
        Parameters:
        -----------
        t : float
            Time point
            
        Returns:
        --------
        SparsePauliOp
            Time-dependent Hamiltonian
        """
        if self.static_hamiltonian is None or self.total_dipole_operator is None:
            raise ValueError("Operators must be built first")
        
        E_t = self.electric_field(t)
        return self.static_hamiltonian + E_t * self.total_dipole_operator
    
    def time_dependent_hamiltonian_matrix(self, t: float) -> npt.NDArray[np.complex128]:
        """
        Compute time-dependent Hamiltonian matrix H(t) = H₀ + E(t) * μ.
        
        Parameters:
        -----------
        t : float
            Time point
            
        Returns:
        --------
        np.ndarray
            Time-dependent Hamiltonian matrix
        """
        if self.static_hamiltonian_matrix is None or self.dipole_matrix is None:
            raise ValueError("Matrices must be built first")
        
        E_t = self.electric_field(t)
        return self.static_hamiltonian_matrix + E_t * self.dipole_matrix
    
    def get_system_info(self) -> Dict[str, any]:
        """Get information about the quantum system."""
        if self.problem is None:
            return {"error": "System not initialized"}
        
        return {
            "num_qubits": self.static_hamiltonian.num_qubits if self.static_hamiltonian else None,
            "num_spatial_orbitals": self.problem.num_spatial_orbitals,
            "num_particles": self.problem.num_particles,
            "ground_energy": self.ground_energy,
            "molecule": self.config.atom_string,
            "basis": self.config.basis
        }


def create_quantum_system(config: Optional[SimulationConfig] = None) -> QuantumSystem:
    """
    Factory function to create and initialize a quantum system.
    
    Parameters:
    -----------
    config : SimulationConfig, optional
        Configuration parameters
        
    Returns:
    --------
    QuantumSystem
        Initialized quantum system
    """
    system = QuantumSystem(config)
    system.setup_molecular_system()
    system.build_operators()
    return system 