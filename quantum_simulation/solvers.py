"""
Time evolution solvers for quantum simulation.

This module contains various algorithms for simulating quantum time evolution:
- Exact ODE solver
- Variational Quantum Algorithm (VQA) compilation
- Trotter decomposition (1st and 2nd order)
- Magnus expansion (2nd order)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import time
import math
from numpy.random import Generator, PCG64

# Qiskit imports
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

# Pennylane imports
import pennylane as qml
import pennylane.numpy as pnp

from .config import SimulationConfig, DEFAULT_CONFIG
from .hamiltonians import QuantumSystem


class TimeEvolutionSolver(ABC):
    """Abstract base class for time evolution solvers."""
    
    def __init__(self, quantum_system: QuantumSystem, config: Optional[SimulationConfig] = None):
        """
        Initialize solver with quantum system and configuration.
        
        Parameters:
        -----------
        quantum_system : QuantumSystem
            The quantum system to evolve
        config : SimulationConfig, optional
            Configuration parameters
        """
        self.system = quantum_system
        self.config = config or DEFAULT_CONFIG
        self.results: Dict[str, Any] = {}
        
    @abstractmethod
    def evolve(self, time_points: npt.NDArray[np.float64], 
               initial_state: npt.NDArray[np.complex128]) -> Dict[float, npt.NDArray[np.complex128]]:
        """
        Evolve the quantum state over given time points.
        
        Parameters:
        -----------
        time_points : np.ndarray
            Array of time points for evolution
        initial_state : np.ndarray
            Initial quantum state
            
        Returns:
        --------
        Dict[float, np.ndarray]
            Dictionary mapping time points to evolved states
        """
        pass
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get information about the solver."""
        return {
            "solver_type": self.__class__.__name__,
            "config": self.config.to_dict(),
            "results": self.results
        }


class ExactODESolver(TimeEvolutionSolver):
    """Exact time evolution using ODE integration."""
    
    def __init__(self, quantum_system: QuantumSystem, config: Optional[SimulationConfig] = None):
        super().__init__(quantum_system, config)
        self.method = config.ode_method if config else DEFAULT_CONFIG.ode_method
        
    def _schrodinger_rhs(self, t: float, psi_flat: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """
        Right-hand side of Schrödinger equation: dψ/dt = -i H(t) ψ.
        
        Parameters:
        -----------
        t : float
            Time point
        psi_flat : np.ndarray
            Flattened state vector
            
        Returns:
        --------
        np.ndarray
            Time derivative of state vector
        """
        H_t = self.system.time_dependent_hamiltonian_matrix(t)
        return -1j * H_t @ psi_flat
    
    def evolve(self, time_points: npt.NDArray[np.float64], 
               initial_state: npt.NDArray[np.complex128]) -> Dict[float, npt.NDArray[np.complex128]]:
        """Evolve using exact ODE integration."""
        print(f"Starting exact ODE evolution using {self.method} method...")
        start_time = time.time()
        
        t_span = [time_points[0], time_points[-1]]
        
        try:
            solution = solve_ivp(
                self._schrodinger_rhs,
                t_span,
                y0=initial_state,
                t_eval=time_points,
                method=self.method,
                rtol=self.config.ode_rtol,
                atol=self.config.ode_atol
            )
            
            if not solution.success:
                raise RuntimeError(f"ODE integration failed: {solution.message}")
            
            # Convert solution to dictionary
            evolved_states = {}
            for i, t in enumerate(time_points):
                evolved_states[t] = solution.y[:, i]
            
            end_time = time.time()
            self.results = {
                "evolution_time": end_time - start_time,
                "success": True,
                "num_time_points": len(time_points),
                "method": self.method
            }
            
            print(f"Exact ODE evolution completed in {end_time - start_time:.2f} seconds")
            return evolved_states
            
        except Exception as e:
            self.results = {"success": False, "error": str(e)}
            raise RuntimeError(f"Exact ODE evolution failed: {e}")


class VQACompilationSolver(TimeEvolutionSolver):
    """Variational Quantum Algorithm (VQA) compilation solver."""
    
    def __init__(self, quantum_system: QuantumSystem, config: Optional[SimulationConfig] = None):
        super().__init__(quantum_system, config)
        self.num_layers = config.num_layers if config else DEFAULT_CONFIG.num_layers
        self.max_steps = config.max_optimization_steps if config else DEFAULT_CONFIG.max_optimization_steps
        self.learning_rate = config.learning_rate if config else DEFAULT_CONFIG.learning_rate
        self.error_threshold = config.error_threshold if config else DEFAULT_CONFIG.error_threshold
        
        # Get Pauli labels for ansatz
        self.pauli_labels = self._get_pauli_labels()
        self.num_thetas = len(self.pauli_labels) * self.num_layers
        
        # Initialize random parameters
        rng = Generator(PCG64())
        self.init_thetas = pnp.array(
            2 * math.pi * rng.random(size=self.num_thetas), 
            requires_grad=True
        )
        
    def _get_pauli_labels(self) -> List[str]:
        """Get combined unique Pauli labels from Hamiltonian and dipole operators."""
        static_labels = self.system.static_hamiltonian.paulis.to_labels()
        dipole_labels = self.system.total_dipole_operator.paulis.to_labels()
        return list(dict.fromkeys(static_labels + dipole_labels))
    
    def _pennylane_ansatz_from_qiskit_pauli_evo(self, thetas: pnp.ndarray, num_qubits: int, 
                                                pauli_labels: List[str], num_layers: int) -> None:
        """
        Create Pennylane ansatz equivalent to Qiskit PauliEvolutionGate sequence.
        
        Parameters:
        -----------
        thetas : pnp.ndarray
            Parameter array
        num_qubits : int
            Number of qubits
        pauli_labels : List[str]
            List of Pauli strings
        num_layers : int
            Number of layers
        """
        if not pauli_labels:
            raise ValueError("pauli_labels must not be empty")
        
        num_params_per_layer = len(pauli_labels)
        num_params_expected = num_params_per_layer * num_layers
        
        if len(thetas) != num_params_expected:
            raise ValueError(f"Expected {num_params_expected} parameters, got {len(thetas)}")
        
        param_idx = 0
        for _ in range(num_layers):
            for pauli_qiskit_str in pauli_labels:
                if len(pauli_qiskit_str) != num_qubits:
                    raise ValueError(f"Pauli string '{pauli_qiskit_str}' does not match num_qubits={num_qubits}")
                
                # Build Pennylane observable
                pennylane_observable = self._build_pennylane_observable(pauli_qiskit_str, num_qubits)
                
                if pennylane_observable is not None:
                    qml.exp(pennylane_observable, -1j * thetas[param_idx])
                
                param_idx += 1
    
    def _build_pennylane_observable(self, pauli_str: str, num_qubits: int):
        """Build Pennylane observable from Pauli string."""
        if all(c == 'I' for c in pauli_str):
            # Identity operator
            if num_qubits > 0:
                pennylane_observable = qml.Identity(0)
                for i in range(1, num_qubits):
                    pennylane_observable @= qml.Identity(i)
                return pennylane_observable
            return qml.Identity(0) if num_qubits == 0 else None
        
        # Try using string_to_pauli_word if available
        try:
            return qml.pauli.string_to_pauli_word(pauli_str)
        except AttributeError:
            # Fallback: build manually
            ops_list = []
            for i, char in enumerate(pauli_str):
                if char == 'X':
                    ops_list.append(qml.PauliX(i))
                elif char == 'Y':
                    ops_list.append(qml.PauliY(i))
                elif char == 'Z':
                    ops_list.append(qml.PauliZ(i))
                elif char == 'I':
                    continue
                else:
                    raise ValueError(f"Invalid Pauli character '{char}'")
            
            if not ops_list and num_qubits > 0:
                pennylane_observable = qml.Identity(0)
                for i in range(1, num_qubits):
                    pennylane_observable @= qml.Identity(i)
                return pennylane_observable
            elif ops_list:
                return qml.prod(*ops_list)
            else:
                raise ValueError(f"Cannot create observable from '{pauli_str}'")
    
    def _vqa_cost(self, thetas: pnp.ndarray, target_unitary: npt.NDArray[np.complex128]) -> float:
        """Compute VQA cost function."""
        num_qubits = self.system.static_hamiltonian.num_qubits
        ansatz_matrix = qml.matrix(
            self._pennylane_ansatz_from_qiskit_pauli_evo, 
            wire_order=list(range(num_qubits))
        )(thetas, num_qubits, self.pauli_labels, self.num_layers)
        
        # Hilbert-Schmidt test cost function
        return self._hilbert_schmidt_cost(ansatz_matrix, target_unitary)
    
    def _hilbert_schmidt_cost(self, U_ansatz: npt.NDArray[np.complex128], 
                             U_target: npt.NDArray[np.complex128]) -> float:
        """Compute Hilbert-Schmidt test cost function."""
        dim = U_target.shape[0]
        diff = U_ansatz - U_target
        return np.real(np.trace(diff.conj().T @ diff)) / dim
    
    def _train_vqa_for_time(self, target_unitary: npt.NDArray[np.complex128]) -> Tuple[pnp.ndarray, List[float]]:
        """Train VQA for a single time point."""
        optimizer = qml.AdamOptimizer(stepsize=self.learning_rate)
        thetas = self.init_thetas.copy()
        cost_history = []
        
        for step in range(self.max_steps):
            cost_fn = lambda th: self._vqa_cost(th, target_unitary)
            thetas, prev_cost = optimizer.step_and_cost(cost_fn, thetas)
            
            if (step + 1) % (self.max_steps // 2 if self.max_steps >= 2 else 1) == 0:
                print(f"    Optimization step {step+1}/{self.max_steps}, Cost: {prev_cost:.6f}")
            
            if prev_cost < self.error_threshold:
                print(f"    Reached error threshold at step {step+1}")
                break
                
            cost_history.append(prev_cost)
        
        return thetas, cost_history
    
    def evolve(self, time_points: npt.NDArray[np.float64], 
               initial_state: npt.NDArray[np.complex128]) -> Dict[float, npt.NDArray[np.complex128]]:
        """Evolve using VQA compilation approach."""
        print("Starting VQA compilation time evolution...")
        start_time = time.time()
        
        # First, compute target unitaries using exact method
        exact_solver = ExactODESolver(self.system, self.config)
        
        # Compute target unitaries
        print("Computing target unitaries...")
        target_unitaries = self._compute_target_unitaries(time_points)
        
        evolved_states = {}
        optimized_unitaries = {}
        all_costs = []
        
        num_qubits = self.system.static_hamiltonian.num_qubits
        
        for i, t in enumerate(time_points):
            print(f"\n--- Time t = {t:.4f} ---")
            target_unitary = target_unitaries[i]
            
            # Train VQA for this time point
            optimal_thetas, cost_history = self._train_vqa_for_time(target_unitary)
            all_costs.extend(cost_history)
            
            # Get optimized unitary
            U_optimized = qml.matrix(
                self._pennylane_ansatz_from_qiskit_pauli_evo,
                wire_order=range(num_qubits)
            )(optimal_thetas, num_qubits, self.pauli_labels, self.num_layers)
            
            optimized_unitaries[t] = U_optimized.numpy() if hasattr(U_optimized, 'numpy') else np.asarray(U_optimized)
            
            # Apply to initial state
            psi_t = optimized_unitaries[t] @ initial_state
            evolved_states[t] = psi_t
        
        end_time = time.time()
        self.results = {
            "evolution_time": end_time - start_time,
            "success": True,
            "num_time_points": len(time_points),
            "num_layers": self.num_layers,
            "max_optimization_steps": self.max_steps,
            "final_costs": all_costs[-len(time_points):] if all_costs else [],
            "optimized_unitaries": optimized_unitaries
        }
        
        print(f"VQA compilation completed in {end_time - start_time:.2f} seconds")
        return evolved_states
    
    def _compute_target_unitaries(self, time_points: npt.NDArray[np.float64]) -> List[npt.NDArray[np.complex128]]:
        """Compute target unitaries using ODE integration."""
        dim = self.system.static_hamiltonian_matrix.shape[0]
        U0_flat = np.eye(dim, dtype=complex).flatten()
        
        def unitary_rhs(t: float, U_flat: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
            U = U_flat.reshape(dim, dim)
            H_t = self.system.time_dependent_hamiltonian_matrix(t)
            dU_dt = -1j * (H_t @ U)
            return dU_dt.flatten()
        
        t_span = [time_points[0], time_points[-1]]
        
        solution = solve_ivp(
            unitary_rhs,
            t_span,
            U0_flat,
            t_eval=time_points,
            method=self.config.ode_method,
            rtol=self.config.ode_rtol,
            atol=self.config.ode_atol
        )
        
        if not solution.success:
            raise RuntimeError(f"Target unitary computation failed: {solution.message}")
        
        target_unitaries = []
        unitaries_flat = solution.y.T
        for u_flat in unitaries_flat:
            target_unitaries.append(u_flat.reshape(dim, dim))
        
        return target_unitaries


class TrotterSolver(TimeEvolutionSolver):
    """Trotter decomposition solver for time evolution."""
    
    def __init__(self, quantum_system: QuantumSystem, config: Optional[SimulationConfig] = None, order: int = 1):
        super().__init__(quantum_system, config)
        self.order = order
        self.steps_per_au = config.trotter_steps_per_au if config else DEFAULT_CONFIG.trotter_steps_per_au
        
    def _run_trotter_simulation(self, total_time: float, num_steps: int, 
                               initial_state: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        """Run Trotter simulation for given total time."""
        num_qubits = self.system.static_hamiltonian.num_qubits
        dt = total_time / num_steps
        
        # Create quantum circuit
        circuit = QuantumCircuit(num_qubits)
        circuit.initialize(initial_state, range(num_qubits))
        
        # Apply Trotter steps
        for i in range(num_steps):
            t = (i + 0.5) * dt  # Electric field at middle of time step
            E_t = self.system.electric_field(t)
            
            if self.order == 1:
                # Trotter 1st order: U(dt) ≈ exp(-i*H0*dt) * exp(-i*V(t)*dt)
                circuit.append(
                    PauliEvolutionGate(self.system.static_hamiltonian, time=dt), 
                    range(num_qubits)
                )
                circuit.append(
                    PauliEvolutionGate(self.system.total_dipole_operator, time=E_t * dt), 
                    range(num_qubits)
                )
            elif self.order == 2:
                # Trotter 2nd order: U(dt) ≈ exp(-i*H0*dt/2) * exp(-i*V(t)*dt) * exp(-i*H0*dt/2)
                circuit.append(
                    PauliEvolutionGate(self.system.static_hamiltonian, time=dt/2), 
                    range(num_qubits)
                )
                circuit.append(
                    PauliEvolutionGate(self.system.total_dipole_operator, time=E_t * dt), 
                    range(num_qubits)
                )
                circuit.append(
                    PauliEvolutionGate(self.system.static_hamiltonian, time=dt/2), 
                    range(num_qubits)
                )
            else:
                raise ValueError("Trotter order must be 1 or 2")
        
        # Get final state
        final_state = Statevector(circuit)
        return final_state.data
    
    def evolve(self, time_points: npt.NDArray[np.float64], 
               initial_state: npt.NDArray[np.complex128]) -> Dict[float, npt.NDArray[np.complex128]]:
        """Evolve using Trotter decomposition."""
        print(f"Starting Trotter {self.order}{'st' if self.order == 1 else 'nd'} order evolution...")
        start_time = time.time()
        
        evolved_states = {}
        
        for t in time_points:
            if t == 0:
                evolved_states[t] = initial_state
            else:
                # Number of Trotter steps proportional to time for accuracy
                num_steps = max(1, int(t * self.steps_per_au))
                psi_t = self._run_trotter_simulation(t, num_steps, initial_state)
                evolved_states[t] = psi_t
        
        end_time = time.time()
        self.results = {
            "evolution_time": end_time - start_time,
            "success": True,
            "num_time_points": len(time_points),
            "trotter_order": self.order,
            "steps_per_au": self.steps_per_au
        }
        
        print(f"Trotter {self.order}{'st' if self.order == 1 else 'nd'} order evolution completed in {end_time - start_time:.2f} seconds")
        return evolved_states


class MagnusSolver(TimeEvolutionSolver):
    """Magnus expansion solver (2nd order) for time evolution."""
    
    def __init__(self, quantum_system: QuantumSystem, config: Optional[SimulationConfig] = None):
        super().__init__(quantum_system, config)
        self.dt_magnus = config.magnus_dt if config else DEFAULT_CONFIG.magnus_dt
        self.inner_integral_points = config.magnus_inner_integral_points if config else DEFAULT_CONFIG.magnus_inner_integral_points
        
        # Pre-compute commutator [H0, D] for efficiency
        self.commutator_H0_D = (
            self.system.static_hamiltonian_matrix @ self.system.dipole_matrix - 
            self.system.dipole_matrix @ self.system.static_hamiltonian_matrix
        )
        
    def _trapezoidal_weights(self, n_points: int) -> npt.NDArray[np.float64]:
        """Compute trapezoidal integration weights."""
        if n_points <= 1:
            return np.array([1.0]) if n_points == 1 else np.array([])
        weights = np.ones(n_points)
        weights[0] = weights[-1] = 0.5
        return weights * (1.0 / (n_points - 1))
    
    def _f_lorentzian(self, t: float) -> float:
        """Lorentzian electric field function."""
        return self.system.electric_field(t)
    
    def _f_integral(self, t_start: float, t_end: float) -> float:
        """Analytical integral of Lorentzian from t_start to t_end."""
        return (self.config.e0 / np.pi) * (
            np.arctan(t_end / self.config.gamma) - np.arctan(t_start / self.config.gamma)
        )
    
    def _compute_magnus_step(self, t_start: float, h_step: float) -> npt.NDArray[np.complex128]:
        """Compute single Magnus step unitary."""
        if h_step == 0:
            dim = self.system.static_hamiltonian_matrix.shape[0]
            return np.eye(dim, dtype=complex)
        
        # First order Magnus term: Ω₁ = -i ∫ H(t) dt
        Omega1 = -1j * (
            self.system.static_hamiltonian_matrix * h_step + 
            self.system.dipole_matrix * self._f_integral(t_start, t_start + h_step)
        )
        
        # Second order Magnus term: Ω₂ = -(1/2) ∫∫ [H(t₁), H(t₂)] dt₁ dt₂
        Omega2 = np.zeros_like(Omega1, dtype=complex)
        
        if self.inner_integral_points > 1:
            t_points = np.linspace(t_start, t_start + h_step, self.inner_integral_points)
            delta_t = h_step / (self.inner_integral_points - 1)
            weights = self._trapezoidal_weights(self.inner_integral_points) * delta_t
            f_values = np.array([self._f_lorentzian(t) for t in t_points])
            
            for p in range(self.inner_integral_points):
                t1 = t_points[p]
                f_t1 = f_values[p]
                integral_f_up_to_t1 = self._f_integral(t_start, t1)
                term = integral_f_up_to_t1 - (f_t1 * (t1 - t_start))
                Omega2 += term * self.commutator_H0_D * weights[p]
        
        Omega2 *= -0.5
        
        # Total Magnus generator
        Omega = Omega1 + Omega2
        
        # Compute unitary: U = exp(Ω)
        U_step = expm(Omega)
        return U_step
    
    def evolve(self, time_points: npt.NDArray[np.float64], 
               initial_state: npt.NDArray[np.complex128]) -> Dict[float, npt.NDArray[np.complex128]]:
        """Evolve using Magnus expansion (2nd order)."""
        print("Starting Magnus 2nd order expansion evolution...")
        start_time = time.time()
        
        evolved_states = {}
        current_t = 0.0
        current_state = Statevector(initial_state)
        
        # Save initial state
        evolved_states[0.0] = initial_state
        plot_idx = 1
        
        while current_t < time_points[-1] and plot_idx < len(time_points):
            # Time step to next plot point
            h = time_points[plot_idx] - current_t
            
            # Divide into smaller Magnus steps
            num_sub_steps = max(1, int(np.ceil(h / self.dt_magnus)))
            dt_sub_step = h / num_sub_steps
            
            state_at_h = current_state
            
            for i in range(num_sub_steps):
                t_sub_start = current_t + i * dt_sub_step
                U_sub_step = self._compute_magnus_step(t_sub_start, dt_sub_step)
                state_at_h = state_at_h.evolve(Operator(U_sub_step))
            
            # Save result at plot point
            evolved_states[time_points[plot_idx]] = state_at_h.data
            
            # Update for next iteration
            current_t = time_points[plot_idx]
            current_state = state_at_h
            plot_idx += 1
        
        end_time = time.time()
        self.results = {
            "evolution_time": end_time - start_time,
            "success": True,
            "num_time_points": len(time_points),
            "magnus_dt": self.dt_magnus,
            "inner_integral_points": self.inner_integral_points
        }
        
        print(f"Magnus 2nd order evolution completed in {end_time - start_time:.2f} seconds")
        return evolved_states


# Factory functions for easy solver creation
def create_exact_solver(quantum_system: QuantumSystem, 
                       config: Optional[SimulationConfig] = None) -> ExactODESolver:
    """Create exact ODE solver."""
    return ExactODESolver(quantum_system, config)

def create_vqa_solver(quantum_system: QuantumSystem, 
                     config: Optional[SimulationConfig] = None) -> VQACompilationSolver:
    """Create VQA compilation solver."""
    return VQACompilationSolver(quantum_system, config)

def create_trotter_solver(quantum_system: QuantumSystem, 
                         config: Optional[SimulationConfig] = None, 
                         order: int = 1) -> TrotterSolver:
    """Create Trotter solver."""
    return TrotterSolver(quantum_system, config, order)

def create_magnus_solver(quantum_system: QuantumSystem, 
                        config: Optional[SimulationConfig] = None) -> MagnusSolver:
    """Create Magnus expansion solver."""
    return MagnusSolver(quantum_system, config) 