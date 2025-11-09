"""
Configuration file for quantum simulation.

This module contains all constants, parameters, and configuration settings
used throughout the quantum simulation project.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from qiskit_nature.units import DistanceUnit


# Physical constants
GAMMA = 0.25
E0 = 0.01
HARTREE_TO_EV = 27.211386245988
SPEED_OF_LIGHT_AU = 137.036  # Speed of light in atomic units

# Molecular system parameters
MOLECULE_ATOM_STRING = "H 0 0 0; H 0 0 0.735"  # H2 molecule
BASIS_SET = "sto3g"
MOLECULAR_CHARGE = 0
MOLECULAR_SPIN = 0
DISTANCE_UNIT = DistanceUnit.ANGSTROM

# VQA optimization parameters
VQA_NUM_LAYERS = 6
VQA_MAX_STEPS = 300
VQA_LEARNING_RATE = 0.01
VQA_ERROR_THRESHOLD = 1e-6

# Time evolution parameters
SIMULATION_TIME_START = 0.0
SIMULATION_TIME_END = 300.0
NUM_TIME_POINTS = 50
TROTTER_STEPS_PER_AU = 10
MAGNUS_INNER_INTEGRAL_POINTS = 50
MAGNUS_TIME_STEP = 0.1

# ODE solver parameters
ODE_RTOL = 1e-8
ODE_ATOL = 1e-8
ODE_METHOD = 'RK45'

# FFT and spectrum analysis parameters
FFT_DAMPING_GAMMA = 0.001
FFT_NORMALIZE = True


@dataclass
class SimulationConfig:
    """Configuration class for quantum simulation parameters."""
    
    # Physical parameters
    gamma: float = GAMMA
    e0: float = E0
    
    # Molecular system
    atom_string: str = MOLECULE_ATOM_STRING
    basis: str = BASIS_SET
    charge: int = MOLECULAR_CHARGE
    spin: int = MOLECULAR_SPIN
    unit: DistanceUnit = DISTANCE_UNIT
    
    # VQA parameters
    num_layers: int = VQA_NUM_LAYERS
    max_optimization_steps: int = VQA_MAX_STEPS
    learning_rate: float = VQA_LEARNING_RATE
    error_threshold: float = VQA_ERROR_THRESHOLD
    
    # Time evolution
    time_start: float = SIMULATION_TIME_START
    time_end: float = SIMULATION_TIME_END
    num_time_points: int = NUM_TIME_POINTS
    trotter_steps_per_au: int = TROTTER_STEPS_PER_AU
    magnus_dt: float = MAGNUS_TIME_STEP
    magnus_inner_integral_points: int = MAGNUS_INNER_INTEGRAL_POINTS
    
    # ODE solver
    ode_rtol: float = ODE_RTOL
    ode_atol: float = ODE_ATOL
    ode_method: str = ODE_METHOD
    
    def get_time_points(self) -> np.ndarray:
        """Get array of time points for simulation."""
        return np.linspace(self.time_start, self.time_end, self.num_time_points)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'gamma': self.gamma,
            'e0': self.e0,
            'atom_string': self.atom_string,
            'basis': self.basis,
            'charge': self.charge,
            'spin': self.spin,
            'num_layers': self.num_layers,
            'max_optimization_steps': self.max_optimization_steps,
            'learning_rate': self.learning_rate,
            'error_threshold': self.error_threshold,
            'time_start': self.time_start,
            'time_end': self.time_end,
            'num_time_points': self.num_time_points,
            'trotter_steps_per_au': self.trotter_steps_per_au,
            'magnus_dt': self.magnus_dt,
            'magnus_inner_integral_points': self.magnus_inner_integral_points,
            'ode_rtol': self.ode_rtol,
            'ode_atol': self.ode_atol,
            'ode_method': self.ode_method,
            'fft_damping_gamma': FFT_DAMPING_GAMMA
        }
    
    @property
    def fft_damping_gamma(self) -> float:
        """Get FFT damping parameter."""
        return FFT_DAMPING_GAMMA


# Default configuration instance
DEFAULT_CONFIG = SimulationConfig() 