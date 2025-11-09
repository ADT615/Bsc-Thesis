"""
Analysis tools for quantum simulation results.

This module provides comprehensive analysis capabilities including:
- Quantum state analysis (fidelity, expectation values)
- Error metrics and comparison between methods
- Spectrum analysis (FFT, absorption spectra)
- Performance benchmarking
- Data export and statistical analysis
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import numpy.typing as npt
from scipy.fft import fftfreq, fftshift
from scipy import fftpack
from scipy.signal import windows
from scipy.constants import speed_of_light, physical_constants
import pandas as pd
from pathlib import Path
import time
from dataclasses import dataclass

# Qiskit imports
from qiskit.quantum_info import SparsePauliOp, Statevector

from .config import SimulationConfig, DEFAULT_CONFIG, HARTREE_TO_EV
from .hamiltonians import QuantumSystem


@dataclass
class AnalysisResults:
    """Container for analysis results."""
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: float
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save results to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'AnalysisResults':
        """Load results from file."""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class QuantumStateAnalyzer:
    """
    Analyzer for quantum state properties and comparisons.
    
    This class provides tools for analyzing quantum states including
    fidelity calculations, expectation values, and state comparisons.
    """
    
    def __init__(self, quantum_system: QuantumSystem, config: Optional[SimulationConfig] = None):
        """
        Initialize state analyzer.
        
        Parameters:
        -----------
        quantum_system : QuantumSystem
            The quantum system for context
        config : SimulationConfig, optional
            Configuration parameters
        """
        self.system = quantum_system
        self.config = config or DEFAULT_CONFIG
        
    def calculate_expectation_value(self, 
                                  state: npt.NDArray[np.complex128], 
                                  operator: SparsePauliOp) -> float:
        """
        Calculate expectation value of operator for given state.
        
        Parameters:
        -----------
        state : np.ndarray
            Quantum state vector
        operator : SparsePauliOp
            Quantum operator
            
        Returns:
        --------
        float
            Real expectation value
        """
        try:
            # Ensure state is contiguous for Qiskit
            state_contiguous = np.ascontiguousarray(state)
            statevector = Statevector(state_contiguous)
            expectation = statevector.expectation_value(operator)
            return expectation.real
        except Exception as e:
            print(f"Warning: Expectation value calculation failed: {e}")
            return np.nan
    
    def calculate_fidelity(self, 
                          state1: npt.NDArray[np.complex128], 
                          state2: npt.NDArray[np.complex128]) -> float:
        """
        Calculate fidelity between two quantum states.
        
        Parameters:
        -----------
        state1, state2 : np.ndarray
            Quantum state vectors
            
        Returns:
        --------
        float
            Fidelity value between 0 and 1
        """
        if state1.shape != state2.shape:
            raise ValueError("States must have same dimensions")
        
        inner_product = np.vdot(state1, state2)
        fidelity = np.abs(inner_product)**2
        return fidelity.real
    
    def calculate_state_overlap(self, 
                               state1: npt.NDArray[np.complex128], 
                               state2: npt.NDArray[np.complex128]) -> complex:
        """
        Calculate complex overlap between two states.
        
        Parameters:
        -----------
        state1, state2 : np.ndarray
            Quantum state vectors
            
        Returns:
        --------
        complex
            Complex overlap <state1|state2>
        """
        return np.vdot(state1, state2)
    
    def calculate_state_distance(self, 
                                state1: npt.NDArray[np.complex128], 
                                state2: npt.NDArray[np.complex128]) -> float:
        """
        Calculate Euclidean distance between states.
        
        Parameters:
        -----------
        state1, state2 : np.ndarray
            Quantum state vectors
            
        Returns:
        --------
        float
            Euclidean distance ||state1 - state2||
        """
        return np.linalg.norm(state1 - state2)
    
    def analyze_dipole_moment_time_series(self, 
                                        evolved_states: Dict[float, npt.NDArray[np.complex128]],
                                        dipole_operator: Optional[SparsePauliOp] = None) -> Dict[str, npt.NDArray]:
        """
        Analyze dipole moment evolution over time.
        
        Parameters:
        -----------
        evolved_states : Dict[float, np.ndarray]
            Dictionary mapping time points to evolved states
        dipole_operator : SparsePauliOp, optional
            Dipole moment operator (uses system's total dipole if None)
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with 'times' and 'dipole_moments' arrays
        """
        if dipole_operator is None:
            dipole_operator = self.system.total_dipole_operator
        
        times = sorted(evolved_states.keys())
        dipole_moments = []
        
        for t in times:
            state = evolved_states[t]
            dipole_val = self.calculate_expectation_value(state, dipole_operator)
            dipole_moments.append(dipole_val)
        
        return {
            'times': np.array(times),
            'dipole_moments': np.array(dipole_moments)
        }
    
    def compare_methods(self, 
                       results_dict: Dict[str, Dict[float, npt.NDArray[np.complex128]]],
                       reference_method: str = 'exact') -> AnalysisResults:
        """
        Compare multiple simulation methods against a reference.
        
        Parameters:
        -----------
        results_dict : Dict[str, Dict[float, np.ndarray]]
            Dict mapping method names to their evolved states
        reference_method : str
            Name of reference method for comparison
            
        Returns:
        --------
        AnalysisResults
            Comprehensive comparison results
        """
        if reference_method not in results_dict:
            raise ValueError(f"Reference method '{reference_method}' not found in results")
        
        reference_states = results_dict[reference_method]
        times = sorted(reference_states.keys())
        
        comparison_data = {
            'times': times,
            'methods': list(results_dict.keys()),
            'fidelities': {},
            'distances': {},
            'dipole_moments': {}
        }
        
        # Calculate dipole moments for all methods
        for method_name, states in results_dict.items():
            dipole_data = self.analyze_dipole_moment_time_series(states)
            comparison_data['dipole_moments'][method_name] = dipole_data['dipole_moments']
        
        # Calculate fidelities and distances vs reference
        for method_name, states in results_dict.items():
            if method_name == reference_method:
                comparison_data['fidelities'][method_name] = np.ones(len(times))
                comparison_data['distances'][method_name] = np.zeros(len(times))
                continue
            
            fidelities = []
            distances = []
            
            for t in times:
                if t in states and t in reference_states:
                    fidelity = self.calculate_fidelity(reference_states[t], states[t])
                    distance = self.calculate_state_distance(reference_states[t], states[t])
                    fidelities.append(fidelity)
                    distances.append(distance)
                else:
                    fidelities.append(np.nan)
                    distances.append(np.nan)
            
            comparison_data['fidelities'][method_name] = np.array(fidelities)
            comparison_data['distances'][method_name] = np.array(distances)
        
        # Calculate error metrics for dipole moments
        ref_dipole = comparison_data['dipole_moments'][reference_method]
        dipole_errors = {}
        
        for method_name, dipole_vals in comparison_data['dipole_moments'].items():
            if method_name == reference_method:
                continue
            
            mae = np.mean(np.abs(dipole_vals - ref_dipole))
            max_error = np.max(np.abs(dipole_vals - ref_dipole))
            rmse = np.sqrt(np.mean((dipole_vals - ref_dipole)**2))
            
            dipole_errors[method_name] = {
                'mae': mae,
                'max_error': max_error,
                'rmse': rmse
            }
        
        comparison_data['dipole_errors'] = dipole_errors
        
        metadata = {
            'reference_method': reference_method,
            'num_methods': len(results_dict),
            'num_time_points': len(times),
            'time_range': (min(times), max(times))
        }
        
        return AnalysisResults(
            data=comparison_data,
            metadata=metadata,
            timestamp=time.time()
        )


class SpectrumAnalyzer:
    """
    Analyzer for frequency domain and absorption spectrum analysis.
    
    This class provides tools for FFT analysis and absorption spectrum
    calculation from time-domain dipole moment data.
    """
    
    def __init__(self, quantum_system: QuantumSystem, config: Optional[SimulationConfig] = None):
        """Initialize spectrum analyzer."""
        self.system = quantum_system
        self.config = config or DEFAULT_CONFIG
        
        # Physical constants
        self.speed_of_light_au = speed_of_light / physical_constants["atomic unit of velocity"][0]
    
    def apply_damping(self, signal: npt.NDArray, dt: float, gamma: float) -> npt.NDArray:
        """
        Apply exponential damping to time signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Time domain signal
        dt : float
            Time step
        gamma : float
            Damping parameter
            
        Returns:
        --------
        np.ndarray
            Damped signal
        """
        t = np.arange(len(signal)) * dt
        return signal * np.exp(-gamma * t)
    
    def apply_window(self, signal: npt.NDArray, window_type: str = 'blackman') -> npt.NDArray:
        """
        Apply window function to reduce spectral leakage.
        
        Parameters:
        -----------
        signal : np.ndarray
            Time domain signal
        window_type : str
            Type of window ('blackman', 'hann', 'hamming', 'kaiser')
            
        Returns:
        --------
        np.ndarray
            Windowed signal
        """
        if window_type == 'blackman':
            return signal * windows.blackman(len(signal))
        elif window_type == 'hann':
            return signal * windows.hann(len(signal))
        elif window_type == 'hamming':
            return signal * windows.hamming(len(signal))
        elif window_type == 'kaiser':
            return signal * windows.kaiser(len(signal), beta=8.6)
        else:
            return signal  # No windowing
    
    def fourier_transform(self, signal: npt.NDArray, dt: float) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Perform Fourier transform with proper normalization.
        
        Parameters:
        -----------
        signal : np.ndarray
            Time domain signal
        dt : float
            Time step
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Frequency array and Fourier coefficients
        """
        c = dt / (2 * np.pi)
        M = len(signal) // 2
        N = 2 * M  # Ensure even number of points
        
        # Truncate signal to even length if necessary
        signal_truncated = signal[:N]
        
        omega = fftpack.fftfreq(N, c)[:M]
        F = N * c * fftpack.ifft(signal_truncated, overwrite_x=True)[:M]
        
        return omega, F
    
    def calculate_absorption_spectrum(self, 
                                    dipole_omega: npt.NDArray[np.complex128],
                                    omega: npt.NDArray[np.float64],
                                    E_omega: npt.NDArray[np.complex128],
                                    normalize: bool = True) -> npt.NDArray[np.float64]:
        """
        Calculate absorption spectrum from dipole and field spectra.
        
        Parameters:
        -----------
        dipole_omega : np.ndarray
            Frequency domain dipole moment
        omega : np.ndarray
            Frequency array
        E_omega : np.ndarray
            Frequency domain electric field
        normalize : bool
            Whether to normalize the spectrum
            
        Returns:
        --------
        np.ndarray
            Absorption spectrum
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = np.zeros_like(dipole_omega, dtype=complex)
            valid = np.abs(E_omega) > 1e-9
            alpha[valid] = dipole_omega[valid] / E_omega[valid]
            
            # Absorption cross-section
            sigma = (4 * np.pi * omega / self.speed_of_light_au) * np.imag(alpha)
            sigma[sigma < 0] = 0  # Remove negative values
        
        if normalize and sigma.max() > 0:
            sigma = sigma / sigma.max()
        
        return sigma
    
    def analyze_absorption_spectrum(self, 
                                  times: npt.NDArray[np.float64],
                                  dipole_moments: npt.NDArray[np.float64],
                                  damping_gamma: Optional[float] = None,
                                  window_type: str = 'blackman',
                                  normalize: bool = True) -> Dict[str, npt.NDArray]:
        """
        Complete absorption spectrum analysis from time-domain data.
        
        Parameters:
        -----------
        times : np.ndarray
            Time points
        dipole_moments : np.ndarray
            Dipole moment values
        damping_gamma : float, optional
            Damping parameter (uses config default if None)
        window_type : str
            Window function type
        normalize : bool
            Whether to normalize spectrum
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with frequency and spectrum data
        """
        if len(times) < 2:
            raise ValueError("Need at least 2 time points for spectrum analysis")
        
        dt = times[1] - times[0]
        if damping_gamma is None:
            damping_gamma = getattr(self.config, 'fft_damping_gamma', 0.001)
        
        # Prepare dipole signal
        dipole_centered = dipole_moments - np.mean(dipole_moments)
        dipole_damped = self.apply_damping(dipole_centered, dt, damping_gamma)
        dipole_windowed = self.apply_window(dipole_damped, window_type)
        
        # Prepare electric field signal
        E_t = np.array([self.system.electric_field(t) for t in times])
        E_windowed = self.apply_window(E_t, window_type)
        
        # Fourier transforms
        omega, dipole_omega = self.fourier_transform(dipole_windowed, dt)
        _, E_omega = self.fourier_transform(E_windowed, dt)
        
        # Absorption spectrum
        sigma = self.calculate_absorption_spectrum(dipole_omega, omega, E_omega, normalize)
        
        return {
            'omega_au': omega,
            'omega_ev': omega * HARTREE_TO_EV,
            'absorption_spectrum': sigma,
            'dipole_spectrum': dipole_omega,
            'field_spectrum': E_omega
        }


class PerformanceAnalyzer:
    """
    Analyzer for benchmarking and performance comparison of solvers.
    
    This class provides tools for timing analysis, accuracy benchmarking,
    and performance profiling of different quantum simulation methods.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.benchmark_results: List[Dict[str, Any]] = []
    
    def benchmark_solver(self, 
                        solver_func: callable,
                        solver_name: str,
                        *args, **kwargs) -> Dict[str, Any]:
        """
        Benchmark a single solver.
        
        Parameters:
        -----------
        solver_func : callable
            Solver function to benchmark
        solver_name : str
            Name of the solver
        *args, **kwargs
            Arguments for solver function
            
        Returns:
        --------
        Dict[str, Any]
            Benchmark results
        """
        start_time = time.time()
        start_cpu = time.process_time()
        
        try:
            result = solver_func(*args, **kwargs)
            success = True
            error_msg = None
        except Exception as e:
            result = None
            success = False
            error_msg = str(e)
        
        end_time = time.time()
        end_cpu = time.process_time()
        
        benchmark_data = {
            'solver_name': solver_name,
            'wall_time': end_time - start_time,
            'cpu_time': end_cpu - start_cpu,
            'success': success,
            'error_message': error_msg,
            'timestamp': start_time
        }
        
        self.benchmark_results.append(benchmark_data)
        return benchmark_data
    
    def compare_solver_performance(self, 
                                 benchmark_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create performance comparison table.
        
        Parameters:
        -----------
        benchmark_results : List[Dict[str, Any]]
            List of benchmark results
            
        Returns:
        --------
        pd.DataFrame
            Performance comparison table
        """
        df = pd.DataFrame(benchmark_results)
        
        if not df.empty:
            # Add relative performance metrics
            if 'wall_time' in df.columns:
                min_time = df[df['success']]['wall_time'].min()
                df['relative_speed'] = min_time / df['wall_time']
                df['speedup'] = df['wall_time'] / min_time
        
        return df
    
    def generate_performance_report(self, 
                                  accuracy_results: Optional[AnalysisResults] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Parameters:
        -----------
        accuracy_results : AnalysisResults, optional
            Accuracy analysis results to include
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive performance report
        """
        performance_df = self.compare_solver_performance(self.benchmark_results)
        
        report = {
            'performance_summary': performance_df,
            'total_benchmarks': len(self.benchmark_results),
            'successful_runs': sum(1 for r in self.benchmark_results if r['success']),
            'failed_runs': sum(1 for r in self.benchmark_results if not r['success'])
        }
        
        if accuracy_results is not None:
            report['accuracy_analysis'] = accuracy_results.data
        
        # Calculate performance statistics
        successful_results = [r for r in self.benchmark_results if r['success']]
        if successful_results:
            wall_times = [r['wall_time'] for r in successful_results]
            report['timing_stats'] = {
                'mean_wall_time': np.mean(wall_times),
                'std_wall_time': np.std(wall_times),
                'min_wall_time': np.min(wall_times),
                'max_wall_time': np.max(wall_times)
            }
        
        return report


class DataExporter:
    """
    Utility class for exporting analysis results to various formats.
    
    This class provides methods to export quantum simulation results
    to CSV, HDF5, and other formats for further analysis.
    """
    
    @staticmethod
    def export_time_series_csv(data: Dict[str, npt.NDArray], 
                              filepath: Union[str, Path],
                              time_column: str = 'time') -> None:
        """
        Export time series data to CSV.
        
        Parameters:
        -----------
        data : Dict[str, np.ndarray]
            Dictionary with time series data
        filepath : str or Path
            Output file path
        time_column : str
            Name of time column
        """
        df = pd.DataFrame(data)
        if time_column in df.columns:
            df = df.set_index(time_column)
        df.to_csv(filepath)
    
    @staticmethod
    def export_comparison_results(analysis_results: AnalysisResults,
                                filepath: Union[str, Path],
                                format: str = 'csv') -> None:
        """
        Export comparison results to file.
        
        Parameters:
        -----------
        analysis_results : AnalysisResults
            Analysis results to export
        filepath : str or Path
            Output file path
        format : str
            Export format ('csv', 'hdf5', 'pickle')
        """
        data = analysis_results.data
        
        if format.lower() == 'csv':
            # Export main comparison data
            comparison_df = pd.DataFrame({
                'time': data['times'],
                **{f'dipole_{method}': dipole_vals 
                   for method, dipole_vals in data['dipole_moments'].items()},
                **{f'fidelity_{method}': fid_vals 
                   for method, fid_vals in data['fidelities'].items()},
                **{f'distance_{method}': dist_vals 
                   for method, dist_vals in data['distances'].items()}
            })
            comparison_df.to_csv(filepath, index=False)
            
        elif format.lower() == 'hdf5':
            with pd.HDFStore(filepath, 'w') as store:
                # Store time series data
                for method, dipole_vals in data['dipole_moments'].items():
                    df = pd.DataFrame({
                        'time': data['times'],
                        'dipole_moment': dipole_vals
                    })
                    store[f'dipole_{method}'] = df
                
                # Store error metrics
                if 'dipole_errors' in data:
                    error_df = pd.DataFrame(data['dipole_errors']).T
                    store['error_metrics'] = error_df
                    
        elif format.lower() == 'pickle':
            analysis_results.save(filepath)
        
        else:
            raise ValueError(f"Unsupported format: {format}")


# Factory functions for easy analyzer creation
def create_state_analyzer(quantum_system: QuantumSystem, 
                         config: Optional[SimulationConfig] = None) -> QuantumStateAnalyzer:
    """Create quantum state analyzer."""
    return QuantumStateAnalyzer(quantum_system, config)

def create_spectrum_analyzer(quantum_system: QuantumSystem, 
                           config: Optional[SimulationConfig] = None) -> SpectrumAnalyzer:
    """Create spectrum analyzer."""
    return SpectrumAnalyzer(quantum_system, config)

def create_performance_analyzer() -> PerformanceAnalyzer:
    """Create performance analyzer."""
    return PerformanceAnalyzer() 