"""
Quantum Simulation Package

A comprehensive package for quantum molecular simulation including:
- Hamiltonian construction
- Time evolution algorithms (VQA, Trotter, Magnus)
- Analysis and visualization tools
"""

from .config import SimulationConfig, DEFAULT_CONFIG
from .hamiltonians import QuantumSystem, create_quantum_system
from .solvers import (
    TimeEvolutionSolver,
    ExactODESolver,
    VQACompilationSolver,
    TrotterSolver,
    MagnusSolver,
    create_exact_solver,
    create_vqa_solver,
    create_trotter_solver,
    create_magnus_solver
)
from .analysis import (
    AnalysisResults,
    QuantumStateAnalyzer,
    SpectrumAnalyzer,
    PerformanceAnalyzer,
    DataExporter,
    create_state_analyzer,
    create_spectrum_analyzer,
    create_performance_analyzer
)
from .visualization import (
    PlotStyle,
    QuantumPlotter,
    TimeEvolutionPlotter,
    SpectrumPlotter,
    ErrorAnalysisPlotter,
    PerformancePlotter,
    create_time_evolution_plotter,
    create_spectrum_plotter,
    create_error_analysis_plotter,
    create_performance_plotter,
    create_comprehensive_analysis_plots
)
from .main import SimulationWorkflow

__version__ = "1.0.0"
__author__ = "Quantum Simulation Team"

__all__ = [
    "SimulationConfig",
    "DEFAULT_CONFIG", 
    "QuantumSystem",
    "create_quantum_system",
    "TimeEvolutionSolver",
    "ExactODESolver",
    "VQACompilationSolver", 
    "TrotterSolver",
    "MagnusSolver",
    "create_exact_solver",
    "create_vqa_solver",
    "create_trotter_solver",
    "create_magnus_solver",
    "AnalysisResults",
    "QuantumStateAnalyzer",
    "SpectrumAnalyzer", 
    "PerformanceAnalyzer",
    "DataExporter",
    "create_state_analyzer",
    "create_spectrum_analyzer",
    "create_performance_analyzer",
    "PlotStyle",
    "QuantumPlotter",
    "TimeEvolutionPlotter",
    "SpectrumPlotter",
    "ErrorAnalysisPlotter",
    "PerformancePlotter",
    "create_time_evolution_plotter",
    "create_spectrum_plotter",
    "create_error_analysis_plotter",
    "create_performance_plotter",
    "create_comprehensive_analysis_plots",
    "SimulationWorkflow"
] 