# Quantum Simulation Package

A professional Python package for quantum molecular dynamics simulation and analysis, featuring multiple time evolution algorithms, comprehensive analysis tools, and publication-quality visualization.

## ✨ Features

### 🧬 Quantum System Setup
- **Molecular system initialization** with PySCF integration
- **Hamiltonian construction** for quantum chemistry
- **Electric dipole moment** operators
- **Ground state computation** via VQE and exact diagonalization

### ⚡ Time Evolution Algorithms
- **Exact ODE Integration** - Numerical reference solution
- **Trotter Decomposition** - 1st and 2nd order implementations  
- **Magnus Expansion** - 2nd order with optimized integration
- **VQA Compilation** - Variational quantum algorithm approach

### 📊 Analysis & Visualization
- **State fidelity analysis** and error metrics
- **Absorption spectrum** calculation via FFT
- **Performance benchmarking** and comparison
- **Publication-quality plots** with customizable styling
- **Comprehensive data export** (CSV, Pickle, JSON)

### 🚀 Workflow Automation
- **Command-line interface** for batch processing
- **Configuration management** via YAML/JSON
- **Automated report generation** 
- **Organized output structure**

## 📦 Package Structure

```
quantum_simulation/
├── config.py          # Configuration management
├── hamiltonians.py     # Quantum system setup
├── solvers.py          # Time evolution algorithms  
├── analysis.py         # Results analysis tools
├── visualization.py    # Publication-quality plotting
├── main.py            # Workflow orchestration
├── __main__.py        # CLI entry point
└── __init__.py        # Package initialization
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install qiskit qiskit-nature pyscf pennylane matplotlib seaborn pandas scipy numpy

# Clone or download the package
git clone <repository-url>
cd quantum-simulation
```

### Basic Usage

```python
from quantum_simulation import (
    SimulationConfig, 
    create_quantum_system,
    create_exact_solver,
    create_state_analyzer
)

# Create configuration
config = SimulationConfig(
    atom_string="H 0 0 0; H 0 0 0.735",
    time_end=100.0,
    num_time_points=50
)

# Setup quantum system
system = create_quantum_system(config)
ground_state, ground_energy = system.compute_ground_state_vqe()

# Run time evolution
solver = create_exact_solver(system, config)
time_points = config.get_time_points()
evolved_states = solver.evolve(time_points, ground_state)

# Analyze results
analyzer = create_state_analyzer(system, config)
results = analyzer.analyze_dipole_moment_time_series(time_points, evolved_states)
```

### Command Line Interface

```bash
# Basic simulation
python -m quantum_simulation --methods exact trotter1 trotter2

# With configuration file  
python -m quantum_simulation --config my_config.yaml

# Custom parameters
python -m quantum_simulation \
    --molecule "Li 0 0 0; H 0 0 1.6" \
    --basis cc-pvdz \
    --methods exact trotter1 trotter2 magnus \
    --output results_LiH \
    --time-end 200
```

## 📋 Configuration

Create a YAML configuration file:

```yaml
# Molecular system
atom_string: "H 0 0 0; H 0 0 0.735"
basis: "sto3g"

# Time evolution  
time_end: 100.0
num_time_points: 50

# Electric field
e0: 0.01
gamma: 0.25

# VQA parameters
num_layers: 6
max_optimization_steps: 300
```

## 🔬 Available Solvers

| Method | Description | Accuracy | Speed |
|--------|-------------|----------|-------|
| `exact` | ODE integration | Highest | Slowest |
| `trotter1` | 1st order Trotter | Medium | Fast |
| `trotter2` | 2nd order Trotter | High | Medium |
| `magnus` | Magnus expansion | High | Medium |
| `vqa` | VQA compilation | Variable | Slowest |

## 📊 Analysis Capabilities

### Quantum State Analysis
- **Fidelity calculation** between quantum states
- **Expectation value computation** for observables  
- **State overlap** and distance metrics
- **Method comparison** with error analysis

### Spectrum Analysis
- **Fourier transform** of dipole moment time series
- **Absorption spectrum** calculation
- **Signal processing** with damping and windowing
- **Peak identification** and analysis

### Performance Analysis
- **Execution time benchmarking**
- **Success rate tracking**
- **Scalability analysis**
- **Method comparison reports**

## 🎨 Visualization Examples

### Time Evolution Plots
```python
from quantum_simulation import create_time_evolution_plotter

plotter = create_time_evolution_plotter()
fig = plotter.plot_dipole_evolution(times, dipole_data)
fig = plotter.plot_state_fidelity_evolution(times, fidelities)
```

### Spectrum Analysis
```python
from quantum_simulation import create_spectrum_plotter

plotter = create_spectrum_plotter()  
fig = plotter.plot_absorption_spectra(spectrum_data)
```

### Error Analysis
```python
from quantum_simulation import create_error_analysis_plotter

plotter = create_error_analysis_plotter()
fig = plotter.plot_error_metrics_comparison(error_data)
```

## 📁 Output Structure

The workflow automatically organizes results:

```
simulation_results/
├── data/
│   ├── dipole_evolution.csv
│   ├── absorption_spectra.csv
│   └── performance_benchmarks.csv
├── plots/
│   ├── dipole_evolution.png
│   ├── fidelity_evolution.png
│   └── absorption_spectra.png
├── reports/
│   ├── simulation_report.md
│   └── simulation_summary.json
└── simulation_config.json
```

## 🔧 Advanced Usage

### Custom Workflow
```python
from quantum_simulation import SimulationWorkflow

workflow = SimulationWorkflow(config=my_config, output_dir="results")
workflow.run_complete_workflow(['exact', 'trotter2', 'magnus'])
```

### Batch Processing
```python
# Process multiple configurations
configs = [config1, config2, config3]
for i, config in enumerate(configs):
    workflow = SimulationWorkflow(config, f"results_{i}")
    workflow.run_complete_workflow(['exact', 'trotter2'])
```

## 📚 Documentation

Detailed documentation is available in the demo notebooks:

- `refactored_demo.ipynb` - Basic module demonstration
- `solvers_demo.ipynb` - Solver comparison
- `analysis_demo.ipynb` - Analysis tools
- `visualization_demo.ipynb` - Plotting capabilities  
- `main_workflow_demo.ipynb` - Complete workflow

## 🎯 Use Cases

- **Quantum chemistry research** - Molecular dynamics simulation
- **Algorithm benchmarking** - Compare time evolution methods
- **Educational purposes** - Learn quantum simulation techniques
- **Method development** - Test new quantum algorithms
- **Production workflows** - Automated analysis pipelines

## 🤝 Contributing

This package follows professional software development practices:

- **Modular architecture** for easy extension
- **Comprehensive testing** via demo notebooks
- **Type hints** throughout codebase
- **Consistent API design**
- **Detailed documentation**

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

Built using:
- [Qiskit](https://qiskit.org/) - Quantum computing framework
- [PySCF](https://pyscf.org/) - Quantum chemistry calculations
- [PennyLane](https://pennylane.ai/) - Quantum machine learning
- [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Matplotlib](https://matplotlib.org/)

---

**From research notebook to production package** 🚀

*Professional quantum simulation made simple.* 