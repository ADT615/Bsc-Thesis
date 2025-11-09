# Quantum Simulation Analysis Report

Generated: 2025-07-30 16:34:52

Simulation Package: quantum_simulation v1.0.0

## System Configuration
- **Molecule**: H 0 0 0; H 0 0 0.735
- **Basis Set**: sto3g
- **Qubits**: 4
- **Ground Energy**: -1.857275 hartree

## Time Evolution Parameters
- **Time Range**: 0.0 to 50.0 a.u.
- **Time Points**: 20
- **Electric Field**: E₀ = 0.01, Γ = 0.25

## Methods Comparison
- **Trotter1**: MAE = 3.86e-04, Max = 7.83e-04
- **Trotter2**: MAE = 5.51e-05, Max = 1.28e-04
- **Magnus**: MAE = 1.38e-05, Max = 2.49e-05

## Performance Results
- **exact**: ✅ 0.039s
- **trotter1**: ✅ 45.269s
- **trotter2**: ✅ 74.505s
- **magnus**: ✅ 0.185s
- **vqa**: ❌ 0.112s

## Generated Outputs
- **Data Files**: `data/` directory
- **Visualizations**: `plots/` directory
  - dipole_evolution.png
  - dipole_with_error.png
  - fidelity_evolution.png
  - error_metrics.png
  - absorption_spectra.png
  - spectrum_detailed.png
  - performance.png
  - accuracy_vs_performance.png

## Analysis Summary
- **Methods Tested**: 5
- **Successful Runs**: 4
- **Total Runtime**: 133.6 seconds