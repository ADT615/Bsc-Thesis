"""
Main orchestration script for quantum simulation workflows.

This module provides command-line interface and workflow automation for
comprehensive quantum simulation analysis including:
- Automated simulation running
- Method comparison and benchmarking
- Report generation
- Batch processing capabilities
- Configuration management
- Progress tracking and logging
"""

import argparse
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import traceback
from datetime import datetime
import yaml

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for CLI
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error importing required packages: {e}")
    sys.exit(1)

from .config import SimulationConfig, DEFAULT_CONFIG
from .hamiltonians import create_quantum_system, QuantumSystem
from .solvers import (
    create_exact_solver,
    create_vqa_solver, 
    create_trotter_solver,
    create_magnus_solver
)
from .analysis import (
    create_state_analyzer,
    create_spectrum_analyzer,
    create_performance_analyzer,
    DataExporter
)
from .visualization import (
    create_comprehensive_analysis_plots,
    PlotStyle
)


class SimulationWorkflow:
    """
    Main workflow orchestrator for quantum simulations.
    
    This class manages the complete simulation pipeline from
    setup through analysis and visualization.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None, 
                 output_dir: Optional[Union[str, Path]] = None,
                 log_level: str = 'INFO'):
        """
        Initialize simulation workflow.
        
        Parameters:
        -----------
        config : SimulationConfig, optional
            Simulation configuration
        output_dir : str or Path, optional
            Output directory for results
        log_level : str
            Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.config = config or DEFAULT_CONFIG
        self.output_dir = Path(output_dir) if output_dir else Path("simulation_results")
        self.setup_logging(log_level)
        
        # Create output directory structure
        self.setup_output_directory()
        
        # Initialize components
        self.system: Optional[QuantumSystem] = None
        self.solvers: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.analysis_results: Optional[Any] = None
        
        # Workflow state
        self.workflow_start_time: Optional[float] = None
        self.workflow_status: str = 'initialized'
        
    def setup_logging(self, log_level: str) -> None:
        """Setup logging configuration."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=getattr(logging, log_level.upper()),
                          format=log_format)
        self.logger = logging.getLogger(__name__)
        
    def setup_output_directory(self) -> None:
        """Create output directory structure."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "plots").mkdir(exist_ok=True)
            (self.output_dir / "data").mkdir(exist_ok=True)
            (self.output_dir / "reports").mkdir(exist_ok=True)
            (self.output_dir / "logs").mkdir(exist_ok=True)
            self.logger.info(f"Output directory created: {self.output_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise
    
    def save_configuration(self) -> None:
        """Save current configuration to file."""
        try:
            config_path = self.output_dir / "simulation_config.json"
            config_dict = self.config.to_dict()
            config_dict['timestamp'] = datetime.now().isoformat()
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            self.logger.info(f"Configuration saved to: {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def initialize_system(self) -> None:
        """Initialize quantum system."""
        try:
            self.logger.info("Initializing quantum system...")
            self.system = create_quantum_system(self.config)
            self.logger.info(f"System initialized: {self.system.static_hamiltonian.num_qubits} qubits")
            
            # Compute ground state
            ground_state, ground_energy = self.system.compute_ground_state_vqe()
            self.results['ground_state'] = ground_state
            self.results['ground_energy'] = ground_energy
            self.logger.info(f"Ground state computed: E = {ground_energy:.6f} hartree")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise
    
    def initialize_solvers(self, methods: List[str]) -> None:
        """
        Initialize requested solvers.
        
        Parameters:
        -----------
        methods : List[str]
            List of solver methods to initialize
        """
        self.logger.info(f"Initializing solvers: {methods}")
        
        solver_factories = {
            'exact': create_exact_solver,
            'trotter1': lambda sys, cfg: create_trotter_solver(sys, cfg, order=1),
            'trotter2': lambda sys, cfg: create_trotter_solver(sys, cfg, order=2),
            'magnus': create_magnus_solver,
            'vqa': create_vqa_solver
        }
        
        for method in methods:
            if method in solver_factories:
                try:
                    self.solvers[method] = solver_factories[method](self.system, self.config)
                    self.logger.info(f"✅ {method.capitalize()} solver initialized")
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize {method} solver: {e}")
            else:
                self.logger.warning(f"Unknown solver method: {method}")
    
    def run_simulations(self) -> None:
        """Run all initialized solvers."""
        if not self.solvers:
            raise ValueError("No solvers initialized")
        
        if 'ground_state' not in self.results:
            raise ValueError("Ground state not computed")
        
        time_points = self.config.get_time_points()
        ground_state = self.results['ground_state']
        
        self.logger.info(f"Running simulations for {len(self.solvers)} methods...")
        self.logger.info(f"Time evolution: {len(time_points)} points from {time_points[0]} to {time_points[-1]}")
        
        simulation_results = {}
        performance_data = []
        
        for method_name, solver in self.solvers.items():
            self.logger.info(f"Running {method_name}...")
            start_time = time.time()
            
            try:
                evolved_states = solver.evolve(time_points, ground_state)
                wall_time = time.time() - start_time
                
                simulation_results[method_name] = evolved_states
                performance_data.append({
                    'solver_name': method_name,
                    'wall_time': wall_time,
                    'success': True,
                    'error_message': None,
                    'num_time_points': len(time_points)
                })
                
                self.logger.info(f"✅ {method_name} completed in {wall_time:.3f}s")
                
            except Exception as e:
                wall_time = time.time() - start_time
                performance_data.append({
                    'solver_name': method_name,
                    'wall_time': wall_time,
                    'success': False,
                    'error_message': str(e),
                    'num_time_points': len(time_points)
                })
                self.logger.error(f"❌ {method_name} failed: {e}")
        
        self.results['evolved_states'] = simulation_results
        self.results['performance_data'] = performance_data
        self.results['time_points'] = time_points
        
        successful_methods = [r['solver_name'] for r in performance_data if r['success']]
        self.logger.info(f"Simulations completed: {len(successful_methods)}/{len(self.solvers)} successful")
    
    def run_analysis(self) -> None:
        """Run comprehensive analysis of simulation results."""
        if 'evolved_states' not in self.results:
            raise ValueError("No simulation results available for analysis")
        
        self.logger.info("Running comprehensive analysis...")
        
        try:
            # State analysis
            state_analyzer = create_state_analyzer(self.system, self.config)
            comparison_results = state_analyzer.compare_methods(
                self.results['evolved_states'], 
                reference_method='exact'
            )
            self.analysis_results = comparison_results
            self.logger.info("✅ State analysis completed")
            
            # Spectrum analysis
            spectrum_analyzer = create_spectrum_analyzer(self.system, self.config)
            spectrum_results = {}
            
            for method, dipole_vals in comparison_results.data['dipole_moments'].items():
                try:
                    spectrum_data = spectrum_analyzer.analyze_absorption_spectrum(
                        times=np.array(self.results['time_points']),
                        dipole_moments=dipole_vals,
                        normalize=True
                    )
                    spectrum_results[method] = spectrum_data
                except Exception as e:
                    self.logger.warning(f"Spectrum analysis failed for {method}: {e}")
            
            self.results['spectrum_analysis'] = spectrum_results
            self.logger.info(f"✅ Spectrum analysis completed for {len(spectrum_results)} methods")
            
            # Performance analysis
            if self.results['performance_data']:
                perf_analyzer = create_performance_analyzer()
                for perf_data in self.results['performance_data']:
                    perf_analyzer.benchmark_results.append(perf_data)
                
                performance_report = perf_analyzer.generate_performance_report(comparison_results)
                self.results['performance_analysis'] = performance_report
                self.logger.info("✅ Performance analysis completed")
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise
    
    def generate_visualizations(self) -> None:
        """Generate comprehensive visualizations."""
        if not self.analysis_results:
            raise ValueError("No analysis results available for visualization")
        
        self.logger.info("Generating visualizations...")
        
        try:
            plot_style = PlotStyle(
                figsize=(12, 8),
                dpi=300,
                save_format='png',
                save_dpi=300
            )
            
            plots_dir = self.output_dir / "plots"
            spectrum_data = self.results.get('spectrum_analysis')
            performance_df = None
            
            if self.results.get('performance_analysis'):
                performance_df = self.results['performance_analysis']['performance_summary']
            
            # Generate comprehensive plots
            figures = create_comprehensive_analysis_plots(
                analysis_results=self.analysis_results,
                spectrum_data=spectrum_data,
                performance_df=performance_df,
                output_dir=plots_dir,
                style=plot_style
            )
            
            self.results['generated_plots'] = list(figures.keys())
            self.logger.info(f"✅ Generated {len(figures)} visualization plots")
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            raise
    
    def export_data(self) -> None:
        """Export analysis results to various formats."""
        if not self.analysis_results:
            raise ValueError("No analysis results available for export")
        
        self.logger.info("Exporting data...")
        data_dir = self.output_dir / "data"
        
        try:
            # Export comparison results
            DataExporter.export_comparison_results(
                self.analysis_results,
                data_dir / "comparison_results.csv",
                format='csv'
            )
            
            # Export time series
            dipole_data = {
                'time': self.analysis_results.data['times'],
                **{f'dipole_{method}': vals 
                   for method, vals in self.analysis_results.data['dipole_moments'].items()}
            }
            DataExporter.export_time_series_csv(dipole_data, data_dir / "dipole_evolution.csv")
            
            # Export spectrum data
            if self.results.get('spectrum_analysis'):
                spectrum_export = {}
                for method, spectrum_data in self.results['spectrum_analysis'].items():
                    spectrum_export[f'energy_eV'] = spectrum_data['omega_ev']
                    spectrum_export[f'spectrum_{method}'] = spectrum_data['absorption_spectrum']
                
                DataExporter.export_time_series_csv(
                    spectrum_export, 
                    data_dir / "absorption_spectra.csv",
                    time_column='energy_eV'
                )
            
            # Export performance data
            if self.results.get('performance_data'):
                perf_df = pd.DataFrame(self.results['performance_data'])
                perf_df.to_csv(data_dir / "performance_benchmarks.csv", index=False)
            
            self.logger.info("✅ Data export completed")
            
        except Exception as e:
            self.logger.error(f"Data export failed: {e}")
            raise
    
    def generate_report(self) -> None:
        """Generate comprehensive analysis report."""
        self.logger.info("Generating analysis report...")
        
        try:
            report_path = self.output_dir / "reports" / "simulation_report.md"
            
            with open(report_path, 'w') as f:
                f.write(self._create_markdown_report())
            
            # Also generate JSON summary
            summary_path = self.output_dir / "reports" / "simulation_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(self._create_json_summary(), f, indent=2)
            
            self.logger.info(f"✅ Reports generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise
    
    def _create_markdown_report(self) -> str:
        """Create comprehensive markdown report."""
        report = []
        report.append("# Quantum Simulation Analysis Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nSimulation Package: quantum_simulation v1.0.0")
        
        # System configuration
        report.append("\n## System Configuration")
        report.append(f"- **Molecule**: {self.config.atom_string}")
        report.append(f"- **Basis Set**: {self.config.basis}")
        report.append(f"- **Qubits**: {self.system.static_hamiltonian.num_qubits}")
        report.append(f"- **Ground Energy**: {self.results['ground_energy']:.6f} hartree")
        
        # Time evolution parameters
        report.append("\n## Time Evolution Parameters")
        report.append(f"- **Time Range**: {self.config.time_start} to {self.config.time_end} a.u.")
        report.append(f"- **Time Points**: {self.config.num_time_points}")
        report.append(f"- **Electric Field**: E₀ = {self.config.e0}, Γ = {self.config.gamma}")
        
        # Methods comparison
        if self.analysis_results:
            report.append("\n## Methods Comparison")
            error_data = self.analysis_results.data.get('dipole_errors', {})
            for method, errors in error_data.items():
                report.append(f"- **{method.capitalize()}**: MAE = {errors['mae']:.2e}, Max = {errors['max_error']:.2e}")
        
        # Performance results
        if self.results.get('performance_data'):
            report.append("\n## Performance Results")
            for perf in self.results['performance_data']:
                status = "✅" if perf['success'] else "❌"
                report.append(f"- **{perf['solver_name']}**: {status} {perf['wall_time']:.3f}s")
        
        # Generated outputs
        report.append("\n## Generated Outputs")
        report.append("- **Data Files**: `data/` directory")
        report.append("- **Visualizations**: `plots/` directory")
        if self.results.get('generated_plots'):
            for plot_name in self.results['generated_plots']:
                report.append(f"  - {plot_name}.png")
        
        report.append("\n## Analysis Summary")
        successful_methods = len([p for p in self.results.get('performance_data', []) if p['success']])
        total_methods = len(self.results.get('performance_data', []))
        report.append(f"- **Methods Tested**: {total_methods}")
        report.append(f"- **Successful Runs**: {successful_methods}")
        
        if self.workflow_start_time:
            total_time = time.time() - self.workflow_start_time
            report.append(f"- **Total Runtime**: {total_time:.1f} seconds")
        
        return '\n'.join(report)
    
    def _create_json_summary(self) -> Dict[str, Any]:
        """Create JSON summary of results."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config.to_dict(),
            'system_info': self.system.get_system_info() if self.system else {},
            'results_summary': {}
        }
        
        if self.analysis_results:
            summary['results_summary']['error_metrics'] = self.analysis_results.data.get('dipole_errors', {})
            summary['results_summary']['num_methods'] = self.analysis_results.metadata.get('num_methods', 0)
        
        if self.results.get('performance_data'):
            summary['results_summary']['performance'] = self.results['performance_data']
        
        summary['results_summary']['output_files'] = {
            'data_directory': str(self.output_dir / "data"),
            'plots_directory': str(self.output_dir / "plots"),
            'reports_directory': str(self.output_dir / "reports")
        }
        
        return summary
    
    def run_complete_workflow(self, methods: List[str]) -> None:
        """
        Run complete simulation workflow.
        
        Parameters:
        -----------
        methods : List[str]
            List of solver methods to run
        """
        self.workflow_start_time = time.time()
        self.workflow_status = 'running'
        
        try:
            self.logger.info("🚀 Starting complete quantum simulation workflow")
            
            # Save configuration
            self.save_configuration()
            
            # Initialize system
            self.initialize_system()
            
            # Initialize and run solvers
            self.initialize_solvers(methods)
            self.run_simulations()
            
            # Analysis
            self.run_analysis()
            
            # Visualization
            self.generate_visualizations()
            
            # Data export
            self.export_data()
            
            # Report generation
            self.generate_report()
            
            # Final summary
            total_time = time.time() - self.workflow_start_time
            self.workflow_status = 'completed'
            
            self.logger.info(f"🎉 Workflow completed successfully in {total_time:.1f} seconds")
            self.logger.info(f"📁 Results saved to: {self.output_dir}")
            
        except Exception as e:
            self.workflow_status = 'failed'
            self.logger.error(f"💥 Workflow failed: {e}")
            self.logger.debug(traceback.format_exc())
            raise


def create_config_from_args(args: argparse.Namespace) -> SimulationConfig:
    """Create simulation configuration from command line arguments."""
    config_kwargs = {}
    
    # Map command line arguments to configuration parameters
    arg_mapping = {
        'molecule': 'atom_string',
        'basis': 'basis',
        'time_end': 'time_end',
        'num_time_points': 'num_time_points',
        'e0': 'e0',
        'gamma': 'gamma',
        'vqa_layers': 'num_layers',
        'vqa_steps': 'max_optimization_steps'
    }
    
    for arg_name, config_name in arg_mapping.items():
        if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
            config_kwargs[config_name] = getattr(args, arg_name)
    
    return SimulationConfig(**config_kwargs)


def load_config_from_file(config_file: Union[str, Path]) -> SimulationConfig:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError("Configuration file must be YAML (.yml, .yaml) or JSON (.json)")
    
    return SimulationConfig(**config_dict)


def main():
    """Main entry point for quantum simulation CLI."""
    parser = argparse.ArgumentParser(
        description="Quantum Simulation Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation with default settings
  python -m quantum_simulation.main --methods exact trotter1 trotter2
  
  # Custom molecule and parameters
  python -m quantum_simulation.main --molecule "Li 0 0 0; H 0 0 1.6" --basis cc-pvdz --methods exact vqa
  
  # Load configuration from file
  python -m quantum_simulation.main --config simulation_config.yaml
  
  # Full analysis with custom output directory
  python -m quantum_simulation.main --methods exact trotter1 trotter2 magnus vqa --output results_H2 --time-end 100
        """
    )
    
    # Configuration options
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', type=str, 
                            help='Load configuration from YAML/JSON file')
    config_group.add_argument('--molecule', type=str,
                            help='Molecule specification (default: H2)')
    config_group.add_argument('--basis', type=str,
                            help='Basis set (default: sto3g)')
    config_group.add_argument('--time-end', type=float, dest='time_end',
                            help='End time for simulation (default: 300)')
    config_group.add_argument('--num-time-points', type=int, dest='num_time_points',
                            help='Number of time points (default: 50)')
    config_group.add_argument('--e0', type=float,
                            help='Electric field strength (default: 0.01)')
    config_group.add_argument('--gamma', type=float,
                            help='Field pulse width (default: 0.25)')
    
    # VQA specific options
    vqa_group = parser.add_argument_group('VQA Options')
    vqa_group.add_argument('--vqa-layers', type=int, dest='vqa_layers',
                         help='Number of VQA layers (default: 6)')
    vqa_group.add_argument('--vqa-steps', type=int, dest='vqa_steps',
                         help='Max VQA optimization steps (default: 300)')
    
    # Solver selection
    parser.add_argument('--methods', nargs='+', 
                       choices=['exact', 'trotter1', 'trotter2', 'magnus', 'vqa'],
                       default=['exact', 'trotter1', 'trotter2'],
                       help='Simulation methods to run')
    
    # Output options
    parser.add_argument('--output', type=str, default='simulation_results',
                       help='Output directory (default: simulation_results)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    # Workflow options
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip analysis step (only run simulations)')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--skip-export', action='store_true',
                       help='Skip data export')
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        if args.config:
            config = load_config_from_file(args.config)
            print(f"📋 Loaded configuration from: {args.config}")
        else:
            config = create_config_from_args(args)
            print(f"📋 Using configuration from command line arguments")
        
        # Create workflow
        workflow = SimulationWorkflow(
            config=config,
            output_dir=args.output,
            log_level=args.log_level
        )
        
        print(f"🚀 Starting quantum simulation workflow")
        print(f"📁 Output directory: {workflow.output_dir}")
        print(f"🔬 Methods: {', '.join(args.methods)}")
        
        # Run workflow
        if args.skip_analysis:
            workflow.initialize_system()
            workflow.initialize_solvers(args.methods)
            workflow.run_simulations()
        else:
            workflow.run_complete_workflow(args.methods)
        
        print(f"\n🎉 Simulation completed successfully!")
        print(f"📊 Check results in: {workflow.output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Simulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Simulation failed: {e}")
        if args.log_level == 'DEBUG':
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 