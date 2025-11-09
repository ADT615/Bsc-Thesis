"""
Visualization tools for quantum simulation results.

This module provides comprehensive plotting capabilities including:
- Time evolution visualization (dipole moments, states)
- Method comparison plots
- Absorption spectrum visualization
- Error analysis plots
- Performance benchmarking charts
- Publication-quality figure generation
- Interactive plotting capabilities
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd
from pathlib import Path
import warnings
from dataclasses import dataclass

# Optional imports for enhanced visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .config import SimulationConfig, DEFAULT_CONFIG, HARTREE_TO_EV
from .analysis import AnalysisResults


@dataclass
class PlotStyle:
    """Configuration for plot styling."""
    figsize: Tuple[float, float] = (10, 6)
    dpi: int = 300
    style: str = 'seaborn-v0_8'
    color_palette: str = 'Set2'
    font_size: int = 12
    title_size: int = 14
    label_size: int = 12
    legend_size: int = 10
    line_width: float = 2.0
    marker_size: float = 6.0
    grid_alpha: float = 0.3
    save_format: str = 'png'
    save_dpi: int = 300
    transparent: bool = False


class QuantumPlotter:
    """
    Base class for quantum simulation plotting.
    
    This class provides common functionality and styling for
    all quantum simulation visualizations.
    """
    
    def __init__(self, style: Optional[PlotStyle] = None):
        """
        Initialize plotter with styling configuration.
        
        Parameters:
        -----------
        style : PlotStyle, optional
            Plotting style configuration
        """
        self.style = style or PlotStyle()
        self._setup_matplotlib_style()
        
        # Color palettes for different methods
        self.method_colors = {
            'exact': '#1f77b4',      # Blue
            'trotter1': '#ff7f0e',   # Orange  
            'trotter2': '#2ca02c',   # Green
            'magnus': '#d62728',     # Red
            'magnus2': '#d62728',    # Red
            'vqa': '#9467bd',        # Purple
            'compilation': '#9467bd'  # Purple
        }
        
        # Line styles for different methods
        self.method_linestyles = {
            'exact': '-',
            'trotter1': '--',
            'trotter2': '-.',
            'magnus': ':',
            'magnus2': ':',
            'vqa': (0, (3, 1, 1, 1)),
            'compilation': (0, (3, 1, 1, 1))
        }
        
        # Marker styles
        self.method_markers = {
            'exact': 'o',
            'trotter1': 's',
            'trotter2': '^',
            'magnus': 'D',
            'magnus2': 'D',
            'vqa': 'v',
            'compilation': 'v'
        }
    
    def _setup_matplotlib_style(self) -> None:
        """Setup matplotlib styling."""
        try:
            plt.style.use(self.style.style)
        except OSError:
            # Fallback to default if style not available
            plt.style.use('default')
        
        # Set font sizes
        plt.rcParams.update({
            'font.size': self.style.font_size,
            'axes.titlesize': self.style.title_size,
            'axes.labelsize': self.style.label_size,
            'legend.fontsize': self.style.legend_size,
            'lines.linewidth': self.style.line_width,
            'lines.markersize': self.style.marker_size,
            'figure.dpi': self.style.dpi,
            'savefig.dpi': self.style.save_dpi,
            'savefig.transparent': self.style.transparent
        })
        
        # Set color palette
        sns.set_palette(self.style.color_palette)
    
    def _get_method_style(self, method: str) -> Dict[str, Any]:
        """Get styling for a specific method."""
        method_lower = method.lower()
        
        return {
            'color': self.method_colors.get(method_lower, '#333333'),
            'linestyle': self.method_linestyles.get(method_lower, '-'),
            'marker': self.method_markers.get(method_lower, 'o'),
            'linewidth': self.style.line_width,
            'markersize': self.style.marker_size,
            'alpha': 0.8
        }
    
    def save_figure(self, fig: Figure, filepath: Union[str, Path], 
                   format: Optional[str] = None, **kwargs) -> None:
        """
        Save figure to file with proper formatting.
        
        Parameters:
        -----------
        fig : Figure
            Matplotlib figure to save
        filepath : str or Path
            Output file path
        format : str, optional
            File format (png, pdf, svg, eps)
        **kwargs
            Additional arguments for savefig
        """
        if format is None:
            format = self.style.save_format
        
        save_kwargs = {
            'dpi': self.style.save_dpi,
            'bbox_inches': 'tight',
            'transparent': self.style.transparent,
            'format': format
        }
        save_kwargs.update(kwargs)
        
        fig.savefig(filepath, **save_kwargs)
        print(f"Figure saved to: {filepath}")


class TimeEvolutionPlotter(QuantumPlotter):
    """
    Plotter for time evolution results.
    
    This class provides specialized plotting for time-dependent
    quantum simulation results.
    """
    
    def plot_dipole_evolution(self, 
                             times: npt.NDArray[np.float64],
                             dipole_data: Dict[str, npt.NDArray[np.float64]], 
                             title: str = "Dipole Moment Evolution",
                             save_path: Optional[Union[str, Path]] = None) -> Figure:
        """
        Plot dipole moment evolution over time.
        
        Parameters:
        -----------
        times : np.ndarray
            Time points
        dipole_data : Dict[str, np.ndarray]  
            Dictionary mapping method names to dipole values
        title : str
            Plot title
        save_path : str or Path, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.style.figsize)
        
        for method, dipole_vals in dipole_data.items():
            style = self._get_method_style(method)
            ax.plot(times, dipole_vals, label=method.capitalize(), **style)
        
        ax.set_xlabel('Time (a.u.)')
        ax.set_ylabel('Dipole moment (a.u.)')
        ax.set_title(title)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=self.style.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig
    
    def plot_dipole_comparison_with_error(self,
                                         times: npt.NDArray[np.float64],
                                         dipole_data: Dict[str, npt.NDArray[np.float64]],
                                         reference_method: str = 'exact',
                                         title: str = "Dipole Evolution with Error Analysis",
                                         save_path: Optional[Union[str, Path]] = None) -> Figure:
        """
        Plot dipole evolution with error analysis subplot.
        
        Parameters:
        -----------
        times : np.ndarray
            Time points
        dipole_data : Dict[str, np.ndarray]
            Dictionary mapping method names to dipole values
        reference_method : str
            Reference method for error calculation
        title : str
            Plot title
        save_path : str or Path, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.style.figsize[0], self.style.figsize[1]*1.5))
        
        reference_data = dipole_data.get(reference_method)
        if reference_data is None:
            raise ValueError(f"Reference method '{reference_method}' not found in data")
        
        # Top plot: Dipole evolution
        for method, dipole_vals in dipole_data.items():
            style = self._get_method_style(method)
            ax1.plot(times, dipole_vals, label=method.capitalize(), **style)
        
        ax1.set_ylabel('Dipole moment (a.u.)')
        ax1.set_title(title)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=self.style.grid_alpha)
        
        # Bottom plot: Error analysis
        for method, dipole_vals in dipole_data.items():
            if method != reference_method:
                error = np.abs(dipole_vals - reference_data)
                style = self._get_method_style(method)
                ax2.semilogy(times, error, label=f'{method.capitalize()} error', **style)
        
        ax2.set_xlabel('Time (a.u.)')
        ax2.set_ylabel('Absolute Error (log scale)')
        ax2.set_title(f'Absolute Error vs {reference_method.capitalize()}')
        ax2.legend(frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=self.style.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig
    
    def plot_state_fidelity_evolution(self,
                                     times: npt.NDArray[np.float64],
                                     fidelity_data: Dict[str, npt.NDArray[np.float64]],
                                     reference_method: str = 'exact',
                                     title: str = "State Fidelity Evolution",
                                     save_path: Optional[Union[str, Path]] = None) -> Figure:
        """
        Plot quantum state fidelity evolution over time.
        
        Parameters:
        -----------
        times : np.ndarray
            Time points
        fidelity_data : Dict[str, np.ndarray]
            Dictionary mapping method names to fidelity values
        reference_method : str
            Reference method name
        title : str
            Plot title
        save_path : str or Path, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.style.figsize)
        
        for method, fidelity_vals in fidelity_data.items():
            if method != reference_method:
                style = self._get_method_style(method)
                ax.plot(times, fidelity_vals, label=f'{method.capitalize()} vs {reference_method}', **style)
        
        ax.set_xlabel('Time (a.u.)')
        ax.set_ylabel('Fidelity')
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=self.style.grid_alpha)
        
        # Add horizontal line at perfect fidelity
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Perfect fidelity')
        
        plt.tight_layout()
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig


class SpectrumPlotter(QuantumPlotter):
    """
    Plotter for frequency domain and absorption spectrum results.
    
    This class provides specialized plotting for spectroscopic analysis.
    """
    
    def plot_absorption_spectra(self,
                               spectrum_data: Dict[str, Dict[str, npt.NDArray]],
                               title: str = "Absorption Spectra Comparison",
                               energy_range: Optional[Tuple[float, float]] = None,
                               save_path: Optional[Union[str, Path]] = None) -> Figure:
        """
        Plot absorption spectra for multiple methods.
        
        Parameters:
        -----------
        spectrum_data : Dict[str, Dict[str, np.ndarray]]
            Dictionary mapping method names to spectrum dictionaries
            Each spectrum dict should have 'omega_ev' and 'absorption_spectrum'
        title : str
            Plot title
        energy_range : Tuple[float, float], optional
            Energy range to plot (eV)
        save_path : str or Path, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.style.figsize)
        
        for method, spectrum_dict in spectrum_data.items():
            omega_ev = spectrum_dict['omega_ev']
            sigma = spectrum_dict['absorption_spectrum']
            
            # Apply energy range filter if specified
            if energy_range:
                mask = (omega_ev >= energy_range[0]) & (omega_ev <= energy_range[1])
                omega_ev = omega_ev[mask]
                sigma = sigma[mask]
            
            style = self._get_method_style(method)
            ax.plot(omega_ev, sigma, label=method.capitalize(), **style)
        
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Normalized Absorption')
        ax.set_title(title)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=self.style.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig
    
    def plot_spectrum_comparison_detailed(self,
                                        spectrum_data: Dict[str, Dict[str, npt.NDArray]],
                                        title: str = "Detailed Spectrum Analysis",
                                        save_path: Optional[Union[str, Path]] = None) -> Figure:
        """
        Plot detailed spectrum comparison with zoomed regions.
        
        Parameters:
        -----------
        spectrum_data : Dict[str, Dict[str, np.ndarray]]
            Dictionary mapping method names to spectrum dictionaries
        title : str
            Plot title
        save_path : str or Path, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(self.style.figsize[0]*1.5, self.style.figsize[1]*1.2))
        
        # Create subplot layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])  # Full spectrum
        ax2 = fig.add_subplot(gs[1, 0])  # Low energy zoom
        ax3 = fig.add_subplot(gs[1, 1])  # Peak region zoom
        
        # Full spectrum plot
        for method, spectrum_dict in spectrum_data.items():
            omega_ev = spectrum_dict['omega_ev']
            sigma = spectrum_dict['absorption_spectrum']
            style = self._get_method_style(method)
            ax1.plot(omega_ev, sigma, label=method.capitalize(), **style)
        
        ax1.set_xlabel('Energy (eV)')
        ax1.set_ylabel('Normalized Absorption')
        ax1.set_title(f'{title} - Full Spectrum')
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=self.style.grid_alpha)
        
        # Low energy zoom (0-1 eV)
        for method, spectrum_dict in spectrum_data.items():
            omega_ev = spectrum_dict['omega_ev']
            sigma = spectrum_dict['absorption_spectrum']
            mask = (omega_ev >= 0) & (omega_ev <= 1)
            if np.any(mask):
                style = self._get_method_style(method)
                ax2.plot(omega_ev[mask], sigma[mask], **style)
        
        ax2.set_xlabel('Energy (eV)')
        ax2.set_ylabel('Normalized Absorption')  
        ax2.set_title('Low Energy Region (0-1 eV)')
        ax2.grid(True, alpha=self.style.grid_alpha)
        
        # Peak region zoom (find peak automatically)
        peak_energies = []
        for spectrum_dict in spectrum_data.values():
            omega_ev = spectrum_dict['omega_ev']
            sigma = spectrum_dict['absorption_spectrum']
            peak_idx = np.argmax(sigma)
            peak_energies.append(omega_ev[peak_idx])
        
        if peak_energies:
            center_energy = np.mean(peak_energies)
            zoom_range = 0.5  # ±0.5 eV around peak
            
            for method, spectrum_dict in spectrum_data.items():
                omega_ev = spectrum_dict['omega_ev']
                sigma = spectrum_dict['absorption_spectrum']
                mask = (omega_ev >= center_energy - zoom_range) & (omega_ev <= center_energy + zoom_range)
                if np.any(mask):
                    style = self._get_method_style(method)
                    ax3.plot(omega_ev[mask], sigma[mask], **style)
            
            ax3.set_xlabel('Energy (eV)')
            ax3.set_ylabel('Normalized Absorption')
            ax3.set_title(f'Peak Region ({center_energy-zoom_range:.1f}-{center_energy+zoom_range:.1f} eV)')
            ax3.grid(True, alpha=self.style.grid_alpha)
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig


class ErrorAnalysisPlotter(QuantumPlotter):
    """
    Plotter for error analysis and method comparison.
    
    This class provides specialized plotting for accuracy analysis
    and method performance comparison.
    """
    
    def plot_error_metrics_comparison(self,
                                    error_data: Dict[str, Dict[str, float]],
                                    title: str = "Error Metrics Comparison",
                                    save_path: Optional[Union[str, Path]] = None) -> Figure:
        """
        Plot error metrics comparison between methods.
        
        Parameters:
        -----------
        error_data : Dict[str, Dict[str, float]]
            Dictionary mapping method names to error metrics
            Each error dict should have 'mae', 'max_error', 'rmse'
        title : str
            Plot title
        save_path : str or Path, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(error_data).T
        
        fig, axes = plt.subplots(1, 3, figsize=(self.style.figsize[0]*1.8, self.style.figsize[1]))
        
        error_types = ['mae', 'max_error', 'rmse']
        error_labels = ['Mean Absolute Error', 'Maximum Error', 'Root Mean Square Error']
        
        for i, (error_type, label) in enumerate(zip(error_types, error_labels)):
            if error_type in df.columns:
                bars = axes[i].bar(df.index, df[error_type], 
                                 color=[self.method_colors.get(method.lower(), '#333333') 
                                       for method in df.index])
                axes[i].set_ylabel(label)
                axes[i].set_title(f'{label} by Method')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].set_yscale('log')
                axes[i].grid(True, alpha=self.style.grid_alpha)
                
                # Add value labels on bars
                for bar, value in zip(bars, df[error_type]):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.2e}', ha='center', va='bottom', rotation=0)
        
        plt.suptitle(title, fontsize=self.style.title_size)
        plt.tight_layout()
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig
    
    def plot_accuracy_vs_performance(self,
                                    accuracy_data: Dict[str, float],
                                    performance_data: Dict[str, float],
                                    title: str = "Accuracy vs Performance Tradeoff",
                                    save_path: Optional[Union[str, Path]] = None) -> Figure:
        """
        Plot accuracy vs performance tradeoff scatter plot.
        
        Parameters:
        -----------
        accuracy_data : Dict[str, float]
            Dictionary mapping method names to accuracy metrics (lower is better)
        performance_data : Dict[str, float]
            Dictionary mapping method names to performance metrics (time in seconds)
        title : str
            Plot title
        save_path : str or Path, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.style.figsize)
        
        methods = set(accuracy_data.keys()) & set(performance_data.keys())
        
        for method in methods:
            accuracy = accuracy_data[method]
            performance = performance_data[method]
            style = self._get_method_style(method)
            
            ax.scatter(performance, accuracy, 
                      color=style['color'], 
                      marker=style['marker'],
                      s=self.style.marker_size**2,
                      alpha=0.8,
                      label=method.capitalize())
            
            # Add method label
            ax.annotate(method.capitalize(), 
                       (performance, accuracy),
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=self.style.font_size-1)
        
        ax.set_xlabel('Computation Time (s)')
        ax.set_ylabel('Error (log scale)')
        ax.set_yscale('log')
        ax.set_title(title)
        ax.grid(True, alpha=self.style.grid_alpha)
        
        # Add Pareto frontier line (conceptual)
        sorted_methods = sorted(methods, key=lambda x: performance_data[x])
        if len(sorted_methods) > 1:
            pareto_x = [performance_data[m] for m in sorted_methods]
            pareto_y = [accuracy_data[m] for m in sorted_methods]
            ax.plot(pareto_x, pareto_y, '--', alpha=0.5, color='gray', 
                   label='Performance trend')
        
        ax.legend(frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig


class PerformancePlotter(QuantumPlotter):
    """
    Plotter for performance benchmarking results.
    
    This class provides specialized plotting for timing analysis
    and performance comparison.
    """
    
    def plot_performance_comparison(self,
                                  performance_df: pd.DataFrame,
                                  title: str = "Performance Comparison",
                                  save_path: Optional[Union[str, Path]] = None) -> Figure:
        """
        Plot performance comparison bar chart.
        
        Parameters:
        -----------
        performance_df : pd.DataFrame
            Performance data from PerformanceAnalyzer
        title : str
            Plot title
        save_path : str or Path, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.style.figsize[0]*1.5, self.style.figsize[1]))
        
        # Filter successful runs
        success_df = performance_df[performance_df['success']]
        
        if not success_df.empty:
            # Wall time comparison
            bars1 = ax1.bar(success_df['solver_name'], success_df['wall_time'],
                           color=[self.method_colors.get(name.lower(), '#333333') 
                                 for name in success_df['solver_name']])
            ax1.set_ylabel('Wall Time (s)')
            ax1.set_title('Computation Time by Method')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=self.style.grid_alpha)
            
            # Add value labels on bars
            for bar, value in zip(bars1, success_df['wall_time']):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}s', ha='center', va='bottom')
            
            # Speedup comparison (if available)
            if 'speedup' in success_df.columns:
                bars2 = ax2.bar(success_df['solver_name'], success_df['speedup'],
                               color=[self.method_colors.get(name.lower(), '#333333') 
                                     for name in success_df['solver_name']])
                ax2.set_ylabel('Speedup Factor')
                ax2.set_title('Relative Speedup')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=self.style.grid_alpha)
                ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline')
                
                # Add value labels on bars
                for bar, value in zip(bars2, success_df['speedup']):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.2f}x', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=self.style.title_size)
        plt.tight_layout()
        
        if save_path:
            self.save_figure(fig, save_path)
        
        return fig


# Factory functions for easy plotter creation
def create_time_evolution_plotter(style: Optional[PlotStyle] = None) -> TimeEvolutionPlotter:
    """Create time evolution plotter."""
    return TimeEvolutionPlotter(style)

def create_spectrum_plotter(style: Optional[PlotStyle] = None) -> SpectrumPlotter:
    """Create spectrum plotter."""
    return SpectrumPlotter(style)

def create_error_analysis_plotter(style: Optional[PlotStyle] = None) -> ErrorAnalysisPlotter:
    """Create error analysis plotter."""
    return ErrorAnalysisPlotter(style)

def create_performance_plotter(style: Optional[PlotStyle] = None) -> PerformancePlotter:
    """Create performance plotter."""
    return PerformancePlotter(style)


# Convenient all-in-one plotting function
def create_comprehensive_analysis_plots(analysis_results: AnalysisResults,
                                       spectrum_data: Optional[Dict[str, Dict[str, npt.NDArray]]] = None,
                                       performance_df: Optional[pd.DataFrame] = None,
                                       output_dir: Optional[Union[str, Path]] = None,
                                       style: Optional[PlotStyle] = None) -> Dict[str, Figure]:
    """
    Create comprehensive analysis plots from results.
    
    Parameters:
    -----------
    analysis_results : AnalysisResults
        Results from quantum state analysis
    spectrum_data : Dict, optional
        Spectrum analysis results
    performance_df : pd.DataFrame, optional
        Performance benchmarking results
    output_dir : str or Path, optional
        Directory to save plots
    style : PlotStyle, optional
        Plotting style configuration
        
    Returns:
    --------
    Dict[str, Figure]
        Dictionary mapping plot names to figure objects
    """
    figures = {}
    
    # Create plotters
    time_plotter = create_time_evolution_plotter(style)
    spectrum_plotter = create_spectrum_plotter(style)
    error_plotter = create_error_analysis_plotter(style)
    perf_plotter = create_performance_plotter(style)
    
    # Extract data
    times = np.array(analysis_results.data['times'])
    dipole_data = analysis_results.data['dipole_moments']
    fidelity_data = analysis_results.data['fidelities']
    error_data = analysis_results.data.get('dipole_errors', {})
    
    # Time evolution plots
    figures['dipole_evolution'] = time_plotter.plot_dipole_evolution(
        times, dipole_data,
        save_path=Path(output_dir) / 'dipole_evolution.png' if output_dir else None
    )
    
    figures['dipole_with_error'] = time_plotter.plot_dipole_comparison_with_error(
        times, dipole_data,
        save_path=Path(output_dir) / 'dipole_with_error.png' if output_dir else None
    )
    
    figures['fidelity_evolution'] = time_plotter.plot_state_fidelity_evolution(
        times, fidelity_data,
        save_path=Path(output_dir) / 'fidelity_evolution.png' if output_dir else None
    )
    
    # Error analysis plots
    if error_data:
        figures['error_metrics'] = error_plotter.plot_error_metrics_comparison(
            error_data,
            save_path=Path(output_dir) / 'error_metrics.png' if output_dir else None
        )
    
    # Spectrum plots
    if spectrum_data:
        figures['absorption_spectra'] = spectrum_plotter.plot_absorption_spectra(
            spectrum_data,
            save_path=Path(output_dir) / 'absorption_spectra.png' if output_dir else None
        )
        
        figures['spectrum_detailed'] = spectrum_plotter.plot_spectrum_comparison_detailed(
            spectrum_data,
            save_path=Path(output_dir) / 'spectrum_detailed.png' if output_dir else None
        )
    
    # Performance plots
    if performance_df is not None and not performance_df.empty:
        figures['performance'] = perf_plotter.plot_performance_comparison(
            performance_df,
            save_path=Path(output_dir) / 'performance_comparison.png' if output_dir else None
        )
        
        # Accuracy vs performance tradeoff
        if error_data and 'wall_time' in performance_df.columns:
            accuracy_dict = {method: errors['mae'] for method, errors in error_data.items()}
            performance_dict = dict(zip(performance_df['solver_name'].str.lower(), 
                                      performance_df['wall_time']))
            
            figures['accuracy_vs_performance'] = error_plotter.plot_accuracy_vs_performance(
                accuracy_dict, performance_dict,
                save_path=Path(output_dir) / 'accuracy_vs_performance.png' if output_dir else None
            )
    
    return figures 