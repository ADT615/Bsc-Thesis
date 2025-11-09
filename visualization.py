"""
Visualization module
Chứa các hàm để vẽ biểu đồ và visualization
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_optimization_convergence(costs, fidelities=None, title="Optimization Convergence"):
    """
    Plot optimization convergence
    
    Args:
        costs: Array of cost values
        fidelities: Array of fidelity values (optional)
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2 if fidelities is not None else 1, figsize=(12, 4))
    
    if fidelities is not None:
        # Plot cost
        axes[0].plot(costs, 'b-', linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Cost')
        axes[0].set_title('Cost Function')
        axes[0].grid(True, alpha=0.3)
        
        # Plot fidelity
        axes[1].plot(fidelities, 'r-', linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Fidelity')
        axes[1].set_title('Fidelity')
        axes[1].grid(True, alpha=0.3)
    else:
        axes.plot(costs, 'b-', linewidth=2)
        axes.set_xlabel('Iteration')
        axes.set_ylabel('Cost')
        axes.set_title('Cost Function')
        axes.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_energy_convergence(energies, exact_energy=None, title="Energy Convergence"):
    """
    Plot energy convergence during VQE
    
    Args:
        energies: Array of energy values
        exact_energy: Exact ground state energy for reference
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(energies, 'b-', linewidth=2, label='VQE Energy')
    
    if exact_energy is not None:
        plt.axhline(y=exact_energy, color='r', linestyle='--', linewidth=2, label='Exact Energy')
    
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_fidelity_vs_time(times, fidelities, title="Fidelity vs Time"):
    """
    Plot fidelity as a function of time
    
    Args:
        times: Time array
        fidelities: Fidelity array
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(times, fidelities, 'g-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Time')
    plt.ylabel('Fidelity')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


def plot_electric_field(times, title="Electric Field Profile"):
    """
    Plot electric field as a function of time
    
    Args:
        times: Time array
        title: Plot title
    """
    from time_evolution import E_field
    
    E_values = [E_field(t) for t in times]
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, E_values, 'purple', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Electric Field')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_absorption_spectrum(frequencies, intensities, title="Absorption Spectrum"):
    """
    Plot absorption spectrum
    
    Args:
        frequencies: Frequency array
        intensities: Intensity array
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, intensities, 'b-', linewidth=2)
    plt.xlabel('Frequency (eV)')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_comparison(x_data, y_data_list, labels, title="Comparison Plot", xlabel="X", ylabel="Y"):
    """
    Plot comparison between multiple datasets
    
    Args:
        x_data: X-axis data
        y_data_list: List of Y-axis datasets
        labels: List of labels for each dataset
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    linestyles = ['-', '--', '-.', ':']
    
    for i, (y_data, label) in enumerate(zip(y_data_list, labels)):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        plt.plot(x_data, y_data, color=color, linestyle=linestyle, 
                linewidth=2, label=label, marker='o', markersize=4)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_plot(filename, dpi=300):
    """
    Save current plot to file
    
    Args:
        filename: Output filename
        dpi: Resolution in DPI
    """
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to {filename}")


def setup_matplotlib_style():
    """
    Setup matplotlib style for better plots
    """
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.major.size': 5,
        'ytick.minor.size': 3,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'figure.dpi': 100
    })
