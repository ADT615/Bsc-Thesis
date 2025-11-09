"""
Entry point for running quantum_simulation as a module.

Usage:
    python -m quantum_simulation --methods exact trotter1 trotter2
    python -m quantum_simulation --config my_config.yaml
"""

from .main import main

if __name__ == "__main__":
    main() 