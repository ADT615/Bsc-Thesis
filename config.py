"""
Configuration file for quantum simulation project
Chứa các constants và parameters cần thiết
"""

import numpy as np

# Physical parameters
GAMMA = 0.25
E0 = 0.01

# Molecular geometry
MOLECULE_GEOMETRY = "H 0 0 0; H 0 0 2.0"
BASIS_SET = "sto3g"
CHARGE = 0
SPIN = 0

# Optimization parameters
OPTIMIZER_MAXITER = 300
ODE_RTOL = 1e-8
ODE_ATOL = 1e-8

# Training parameters
TIME_RANGE = [0, 300]
NUM_TIME_POINTS = 50

# VQA parameters
VQA_NUM_LAYERS = 6
VQA_STEPS = 300
VQA_LEARNING_RATE = 0.01
VQA_ERROR_THRESHOLD = 1e-6

# Ansatz parameters
EXCITATIONS = 'sd'
REPS = 1

# Field parameters
FIELD_TIME_CUTOFF = 200  # Time cutoff for E-field calculation
