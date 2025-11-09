"""
Time evolution module
Chứa các hàm tính toán time evolution của quantum system
"""

import numpy as np
from scipy.integrate import solve_ivp
import time

from config import E0, GAMMA, FIELD_TIME_CUTOFF, TIME_RANGE, NUM_TIME_POINTS, ODE_RTOL, ODE_ATOL


def E_field(t):
    """
    Electric field as a function of time
    
    Args:
        t: Time
        
    Returns:
        float: Electric field value
    """
    if t < -FIELD_TIME_CUTOFF or t > FIELD_TIME_CUTOFF:
        return 0.0
    return (E0 / np.pi) * GAMMA / (GAMMA**2 + t**2)


def H_t_matrix(t, H0_matrix, Dz_matrix):
    """
    Time-dependent Hamiltonian matrix
    
    Args:
        t: Time
        H0_matrix: Static Hamiltonian matrix
        Dz_matrix: Dipole matrix
        
    Returns:
        ndarray: Time-dependent Hamiltonian matrix
    """
    return H0_matrix + E_field(t) * Dz_matrix


def unitary_rhs(t, U_flat, H0_matrix, Dz_matrix):
    """
    Right-hand side of the differential equation for unitary evolution
    dU/dt = -i * H(t) * U
    
    Args:
        t: Time
        U_flat: Flattened unitary matrix
        H0_matrix: Static Hamiltonian matrix
        Dz_matrix: Dipole matrix
        
    Returns:
        ndarray: Flattened derivative of unitary matrix
    """
    dim = H0_matrix.shape[0]
    U = U_flat.reshape(dim, dim)  # Convert vector back to matrix
    H = H_t_matrix(t, H0_matrix, Dz_matrix)
    dU_dt = -1j * (H @ U)
    return dU_dt.flatten()  # Convert back to vector for solver


def compute_target_unitaries(H0_matrix, Dz_matrix, times_for_training=None):
    """
    Compute target unitary operators at specified times using ODE solver
    
    Args:
        H0_matrix: Static Hamiltonian matrix
        Dz_matrix: Dipole matrix
        times_for_training: Time points for computation (optional)
        
    Returns:
        list: List of target unitary matrices
        ndarray: Time points used
    """
    if times_for_training is None:
        times_for_training = np.linspace(TIME_RANGE[0], TIME_RANGE[1], NUM_TIME_POINTS)
    
    dim = H0_matrix.shape[0]
    U0_flat = np.eye(dim, dtype=complex).flatten()  # U(0) = I, flattened for solver
    
    t_span = [times_for_training[0], times_for_training[-1]]
    
    print("Bắt đầu tính toán Target Unitaries bằng ODE Solver...")
    start_time = time.time()
    
    sol_U = solve_ivp(
        lambda t, U_flat: unitary_rhs(t, U_flat, H0_matrix, Dz_matrix),
        t_span,
        U0_flat,
        t_eval=times_for_training,
        method='RK45',  # Runge-Kutta method of order 4(5)
        rtol=ODE_RTOL,  # High tolerance for accurate results
        atol=ODE_ATOL
    )
    
    end_time = time.time()
    print(f"Hoàn thành trong {end_time - start_time:.2f} giây.")
    
    # Process results
    target_unitaries_list = []
    if sol_U.success:
        # sol_U.y has shape (dim*dim, N_points)
        unitaries_flat = sol_U.y.T  # Transpose to (N_points, dim*dim)
        for u_flat in unitaries_flat:
            target_unitaries_list.append(u_flat.reshape(dim, dim))
        print(f"Đã tạo thành công {len(target_unitaries_list)} target unitaries.")
    else:
        print("Lỗi: Bộ giải ODE không hội tụ khi tính toán Unitary.")
    
    return target_unitaries_list, times_for_training


def create_time_evolution_function(H_0, dipole_qubit):
    """
    Create a time evolution function for given operators
    
    Args:
        H_0: Static Hamiltonian operator
        dipole_qubit: Dipole operator
        
    Returns:
        function: Time evolution function H(t)
    """
    def H_time(t):
        from quantum_setup import Hamilton_SP
        return Hamilton_SP(t, H_0, dipole_qubit, E0, GAMMA)
    
    return H_time
