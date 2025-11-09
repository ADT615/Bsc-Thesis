from qiskit.quantum_info import SparsePauliOp
import sys, qiskit
import numpy as np
from scipy.linalg import expm
#from qiskit.extensions import UnitaryGate

def ising(N,J,h):
    """create Hamiltonian for Ising model
        H = J * sum_{i = 0}^{N-1) Z_i*Z_{i+1} + h*sum_{i=0}^{N-1}X_i
    """

    pauli_op_list = []  
    for j in range(N):
        s = ('I' * (N - j - 1)) + 'X' + ('I' * j)
        pauli_op_list.append((s, h))

    for j in range(N-1):
        s = ('I' * (N - j - 2)) + 'ZZ' + ('I' * j)
        pauli_op_list.append((s, J))
        
    h_op = SparsePauliOp.from_list(pauli_op_list)

    return h_op

def XYZ_bare(N,J,u,h):
    """create Hamiltonian for Ising model
        H = J * sum_{i = 0}^{N-1) Z_i*Z_{i+1} + h*sum_{i=0}^{N-1}X_i
    """
    

    pauli_op_list = []  
    for j in range(N):
        s = ('I' * (N - j - 1)) + 'X' + ('I' * j)
        pauli_op_list.append((s, h[j]))
    
    for j in range(N-1):
        s = ('I' * (N - j - 2)) + 'ZZ' + ('I' * j)
        pauli_op_list.append((s, u))
    
    for j in range(N-1):
        s = ('I' * (N - j - 2)) + 'XX' + ('I' * j)
        pauli_op_list.append((s, -J))

    for j in range(N-1):
        s = ('I' * (N - j - 2)) + 'YY' + ('I' * j)
        pauli_op_list.append((s, -J))
        
    h_op = SparsePauliOp.from_list(pauli_op_list)

    return h_op

def XYZ_model(N,J,u,h,T,t):
    """create Hamiltonian for Ising model
        H = J * sum_{i = 0}^{N-1) Z_i*Z_{i+1} + h*sum_{i=0}^{N-1}X_i
    """
    

    pauli_op_list = []  
    for j in range(N):
        s = ('I' * (N - j - 1)) + 'X' + ('I' * j)
        pauli_op_list.append((s, h[j]))
    
    for j in range(N-1):
        s = ('I' * (N - j - 2)) + 'ZZ' + ('I' * j)
        pauli_op_list.append((s, u))
    
    for j in range(N-1):
        s = ('I' * (N - j - 2)) + 'XX' + ('I' * j)
        pauli_op_list.append((s, -J*(1-t/T)/2.0))

    for j in range(N-1):
        s = ('I' * (N - j - 2)) + 'YY' + ('I' * j)
        pauli_op_list.append((s, -J*(1+t/T)/2.0))
        
    h_op = SparsePauliOp.from_list(pauli_op_list)

    return h_op

def pauli_oper(N, oper = None):
    
    pauli_op_list = []  
    s = ('I' * (N - 1)) + oper
    pauli_op_list.append((s, 1.0))
        
    h_op = SparsePauliOp.from_list(pauli_op_list)
    return h_op  

def pauli_ij(N, oper, pos):
    pauli_op_list = []
    s = 'I' * N

    for k in range(len(oper)):

        if oper[k] == s[N - pos[k]]:
            s = s[:N - pos[k]] + 'I' + s[N - pos[k] + 1:]
        else:
            s = s[:N - pos[k]] + oper[k] + s[N - pos[k]+ 1:]

    pauli_op_list.append((s, 1.0))

    h_op = SparsePauliOp.from_list(pauli_op_list)
    return h_op 

def pauli_collect(N, oper = None):
    
    pauli_op_list = []  
    for j in range(N):
        s = ('I' * (N - j - 1)) + oper + ('I' * j)
        print(s)
        pauli_op_list.append((s, 1.0))
        
    h_op = SparsePauliOp.from_list(pauli_op_list)
    return h_op  

def time_dependent_qc(num_qubits: int,h_opt, t):
    """create U circuit from h_opt and time t
    
    Args:
        - qc (QuantumCircuit): Init circuit
        - h_opt: Hamiltonian
        - t (float): time
        
    Returns:
        - QuantumCircuit: the added circuit
    """
    #Create circuit
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for qubit in range(num_qubits//2, num_qubits):
        qc.x(qubit)
    
    # Ensure h_opt is Hermitian
    if not np.allclose(h_opt(t).to_matrix(), np.conj(h_opt(t).to_matrix()).T):
        raise ValueError("The Hamiltonian is not Hermitian.")

    time_points = np.linspace(0, t, 100)
    # Calculate the integral of H(t) using numerical approximation (e.g., trapezoidal rule)
    integral = np.zeros_like(h_opt, dtype=complex)  # Initialize integral as a matrix
    
    for i in range(len(time_points) - 1):
        dt = time_points[i + 1] - time_points[i]
        integral += (h_opt(time_points[i]) + h_opt(time_points[i + 1])) / 2 * dt

    # Compute the matrix exponential
    U = expm(-1j * integral)

    # Check if U is unitary
    if not np.allclose(U @ U.conj().T, np.eye(U.shape[0])):
        raise ValueError("The resulting matrix U is not unitary.")

    #return U matrix
    """
    # Create a UnitaryGate from the unitary_matrix
    unitary_gate = UnitaryGate(U)

    # Append the unitary_gate to the quantum circuit
    qc.append(unitary_gate, range(qc.num_qubits))
    """
    return U

def time_dependent_integral(h_opt, t):
    """create U circuit from h_opt and time t
    
    Args:
        - qc (QuantumCircuit): Init circuit
        - h_opt: Hamiltonian
        - t (float): time
        
    Returns:
        - QuantumCircuit: the added circuit
    """
    # Ensure h_opt is Hermitian
    if not np.allclose(h_opt(t).to_matrix(), np.conj(h_opt(t).to_matrix()).T):
        raise ValueError("The Hamiltonian is not Hermitian.")

    time_points = np.linspace(0, t, 100)
    # Calculate the integral of H(t) using numerical approximation (e.g., trapezoidal rule)
    integral = np.zeros_like(h_opt, dtype=complex)  # Initialize integral as a matrix
    
    for i in range(len(time_points) - 1):
        dt = time_points[i + 1] - time_points[i]
        integral += (h_opt(time_points[i]) + h_opt(time_points[i + 1])) / 2 * dt

    return integral