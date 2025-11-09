import sys
import matplotlib.pyplot as plt
import numpy as np
from qoop.compilation.qsp import QuantumStatePreparation
from qiskit.quantum_info import SparsePauliOp
import sys, qiskit
import numpy as np
from scipy.linalg import expm
#from qiskit.extensions import UnitaryGate


def trotter_circuit(nqubits, labels, coeffs, t, M):

    # Convert one Trotter decomposition ,e^{iZ_1Z_2*delta}*e^{iZ_2Z_3*delta}*...e^{iZ_nZ_1*delta} to a quantum gate
    circuit = qiskit.QuantumCircuit(nqubits)
    for qubit in range(nqubits//2, nqubits):
        circuit.x(qubit)

    # Time increment range
    delta = 0.5
        
    for i in range(len(labels)):
        # 'IX', 'IZ', 'IY' case
        if labels[i][0] == 'I':
            if labels[i][1] == 'Z':
                circuit.rz(2*delta*coeffs[i],1)
            elif labels[i][1] == 'X':
                circuit.rx(2*delta*coeffs[i],1)
            elif labels[i][1] == 'Y':
                circuit.ry(2*delta*coeffs[i],1)
    
        # 'XI', 'ZI', 'YI' case
        elif labels[i][1] == 'I':
            if labels[i][0] == 'Z':
                circuit.rz(2*delta*coeffs[i],0)
            elif labels[i][0] == 'X':
                circuit.rx(2*delta*coeffs[i],0)
            elif labels[i][0] == 'Y':
                circuit.ry(2*delta*coeffs[i],0)
    
        # # 'XX', 'ZZ', 'YY' case
        elif labels[i] in ['XX', 'YY', 'ZZ']:
            for j in range(nqubits):
                if labels[i][1] == 'Z':
                    #circuit.cx((j+1)%(nqubits),j)
                    circuit.rzz(2*delta*coeffs[i],(j+1)%nqubits, j) ## RZ(a)=exp(i*a/2*Z)
                    #circuit.cx((j+1)%(nqubits),j)
                elif labels[i][1] == 'X':
                    #circuit.cx((j+1)%(nqubits),j)
                    circuit.rxx(2*delta*coeffs[i],(j+1)%nqubits, j) ## RZ(a)=exp(i*a/2*Z)
                    #circuit.cx((j+1)%(nqubits),j)
                elif labels[i][1] == 'Y':
                    #circuit.cx((j+1)%(nqubits),j)
                    circuit.ryy(2*delta*coeffs[i],(j+1)%nqubits, j) ## RZ(a)=exp(i*a/2*Z)
                    #circuit.cx((j+1)%(nqubits),j)
    return circuit


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
    from qoop.core.state import specific_matrix
    
    return specific_matrix(U)

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

# def system coefs
def coefs(mod):
    #return coefs
    N=2
    J=1
    T=10
    #test mod
    if mod == 'mod1':  
        u=0
        h=[0,0]
    elif mod == 'mod2':
        u=-1
        h=[0.25,0.25]
    elif mod == 'mod3':
        u=1
        h=[0.25,0.25]
    elif mod == 'mod4':
        u=0.25
        h=[0.25,0.25]  
    elif mod == 'mod5':
        u=-0.25
        h=[0.25,0.25]  
    elif mod == 'mod6':
        u=-0.25
        h=[0,0]    
    elif mod == 'mod7':
        u=0.0
        h=[0.25,0.25] 
    elif mod == 'mod8': #same as 5
        u=-0.25
        h=[-0.25,-0.25]         
    return N, J, u, h, T     

mod = 'mod5'     
def h_time(t):
    N, J, u, h, T = coefs(mod)
    return XYZ_model(N,J,u,h,T,t)
    
p0s = []
N, J, u, h, T = coefs(mod) 
labels = time_dependent_integral(h_time,t=T).paulis.to_labels()
coeffs = time_dependent_integral(h_time,t=T).coeffs
coeffs = np.real(coeffs)
qc = trotter_circuit(N,labels, coeffs, T, M=100)
times = np.linspace(0,10, 4)
for time in times:
    qsp = QuantumStatePreparation(
        u=qc,
        target_state= time_dependent_qc(N,h_time,time).inverse()
        ).fit(num_steps=30, metrics_func=['loss_basic'])
    p0s.append(1-qsp.compiler.metrics['loss_basic'][-1])

print('Mean loss',p0s)










